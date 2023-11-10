import os
import sys
from argparse import Namespace

import numpy as np
import tensorflow as tf
import transformers
from coref_dataset import Dataset

sys.path.append("../../")
from corefscorer import evaluate_coreference


class LinearWarmup(tf.optimizers.schedules.LearningRateSchedule):
    def __init__(self, warmup_steps, following_schedule):
        self._warmup_steps = warmup_steps
        self._warmup = tf.optimizers.schedules.PolynomialDecay(0.0, warmup_steps, following_schedule(0))
        self._following = following_schedule

    def __call__(self, step):
        return tf.cond(
            step < self._warmup_steps,
            lambda: self._warmup(step),
            lambda: self._following(step - self._warmup_steps),
        )


class Model(tf.keras.Model):
    def __init__(self, tags, args: Namespace) -> None:
        super().__init__()

        self._tags = tags
        self._args = args

        # Load the BERT model
        self._bert = transformers.TFAutoModel.from_pretrained(args.bert)

        # Initialize the dense layers for the mention scoring
        MULT = 4
        self._dense_hidden_q = tf.keras.layers.Dense(MULT * self._bert.config.hidden_size, activation=tf.nn.relu)
        self._dense_hidden_k = tf.keras.layers.Dense(MULT * self._bert.config.hidden_size, activation=tf.nn.relu)
        self._dense_hidden_tags = tf.keras.layers.Dense(MULT * self._bert.config.hidden_size, activation=tf.nn.relu)

        # Initialize the dense layers for the attention mechanism
        self._dense_q = tf.keras.layers.Dense(self._bert.config.hidden_size, use_bias=False)
        self._dense_k = tf.keras.layers.Dense(self._bert.config.hidden_size, use_bias=False)

        # Initialize the dense layer for the output tags
        self._dense_tags = tf.keras.layers.Dense(len(tags))

    def compile(self, train: tf.data.Dataset) -> None:
        args = self._args
        warmup_steps = int(args.warmup * args.epochs * len(train))
        learning_rate = tf.optimizers.schedules.PolynomialDecay(
            args.learning_rate, args.epochs * len(train) - warmup_steps, 0.0 if args.learning_rate_decay else args.learning_rate
        )
        if warmup_steps:
            lr = LinearWarmup(warmup_steps, learning_rate)
        super().compile(optimizer=tf.optimizers.Adam(learning_rate=lr))
        # describe self model parameters:
        

    @tf.function(experimental_relax_shapes=True)
    def compute_tags(self, subwords, word_indices, training):
        """
        This function computes the output logits for each tag.

        It takes subwords and their word indices as inputs, applies the BERT model to generate embeddings,
        and then uses a Dense layer to produce logits for each tag.

        Args:
            subwords (tf.Tensor): Subwords for each sentence.
            word_indices (tf.Tensor): Indices of subwords that make up a word.
            training (bool): Whether the model is in training mode.

        Returns:
            A tuple containing the word embeddings and the tag logits.
        """

        attention_mask = tf.sequence_mask(subwords.row_lengths())
        embeddings = self._bert(subwords.to_tensor(), attention_mask=attention_mask, training=training).last_hidden_state
        words = tf.gather(embeddings, word_indices[:, :-1], batch_dims=1)
        logits = self._dense_tags(self._dense_hidden_tags(words))
        return embeddings, logits

    @tf.function(experimental_relax_shapes=True)
    def compute_antecedents(self, embeddings, previous, mentions) -> tf.RaggedTensor:
        """
        This function computes the antecedent weights for each mention.

        It takes the word embeddings, the indices of previous mentions, and the current mention indices,
        and computes a score for each possible antecedent.

        Args:
            embeddings (tf.Tensor): Word embeddings for each sentence.
            previous (tf.Tensor): Indices of previous mentions.
            mentions (tf.Tensor): Indices of current mentions.

        Returns:
            A RaggedTensor with the antecedent weights for each mention.
        """
        mentions_embedded = tf.gather(embeddings, mentions, batch_dims=1).values
        mentions_embedded = tf.reshape(mentions_embedded, [-1, np.prod(mentions_embedded.shape[-2:])])

        # Compute the query tensor for the attention mechanism
        queries = mentions.with_values(self._dense_q(self._dense_hidden_q(mentions_embedded)))

        # Compute the key tensor for the mentions
        keys_mentions = mentions.with_values(self._dense_k(self._dense_hidden_k(mentions_embedded)))

        # Embed the previous mentions
        previous_embedded = tf.gather(embeddings, previous, batch_dims=1).values
        previous_embedded = tf.reshape(previous_embedded, [-1, mentions_embedded.shape[-1]])

        # Compute the key tensor for the previous mentions
        keys_previous = previous.with_values(self._dense_k(self._dense_hidden_k(previous_embedded)))

        # Concatenate the keys for the previous mentions and the current mentions
        keys = tf.concat([keys_previous, keys_mentions], axis=1)

        # Compute the attention weights for each possible antecedent
        weights = tf.matmul(queries.to_tensor(), keys.to_tensor(), transpose_b=True) / (self._dense_q.units**0.5)

        # Return the antecedent weights
        return weights

    def train_step(self, data: tuple):
        # Unpack the data from the tuple
        (subwords, word_indices), (tags, previous, mentions, mask, antecedents) = data

        # Create a GradientTape instance to monitor the operations for automatic differentiation
        with tf.GradientTape() as tape:
            # Tagging part

            # Compute the embeddings and logits by calling the compute_tags method
            embeddings, logits = self.compute_tags(subwords, word_indices, True)

            # Calculate tags loss using categorical cross entropy loss function
            tags_loss = tf.losses.CategoricalCrossentropy(
                from_logits=True, label_smoothing=self._args.label_smoothing, reduction=tf.losses.Reduction.SUM
            )(tf.one_hot(tags.values, len(self._tags)), logits.values) / tf.cast(tf.shape(logits.values)[0], tf.float32)

            # Antecedents part
            # Define the function to compute the antecedent loss
            def antecedent_loss():
                # Compute weights by calling compute_antecedents method
                weights = self.compute_antecedents(embeddings, previous, mentions)

                # Create a mask for the weights
                mask_dense = tf.cast(mask.to_tensor(), tf.float32)
                weights = weights[
                    :, :, : tf.shape(mask_dense)[-1]
                ]  # Handle case when the largest number of mentions have 0 queries

                # Apply the mask to the weights
                weights = mask_dense * weights + (1 - mask_dense) * -1e9

                # Return the antecedent loss, computed as categorical cross entropy loss
                return tf.losses.CategoricalCrossentropy(from_logits=True, reduction=tf.losses.Reduction.SUM)(
                    antecedents.values.to_tensor(), tf.RaggedTensor.from_tensor(weights, antecedents.row_lengths()).values
                ) / tf.cast(tf.math.reduce_sum(antecedents.row_lengths()), tf.float32)

            # Compute the antecedent loss only if there is at least one antecedent, otherwise set to 0
            antecedent_loss = tf.cond(tf.math.reduce_sum(antecedents.row_lengths()) != 0, antecedent_loss, lambda: 0.0)

            # Compute total loss as the sum of the tags loss and antecedent loss
            loss = tags_loss + antecedent_loss
            # loss = antecedent_loss

        # Apply gradients to minimize the total loss
        self.optimizer.minimize(loss, self.trainable_variables, tape=tape)

        # Return a dictionary containing the losses and learning rate
        return {
            "tags_loss": tags_loss,
            "antecedent_loss": antecedent_loss,
            "loss": loss,
            "lr": self.optimizer.learning_rate(self.optimizer.iterations),
        }

    def predict(self, dataset: Dataset, pipeline: tf.data.Dataset):
        tid = len(dataset._treebank_token)
        results, entities = [], 0
        doc_mentions, doc_subwords = [], 0
        for b_subwords, b_word_indices in pipeline:
            b_size = b_subwords.shape[0]
            b_embeddings, b_logits = self.compute_tags(b_subwords, b_word_indices, False)
            b_tags = b_logits.with_values(tf.argmax(b_logits.values, axis=-1))

            b_previous, b_mentions, b_refs = [], [], []
            for b in range(b_size):
                word_indices, tags = b_word_indices[b].numpy(), b_tags[b].numpy()
                if word_indices[0] == 2 + tid:
                    doc_mentions, doc_subwords = [], 0

                # Decode mentions
                mentions, stack = [], []
                for i, tag in enumerate(self._tags[tag] for tag in tags):
                    for command in tag.split(",")[1:]:  # The first is stack depth, which we ignore now
                        if command == "PUSH":
                            stack.append(i)
                        elif command.startswith("POP:"):
                            j = int(command[4:])
                            if len(stack):
                                j = len(stack) - (j if j <= len(stack) else 1)
                                mentions.append((stack.pop(j), i))
                        else:
                            raise ValueError(f"Unknown command '{command}'")
                while len(stack):
                    mentions.append((stack.pop(), len(tags) - 1))
                mentions = [[s, e, None] for s, e in sorted(set(mentions), key=lambda x: (x[0], -x[1]))]

                # Prepare inputs for antecedent prediction
                offset = doc_subwords - (word_indices[0] - 2 - tid)
                results.append([]), b_previous.append([]), b_mentions.append([]), b_refs.append([])
                for doc_mention in doc_mentions:
                    if doc_mention[0] < offset:
                        continue
                    b_previous[-1].append([doc_mention[0] - offset + 1 + tid, doc_mention[1] - offset + 1 + tid])
                    b_refs[-1].append(doc_mention[2])
                for mention in mentions:
                    results[-1].append(mention)
                    b_refs[-1].append(mention)
                    b_mentions[-1].append([word_indices[mention[0]], word_indices[mention[1]]])
                    doc_mentions.append(
                        [
                            doc_subwords + word_indices[mention[0]] - word_indices[0],
                            doc_subwords + word_indices[mention[1]] - word_indices[0],
                            mention,
                        ]
                    )
                doc_subwords += word_indices[-1] - word_indices[0]

            # Decode antecedents
            if sum(len(mentions) for mentions in b_mentions) == 0:
                continue
            b_antecedents = self.compute_antecedents(
                b_embeddings,
                tf.ragged.constant(b_previous, dtype=tf.int32, ragged_rank=1),
                tf.ragged.constant(b_mentions, dtype=tf.int32, ragged_rank=1),
            )
            for b in range(b_size):
                len_prev, mentions, refs, antecedents = len(b_previous[b]), b_mentions[b], b_refs[b], b_antecedents[b].numpy()
                for i in range(len(mentions)):
                    j = i - 1
                    while j >= 0 and mentions[j][0] == mentions[i][0]:
                        antecedents[i, j + len_prev] = antecedents[i, i + len_prev] - 1
                        j -= 1
                    j = np.argmax(antecedents[i, : i + len_prev + 1])
                    if j == i + len_prev:
                        entities += 1
                        refs[i + len_prev][2] = entities
                    else:
                        refs[i + len_prev][2] = refs[j][2]

        return results

    def callback(self, epoch: int, datasets, evaluate: bool) -> None:
        for dataset, pipeline in datasets:
            mentions = self.predict(dataset, pipeline)
            path = os.path.join(self._args.logdir, f"{os.path.splitext(os.path.basename(dataset._path))[0]}.{epoch:02d}.conllu")
            # cant use removesuffix:
            _path = path[:-7] if path.endswith(".conllu") else path
            headsonly_path = f"{_path}.headsonly.conllu"
            if len(mentions) > 0:
                dataset.save_mentions(path, headsonly_path, mentions)
            else:
                print(f"WARNING: No mentions found for {dataset._path}")

            if evaluate:
                for eval_path in [path, headsonly_path]:
                    print(f"Evaluating {dataset._path}")
                    metrics = evaluate_coreference(dataset._path, eval_path)
                    print(metrics)

    def call(self, inputs, training):
        inp, tar = inputs