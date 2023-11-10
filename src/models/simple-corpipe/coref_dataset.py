import argparse
import os
import pickle
import re
import sys

import numpy as np
import tensorflow as tf
import transformers
import udapi
import udapi.block.corefud.movehead
import udapi.block.corefud.removemisc


class Dataset:
    TOKEN_EMPTY = "\\"
    # Use Katakana as treebank ids -- use a subset with embedded "space" characters in Rembert
    TOKEN_TREEBANKS = [chr(i) for i in [*range(0x30A2, 0x30AB, 2), *range(0x30AB, 0x30B0, 2), *range(0x30B3, 0x30BC, 2)]]

    def __init__(self, path: str, tokenizer: transformers.PreTrainedTokenizerFast) -> None:
        self._cls = tokenizer.cls_token_id
        self._sep = tokenizer.sep_token_id
        self._path = path
        self._treebank_token = []

        # Create the tokenized documents if they do not exist
        cache_path = f"{path}.mentions.{os.path.basename(tokenizer.name_or_path)}"
        if not os.path.exists(cache_path):
            # Create flat representation
            if not os.path.exists(f"{path}.flat"):
                with open(path, "r", encoding="utf-8-sig") as data_file:
                    data_original = [line.rstrip("\r\n") for line in data_file.readlines() if not re.match(r"^\d+-", line)]

                # Remove multi-word tokens
                data = [line for line in data_original if not re.match(r"\d+-", line)]

                # Flatten the representation
                flat, i = [], 0
                for line in data:
                    if not line:
                        i = 0
                    elif not line.startswith("#"):
                        columns = line.split("\t")
                        assert len(columns) == 10
                        if "." in columns[0]:
                            columns[1] = (
                                self.TOKEN_EMPTY + " " + (columns[1] if columns[1] and columns[1] != "_" else columns[2])
                            )
                        columns[0] = str(i + 1)
                        columns[6] = "0"
                        line = "\t".join(columns)
                        i += 1
                    flat.append(line)

                with open(f"{path}.flat", "w", encoding="utf-8-sig") as data_file:
                    for line in flat:
                        print(line, file=data_file)

            # Parse with Udapi
            if not os.path.exists(f"{path}.mentions"):
                docs = []
                for doc in udapi.block.read.conllu.Conllu(files=[f"{path}.flat"], split_docs=True).read_documents():
                    new_doc = []
                    for tree in doc.trees:
                        words, coref_mentions = [], set()
                        for node in tree.descendants:
                            words.append(node.form)
                            coref_mentions.update(node.coref_mentions)

                        dense_mentions = []
                        for mention in coref_mentions:
                            span = mention.words
                            start = end = span.index(mention.head)
                            while start > 0 and span[start - 1].ord + 1 == span[start].ord:
                                start -= 1
                            while end < len(span) - 1 and span[end].ord + 1 == span[end + 1].ord:
                                end += 1
                            dense_mentions.append(
                                ((span[start].ord - 1, span[end].ord - 1), mention.entity.eid, start > 0 or end + 1 < len(span))
                            )
                        dense_mentions = sorted(dense_mentions, key=lambda x: (x[0][0], -x[0][1], x[2]))

                        mentions = []
                        for i, mention in enumerate(dense_mentions):
                            if i and dense_mentions[i - 1][0] == mention[0]:
                                print(
                                    f"Multiple same mentions {mention[2]}/{dense_mentions[i-1][2]} in sent_id {tree.sent_id}: {tree.get_sentence()}",
                                    flush=True,
                                )
                                continue
                            mentions.append((mention[0][0], mention[0][1], mention[1]))
                        new_doc.append((words, mentions))
                    docs.append(new_doc)
                with open(f"{path}.mentions", "wb") as cache_file:
                    pickle.dump(docs, cache_file, protocol=3)
            with open(f"{path}.mentions", "rb") as cache_file:
                docs = pickle.load(cache_file)

            # Tokenize the data, generate stack operations and subword mentions
            self.docs = []
            for doc in docs:
                new_doc = []
                for words, mentions in doc:
                    subwords, word_indices, word_tags, subword_mentions, stack = [], [], [], [], []
                    for i in range(len(words)):
                        word_indices.append(len(subwords))
                        word = words[i]
                        subword = tokenizer.encode(word, add_special_tokens=False)
                        assert len(subword) > 0
                        if subword[0] == 6 and "xlm-r" in tokenizer.name_or_path:  # Hack: remove the space-only token in XLM-R
                            subword = subword[1:]
                        assert len(subword) > 0
                        subwords.extend(subword)

                        tag = [str(len(stack))]
                        for _ in range(2):
                            for j in reversed(range(len(stack))):
                                start, end, eid = stack[j]
                                if end == i:
                                    tag.append(f"POP:{len(stack)-j}")
                                    subword_mentions.append((start, word_indices[-1], eid))
                                    stack.pop(j)
                            while mentions and mentions[0][0] == i:
                                tag.append("PUSH")
                                stack.append((word_indices[-1], mentions[0][1], mentions[0][2]))
                                mentions = mentions[1:]
                        word_tags.append(",".join(tag))
                    assert len(stack) == 0
                    subword_mentions = sorted(subword_mentions, key=lambda x: (x[0], -x[1]))

                    new_doc.append((subwords, word_indices, word_tags, subword_mentions))
                self.docs.append(new_doc)

            with open(cache_path, "wb") as cache_file:
                pickle.dump(self.docs, cache_file, protocol=3)
        with open(cache_path, "rb") as cache_file:
            self.docs = pickle.load(cache_file)

    @staticmethod
    def create_tags(trains):
        tags = set()
        for train in trains:
            for doc in train.docs:
                for _, _, word_tags, _ in doc:
                    tags.update(word_tags)
        return sorted(tags)

    def pipeline(self, tags_map, train: bool, args: argparse.Namespace) -> tf.data.Dataset:
        # Define a generator function that yields input and output data for the model
        def generator():
            # Initialize a token ID counter
            tid = len(self._treebank_token)
            # Iterate over all documents in the dataset
            for doc in self.docs:
                # Initialize lists to store subwords and subword mentions
                p_subwords, p_subword_mentions = [], []
                # Iterate over all sentences in the document
                for doc_i, (subwords, word_indices, word_tags, subword_mentions) in enumerate(doc):
                    # Check if the sentence can fit within the maximum segment length
                    if len(subwords) + 4 + tid <= args.segment:
                        # Compute the number of subwords to reserve on the right side of the sentence
                        right_reserve = min((args.segment - 4 - tid - len(subwords)) // 2, args.right or 0)
                        # Compute the number of subwords to include from the previous sentence
                        context = min(args.segment - 4 - tid - len(subwords) - right_reserve, len(p_subwords))
                        # Compute the indices of the words in the input text
                        word_indices = [context + 2 + tid + i for i in word_indices + [len(subwords)]]
                        # Concatenate the subwords from the previous sentence, the current sentence, and the right reserve
                        e_subwords = [self._cls, *self._treebank_token, *p_subwords[-context:], self._sep, *subwords, self._sep]
                        # If there is a right reserve, add subwords from the next sentence until the maximum length is reached
                        if args.right is not None:
                            i = doc_i + 1
                            while i < len(doc) and len(e_subwords) + 1 < args.segment:
                                e_subwords.extend(doc[i][0][: args.segment - len(e_subwords) - 1])
                                i += 1
                        e_subwords.append(self._sep)
                        # Define the input data as a tuple of subwords and word indices
                        output = (e_subwords, word_indices)
                        # If training, define the output data as a tuple of word tags, previous mentions, current mentions, mask, and gold labels
                        if train:
                            # Compute the previous mentions that overlap with the current sentence
                            offset = len(p_subwords) - context
                            prev = [
                                (s - offset + 1 + tid, e - offset + 1 + tid, eid)
                                for s, e, eid in p_subword_mentions
                                if s >= offset
                            ]
                            # Define the previous mention positions and IDs
                            prev_pos = np.array([[s, e] for s, e, _ in prev], dtype=np.int32).reshape([-1, 2])
                            prev_eid = np.array([eid for _, _, eid in prev], dtype=str)
                            # Define the current mention positions and IDs
                            ment = [(context + 2 + tid + s, context + 2 + tid + e, eid) for s, e, eid in subword_mentions]
                            ment_pos = np.array([[s, e] for s, e, _ in ment], dtype=np.int32).reshape([-1, 2])
                            ment_eid = np.array([eid for _, _, eid in ment], dtype=str)
                            # Define the mask that indicates which mention pairs are valid
                            mask = ment_pos[:, 0, None] > np.concatenate([prev_pos[:, 0], ment_pos[:, 0]])[None, :]
                            # Define the gold labels for each mention pair
                            diag = np.pad(np.eye(len(ment_pos)), [[0, 0], [len(prev_pos), 0]])
                            gold = (ment_eid[:, None] == np.concatenate([prev_eid, ment_eid])[None, :]) * mask
                            gold = np.where(np.sum(gold, axis=1, keepdims=True) > 0, gold, diag)
                            # Apply a maximum number of links constraint to the gold labels
                            if args.max_links is not None:
                                max_link_mask = np.cumsum(gold, axis=1)
                                gold *= max_link_mask > max_link_mask[:, -1:] - args.max_links
                            # Normalize the gold labels
                            gold /= np.sum(gold, axis=1, keepdims=True)
                            # Add the diagonal to the mask
                            mask = mask + diag
                            # Apply label smoothing to the gold labels
                            if args.label_smoothing:
                                gold = (1 - args.label_smoothing) * gold + args.label_smoothing * (
                                    mask / np.sum(mask, axis=1, keepdims=True)
                                )
                            # Convert the word tags to integer labels
                            word_tags = [tags_map[tag] for tag in word_tags]
                            # Define the output data as a tuple of word tags, previous mentions, current mentions, mask, and gold labels
                            output = (output, (word_tags, prev_pos, ment_pos, mask, gold))
                        # Yield the input and output data as a tuple
                        yield output
                    # Update the subword mentions and subwords from the previous sentences
                    p_subword_mentions.extend((s + len(p_subwords), e + len(p_subwords), eid) for s, e, eid in subword_mentions)
                    p_subwords.extend(subwords)

        # Define the output signature of the generator function
        output_signature = (tf.TensorSpec([None], tf.int32), tf.TensorSpec([None], tf.int32))
        if train:
            output_signature = (
                output_signature,
                (
                    tf.TensorSpec([None], tf.int32),
                    tf.TensorSpec([None, 2], tf.int32),
                    tf.TensorSpec([None, 2], tf.int32),
                    tf.TensorSpec([None, None], tf.bool),
                    tf.TensorSpec([None, None], tf.float32),
                ),
            )
        # Create a dataset from the generator function
        pipeline = tf.data.Dataset.from_generator(generator, output_signature=output_signature)
        # Cache the dataset for faster access
        pipeline = pipeline.cache()
        # Check the cardinality of the dataset
        pipeline = pipeline.apply(tf.data.experimental.assert_cardinality(sum(1 for _ in pipeline)))
        return pipeline

    def save_mentions(self, path: str, headsonly_path: str, mentions) -> None:
        doc = udapi.block.read.conllu.Conllu(files=[self._path]).read_documents()[0]
        udapi.block.corefud.removemisc.RemoveMisc(attrnames="Entity,SplitAnte,Bridge").apply_on_document(doc)

        entities = {}
        for i, tree in enumerate(doc.trees):
            nodes = tree.descendants_and_empty
            for start, end, eid in mentions[i]:
                if not eid in entities:
                    entities[eid] = udapi.core.coref.CorefEntity(f"c{eid}")
                udapi.core.coref.CorefMention(nodes[start : end + 1], entity=entities[eid])
        doc._eid_to_entity = {entity._eid: entity for entity in sorted(entities.values())}
        udapi.block.corefud.movehead.MoveHead(bugs="ignore").apply_on_document(doc)
        udapi.block.write.conllu.Conllu(files=[path]).apply_on_document(doc)

        for mention in doc.coref_mentions:
            mention.words = [mention.head]
        udapi.block.write.conllu.Conllu(files=[headsonly_path]).apply_on_document(doc)
