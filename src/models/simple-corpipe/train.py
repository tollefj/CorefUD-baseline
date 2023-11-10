import argparse
import contextlib
import datetime
import functools
import json
import os
import shutil

import numpy as np
import tensorflow as tf
import transformers
from coref_dataset import Dataset
from coref_model import Model
from tqdm import tqdm

DATA_BASE_PATH = "../../../data"

tf.get_logger().setLevel("ERROR")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")  # Report only TF errors by default
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

parser = argparse.ArgumentParser()
parser.add_argument("--langs", default=[], nargs="+", type=str, help="Languages to train on.")
parser.add_argument("--batch_size", default=16, type=int, help="Batch size.")
parser.add_argument("--bert", default="xlm-roberta-base", type=str, help="Bert model.")
parser.add_argument("--debug", default=False, action="store_true", help="Debug mode.")
parser.add_argument("--epochs", default=10, type=int, help="Number of epochs.")
parser.add_argument("--exp", default="run", type=str, help="Exp name.")
parser.add_argument("--label_smoothing", default=0.0, type=float, help="Label smoothing.")
parser.add_argument("--learning_rate", default=2e-5, type=float, help="Learning rate.")
parser.add_argument("--learning_rate_decay", default=False, action="store_true", help="Decay LR.")
parser.add_argument("--max_links", default=None, type=int, help="Max antecedent links to train on.")
parser.add_argument("--right", default=50, type=int, help="Reserved space for right context, if any.")
parser.add_argument("--seed", default=42, type=int, help="Random seed.")
parser.add_argument("--segment", default=512, type=int, help="Segment size")
parser.add_argument("--train", default=[], nargs="+", type=str, help="Additional train data.")
parser.add_argument("--warmup", default=0.1, type=float, help="Warmup ratio.")


def main(args: argparse.Namespace) -> None:
    # Fix random seeds and threads
    tf.keras.utils.set_random_seed(args.seed)
    tf.config.threading.set_inter_op_parallelism_threads(0)
    tf.config.threading.set_intra_op_parallelism_threads(0)
    if args.debug:
        tf.config.run_functions_eagerly(True)
        tf.data.experimental.enable_debug_mode()

    # Create logdir name and dump options
    exp = f"{args.exp}-"
    file_name = os.path.splitext(os.path.basename(globals().get("__file__", "notebook")))[0]
    job_id = os.environ.get("JOB_ID", "")
    current_time = datetime.datetime.now().strftime("%y%m%d_%H%M%S")
    result = f"{exp}{file_name}{job_id}{current_time}"
    args.logdir = os.path.join("logs", result)

    os.makedirs(args.logdir)
    shutil.copy2(__file__, os.path.join(args.logdir, os.path.basename(__file__)))

    with open(os.path.join(args.logdir, "options.json"), "w") as json_file:
        json.dump(vars(args), json_file, sort_keys=True, ensure_ascii=False, indent=2)
    print(json.dumps(vars(args), sort_keys=True, ensure_ascii=False, indent=2))

    tokenizer = transformers.AutoTokenizer.from_pretrained(args.bert)

    romance_langs = "ca_ancora es_ancora fr_democrat".split()
    germanic_langs = "de_parcorfull de_potsdamcc en_gum en_parcorfull no_bokmaalnarc no_nynorsknarc".split()
    slavic_baltic_langs = "cs_pcedt cs_pdt pl_pcc lt_lcc ru_rucor".split()
    urgic_turkic_langs = "hu_korkor hu_szegedkoref tr_itcc".split()

    langs = romance_langs + germanic_langs + slavic_baltic_langs + urgic_turkic_langs

    langs_dict = {
        "romance": romance_langs,
        "germanic": germanic_langs,
        "slavic": slavic_baltic_langs,
        "urgic": urgic_turkic_langs,
        "all": langs,
    }

    if len(args.langs) > 0:
        if args.langs[0] in langs_dict:
            langs = langs_dict[args.langs[0]]
        else:
            langs = args.langs
    langs = [f"{DATA_BASE_PATH}/{lang}/{lang}" for lang in langs]

    print("Langs", langs)
    extra_trains = []
    if len(args.train) > 0:
        extra_trains = [f"data/{lang}/{lang}" for lang in args.train]

    print(f"Extra trains: {extra_trains}")
    print("Loading trains + devs")
    trains = []
    devs = []
    tests = []
    devs_blind = []

    def process_dataset(path, split):
        return Dataset(f"{path}-corefud-{split}.conllu", tokenizer)

    for path in tqdm(langs + extra_trains, desc="Processing training data"):
        trains.append(process_dataset(path, "train"))

    for path in tqdm(langs, desc="Processing dev and test"):
        devs.append(process_dataset(path, "dev"))
        tests.append(process_dataset(path, "test.blind"))
        devs_blind.append(process_dataset(path, "dev.blind"))

    tags = Dataset.create_tags(trains)
    with open(os.path.join(args.logdir, "tags.txt"), "w") as tags_file:
        for tag in tags:
            print(tag, file=tags_file)
    tags_map = {tag: i for i, tag in enumerate(tags)}

    strategy_scope = (
        tf.distribute.MirroredStrategy().scope()
        if len(tf.config.list_physical_devices("GPU")) > 1
        else contextlib.nullcontext()
    )
    with strategy_scope:
        def batch(pipeline, drop_remainder=False):
            return pipeline.apply(tf.data.experimental.dense_to_ragged_batch(args.batch_size, drop_remainder)).prefetch(
                tf.data.AUTOTUNE
            )

        print("Creating the pipelines...")
        trains = [train.pipeline(tags_map, True, args) for train in trains]
        print("Concatenating the pipelines...")
        train = functools.reduce(lambda x, y: x.concatenate(y), trains)
        train = batch(train.shuffle(len(train), seed=args.seed), drop_remainder=True)
        devs = [(dev, batch(dev.pipeline(tags_map, False, args))) for dev in devs]
        tests = [(test, batch(test.pipeline(tags_map, False, args))) for test in tests]
        devs_blind = [(dev, batch(dev.pipeline(tags_map, False, args))) for dev in devs_blind]

        # Create and train the model
        print("Creating the model...")
        print("Tags: ", len(tags))
        model = Model(tags, args)

        model.compile(train)
        trainable_weights = sum(w.numpy().size for w in model.trainable_weights)
        print(f"Trainable weights: {trainable_weights}")

        params = sum(np.prod(w.shape) for w in model.trainable_weights)
        print(f"Trainable parameters: {params}")

        trainable_vars = model.trainable_variables
        num_params = sum(map(lambda x: np.prod(x.shape), trainable_vars))
        print(f"Trainable parameters: {num_params}")

        # input size from bert sequence length is 512
        input_size = 512
        model.build(input_shape=(None, input_size))
        model.summary()

        print("Training the model...")
        model.fit(
            train,
            epochs=args.epochs,
            # verbose=os.environ.get("VERBOSE", 2),
            verbose=2,
            callbacks=[
                tf.keras.callbacks.LambdaCallback(on_epoch_end=lambda epoch, _: model.callback(epoch, devs, evaluate=True)),
                tf.keras.callbacks.LambdaCallback(on_epoch_end=lambda epoch, _: model.callback(epoch, tests, evaluate=False)),
                tf.keras.callbacks.LambdaCallback(
                    on_epoch_end=lambda epoch, _: model.callback(epoch, devs_blind, evaluate=False)
                ),
            ],
        )


if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    main(args)
