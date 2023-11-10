import argparse
import sys
from typing import List, Optional, Tuple

from coval.corefud import reader
from coval.eval import evaluator

__author__ = "michnov"


def evaluate_coreference(
    key_file: str,
    sys_file: str,
    metrics: Optional[List[str]] = None,
    keep_singletons: bool = False,
    match: str = "head",
    exact_match: bool = False,
    only_f1: bool = False,
    info: bool = False,
) -> str:
    if metrics is None:
        metrics = ["muc", "bcub", "ceafe"]

    metric_dict = {
        "lea": evaluator.lea,
        "muc": evaluator.muc,
        "bcub": evaluator.b_cubed,
        "ceafe": evaluator.ceafe,
        "ceafm": evaluator.ceafm,
        "blanc": [evaluator.blancc, evaluator.blancn],
        "mor": evaluator.mention_overlap,
        "zero": evaluator.als_zeros,
    }

    metrics = [(name, metric_dict[name]) for name in metrics]

    # --exact-match|-x overrides the --match|-t parameter
    if exact_match:
        match = "exact"

    msg = "The scorer is evaluating coreference {:s} singletons, with {:s} matching of mentions using the following metrics: {:s}.".format(
        "including" if keep_singletons else "excluding", match, ", ".join([name for name, f in metrics])
    )
    if info:
        print(msg)

    coref_infos = reader.get_coref_infos(key_file, sys_file, match, keep_singletons)

    conll = 0
    conll_subparts_num = 0

    output = ""

    for name, metric in metrics:
        recall, precision, f1 = evaluator.evaluate_documents(coref_infos, metric, beta=1, only_split_antecedent=False)
        if name in ["muc", "bcub", "ceafe"]:
            conll += f1
            conll_subparts_num += 1

        output += name + "\n"
        output += "Recall: %.2f" % (recall * 100) + " Precision: %.2f" % (precision * 100) + " F1: %.2f" % (f1 * 100) + "\n"

    if conll_subparts_num == 3:
        conll = (conll / 3) * 100
        output += "CoNLL score: %.2f" % conll + "\n"

    if only_f1:
        return conll
    return output


if __name__ == "__main__":
    argparser = argparse.ArgumentParser(description="Coreference scorer for documents in CorefUD 1.0 scheme")
    argparser.add_argument("key_file", type=str, help="path to the key/reference file")
    argparser.add_argument("sys_file", type=str, help="path to the system/response file")
    argparser.add_argument(
        "-m",
        "--metrics",
        choices=["all", "lea", "muc", "bcub", "ceafe", "ceafm", "blanc", "mor", "zero"],
        nargs="*",
        default=None,
        help="metrics to be used for evaluation",
    )
    argparser.add_argument(
        "-s", "--keep-singletons", action="store_true", default=False, help="evaluate also singletons; ignored otherwise"
    )
    argparser.add_argument(
        "-a",
        "--match",
        type=str,
        choices=["exact", "partial", "head"],
        default="head",
        help="choose the type of mention matching: exact, partial, head",
    )
    argparser.add_argument(
        "-x",
        "--exact-match",
        action="store_true",
        default=False,
        help="use exact match for matching key and system mentions; overrides the value chosen by --match|-t",
    )
    args = argparser.parse_args()

    evaluation_results = evaluate_coreference(
        args.key_file, args.sys_file, args.metrics, args.keep_singletons, args.match, args.exact_match
    )
    print(evaluation_results)