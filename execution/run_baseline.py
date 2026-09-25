"""EXP-001 baseline (directives/baseline_matching.md): exact lowercase+strip name match within country.

  python execution/run_baseline.py --split train   # full train (reproduces the recorded EXP-001 score)
  python execution/run_baseline.py --split val     # frozen validation split
  python execution/run_baseline.py --split test    # writes output/{matching_results,candidate_pairs}.tsv + validates

Kept deliberately naive: it is the floor every other experiment is compared against.
"""
import argparse
import logging
import time

import pandas as pd

from entity_resolution import config
from entity_resolution.data.loader import DataLoader
from entity_resolution.data.validation_split import create_validation_split
from entity_resolution.evaluation.evaluator import Evaluator
from entity_resolution.submission.generator import SubmissionGenerator
from entity_resolution.submission.validator import SubmissionValidator
from entity_resolution.tracking import log_run


def exact_pairs(s1: pd.DataFrame, pool: pd.DataFrame) -> pd.DataFrame:
    def keyed(df):
        return df.assign(norm=df["business_name"].str.lower().str.strip())[["entity_id", "country", "norm"]]
    left, right = keyed(s1), keyed(pool)
    m = left.merge(right[right["norm"] != ""], on=["country", "norm"], suffixes=("_s1", "_c"))
    return pd.DataFrame({"source1_entity_id": m["entity_id_s1"], "candidate_entity_id": m["entity_id_c"]})


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--split", choices=["train", "val", "test"], default="val")
    a = p.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    t0 = time.time()
    loader = DataLoader()
    data_split = "test" if a.split == "test" else "train"
    s1 = loader.load_source(data_split, 1)
    pairs = exact_pairs(s1, loader.load_candidates_pool(data_split))

    if a.split == "test":
        gen = SubmissionGenerator()
        ids = s1["entity_id"].tolist()  # file order
        cpath = gen.generate_candidates(ids, pairs)  # baseline: candidates == matches
        mpath = gen.generate(ids, pairs)
        ok = SubmissionValidator().validate(mpath, cpath)
        log_run({"experiment_id": "EXP-001", "stage": "submission", "split": "test", "valid": ok,
                 "n_pairs": len(pairs), "runtime_s": round(time.time() - t0, 1)})
        raise SystemExit(0 if ok else 1)

    gt = loader.load_ground_truth()
    if a.split == "val":
        _, val_ids = create_validation_split(gt)
        gt = gt[gt["source1_entity_id"].isin(val_ids)]
    pred = (pairs.groupby("source1_entity_id")["candidate_entity_id"].agg(",".join)
            .rename("matched_entity_ids").reset_index())
    metrics = Evaluator().evaluate(gt, pred)
    rec = log_run({"experiment_id": "EXP-001", "stage": "matching", "split": a.split, "metrics": metrics,
                   "runtime_s": round(time.time() - t0, 1)})
    for k, v in metrics.items():
        print(f"{k:22s} {v}")
    print(f"runtime {rec['runtime_s']}s | peak RSS {rec['peak_rss_gb']} GB")


if __name__ == "__main__":
    main()
