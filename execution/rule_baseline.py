"""Rule baseline on the frozen BLK-020 candidates, tuned on the frozen val split with the official metric.

A candidate pair (rank <= R) is predicted a match iff
  score >= t,  score >= a * (its S1's top-1 score),  and  the S1 "owns" the record:
  it is the record's highest-scoring S1 among all rank <= R pairs (in the ground truth every
  S2/S3 record belongs to at most one S1, so a record claimed by two S1s is a sure false merge).
Owners are computed over ALL Source-1 entities that query the same pool: for validation that is
train + val candidates together (val S1 compete with train S1 for the same train S2/S3 records),
which is exactly the situation on test. candidate_pairs.tsv = every rank <= R pair (the scored set).

  python execution/rule_baseline.py tune  --train T.tsv.gz --val V.tsv.gz
  python execution/rule_baseline.py apply --test X.tsv.gz --R 3 --t 0.5 --a 0.9 --out output/
"""
import argparse
import itertools
import json
import logging
import time

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.csv as pacsv

from entity_resolution import config
from entity_resolution.data.loader import DataLoader, explode_id_lists
from entity_resolution.data.validation_split import create_validation_split
from entity_resolution.evaluation.evaluator import Evaluator, f05
from entity_resolution.submission.generator import SubmissionGenerator
from entity_resolution.submission.validator import SubmissionValidator
from entity_resolution.tracking import log_run

R_GRID, T_GRID, A_GRID = (1, 2, 3, 5, 10), np.round(np.arange(0.20, 0.96, 0.05), 2), (0.0, 0.8, 0.9, 0.95)


def read_pairs(path, max_rank):
    """Stream a (gzipped) candidate TSV, keeping rank <= max_rank rows only."""
    types = {"source1_entity_id": pa.string(), "candidate_entity_id": pa.string(),
             "score": pa.float32(), "rank": pa.int16()}
    stream = pa.input_stream(str(path), compression="gzip" if str(path).endswith(".gz") else None)
    reader = pacsv.open_csv(stream, parse_options=pacsv.ParseOptions(delimiter="\t", quote_char=False),
                            convert_options=pacsv.ConvertOptions(column_types=types),
                            read_options=pacsv.ReadOptions(block_size=1 << 26))
    batches = [b.filter(pc.less_equal(b.column(3), max_rank)) for b in reader]
    return pa.Table.from_batches(batches).to_pandas()


def annotate(pairs, R):
    """Rank <= R pairs + ratio to the S1's top-1 score + owner flag (best S1 per record, ties -> first)."""
    p = pairs[pairs["rank"] <= R].reset_index(drop=True)
    top1 = p.loc[p["rank"] == 1].set_index("source1_entity_id")["score"]
    p["ratio"] = (p["score"] / p["source1_entity_id"].map(top1).astype("float32")).astype("float32")
    cand, s1 = pd.factorize(p["candidate_entity_id"])[0], pd.factorize(p["source1_entity_id"])[0]
    order = np.lexsort((s1, -p["score"].to_numpy(), cand))  # per record: highest score first, ties -> first S1 seen
    first = np.r_[True, cand[order][1:] != cand[order][:-1]]
    owner = np.zeros(len(p), bool)
    owner[order[first]] = True
    p["owner"] = owner
    return p


def predict(p, t, a, use_owner=True):
    keep = (p["score"] >= t) & (p["ratio"] >= a)
    return p[keep & p["owner"]] if use_owner else p[keep]


def tune(a):
    t0 = time.time()
    gt = DataLoader().load_ground_truth()
    _, val_ids = create_validation_split(gt)
    val_gt = gt[gt["source1_entity_id"].isin(val_ids)].reset_index(drop=True)
    pairs = pd.concat([read_pairs(a.train, max(R_GRID)), read_pairs(a.val, max(R_GRID))], ignore_index=True)
    logging.info("loaded %d rank<=%d pairs (train+val) in %.0fs", len(pairs), max(R_GRID), time.time() - t0)

    s1_index = pd.Index(val_gt["source1_entity_id"])
    true = explode_id_lists(val_gt, "matched_entity_ids")
    n_true = np.bincount(s1_index.get_indexer(true["source1_entity_id"]), minlength=len(s1_index))
    true_keys = set(zip(true["source1_entity_id"], true["candidate_entity_id"]))

    results = []
    for R in R_GRID:
        p = annotate(pairs, R)
        v = p[p["source1_entity_id"].isin(val_ids)]
        code = s1_index.get_indexer(v["source1_entity_id"])
        hit = np.fromiter((k in true_keys for k in zip(v["source1_entity_id"], v["candidate_entity_id"])),
                          bool, len(v))
        score, ratio, owner = v["score"].to_numpy(), v["ratio"].to_numpy(), v["owner"].to_numpy()
        for t, r_min, use_owner in itertools.product(T_GRID, A_GRID, (True, False)):
            m = (score >= t) & (ratio >= r_min) & (owner if use_owner else True)
            n_pred = np.bincount(code[m], minlength=len(s1_index))
            tp = np.bincount(code[m & hit], minlength=len(s1_index))
            prec = np.divide(tp, n_pred, out=np.zeros(len(tp)), where=n_pred > 0)
            rec = np.divide(tp, n_true, out=np.zeros(len(tp)), where=n_true > 0)
            ent = np.where((n_true == 0) & (n_pred == 0), 1.0, f05(prec, rec))
            results.append({"R": R, "t": float(t), "a": r_min, "owner": use_owner, "f05": float(ent.mean()),
                            "pairs": int(m.sum())})
        logging.info("R=%d done (%.0fs)", R, time.time() - t0)
    res = pd.DataFrame(results).sort_values("f05", ascending=False)
    print(res.head(15).to_string(index=False))
    best = res.iloc[0]
    # confirm the winner with the official evaluator (string-level, exact challenge formula)
    p = annotate(pairs, int(best.R))
    pred = predict(p[p["source1_entity_id"].isin(val_ids)], best.t, best.a, bool(best.owner))
    lists = pred.groupby("source1_entity_id")["candidate_entity_id"].agg(",".join)
    official = Evaluator().evaluate(val_gt, pd.DataFrame({
        "source1_entity_id": val_gt["source1_entity_id"],
        "matched_entity_ids": lists.reindex(val_gt["source1_entity_id"]).fillna("").to_numpy()}))
    print("official evaluator on the winner:", json.dumps(official))
    log_run({"experiment_id": "RULE-001", "stage": "decision-rule-baseline", "blocking": "BLK-020",
             "best": {k: (v.item() if hasattr(v, "item") else v) for k, v in best.items()},
             "official": official, "runtime_s": round(time.time() - t0, 1),
             "top10": res.head(10).to_dict("records")})


def apply(a):
    t0 = time.time()
    pairs = read_pairs(a.test, a.R)
    p = annotate(pairs, a.R)
    pred = predict(p, a.t, a.a, not a.no_owner)
    s1_ids = DataLoader().load_source("test", 1, columns=["entity_id"])["entity_id"].tolist()
    gen = SubmissionGenerator(a.out)
    match = gen.generate(s1_ids, pred)
    cand = gen.generate_candidates(s1_ids, p)
    logging.info("wrote %s (%d pairs, %d S1 matched) and %s (%d pairs) in %.0fs", match, len(pred),
                 pred["source1_entity_id"].nunique(), cand, len(p), time.time() - t0)
    ok = None if a.skip_validate else SubmissionValidator().validate(match, cand, check_ids=True)
    log_run({"experiment_id": "RULE-001", "stage": "decision-rule-apply-test", "R": a.R, "t": a.t, "a": a.a,
             "owner": not a.no_owner, "pred_pairs": len(pred), "candidate_pairs": len(p), "validator_pass": ok})


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("tune")
    t.add_argument("--train", required=True)
    t.add_argument("--val", required=True)
    x = sub.add_parser("apply")
    x.add_argument("--test", required=True)
    x.add_argument("--R", type=int, required=True)
    x.add_argument("--t", type=float, required=True)
    x.add_argument("--a", type=float, required=True)
    x.add_argument("--no-owner", action="store_true")
    x.add_argument("--skip-validate", action="store_true", help="e.g. on AWS, where the official validator is not shipped")
    x.add_argument("--out", default=str(config.OUTPUT_DIR))
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    tune(a) if a.cmd == "tune" else apply(a)


if __name__ == "__main__":
    main()
