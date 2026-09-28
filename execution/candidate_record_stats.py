"""Per-record competition statistics: how contested is each S2/S3 record across ALL Source-1 entities?

For every candidate record: best_score, second_score, n_s1 (how many S1 retrieved it in their top-K),
best_s1 (the S1 with the highest score; ties -> the first one seen). In the ground truth every S2/S3
record belongs to at most one S1, so the matcher joins these on candidate_entity_id and derives
  is_owner = (source1_entity_id == best_s1)      margin_to_best = best_score - score
  owner_gap = best_score - second_score          n_s1
Compute over every S1 that queries the same pool: train + val candidates together (val S1 compete
with train S1 for the same train S2/S3 records), and all test S1 for test.

  python execution/candidate_record_stats.py --split train --inputs T.tsv.gz V.tsv.gz --out stats_trainval.parquet
  python execution/candidate_record_stats.py --split test  --inputs X.tsv.gz --out stats_test.parquet
"""
import argparse
import logging
import time

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.csv as pacsv

from entity_resolution.data.loader import DataLoader
from entity_resolution.tracking import log_run


def _batches(path):
    types = {"source1_entity_id": pa.string(), "candidate_entity_id": pa.string(), "score": pa.float32(), "rank": pa.int16()}
    stream = pa.input_stream(str(path), compression="gzip" if str(path).endswith(".gz") else None)
    yield from pacsv.open_csv(stream, parse_options=pacsv.ParseOptions(delimiter="\t", quote_char=False),
                              convert_options=pacsv.ConvertOptions(column_types=types, include_columns=list(types)[:3]),
                              read_options=pacsv.ReadOptions(block_size=1 << 26))


def record_stats(paths, record_ids, s1_ids):
    """Stream candidate TSVs; top-2 scores, count and argmax S1 per record (exact, batch-mergeable)."""
    rec_index, s1_index = pd.Index(record_ids), pd.Index(s1_ids)
    n = len(rec_index)
    best = np.full(n, -1.0, np.float32)
    second = np.full(n, -1.0, np.float32)
    count = np.zeros(n, np.int32)
    owner = np.full(n, -1, np.int64)
    rows = 0
    for path in paths:
        for b in _batches(path):
            rec = rec_index.get_indexer(b.column(1).to_pandas())
            s1 = s1_index.get_indexer(b.column(0).to_pandas())
            score = b.column(2).to_numpy(zero_copy_only=False)
            if (rec < 0).any() or (s1 < 0).any():
                raise SystemExit(f"{path}: IDs outside the given pool / S1 set")
            order = np.lexsort((np.arange(len(rec)), -score, rec))  # per record: best first, ties -> first seen
            rec, s1, score = rec[order], s1[order], score[order]
            starts = np.r_[0, np.flatnonzero(np.diff(rec)) + 1]
            ends = np.r_[starts[1:], len(rec)]
            r, b1, o1 = rec[starts], score[starts], s1[starts]
            b2 = np.where(ends - starts > 1, score[np.minimum(starts + 1, len(rec) - 1)], -1.0).astype(np.float32)
            new_second = np.maximum(np.minimum(best[r], b1), np.maximum(second[r], b2))
            take = b1 > best[r]  # strictly better: ties keep the earlier owner
            owner[r[take]] = o1[take]
            best[r] = np.maximum(best[r], b1)
            second[r] = new_second
            count[r] += (ends - starts).astype(np.int32)
            rows += len(rec)
        logging.info("%s done, %d rows so far", path, rows)
    seen = count > 0
    return pd.DataFrame({
        "candidate_entity_id": rec_index[seen].astype("string[pyarrow]"),
        "best_score": best[seen], "second_score": np.where(second[seen] < 0, np.nan, second[seen]).astype(np.float32),
        "n_s1": count[seen], "best_s1": pd.Index(s1_ids)[owner[seen]].astype("string[pyarrow]"),
    }), rows


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split", choices=["train", "test"], required=True, help="which pool the candidates index")
    ap.add_argument("--inputs", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    t0 = time.time()
    loader = DataLoader()
    pool = loader.load_candidates_pool(a.split, columns=["entity_id"])["entity_id"]
    s1 = loader.load_source(a.split, 1, columns=["entity_id"])["entity_id"]
    stats, rows = record_stats(a.inputs, pool, s1)
    stats.to_parquet(a.out, index=False)
    summary = {"records": len(stats), "pairs": rows, "contested_share": float((stats["n_s1"] > 1).mean()),
               "median_n_s1": float(stats["n_s1"].median())}
    logging.info("wrote %s %s in %.0fs", a.out, summary, time.time() - t0)
    log_run({"experiment_id": "BLK-020-stats", "stage": "blocking-record-stats", "split": a.split,
             "inputs": a.inputs, "out": a.out, **summary, "runtime_s": round(time.time() - t0, 1)})


if __name__ == "__main__":
    main()
