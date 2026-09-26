"""Generate D2 blocking candidates (the team's blocker) with ONE config for every split.

  python scripts/aws/run_d2_blocking.py --split trainval              # train + validation S1 vs the train S2+S3 index
  python scripts/aws/run_d2_blocking.py --split test                  # test S1 vs the test S2+S3 index
  python scripts/aws/run_d2_blocking.py --split trainval --sample 5000  # timing/recall check on val, writes no TSV
  python scripts/aws/run_d2_blocking.py --split trainval --sample 20000 --blocker word  # same check, another pinned config

Writes artifacts/blocking/{train,validation,test}_candidate_pairs.tsv
(source1_entity_id, candidate_entity_id, score, rank) and records, per split, the blocker
config + hash, generator commit, TSV sha256, pair counts, candidate distribution and
recall / oracle-ceiling F0.5 in artifacts/blocking/blocking_metadata.json.
The index is sharded to disk under ER_CACHE_DIR unless --in-memory (use that on a >=64 GB box).
"""
import argparse
import datetime
import hashlib
import json
import logging
import random
import time

import numpy as np
import pyarrow as pa
import pyarrow.csv as pacsv

from entity_resolution import config
from entity_resolution.blocking.evaluator import BlockingEvaluator
from entity_resolution.blocking.tfidf_blocking import TfidfBlocker
from entity_resolution.data.loader import DataLoader
from entity_resolution.data.validation_split import create_validation_split
from entity_resolution.tracking import git_commit, log_run

# D2 (EXP-002B): country partitions, business_name + business_address, char_wb (3,5), min_df 2, top-200.
D2 = dict(text_fields=["business_name", "business_address"], analyzer="char_wb", ngram_range=[3, 5],
          min_df=2, partition_by_country=True)
# Candidates for comparison against D2: identical except the analyzer (and max_df for the pruned variant).
BLOCKERS = {"d2": D2,
            "word": {**D2, "analyzer": "word", "ngram_range": [1, 1]},
            "word_maxdf02": {**D2, "analyzer": "word", "ngram_range": [1, 1], "max_df": 0.02}}
K = 200
K_SWEEP = (10, 25, 50, 100, 200)
OUT_DIR = config.REPO_ROOT / "artifacts" / "blocking"
SCHEMA = pa.schema([("source1_entity_id", pa.string()), ("candidate_entity_id", pa.string()),
                    ("score", pa.float32()), ("rank", pa.int16())])
# evaluator ratio -> the count it is averaged over when combining batches
WEIGHTS = {"recall_at_": "true", "full_coverage_at_": "pos", "ceiling_f05_at_": "s1"}


def generate(blocker, s1, gt, out_path, batch):
    """Query S1 in batches, stream candidates to TSV, return (per-S1 candidate counts, metrics)."""
    writer = None
    if out_path:  # header written by hand: pyarrow quotes header names even with quoting_style="none"
        sink = open(out_path, "wb")
        sink.write(("\t".join(SCHEMA.names) + "\n").encode())
        writer = pacsv.CSVWriter(sink, SCHEMA, write_options=pacsv.WriteOptions(
            include_header=False, delimiter="\t", quoting_style="none"))
    n_cand, parts = [], []
    for start in range(0, len(s1), batch):
        b = s1.iloc[start:start + batch]
        cands = blocker.generate_candidates(b, K)
        n_cand.append(np.bincount(cands["source1_entity_id"].cat.codes, minlength=len(b)))
        if writer:
            writer.write_table(pa.Table.from_pandas(cands, preserve_index=False).cast(SCHEMA))
        if gt is not None:
            ev = BlockingEvaluator(gt[gt["source1_entity_id"].isin(b["entity_id"])])
            parts.append({"m": ev.evaluate(cands, K_SWEEP), "s1": len(ev.s1), "true": int(ev.n_true.sum()),
                          "pos": int((ev.n_true > 0).sum())})
        logging.info("%s: %d / %d S1 done", out_path.name if out_path else "sample", start + len(b), len(s1))
    if writer:
        writer.close()
        sink.close()
    metrics = {}
    for key in (parts[0]["m"] if parts else {}):
        w = next((w for prefix, w in WEIGHTS.items() if key.startswith(prefix)), None)
        if w:
            metrics[key] = sum(p["m"][key] * p[w] for p in parts) / max(sum(p[w] for p in parts), 1)
    return np.concatenate(n_cand), metrics


def distribution(n_cand, countries):
    out = {"n_s1": len(n_cand), "total_pairs": int(n_cand.sum()), "no_candidates": int((n_cand == 0).sum()),
           **{f"p{q}_cand": float(np.percentile(n_cand, q)) for q in (50, 95, 99)}, "max_cand": int(n_cand.max())}
    for k in K_SWEEP:
        out[f"avg_cand_at_{k}"] = float(np.minimum(n_cand, k).mean())
        out[f"total_pairs_at_{k}"] = int(np.minimum(n_cand, k).sum())
    out["by_country"] = {str(c): {"n_s1": int(m.sum()), "avg_cand": float(n_cand[m].mean()),
                                  "no_candidates": int((n_cand[m] == 0).sum())}
                         for c in np.unique(countries) for m in [countries == c]}
    return out


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 24), b""):
            h.update(block)
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--split", choices=["trainval", "test"], required=True)
    p.add_argument("--sample", type=int, default=0, help="only N random val S1, metrics only (no TSV)")
    p.add_argument("--train-sample", type=int, default=0, help="only N random train S1")
    p.add_argument("--write-sample", action="store_true", help="write TSV even if sampling")
    p.add_argument("--in-memory", action="store_true", help="keep the index in RAM (needs ~25 GB peak)")
    p.add_argument("--batch", type=int, default=250_000, help="S1 per query batch / TSV write")
    p.add_argument("--exp-id", default="EXP-002B-stage2")
    p.add_argument("--blocker", choices=list(BLOCKERS), default="d2")
    a = p.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    commit = git_commit()
    if commit.endswith("-dirty") and not (a.sample or a.train_sample):
        raise SystemExit(f"working tree is dirty ({commit}): commit + push first so the artifacts are reproducible")
    t0 = time.time()

    loader = DataLoader()
    data_split = "test" if a.split == "test" else "train"
    s1 = loader.load_source(data_split, 1)
    pool = loader.load_candidates_pool(data_split, columns=["entity_id", "country"] + D2["text_fields"])
    load_s = time.time() - t0
    gt, jobs = None, []
    if a.split == "test":
        jobs = [("test", s1)]
    else:
        gt = loader.load_ground_truth()
        train_ids, val_ids = create_validation_split(gt)
        if a.train_sample:
            train_ids = set(random.Random(config.SEED).sample(sorted(train_ids), a.train_sample))
            jobs.append(("train_sample", s1[s1["entity_id"].isin(train_ids)]))
        elif a.sample:
            val_ids = set(random.Random(config.SEED).sample(sorted(val_ids), a.sample))
            jobs.append(("validation", s1[s1["entity_id"].isin(val_ids)]))
        else:
            jobs.append(("train", s1[s1["entity_id"].isin(train_ids)]))
            jobs.append(("validation", s1[s1["entity_id"].isin(val_ids)]))

    t_index = time.time()
    blocker = TfidfBlocker(**BLOCKERS[a.blocker], n_jobs=config.N_JOBS,
                           cache_dir=None if a.in_memory else config.CACHE_DIR / "tfidf_index").fit(pool)
    del pool
    index_s = time.time() - t_index
    config_sha = hashlib.sha256(json.dumps(blocker.config, sort_keys=True).encode()).hexdigest()[:12]

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    meta_path = OUT_DIR / "blocking_metadata.json"
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    if "splits" not in meta:  # pre-v2 metadata had no per-split records
        meta = {"splits": {}}
    meta.update(blocker=f"{a.blocker}: TfidfBlocker {BLOCKERS[a.blocker]}",
                generator="scripts/aws/run_d2_blocking.py", top_k=K, validation_split="frozen (manifest-verified)")

    for name, frame in jobs:
        t1 = time.time()
        is_sample = (name == "train_sample") or (a.sample and name == "validation")
        out_path = None if is_sample and not a.write_sample else OUT_DIR / f"{name}_candidate_pairs.tsv"
        n_cand, metrics = generate(blocker, frame, gt, out_path, a.batch)
        record = {"blocker": a.blocker, "config": blocker.config, "config_sha": config_sha,
                  "generator_commit": commit, "generated_at": datetime.datetime.now().isoformat(timespec="seconds"),
                  "load_s": round(load_s, 1), "index_s": round(index_s, 1), "query_s": round(time.time() - t1, 1),
                  "s1_per_s": round(len(frame) / max(time.time() - t1, 1e-9), 1),
                  **distribution(n_cand, frame["country"].fillna("").to_numpy()),
                  **{k: round(v, 5) for k, v in metrics.items()}}
        if out_path:
            record.update(tsv=str(out_path.relative_to(config.REPO_ROOT)).replace("\\", "/"),
                          tsv_sha256=sha256(out_path))
            meta["splits"][name] = record
            meta_path.write_text(json.dumps(meta, indent=1))
        logged = log_run({"experiment_id": a.exp_id, "stage": "blocking-d2", "split": name,
                          "sample": a.sample, **record})
        print(f"\n== {name} ==")
        for key in ("n_s1", "total_pairs", "avg_cand_at_200", "p99_cand", "no_candidates", "recall_at_10",
                    "recall_at_50", "recall_at_200", "ceiling_f05_at_200", "index_s", "query_s", "s1_per_s"):
            print(f"{key:22s} {record.get(key)}")
        print(f"peak RSS {logged['peak_rss_gb']} GB | commit {commit} | config {config_sha}")


if __name__ == "__main__":
    main()
