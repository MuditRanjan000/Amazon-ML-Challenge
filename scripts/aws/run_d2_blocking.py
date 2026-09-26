"""Generate blocking candidates with ONE pinned config for every split.

FROZEN BLOCKER (2026-09-26, Mudit): word unigram (`--blocker word`, config_sha 657f59868e58).

  python scripts/aws/run_d2_blocking.py --split trainval              # train + validation S1 vs the train S2+S3 index
  python scripts/aws/run_d2_blocking.py --split test                  # test S1 vs the test S2+S3 index
  python scripts/aws/run_d2_blocking.py --split trainval --sample 5000  # timing/recall check on val, writes no TSV
  python scripts/aws/run_d2_blocking.py --split trainval --part 3/16  # one contiguous slice (fan-out), then:
  python scripts/aws/run_d2_blocking.py --merge 16                    # merge parts -> final TSVs + metadata

Writes artifacts/blocking/{train,validation,test}_candidate_pairs.tsv
(source1_entity_id, candidate_entity_id, score, rank) and records, per split, the blocker
config + hash, generator commit, TSV sha256, pair counts, candidate distribution and
recall / oracle-ceiling F0.5 in artifacts/blocking/blocking_metadata.json. Parts are
contiguous slices of each split, so merged output is byte-identical to a single run.
The merge also writes a fixed-seed 2,500-S1 train sample (+ ID manifest) for quick checks.
The index is sharded to disk under ER_CACHE_DIR unless --in-memory.
"""
import argparse
import datetime
import hashlib
import json
import logging
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
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
# Compared against D2 in BLK-019: identical except the analyzer (and max_df for the pruned variant).
BLOCKERS = {"d2": D2,
            "word": {**D2, "analyzer": "word", "ngram_range": [1, 1]},
            "word_maxdf02": {**D2, "analyzer": "word", "ngram_range": [1, 1], "max_df": 0.02}}
K = 200
K_SWEEP = (10, 25, 50, 100, 200)
OUT_DIR = config.REPO_ROOT / "artifacts" / "blocking"
SCHEMA = pa.schema([("source1_entity_id", pa.string()), ("candidate_entity_id", pa.string()),
                    ("score", pa.float32()), ("rank", pa.int16())])
HEADER = ("\t".join(SCHEMA.names) + "\n").encode()
# evaluator ratio -> the count it is averaged over when combining batches / parts
WEIGHTS = {"recall_at_": "true", "full_coverage_at_": "pos", "ceiling_f05_at_": "s1"}


def _open_tsv(path, schema=SCHEMA):
    """Header written by hand: pyarrow quotes header names even with quoting_style="none"."""
    sink = open(path, "wb")
    sink.write(HEADER)
    return sink, pacsv.CSVWriter(sink, schema, write_options=pacsv.WriteOptions(
        include_header=False, delimiter="\t", quoting_style="none"))


def _combine(parts):
    """Weighted mean of evaluator ratios over batches/parts (exact: same weights as one big evaluation)."""
    return {key: sum(p["m"][key] * p["w"][w] for p in parts) / max(sum(p["w"][w] for p in parts), 1)
            for key in (parts[0]["m"] if parts else {})
            for w in [next((w for prefix, w in WEIGHTS.items() if key.startswith(prefix)), None)] if w}


def generate(blocker, s1, gt, out_path, batch):
    """Query S1 in batches, stream candidates to TSV. Returns (per-S1 candidate counts, metrics, weights)."""
    sink, writer = _open_tsv(out_path) if out_path else (None, None)
    n_cand, parts = [], []
    for start in range(0, len(s1), batch):
        b = s1.iloc[start:start + batch]
        cands = blocker.generate_candidates(b, K)
        n_cand.append(np.bincount(cands["source1_entity_id"].cat.codes, minlength=len(b)))
        if writer:
            writer.write_table(pa.Table.from_pandas(cands, preserve_index=False).cast(SCHEMA))
        if gt is not None:
            ev = BlockingEvaluator(gt[gt["source1_entity_id"].isin(b["entity_id"])])
            parts.append({"m": ev.evaluate(cands, K_SWEEP), "w": {"s1": len(ev.s1), "true": int(ev.n_true.sum()),
                                                                  "pos": int((ev.n_true > 0).sum())}})
        logging.info("%s: %d / %d S1 done", Path(out_path).name if out_path else "sample", start + len(b), len(s1))
    if writer:
        writer.close()
        sink.close()
    weights = {w: sum(p["w"][w] for p in parts) for w in ("s1", "true", "pos")}
    return (np.concatenate(n_cand) if n_cand else np.zeros(0, np.int64)), _combine(parts), weights


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


def part_slice(frame, i, n):
    """i-th (1-based) of n contiguous slices: concatenating parts 1..n reproduces the frame order."""
    return frame.iloc[np.array_split(np.arange(len(frame)), n)[i - 1]]


def run_part(blocker, name, frame, gt, batch, stem, base):
    """Candidates for one slice of a split -> <stem>.tsv / .npz (ids, country, n_cand) / .json (record)."""
    t1 = time.time()
    n_cand, metrics, weights = generate(blocker, frame, gt, Path(f"{stem}.tsv"), batch)
    np.savez(f"{stem}.npz", ids=frame["entity_id"].to_numpy(dtype=str),
             country=frame["country"].fillna("").to_numpy(dtype=str), n_cand=n_cand)
    record = {**base, "split": name, "n_s1": len(frame), "query_s": round(time.time() - t1, 1),
              "metrics": metrics, "weights": weights}
    Path(f"{stem}.json").write_text(json.dumps(record))
    return record


def write_sample(out_dir, tsv, ids, n_cand, k):
    """Fixed-seed k-S1 slice of the train TSV (rows byte-identical to the full file) + ID manifest."""
    chosen = sorted(random.Random(config.SEED).sample(sorted(ids.tolist()), min(k, len(ids))))
    counts = n_cand[pd.Index(ids).get_indexer(chosen)]
    ids_path, sample_path = out_dir / "train_sample_s1_ids.csv", out_dir / "train_sample_candidate_pairs.tsv"
    pd.DataFrame({"source1_entity_id": chosen, "n_candidates": counts}).to_csv(ids_path, index=False,
                                                                                lineterminator="\n")
    as_text = pa.schema([(f, pa.string()) for f in SCHEMA.names])
    reader = pacsv.open_csv(tsv, parse_options=pacsv.ParseOptions(delimiter="\t", quote_char=False),
                            convert_options=pacsv.ConvertOptions(column_types=dict(zip(as_text.names, as_text.types)),
                                                                 strings_can_be_null=False),
                            read_options=pacsv.ReadOptions(block_size=1 << 26))
    keep, rows = pa.array(chosen), 0
    sink, writer = _open_tsv(sample_path, as_text)
    for rb in reader:
        rb = rb.filter(pc.is_in(rb.column(0), value_set=keep))
        rows += rb.num_rows
        writer.write_batch(rb)
    writer.close()
    sink.close()
    if rows != int(counts.sum()):
        raise SystemExit(f"sample: {rows} rows written but manifest says {int(counts.sum())}")
    return {"s1": len(chosen), "pairs": rows, "seed": config.SEED, "tsv": sample_path.name,
            "tsv_sha256": sha256(sample_path), "ids_csv": ids_path.name, "ids_sha256": sha256(ids_path)}


def merge_parts(out_dir, n, exp_id, sample_s1=2500, log=True):
    """Concatenate parts 1..n of each split, verify them, and write final TSVs + blocking_metadata.json."""
    part_dir = out_dir / "parts"
    meta_path = out_dir / "blocking_metadata.json"
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    meta = meta if "splits" in meta else {"splits": {}}
    recs = None
    for name in ("train", "validation", "test"):
        stems = [part_dir / f"{name}.part{i:02d}of{n:02d}" for i in range(1, n + 1)]
        have = [Path(f"{s}.json").exists() for s in stems]
        if not any(have):
            continue
        if not all(have):
            raise SystemExit(f"{name}: missing parts {[i + 1 for i, h in enumerate(have) if not h]}")
        recs = [json.loads(Path(f"{s}.json").read_text()) for s in stems]
        for key in ("blocker", "config_sha", "generator_commit"):
            if len({r[key] for r in recs}) != 1:
                raise SystemExit(f"{name}: parts disagree on {key}: {sorted({r[key] for r in recs})}")
        out = out_dir / f"{name}_candidate_pairs.tsv"
        h, rows = hashlib.sha256(HEADER), 0
        with open(out, "wb") as dst:
            dst.write(HEADER)
            for s in stems:
                with open(f"{s}.tsv", "rb") as src:
                    if src.readline() != HEADER:
                        raise SystemExit(f"{s}.tsv: unexpected header")
                    for block in iter(lambda: src.read(1 << 24), b""):
                        dst.write(block)
                        h.update(block)
                        rows += block.count(b"\n")
        arrays = [np.load(f"{s}.npz") for s in stems]
        ids = np.concatenate([a["ids"] for a in arrays])
        n_cand = np.concatenate([a["n_cand"] for a in arrays])
        dist = distribution(n_cand, np.concatenate([a["country"] for a in arrays]))
        if rows != dist["total_pairs"] or len(ids) != sum(r["n_s1"] for r in recs) or len(set(ids)) != len(ids):
            raise SystemExit(f"{name}: row/S1 counts do not add up ({rows} rows vs {dist['total_pairs']} pairs)")
        metrics = _combine([{"m": r["metrics"], "w": r["weights"]} for r in recs])
        record = {k: recs[0][k] for k in ("blocker", "config", "config_sha", "generator_commit")}
        record.update(generated_at=max(r["generated_at"] for r in recs), parts=n,
                      index_s=max(r["index_s"] for r in recs), query_s=max(r["query_s"] for r in recs),
                      **dist, **{k: round(v, 5) for k, v in metrics.items()},
                      tsv=f"artifacts/blocking/{out.name}", tsv_sha256=h.hexdigest())
        if name == "train" and sample_s1:
            record["sample"] = write_sample(out_dir, out, ids, n_cand, sample_s1)
        meta["splits"][name] = record
        if log:
            log_run({"experiment_id": exp_id, "stage": "blocking", "split": name,
                     **{k: v for k, v in record.items() if k != "by_country"}})
        logging.info("%s: %d S1, %d pairs, sha256 %s", name, len(ids), rows, record["tsv_sha256"][:12])
    if recs is None:
        raise SystemExit(f"no parts found in {part_dir} for n={n}")
    meta.update(blocker=f"{recs[0]['blocker']}: TfidfBlocker {recs[0]['config']}",
                generator="scripts/aws/run_d2_blocking.py", top_k=K, validation_split="frozen (manifest-verified)")
    meta_path.write_text(json.dumps(meta, indent=1))
    return meta


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--split", choices=["trainval", "test"])
    p.add_argument("--sample", type=int, default=0, help="only N random val S1, metrics only (no TSV)")
    p.add_argument("--in-memory", action="store_true", help="keep the index in RAM")
    p.add_argument("--batch", type=int, default=250_000, help="S1 per query batch / TSV write")
    p.add_argument("--exp-id", default="BLK-020")
    p.add_argument("--blocker", choices=list(BLOCKERS), default="word")
    p.add_argument("--part", default="1/1", help="i/n: only the i-th of n contiguous slices of each split")
    p.add_argument("--merge", type=int, default=0, help="n: merge parts 1..n into final TSVs + metadata")
    a = p.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    if a.merge:
        merge_parts(OUT_DIR, a.merge, a.exp_id)
        return
    if not a.split:
        p.error("--split is required unless --merge")
    i, n = map(int, a.part.split("/"))
    if not 1 <= i <= n:
        p.error("--part must be i/n with 1 <= i <= n")
    commit = git_commit()
    if commit.endswith("-dirty") and not a.sample:
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
        if a.sample:
            val_ids = set(random.Random(config.SEED).sample(sorted(val_ids), a.sample))
        else:
            jobs.append(("train", s1[s1["entity_id"].isin(train_ids)]))
        jobs.append(("validation", s1[s1["entity_id"].isin(val_ids)]))

    t_index = time.time()
    blocker = TfidfBlocker(**BLOCKERS[a.blocker], n_jobs=config.N_JOBS,
                           cache_dir=None if a.in_memory else config.CACHE_DIR / "tfidf_index").fit(pool)
    del pool
    base = {"blocker": a.blocker, "config": blocker.config,
            "config_sha": hashlib.sha256(json.dumps(blocker.config, sort_keys=True).encode()).hexdigest()[:12],
            "generator_commit": commit, "generated_at": datetime.datetime.now().isoformat(timespec="seconds"),
            "load_s": round(load_s, 1), "index_s": round(time.time() - t_index, 1)}

    (OUT_DIR / "parts").mkdir(parents=True, exist_ok=True)
    for name, frame in jobs:
        if a.sample:
            t1 = time.time()
            n_cand, metrics, _ = generate(blocker, frame, gt, None, a.batch)
            record = {**base, "query_s": round(time.time() - t1, 1),
                      "s1_per_s": round(len(frame) / max(time.time() - t1, 1e-9), 1),
                      **distribution(n_cand, frame["country"].fillna("").to_numpy()),
                      **{k: round(v, 5) for k, v in metrics.items()}}
            log_run({"experiment_id": a.exp_id, "stage": "blocking", "split": name, "sample": a.sample, **record})
        else:
            record = run_part(blocker, name, part_slice(frame, i, n), gt, a.batch,
                              OUT_DIR / "parts" / f"{name}.part{i:02d}of{n:02d}", base)
            record["s1_per_s"] = round(record["n_s1"] / max(record["query_s"], 1e-9), 1)
            record.update({k: round(v, 5) for k, v in record.pop("metrics").items()})
        print(f"\n== {name} (part {i}/{n}) ==")
        for key in ("n_s1", "recall_at_10", "recall_at_50", "recall_at_200", "ceiling_f05_at_200",
                    "index_s", "query_s", "s1_per_s"):
            print(f"{key:22s} {record.get(key)}")
        print(f"commit {commit} | config {base['config_sha']}")
    if not a.sample and n == 1:
        merge_parts(OUT_DIR, 1, a.exp_id)


if __name__ == "__main__":
    main()
