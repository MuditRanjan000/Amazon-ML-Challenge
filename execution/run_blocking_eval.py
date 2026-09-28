"""Benchmark a blocking configuration on the frozen validation split.

Examples:
  # fast dev loop: 20k val S1, forward TF-IDF name 3-grams
  python execution/run_blocking_eval.py --exp-id BLK-010 --method tfidf --direction forward --sample 20000
  # the recommended v1 union (exact + forward + reverse) on the full val split, saving candidates
  python execution/run_blocking_eval.py --exp-id BLK-020 --method tfidf --direction both --k 50 --rev-k 5 --save

Metrics are printed and appended to experiments/results/experiments.jsonl
(git commit, config, runtime, peak RSS included). See directives/blocking.md.
"""
import argparse
import logging
import random
import time

from entity_resolution import config
from entity_resolution.blocking.base import union_candidates
from entity_resolution.blocking.evaluator import BlockingEvaluator
from entity_resolution.blocking.exact_name import ExactNameBlocker
from entity_resolution.blocking.tfidf_blocking import TfidfBlocker, reverse_candidates
from entity_resolution.data.loader import DataLoader
from entity_resolution.data.validation_split import create_validation_split
from entity_resolution.tracking import log_run

K_SWEEP = (10, 25, 50, 100, 200)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--exp-id", required=True)
    p.add_argument("--method", choices=["exact", "tfidf"], default="tfidf")
    p.add_argument("--direction", choices=["forward", "reverse", "both"], default="forward",
                   help="forward: S1 queries S2+S3 index; reverse: S2/S3 query an S1 index; both: union + exact")
    # Defaults = best measured config (3k val, full index): word unigrams on name+address,
    # R@50 0.958 / ceiling F0.5@50 0.985 vs 0.706 / 0.829 for char-3gram name-only.
    p.add_argument("--fields", nargs="+", default=["business_name", "business_address"])
    p.add_argument("--analyzer", choices=["word", "char_wb", "char"], default="word")
    p.add_argument("--ngram", nargs=2, type=int, default=[1, 1])
    p.add_argument("--max-df", type=float, default=1.0, help="drop terms in more than this fraction of index records")
    p.add_argument("--strip-legal", action="store_true")
    p.add_argument("--no-partition", action="store_true", help="do not partition the index by country")
    p.add_argument("--k", type=int, default=200, help="forward top-K per S1")
    p.add_argument("--rev-k", type=int, default=5, help="reverse top-k S1 per S2/S3 record")
    p.add_argument("--sample", type=int, default=0, help="evaluate on N random val S1 (0 = all val)")
    p.add_argument("--save", action="store_true", help="write candidates parquet to output/candidates/")
    p.add_argument("--notes", default="")
    a = p.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    t0 = time.time()

    loader = DataLoader()
    gt = loader.load_ground_truth()
    _, val_ids = create_validation_split(gt)
    if a.sample:
        val_ids = set(random.Random(config.SEED).sample(sorted(val_ids), a.sample))
    val_gt = gt[gt["source1_entity_id"].isin(val_ids)]
    s1_all = loader.load_source("train", 1)  # every train S1 competes in the index, as in test
    s1_val = s1_all[s1_all["entity_id"].isin(val_ids)]
    pool = loader.load_candidates_pool("train")
    logging.info("loaded: %d val S1, %d S1 total, %d S2+S3 (%.0fs)", len(s1_val), len(s1_all), len(pool),
                 time.time() - t0)

    tfidf_kwargs = dict(text_fields=a.fields, analyzer=a.analyzer, ngram_range=a.ngram, max_df=a.max_df,
                        strip_legal=a.strip_legal, partition_by_country=not a.no_partition)
    channels = {}
    if a.method == "exact":
        channels["exact"] = ExactNameBlocker().build_index(pool, pool.iloc[:0]).generate_candidates(s1_val, a.k)
    else:
        if a.direction in ("forward", "both"):
            channels["fwd"] = TfidfBlocker(**tfidf_kwargs).fit(pool).generate_candidates(s1_val, a.k)
        if a.direction in ("reverse", "both"):
            rev = reverse_candidates(s1_all, pool, a.rev_k, **tfidf_kwargs)
            channels["rev"] = rev[rev["source1_entity_id"].isin(val_ids)].rename(
                columns={"rev_rank": "rank", "rev_score": "score"})
        if a.direction == "both":
            channels["exact"] = ExactNameBlocker().build_index(pool, pool.iloc[:0]).generate_candidates(s1_val, a.k)

    evaluator = BlockingEvaluator(val_gt)
    metrics = {}
    for name, frame in channels.items():
        sweep = tuple(k for k in K_SWEEP if k <= (a.rev_k if name == "rev" else a.k)) or (a.k,)
        metrics.update({f"{name}.{m}": v for m, v in evaluator.evaluate(frame, sweep).items()})
    if len(channels) > 1:
        union = union_candidates(channels)
        metrics.update({f"union.{m}": v for m, v in evaluator.evaluate(union, k_values=None).items()})
    else:
        union = next(iter(channels.values()))

    if a.save:
        out = config.OUTPUT_DIR / "candidates" / f"candidates_val_{a.exp_id}.parquet"
        out.parent.mkdir(parents=True, exist_ok=True)
        union.to_parquet(out, index=False)
        logging.info("saved %s", out)

    record = log_run({"experiment_id": a.exp_id, "stage": "blocking", "config": vars(a),
                      "n_val_s1": len(s1_val), "runtime_s": round(time.time() - t0, 1),
                      "metrics": {k: (round(v, 5) if isinstance(v, float) else v) for k, v in metrics.items()}})
    for key, value in record["metrics"].items():
        if any(s in key for s in ("recall_at", "ceiling", "avg_cand", "total_pairs", "coverage", "bucket")):
            print(f"{key:40s} {value}")
    print(f"runtime {record['runtime_s']}s | peak RSS {record['peak_rss_gb']} GB | commit {record['git_commit']}")


if __name__ == "__main__":
    main()
