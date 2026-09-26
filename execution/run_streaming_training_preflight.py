"""Disk-backed, training-only Logistic Regression preflight for supplied candidates.

The command is deliberately unable to evaluate, threshold, or consume
validation IDs.  It is for a versioned training candidate TSV plus the exact
Source-1 sample universe that produced it.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pickle
import sqlite3
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.entity_resolution.features import PairwiseFeatureExtractor
from src.entity_resolution.matching.normalization import normalize_for_matching
from src.entity_resolution.matching.records import SQLiteRecordStore
from src.entity_resolution.models.experiments import (
    process_memory_bytes,
    read_source1_ids,
    select_ground_truth,
    sha256_file,
)
from src.entity_resolution.models.logistic import (
    LogisticRegressionConfig,
    fit_logistic_regression,
    predict_match_probabilities,
)
from src.entity_resolution.models.streaming import CandidatePairSpool, StreamingConfig


def _parse_ids(value: object) -> set[str]:
    if pd.isna(value) or not str(value).strip():
        return set()
    return {part.strip() for part in str(value).split(",") if part.strip()}


def _iter_corpus(database: Path, table: str):
    connection = sqlite3.connect(database)
    try:
        for (text,) in connection.execute(f"SELECT text FROM {table} ORDER BY rowid"):
            yield text
    finally:
        connection.close()


def _candidate_recall(database: Path, spool: CandidatePairSpool, truth_map: dict[str, set[str]]) -> dict[str, int | float]:
    connection = sqlite3.connect(database)
    try:
        connection.execute("CREATE TABLE truth_pairs (source1_entity_id TEXT NOT NULL, candidate_entity_id TEXT NOT NULL, PRIMARY KEY(source1_entity_id, candidate_entity_id))")
        connection.executemany(
            "INSERT INTO truth_pairs VALUES (?, ?)",
            [(source, candidate) for source, candidates in truth_map.items() for candidate in candidates],
        )
        connection.execute("ATTACH DATABASE ? AS candidates", (str(spool.database_path.resolve()),))
        total = int(connection.execute("SELECT COUNT(*) FROM truth_pairs").fetchone()[0])
        retrieved = int(
            connection.execute(
                "SELECT COUNT(*) FROM truth_pairs AS t JOIN candidates.pairs AS p "
                "ON p.source1_entity_id=t.source1_entity_id AND p.candidate_entity_id=t.candidate_entity_id"
            ).fetchone()[0]
        )
        return {
            "total_true_pairs": total,
            "retrieved_true_pairs": retrieved,
            "candidate_recall": retrieved / total if total else 1.0,
            "blocking_misses": total - retrieved,
        }
    finally:
        connection.close()


def parser() -> argparse.ArgumentParser:
    command = argparse.ArgumentParser(description=__doc__)
    command.add_argument("--candidates", required=True, type=Path)
    command.add_argument("--sample-source1-ids", required=True, type=Path)
    command.add_argument("--frozen-train-source1-ids", required=True, type=Path)
    command.add_argument("--ground-truth", required=True, type=Path)
    command.add_argument("--store", required=True, type=Path)
    command.add_argument("--run-dir", required=True, type=Path)
    command.add_argument("--candidate-version", required=True)
    command.add_argument("--generator-commit", default="69a91f1")
    command.add_argument("--config-hash", default="657f59868e58")
    command.add_argument("--batch-size", type=int, default=5_000)
    command.add_argument("--max-pairs", type=int, help="Optional explicit diagnostic guard; excess pairs fail, never truncate.")
    command.add_argument("--logistic-c", type=float, default=1.0)
    command.add_argument("--class-weight", choices=("balanced",), default=None)
    return command


def main() -> None:
    args = parser().parse_args()
    required = [args.candidates, args.sample_source1_ids, args.frozen_train_source1_ids, args.ground_truth, args.store]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Required training-preflight artifact(s) missing: {missing}")
    if args.run_dir.exists() and any(args.run_dir.iterdir()):
        raise FileExistsError(f"Run directory must be empty for a new preflight: {args.run_dir}")
    args.run_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    memory_start = process_memory_bytes()

    sample_ids = read_source1_ids(args.sample_source1_ids)
    frozen_ids = read_source1_ids(args.frozen_train_source1_ids)
    outside = set(sample_ids.astype(str)) - set(frozen_ids.astype(str))
    if outside:
        raise ValueError(f"Training sample contains IDs outside Mudit's frozen train split, e.g. {sorted(outside)[:5]}")
    truth = pd.read_csv(args.ground_truth, sep="\t", dtype="string")
    selected_truth = select_ground_truth(truth, sample_ids)
    truth_map = {str(row.source1_entity_id): _parse_ids(row.matched_entity_ids) for row in selected_truth.itertuples(index=False)}

    spool = CandidatePairSpool(args.run_dir / "candidate_spool.sqlite")
    spool_metadata = spool.build(args.candidates, args.sample_source1_ids)
    pair_count = int(spool_metadata["candidate_pair_count"])
    if args.max_pairs is not None and pair_count > args.max_pairs:
        raise ValueError(f"Candidate artifact has {pair_count:,} pairs, exceeding explicit max_pairs={args.max_pairs:,}; no truncation occurred.")

    corpus_database = args.run_dir / "training_corpus.sqlite"
    corpus = sqlite3.connect(corpus_database)
    corpus.execute("CREATE TABLE names (text TEXT PRIMARY KEY)")
    corpus.execute("CREATE TABLE addresses (text TEXT PRIMARY KEY)")
    labels_path = args.run_dir / "training_labels.uint8.mmap"
    feature_path = args.run_dir / "training_features.float32.mmap"
    feature_columns: list[str] | None = None
    features = None
    labels = np.memmap(labels_path, dtype=np.uint8, mode="w+", shape=(pair_count,))
    row_offset = 0
    corpus_started = time.perf_counter()
    store = SQLiteRecordStore(args.store)
    try:
        for _batch_number, pairs in spool.iter_group_batches(args.batch_size):
            joined, _metadata = next(store.iter_joined_batches(pairs, batch_size=len(pairs)))
            names = pd.concat([joined["source1_business_name"], joined["candidate_business_name"]]).map(normalize_for_matching)
            addresses = pd.concat([joined["source1_business_address"], joined["candidate_business_address"]]).map(normalize_for_matching)
            corpus.executemany("INSERT OR IGNORE INTO names VALUES (?)", ((value,) for value in names if value))
            corpus.executemany("INSERT OR IGNORE INTO addresses VALUES (?)", ((value,) for value in addresses if value))
        corpus.commit()
        extractor = PairwiseFeatureExtractor().fit_from_normalized_corpora(
            _iter_corpus(corpus_database, "names"), _iter_corpus(corpus_database, "addresses")
        )
        corpus_seconds = time.perf_counter() - corpus_started

        feature_started = time.perf_counter()
        for _batch_number, pairs in spool.iter_group_batches(args.batch_size):
            joined, _metadata = next(store.iter_joined_batches(pairs, batch_size=len(pairs)))
            batch_features = extractor.transform(joined)
            if feature_columns is None:
                feature_columns = list(extractor.feature_order)
                if len(feature_columns) != 43:
                    raise AssertionError(f"Expected 43 pairwise features, received {len(feature_columns)}.")
                features = np.memmap(feature_path, dtype=np.float32, mode="w+", shape=(pair_count, len(feature_columns)))
            if list(extractor.feature_order) != feature_columns:
                raise AssertionError("Feature schema changed across training batches.")
            stop = row_offset + len(pairs)
            features[row_offset:stop] = batch_features.loc[:, feature_columns].to_numpy(dtype=np.float32)
            labels[row_offset:stop] = np.fromiter(
                (int(candidate in truth_map[str(source)]) for source, candidate in pairs.itertuples(index=False)),
                dtype=np.uint8,
                count=len(pairs),
            )
            row_offset = stop
        if row_offset != pair_count or feature_columns is None or features is None:
            raise AssertionError("Feature extraction did not cover every candidate pair.")
        features.flush()
        labels.flush()
        extractor.save(args.run_dir / "features.pkl")
        feature_seconds = time.perf_counter() - feature_started

        label_series = pd.Series(np.asarray(labels), dtype="int8")
        if label_series.nunique() < 2:
            raise ValueError("Training candidates contain no usable positive/negative class variation for Logistic Regression.")
        model_started = time.perf_counter()
        feature_frame = pd.DataFrame(features, columns=feature_columns, copy=False)
        model = fit_logistic_regression(
            feature_frame,
            label_series,
            feature_columns,
            LogisticRegressionConfig(c=args.logistic_c, class_weight=args.class_weight),
        )
        with (args.run_dir / "logistic_regression.pkl").open("wb") as handle:
            pickle.dump(model, handle)
        model_seconds = time.perf_counter() - model_started

        score_started = time.perf_counter()
        wrote_header = False
        score_path = args.run_dir / "training_logistic_scores.tsv"
        for _batch_number, pairs in spool.iter_group_batches(args.batch_size):
            joined, _metadata = next(store.iter_joined_batches(pairs, batch_size=len(pairs)))
            batch_features = extractor.transform(joined)
            probabilities = predict_match_probabilities(model, batch_features, feature_columns)
            pd.DataFrame(
                {
                    "source1_entity_id": pairs["source1_entity_id"],
                    "candidate_entity_id": pairs["candidate_entity_id"],
                    "match_probability": probabilities,
                }
            ).to_csv(score_path, sep="\t", index=False, mode="a" if wrote_header else "w", header=not wrote_header)
            wrote_header = True
        score_seconds = time.perf_counter() - score_started
    finally:
        corpus.close()

    recall = _candidate_recall(corpus_database, spool, truth_map)
    report = {
        "status": "training_preflight_complete_no_validation_evaluation",
        "blocker": {
            "candidate_version": args.candidate_version,
            "generator_commit": args.generator_commit,
            "config_hash": args.config_hash,
        },
        "input_hashes": {
            "candidates": sha256_file(args.candidates),
            "sample_source1_ids": sha256_file(args.sample_source1_ids),
            "frozen_train_source1_ids": sha256_file(args.frozen_train_source1_ids),
            "ground_truth": sha256_file(args.ground_truth),
            "record_store": sha256_file(args.store),
        },
        "candidate_spool": spool_metadata,
        "training": {
            "source1_entity_count": len(sample_ids),
            "candidate_pair_count": pair_count,
            "zero_candidate_source1_count": spool_metadata["zero_candidate_source1_count"],
            "positive_pairs": int(np.asarray(labels).sum()),
            "negative_pairs": int(pair_count - np.asarray(labels).sum()),
            "retrieved_positive_pairs": recall["retrieved_true_pairs"],
            "retrieved_candidate_negative_pairs": int(pair_count - np.asarray(labels).sum()),
            "positive_pair_rate": float(np.asarray(labels).mean()),
            "candidate_recall": recall,
            "feature_count": len(feature_columns),
        },
        "model": {"type": "L2 Logistic Regression", "c": args.logistic_c, "class_weight": args.class_weight, "calibration": "none"},
        "runtime_seconds": {
            "corpus_and_tfidf_fit": corpus_seconds,
            "feature_extraction": feature_seconds,
            "model_fit": model_seconds,
            "training_score_output": score_seconds,
            "total": time.perf_counter() - started,
        },
        "process_memory_bytes": {"start": memory_start, "end": process_memory_bytes()},
        "outputs": {"feature_artifact": "features.pkl", "model": "logistic_regression.pkl", "scores": "training_logistic_scores.tsv"},
    }
    (args.run_dir / "training_preflight_report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
