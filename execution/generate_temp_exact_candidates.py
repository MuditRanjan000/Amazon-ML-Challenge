"""Generate bounded temporary exact-name candidates from the full S2/S3 corpus.

This is an Ashank-owned bridge while Aayush's EXP-002B artifacts are pending.
It uses the existing exact-name blocker's lower-case, punctuation/suffix/space
normalisation semantics, but materialises a disk-backed SQLite index so the
10M-record candidate corpus is never loaded as one DataFrame.  It never reads
ground truth while generating candidates; labels are read only afterwards to
report candidate recall.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sqlite3
import sys
import time
from collections import Counter
from contextlib import closing
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.entity_resolution.data.config import resolve_data_dir
from src.entity_resolution.models.experiments import (
    candidate_recall,
    read_source1_ids,
    select_ground_truth,
    sha256_file,
    validate_frozen_split_manifest,
)


_PUNCTUATION = re.compile(r"[^\w\s]", flags=re.UNICODE)
_SUFFIXES = re.compile(r"\b(ltd|limited|pvt|private|corp|corporation)\b", flags=re.IGNORECASE)
_WHITESPACE = re.compile(r"\s+")


def normalize_exact_name(value: object) -> str:
    """Equivalent scalar form of ``ExactNameBlocker._normalize``."""

    text = "" if value is None else str(value).lower()
    text = _PUNCTUATION.sub(" ", text)
    text = _SUFFIXES.sub(" ", text)
    return _WHITESPACE.sub(" ", text).strip()


def _index_exists(path: Path) -> bool:
    if not path.is_file():
        return False
    with sqlite3.connect(path) as connection:
        return connection.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name='exact_name_candidates'"
        ).fetchone() is not None


def build_index(index_path: Path, source_paths: list[Path], rebuild: bool) -> dict[str, int]:
    if _index_exists(index_path) and not rebuild:
        raise FileExistsError(f"Temporary exact-name index already exists: {index_path}. Use --rebuild-index to replace it.")
    if index_path.exists() and rebuild:
        # SQLite removes/recreates its own table; no broad filesystem deletion is used.
        pass
    index_path.parent.mkdir(parents=True, exist_ok=True)
    counts: Counter[str] = Counter()
    with closing(sqlite3.connect(index_path)) as connection, connection:
        connection.execute("DROP TABLE IF EXISTS exact_name_candidates")
        connection.execute(
            "CREATE TABLE exact_name_candidates (country TEXT NOT NULL, normalized_name TEXT NOT NULL, entity_id TEXT PRIMARY KEY)"
        )
        for source_path in source_paths:
            rows: list[tuple[str, str, str]] = []
            with source_path.open(encoding="utf-8", newline="") as handle:
                for row in csv.DictReader(handle, delimiter="\t"):
                    normalized = normalize_exact_name(row.get("business_name"))
                    if normalized:
                        rows.append(((row.get("country") or "").strip(), normalized, (row.get("entity_id") or "").strip()))
                    if len(rows) >= 10_000:
                        connection.executemany("INSERT INTO exact_name_candidates VALUES (?, ?, ?)", rows)
                        counts["indexed_records"] += len(rows)
                        rows.clear()
                if rows:
                    connection.executemany("INSERT INTO exact_name_candidates VALUES (?, ?, ?)", rows)
                    counts["indexed_records"] += len(rows)
            connection.commit()
        connection.execute(
            "CREATE INDEX exact_name_lookup ON exact_name_candidates(country, normalized_name, entity_id)"
        )
        connection.commit()
    return dict(counts)


def _bounded_ids(path: Path, maximum: int) -> pd.Series:
    if maximum <= 0:
        raise ValueError("max_source1_entities must be positive.")
    return read_source1_ids(path).iloc[:maximum].reset_index(drop=True)


def generate_candidates(
    *,
    index_path: Path,
    source1_path: Path,
    source1_ids: pd.Series,
    output_path: Path,
    max_pairs: int,
) -> tuple[pd.DataFrame, dict[str, int | float]]:
    """Stream selected S1 records, retaining every exact-name candidate group."""

    if output_path.exists():
        raise FileExistsError(f"Refusing to overwrite candidate artifact: {output_path}")
    allowed = set(source1_ids.astype(str))
    seen: set[str] = set()
    pair_rows: list[dict[str, object]] = []
    counts: Counter[str] = Counter()
    started = time.perf_counter()
    with closing(sqlite3.connect(index_path)) as connection, source1_path.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            source_id = (row.get("entity_id") or "").strip()
            if source_id not in allowed:
                continue
            seen.add(source_id)
            normalized = normalize_exact_name(row.get("business_name"))
            if not normalized:
                continue
            candidates = connection.execute(
                "SELECT entity_id FROM exact_name_candidates WHERE country = ? AND normalized_name = ? ORDER BY entity_id",
                ((row.get("country") or "").strip(), normalized),
            ).fetchall()
            counts["entities_with_candidates"] += int(bool(candidates))
            counts["candidate_pairs"] += len(candidates)
            if counts["candidate_pairs"] > max_pairs:
                raise ValueError(
                    f"Complete candidate groups exceed max_pairs={max_pairs:,}; reduce selected entities or raise the bound."
                )
            pair_rows.extend(
                {
                    "source1_entity_id": source_id,
                    "candidate_entity_id": candidate_id,
                    "score": 1.0,
                    "rank": rank,
                }
                for rank, (candidate_id,) in enumerate(candidates, start=1)
            )
    if seen != allowed:
        raise ValueError(f"Source1 record file lacks selected IDs, e.g. {sorted(allowed - seen)[:5]}.")
    candidates = pd.DataFrame(pair_rows, columns=["source1_entity_id", "candidate_entity_id", "score", "rank"])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    candidates.to_csv(output_path, sep="\t", index=False)
    counts.update(
        {
            "selected_source1_entities": len(source1_ids),
            "entities_without_candidates": len(source1_ids) - counts["entities_with_candidates"],
            "average_candidates_per_selected_entity": len(candidates) / len(source1_ids),
            "generation_seconds": time.perf_counter() - started,
        }
    )
    return candidates, dict(counts)


def parser() -> argparse.ArgumentParser:
    command = argparse.ArgumentParser(description=__doc__)
    command.add_argument("--split", choices=("train", "validation"), required=True)
    command.add_argument("--source1-ids", type=Path, required=True)
    command.add_argument("--other-source1-ids", type=Path, required=True, help="Other frozen split file, used for manifest disjointness verification.")
    command.add_argument("--split-manifest", type=Path, required=True)
    command.add_argument("--ground-truth", type=Path, required=True)
    command.add_argument("--index", type=Path, required=True)
    command.add_argument("--rebuild-index", action="store_true")
    command.add_argument("--output", type=Path, required=True)
    command.add_argument("--data-dir", type=Path)
    command.add_argument("--max-source1-entities", type=int, default=1_000)
    command.add_argument("--max-pairs", type=int, default=50_000)
    command.add_argument("--candidate-version", default="TEMP-EXACT-NAME-v1")
    return command


def main() -> None:
    args = parser().parse_args()
    required = [args.source1_ids, args.other_source1_ids, args.split_manifest, args.ground_truth]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Required frozen artifact(s) missing: {missing}")
    train_ids_path, validation_ids_path = (
        (args.source1_ids, args.other_source1_ids)
        if args.split == "train"
        else (args.other_source1_ids, args.source1_ids)
    )
    split_manifest = validate_frozen_split_manifest(args.split_manifest, train_ids_path, validation_ids_path)
    data_dir = resolve_data_dir(args.data_dir)
    source_paths = [data_dir / "train" / "train_source2.tsv", data_dir / "train" / "train_source3.tsv"]
    if args.rebuild_index or not _index_exists(args.index):
        index_stats = build_index(args.index, source_paths, rebuild=args.rebuild_index)
    else:
        index_stats = {"indexed_records": "existing_index_reused"}
    selected_ids = _bounded_ids(args.source1_ids, args.max_source1_entities)
    candidates, generation = generate_candidates(
        index_path=args.index,
        source1_path=data_dir / "train" / "train_source1.tsv",
        source1_ids=selected_ids,
        output_path=args.output,
        max_pairs=args.max_pairs,
    )
    ground_truth = pd.read_csv(args.ground_truth, sep="\t", dtype="string")
    truth = select_ground_truth(ground_truth, selected_ids)
    recall = candidate_recall(candidates, truth)
    recall.pop("missing_rows")
    report = {
        "candidate_version": args.candidate_version,
        "generator": "temporary_exact_name_sqlite_v1",
        "normalization": "existing ExactNameBlocker lower/punctuation/suffix/whitespace semantics",
        "split": args.split,
        "frozen_split_manifest": split_manifest,
        "input_hashes": {
            "source1_ids": sha256_file(args.source1_ids),
            "other_source1_ids": sha256_file(args.other_source1_ids),
            "ground_truth": sha256_file(args.ground_truth),
        },
        "index": {"path": str(args.index), **index_stats},
        "generation": generation,
        "candidate_recall": recall,
    }
    report_path = args.output.with_suffix(args.output.suffix + ".json")
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({"candidate_output": str(args.output), "report": str(report_path), **generation, **recall}, indent=2))


if __name__ == "__main__":
    main()
