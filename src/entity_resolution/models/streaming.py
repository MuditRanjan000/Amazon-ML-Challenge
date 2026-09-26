"""Disk-backed scoring and decision adapters for large candidate artifacts.

This module intentionally does not implement a new evaluator or validation
split.  It keeps candidate pairs in SQLite, scores complete Source-1 groups in
bounded batches, and emits the existing integration contracts.  The shared
``Evaluator`` remains the authority once a complete decision file is built.
"""

from __future__ import annotations

import csv
import gzip
import hashlib
import json
import os
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import numpy as np
import pandas as pd

from src.entity_resolution.evaluation.evaluator import Evaluator

from ..features import PairwiseFeatureExtractor
from ..matching.contracts import PROBABILITY_SCORE_COLUMNS, RAW_SCORE_COLUMNS, validate_complete_source1_ids
from ..matching.records import SQLiteRecordStore
from .logistic import predict_match_probabilities
from .rule import DeterministicScorer


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, value: dict[str, object]) -> None:
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _source_id_column(fieldnames: list[str] | None, path: Path) -> str:
    for column in ("source1_entity_id", "entity_id"):
        if fieldnames and column in fieldnames:
            return column
    raise ValueError(f"{path} must contain source1_entity_id or entity_id.")


def _read_source_ids(path: str | Path) -> Iterator[tuple[int, str]]:
    source_path = Path(path)
    delimiter = "," if source_path.suffix.lower() == ".csv" else "\t"
    with source_path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter=delimiter)
        column = _source_id_column(reader.fieldnames, source_path)
        for ordinal, row in enumerate(reader):
            entity_id = (row.get(column) or "").strip()
            if not entity_id.startswith("S1-"):
                raise ValueError(f"Invalid Source-1 ID at {source_path}:{ordinal + 2}: {entity_id!r}.")
            yield ordinal, entity_id


def write_source1_prefix(source1_ids_path: str | Path, output_path: str | Path, count: int) -> dict[str, object]:
    """Write a reproducible diagnostic prefix without modifying frozen IDs."""

    if count <= 0:
        raise ValueError("count must be positive.")
    selected = []
    for ordinal, entity_id in _read_source_ids(source1_ids_path):
        if ordinal >= count:
            break
        selected.append(entity_id)
    if len(selected) != count:
        raise ValueError(f"Requested {count:,} Source-1 IDs but input contains only {len(selected):,}.")
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(["source1_entity_id"])
        writer.writerows((entity_id,) for entity_id in selected)
    return {
        "selection": "first_n_frozen_source1_ids_diagnostic_only",
        "source1_id_count": count,
        "original_source1_ids_sha256": sha256_file(source1_ids_path),
        "selected_source1_ids_sha256": sha256_file(output),
        "path": str(output),
    }


@dataclass(frozen=True)
class StreamingConfig:
    """Bounded batch and output settings for an immutable candidate artifact."""

    pair_batch_size: int = 5_000
    include_rule_raw_scores: bool = True

    def __post_init__(self) -> None:
        if self.pair_batch_size <= 0:
            raise ValueError("pair_batch_size must be positive.")


class CandidatePairSpool:
    """SQLite representation that de-duplicates pairs and restores whole groups.

    Candidate input order need not be grouped.  The spool sorts by Source-1 ID
    for scoring, so a group that crosses an input TSV chunk is still scored as a
    unit.  Source-1 IDs live in a separate table, retaining zero-candidate
    entities for decision construction.
    """

    schema_version = 1

    def __init__(self, database_path: str | Path):
        self.database_path = Path(database_path)

    @property
    def metadata_path(self) -> Path:
        return self.database_path.with_suffix(self.database_path.suffix + ".json")

    def build(
        self, candidate_path: str | Path, source1_ids_path: str | Path, *, reuse: bool = True, max_rank: int | None = None
    ) -> dict[str, object]:
        candidate_path = Path(candidate_path)
        source1_ids_path = Path(source1_ids_path)
        if max_rank is not None and max_rank <= 0:
            raise ValueError("max_rank must be positive when supplied.")
        identity = {
            "schema_version": self.schema_version,
            "candidate_path": str(candidate_path.resolve()),
            "candidate_sha256": sha256_file(candidate_path),
            "source1_ids_path": str(source1_ids_path.resolve()),
            "source1_ids_sha256": sha256_file(source1_ids_path),
            "max_rank": max_rank,
        }
        if self.database_path.exists() or self.metadata_path.exists():
            if not (self.database_path.exists() and self.metadata_path.exists()):
                raise FileExistsError("Incomplete candidate spool; remove it explicitly before rebuilding.")
            existing = _json(self.metadata_path)
            if reuse and all(existing.get(key) == value for key, value in identity.items()):
                return existing
            raise FileExistsError("Candidate spool identity differs; use a new spool path rather than overwriting it.")

        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        connection = sqlite3.connect(self.database_path)
        try:
            connection.execute("CREATE TABLE source1_ids (ordinal INTEGER PRIMARY KEY, source1_entity_id TEXT UNIQUE NOT NULL)")
            connection.execute(
                "CREATE TABLE pairs (source1_entity_id TEXT NOT NULL, candidate_entity_id TEXT NOT NULL, "
                "PRIMARY KEY (source1_entity_id, candidate_entity_id))"
            )
            source_count = 0
            source_rows: list[tuple[int, str]] = []
            for row in _read_source_ids(source1_ids_path):
                source_rows.append(row)
                if len(source_rows) >= 10_000:
                    connection.executemany("INSERT INTO source1_ids VALUES (?, ?)", source_rows)
                    source_count += len(source_rows)
                    source_rows.clear()
            if source_rows:
                connection.executemany("INSERT INTO source1_ids VALUES (?, ?)", source_rows)
                source_count += len(source_rows)

            pair_count = 0
            opener = gzip.open if candidate_path.suffix.lower() == ".gz" else Path.open
            with opener(candidate_path, "rt" if opener is gzip.open else "r", encoding="utf-8", newline="") as handle:
                reader = csv.DictReader(handle, delimiter="\t")
                required = {"source1_entity_id", "candidate_entity_id"}
                if not reader.fieldnames or not required.issubset(reader.fieldnames):
                    raise ValueError(f"{candidate_path} must contain exactly the required pair columns at minimum.")
                if max_rank is not None and "rank" not in reader.fieldnames:
                    raise ValueError(f"{candidate_path} must contain a rank column when max_rank is supplied.")
                rows: list[tuple[str, str]] = []
                for line_number, row in enumerate(reader, start=2):
                    source = (row.get("source1_entity_id") or "").strip()
                    candidate = (row.get("candidate_entity_id") or "").strip()
                    if not source.startswith("S1-") or not candidate.startswith(("S2-", "S3-")):
                        raise ValueError(f"Invalid candidate ID at {candidate_path}:{line_number}.")
                    if max_rank is not None:
                        try:
                            rank = int((row.get("rank") or "").strip())
                        except ValueError as error:
                            raise ValueError(f"Invalid rank at {candidate_path}:{line_number}.") from error
                        if rank > max_rank:
                            continue
                    rows.append((source, candidate))
                    if len(rows) >= 10_000:
                        pair_count += self._insert_pairs(connection, rows)
                        rows.clear()
                if rows:
                    pair_count += self._insert_pairs(connection, rows)
            unknown_source_ids = connection.execute(
                "SELECT p.source1_entity_id FROM pairs AS p LEFT JOIN source1_ids AS s "
                "ON s.source1_entity_id = p.source1_entity_id WHERE s.source1_entity_id IS NULL LIMIT 5"
            ).fetchall()
            if unknown_source_ids:
                raise ValueError(
                    "Candidate file contains Source-1 IDs outside the supplied universe, e.g. "
                    f"{[row[0] for row in unknown_source_ids]}."
                )
            connection.execute("CREATE INDEX pairs_by_source1 ON pairs(source1_entity_id, candidate_entity_id)")
            connection.commit()
        except Exception:
            connection.close()
            # Do not delete a partially constructed file implicitly; it is evidence
            # of the failed input and prevents accidental reuse.
            raise
        else:
            connection.close()

        metadata = {
            **identity,
            "source1_entity_count": source_count,
            "candidate_pair_count": pair_count,
            "zero_candidate_source1_count": source_count - self.pair_source_count(),
            "max_rank": max_rank,
        }
        _write_json(self.metadata_path, metadata)
        return metadata

    @staticmethod
    def _insert_pairs(connection: sqlite3.Connection, rows: list[tuple[str, str]]) -> int:
        try:
            connection.executemany("INSERT INTO pairs VALUES (?, ?)", rows)
        except sqlite3.IntegrityError as error:
            raise ValueError("Duplicate candidate pairs are not allowed in a streaming artifact.") from error
        return len(rows)

    def pair_source_count(self) -> int:
        connection = sqlite3.connect(self.database_path)
        try:
            return int(connection.execute("SELECT COUNT(DISTINCT source1_entity_id) FROM pairs").fetchone()[0])
        finally:
            connection.close()

    def iter_group_batches(self, pair_batch_size: int) -> Iterator[tuple[int, pd.DataFrame]]:
        if pair_batch_size <= 0:
            raise ValueError("pair_batch_size must be positive.")
        connection = sqlite3.connect(self.database_path)
        try:
            cursor = connection.execute("SELECT source1_entity_id, candidate_entity_id FROM pairs ORDER BY source1_entity_id, candidate_entity_id")
            batch: list[tuple[str, str]] = []
            group: list[tuple[str, str]] = []
            current: str | None = None
            batch_number = 0
            for source, candidate in cursor:
                if current is None:
                    current = source
                if source != current:
                    if len(group) > pair_batch_size:
                        raise ValueError(
                            f"Source-1 group {current!r} has {len(group):,} pairs, exceeding pair_batch_size={pair_batch_size:,}; "
                            "raise the explicit batch size rather than splitting the group."
                        )
                    if batch and len(batch) + len(group) > pair_batch_size:
                        yield batch_number, pd.DataFrame(batch, columns=["source1_entity_id", "candidate_entity_id"])
                        batch_number += 1
                        batch = []
                    batch.extend(group)
                    group = []
                    current = source
                group.append((source, candidate))
            if group:
                if len(group) > pair_batch_size:
                    raise ValueError(
                        f"Source-1 group {current!r} has {len(group):,} pairs, exceeding pair_batch_size={pair_batch_size:,}; "
                        "raise the explicit batch size rather than splitting the group."
                    )
                if batch and len(batch) + len(group) > pair_batch_size:
                    yield batch_number, pd.DataFrame(batch, columns=["source1_entity_id", "candidate_entity_id"])
                    batch_number += 1
                    batch = []
                batch.extend(group)
            if batch:
                yield batch_number, pd.DataFrame(batch, columns=["source1_entity_id", "candidate_entity_id"])
        finally:
            connection.close()


def _artifact_identity(path: str | Path) -> dict[str, str]:
    file_path = Path(path)
    return {"path": str(file_path.resolve()), "sha256": sha256_file(file_path)}


def score_logistic_stream(
    *,
    spool: CandidatePairSpool,
    record_store: SQLiteRecordStore,
    extractor: PairwiseFeatureExtractor,
    model: object,
    run_dir: str | Path,
    config: StreamingConfig | None = None,
    model_path: str | Path | None = None,
    feature_artifact_path: str | Path | None = None,
) -> dict[str, object]:
    """Score all spooled pairs in bounded, resumable Source-1 group batches."""

    config = config or StreamingConfig()
    run_path = Path(run_dir)
    parts_path = run_path / "score_parts"
    parts_path.mkdir(parents=True, exist_ok=True)
    spool_metadata = _json(spool.metadata_path)
    identity: dict[str, object] = {
        "streaming_config": {"pair_batch_size": config.pair_batch_size, "include_rule_raw_scores": config.include_rule_raw_scores},
        "spool": spool_metadata,
        "record_store": _artifact_identity(record_store.database_path),
    }
    if model_path:
        identity["model"] = _artifact_identity(model_path)
    if feature_artifact_path:
        identity["feature_artifact"] = _artifact_identity(feature_artifact_path)
    state_path = run_path / "score_state.json"
    if state_path.exists():
        state = _json(state_path)
        if state.get("identity") != identity:
            raise ValueError("Existing streaming run identity differs; use a new run directory.")
    else:
        state = {"identity": identity, "completed_parts": {}, "created_at_unix": time.time()}
        _write_json(state_path, state)

    # Import here to avoid coupling the streaming path to the bounded runner at
    # module-import time; this is an OS working-set measurement, not tracemalloc.
    from .experiments import process_memory_bytes

    started = time.perf_counter()
    memory_start = process_memory_bytes()
    completed = state["completed_parts"]
    if not isinstance(completed, dict):
        raise ValueError("Invalid streaming state file.")
    newly_scored = 0
    for number, pairs in spool.iter_group_batches(config.pair_batch_size):
        key = str(number)
        probability_path = parts_path / f"logistic_{number:07d}.tsv"
        raw_path = parts_path / f"rule_raw_{number:07d}.tsv"
        expected = completed.get(key)
        if expected:
            if not probability_path.exists() or sha256_file(probability_path) != expected.get("probability_sha256"):
                raise ValueError(f"Checkpoint for batch {number} does not match its probability part.")
            if config.include_rule_raw_scores and (
                not raw_path.exists() or sha256_file(raw_path) != expected.get("raw_sha256")
            ):
                raise ValueError(f"Checkpoint for batch {number} does not match its raw-score part.")
            continue
        if probability_path.exists() or raw_path.exists():
            raise FileExistsError(f"Uncheckpointed output part exists for batch {number}; inspect it before resuming.")
        joined_batches = list(record_store.iter_joined_batches(pairs, batch_size=len(pairs)))
        if len(joined_batches) != 1:
            raise AssertionError("Expected one bounded record join per streaming batch.")
        joined, _ = joined_batches[0]
        features = extractor.transform(joined)
        if len(features) != len(pairs) or list(extractor.feature_order) != list(features.columns[2:]):
            raise AssertionError("Streaming feature schema or candidate alignment changed.")
        probabilities = predict_match_probabilities(model, features, extractor.feature_order)
        probability_frame = pd.DataFrame(
            {
                "source1_entity_id": pairs["source1_entity_id"].astype("string"),
                "candidate_entity_id": pairs["candidate_entity_id"].astype("string"),
                "match_probability": probabilities,
            }
        ).loc[:, PROBABILITY_SCORE_COLUMNS]
        probability_frame.to_csv(probability_path, sep="\t", index=False)
        raw_hash: str | None = None
        if config.include_rule_raw_scores:
            raw = DeterministicScorer().score(features)
            raw.to_csv(raw_path, sep="\t", index=False)
            raw_hash = sha256_file(raw_path)
        completed[key] = {
            "pair_count": len(pairs),
            "probability_sha256": sha256_file(probability_path),
            "raw_sha256": raw_hash,
        }
        newly_scored += len(pairs)
        _write_json(state_path, state)

    ordered = [completed[str(number)] for number, _ in spool.iter_group_batches(config.pair_batch_size)]
    state.update(
        {
            "completed_pair_count": sum(int(part["pair_count"]) for part in ordered),
            "batch_count": len(ordered),
            "newly_scored_pair_count": newly_scored,
            "elapsed_seconds_this_invocation": time.perf_counter() - started,
            "process_memory_start_bytes": memory_start,
            "process_memory_end_bytes": process_memory_bytes(),
            "complete": True,
        }
    )
    _write_json(state_path, state)
    return state


def assemble_score_output(run_dir: str | Path, output_path: str | Path, *, raw: bool = False) -> dict[str, object]:
    """Concatenate checked score parts without holding pair rows in memory."""

    run_path, output = Path(run_dir), Path(output_path)
    state = _json(run_path / "score_state.json")
    parts = run_path / "score_parts"
    prefix = "rule_raw_" if raw else "logistic_"
    header = "\t".join(RAW_SCORE_COLUMNS if raw else PROBABILITY_SCORE_COLUMNS)
    temporary = output.with_suffix(output.suffix + ".partial")
    output.parent.mkdir(parents=True, exist_ok=True)
    rows = 0
    with temporary.open("w", encoding="utf-8", newline="") as target:
        target.write(header + "\n")
        for number in range(int(state["batch_count"])):
            part = parts / f"{prefix}{number:07d}.tsv"
            if not part.exists():
                raise FileNotFoundError(f"Missing streaming output part: {part}")
            with part.open(encoding="utf-8", newline="") as source:
                part_header = source.readline().rstrip("\r\n")
                if part_header != header:
                    raise ValueError(f"Unexpected output schema in {part}.")
                for line in source:
                    target.write(line)
                    rows += 1
    os.replace(temporary, output)
    return {"path": str(output), "sha256": sha256_file(output), "pair_count": rows, "columns": header.split("\t")}


def build_threshold_decisions(
    *,
    score_path: str | Path,
    source1_ids_path: str | Path,
    output_path: str | Path,
    threshold: float,
    working_database: str | Path,
) -> dict[str, object]:
    """Create one decision row per frozen Source-1 entity without top-1 logic."""

    if not 0 <= threshold <= 1:
        raise ValueError("threshold must be within [0, 1].")
    database = Path(working_database)
    if database.exists():
        raise FileExistsError(f"Decision database already exists: {database}; use a run-specific path.")
    database.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(database)
    accepted = 0
    try:
        connection.execute("CREATE TABLE source1_ids (ordinal INTEGER PRIMARY KEY, source1_entity_id TEXT UNIQUE NOT NULL)")
        connection.execute("CREATE TABLE accepted (source1_entity_id TEXT NOT NULL, candidate_entity_id TEXT NOT NULL, PRIMARY KEY(source1_entity_id, candidate_entity_id))")
        source_rows = list(_read_source_ids(source1_ids_path))
        connection.executemany("INSERT INTO source1_ids VALUES (?, ?)", source_rows)
        with Path(score_path).open(encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle, delimiter="\t")
            if reader.fieldnames != list(PROBABILITY_SCORE_COLUMNS):
                raise ValueError("Score file must have exactly the pairwise probability output columns.")
            rows: list[tuple[str, str]] = []
            for line_number, row in enumerate(reader, start=2):
                probability = float(row["match_probability"])
                if not np.isfinite(probability) or not 0 <= probability <= 1:
                    raise ValueError(f"Invalid match_probability at {score_path}:{line_number}.")
                if probability >= threshold:
                    rows.append((row["source1_entity_id"], row["candidate_entity_id"]))
                if len(rows) >= 10_000:
                    connection.executemany("INSERT INTO accepted VALUES (?, ?)", rows)
                    accepted += len(rows)
                    rows.clear()
            if rows:
                connection.executemany("INSERT INTO accepted VALUES (?, ?)", rows)
                accepted += len(rows)
        connection.commit()
        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
            writer.writerow(["source1_entity_id", "matched_entity_ids"])
            for source, matches in connection.execute(
                "SELECT s.source1_entity_id, COALESCE(GROUP_CONCAT(a.candidate_entity_id, ','), '') "
                "FROM source1_ids AS s LEFT JOIN accepted AS a ON a.source1_entity_id = s.source1_entity_id "
                "GROUP BY s.ordinal, s.source1_entity_id ORDER BY s.ordinal"
            ):
                writer.writerow([source, matches])
    finally:
        connection.close()
    return {
        "threshold": threshold,
        "source1_entity_count": len(source_rows),
        "accepted_pair_count": accepted,
        "output": str(output_path),
        "output_sha256": sha256_file(output_path),
        "decision_database": str(database),
    }


def evaluate_with_shared_evaluator(
    *, ground_truth_path: str | Path, source1_ids_path: str | Path, decisions_path: str | Path
) -> dict[str, float | int]:
    """Invoke Mudit's evaluator after proving decisions cover the full S1 universe.

    This adapter deliberately materialises only entity-level truth/decision rows;
    it never replaces the evaluator's macro-F0.5 implementation or filters out
    entities with empty predictions.
    """

    from .experiments import read_source1_ids, select_ground_truth

    source_ids = read_source1_ids(source1_ids_path)
    decisions = pd.read_csv(decisions_path, sep="\t", dtype="string", keep_default_na=False)
    required = ["source1_entity_id", "matched_entity_ids"]
    if list(decisions.columns) != required:
        raise ValueError("Decision output must have exactly source1_entity_id and matched_entity_ids columns.")
    decision_ids = validate_complete_source1_ids(decisions["source1_entity_id"])
    if len(decision_ids) != len(source_ids) or set(decision_ids.astype(str)) != set(source_ids.astype(str)):
        raise ValueError("Decision output does not cover exactly the frozen Source-1 universe.")
    ground_truth = pd.read_csv(ground_truth_path, sep="\t", dtype="string", keep_default_na=False)
    selected_truth = select_ground_truth(ground_truth, source_ids)
    return Evaluator().evaluate(selected_truth, decisions)


def compute_disk_backed_error_counts(
    *, spool: CandidatePairSpool, ground_truth_path: str | Path, decision_database: str | Path
) -> dict[str, int | float]:
    """Count blocker and matcher errors without materialising pair labels.

    ``blocking_misses`` are true pairs absent from the supplied candidate set;
    ``matcher_rejections`` are retrieved true pairs that fall below the fixed
    threshold.  Their sum is pairwise false negatives, matching the bounded
    experiment's definitions.
    """

    decision_database = Path(decision_database)
    if not decision_database.is_file():
        raise FileNotFoundError(f"Decision database not found: {decision_database}")
    connection = sqlite3.connect(decision_database)
    try:
        if connection.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='accepted'").fetchone() is None:
            raise ValueError("Decision database does not contain thresholded accepted pairs.")
        if connection.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='true_pairs'").fetchone():
            raise FileExistsError("Error counts already exist for this decision database; use the recorded result.")
        connection.execute("CREATE TABLE true_pairs (source1_entity_id TEXT NOT NULL, candidate_entity_id TEXT NOT NULL, PRIMARY KEY(source1_entity_id, candidate_entity_id))")
        source_ids = {row[0] for row in connection.execute("SELECT source1_entity_id FROM source1_ids")}
        with Path(ground_truth_path).open(encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle, delimiter="\t")
            if not reader.fieldnames or not {"source1_entity_id", "matched_entity_ids"}.issubset(reader.fieldnames):
                raise ValueError("Ground truth TSV has an unexpected schema.")
            rows: list[tuple[str, str]] = []
            for row in reader:
                source = (row.get("source1_entity_id") or "").strip()
                if source not in source_ids:
                    continue
                matches = (row.get("matched_entity_ids") or "").strip()
                for candidate in (value.strip() for value in matches.split(",") if value.strip()):
                    if not candidate.startswith(("S2-", "S3-")):
                        raise ValueError(f"Invalid ground-truth candidate ID for {source!r}: {candidate!r}.")
                    rows.append((source, candidate))
                if len(rows) >= 10_000:
                    connection.executemany("INSERT INTO true_pairs VALUES (?, ?)", rows)
                    rows.clear()
            if rows:
                connection.executemany("INSERT INTO true_pairs VALUES (?, ?)", rows)
        connection.execute("ATTACH DATABASE ? AS candidate_spool", (str(spool.database_path.resolve()),))
        total_true = int(connection.execute("SELECT COUNT(*) FROM true_pairs").fetchone()[0])
        retrieved = int(
            connection.execute(
                "SELECT COUNT(*) FROM true_pairs AS t JOIN candidate_spool.pairs AS c "
                "ON c.source1_entity_id = t.source1_entity_id AND c.candidate_entity_id = t.candidate_entity_id"
            ).fetchone()[0]
        )
        true_positive = int(
            connection.execute(
                "SELECT COUNT(*) FROM true_pairs AS t JOIN accepted AS a "
                "ON a.source1_entity_id = t.source1_entity_id AND a.candidate_entity_id = t.candidate_entity_id"
            ).fetchone()[0]
        )
        accepted = int(connection.execute("SELECT COUNT(*) FROM accepted").fetchone()[0])
        singleton_total = int(
            connection.execute(
                "SELECT COUNT(*) FROM source1_ids AS s WHERE NOT EXISTS "
                "(SELECT 1 FROM true_pairs AS t WHERE t.source1_entity_id = s.source1_entity_id)"
            ).fetchone()[0]
        )
        singleton_correct = int(
            connection.execute(
                "SELECT COUNT(*) FROM source1_ids AS s WHERE NOT EXISTS "
                "(SELECT 1 FROM true_pairs AS t WHERE t.source1_entity_id = s.source1_entity_id) "
                "AND NOT EXISTS (SELECT 1 FROM accepted AS a WHERE a.source1_entity_id = s.source1_entity_id)"
            ).fetchone()[0]
        )
        connection.commit()
        return {
            "total_true_pairs": total_true,
            "retrieved_true_pairs": retrieved,
            "candidate_recall": retrieved / total_true if total_true else 1.0,
            "blocking_misses": total_true - retrieved,
            "matcher_rejections": retrieved - true_positive,
            "pairwise_tp": true_positive,
            "pairwise_fp": accepted - true_positive,
            "pairwise_fn": total_true - true_positive,
            "pairwise_precision": true_positive / accepted if accepted else 0.0,
            "pairwise_recall": true_positive / total_true if total_true else 0.0,
            "true_singletons": singleton_total,
            "correct_singletons": singleton_correct,
            "singleton_accuracy": singleton_correct / singleton_total if singleton_total else 1.0,
        }
    finally:
        connection.close()
