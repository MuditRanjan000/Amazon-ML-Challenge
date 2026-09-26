"""Bounded, entity-disjoint pairwise matching experiments.

This module deliberately does not create a validation split.  Callers must
provide the frozen train/validation Source-1 universes and candidate artifacts
owned by the team.  It supports the existing deterministic rule score and a
regularised Logistic Regression whose preprocessing is fit on training pairs
only.
"""

from __future__ import annotations

import hashlib
import heapq
import json
import os
import pickle
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Protocol

import numpy as np
import pandas as pd

from src.entity_resolution.evaluation.evaluator import Evaluator

from ..matching.contracts import PROBABILITY_SCORE_COLUMNS, RAW_SCORE_COLUMNS, validate_candidate_pairs, validate_complete_source1_ids
from src.entity_resolution.features import PairwiseFeatureExtractor
from src.entity_resolution.models import (
    DeterministicScorer,
    LogisticRegressionConfig,
    ThresholdDecisionLayer,
    fit_logistic_regression,
    predict_match_probabilities,
)


class RecordAdapter(Protocol):
    def iter_joined_batches(
        self, candidate_frame: pd.DataFrame, batch_size: int = 10_000
    ): ...


@dataclass(frozen=True)
class BoundedExperimentConfig:
    """Local resource and model settings; entities are never partially sampled."""

    batch_size: int = 5_000
    max_train_pairs: int = 50_000
    max_validation_pairs: int = 50_000
    logistic_c: float = 1.0
    logistic_max_iter: int = 500
    class_weight: str | None = None
    threshold_start: float = 0.50
    threshold_stop: float = 0.95
    threshold_step: float = 0.05
    error_limit: int = 100

    def thresholds(self) -> list[float]:
        if not 0 <= self.threshold_start <= self.threshold_stop <= 1:
            raise ValueError("Threshold range must lie within [0, 1].")
        if self.threshold_step <= 0:
            raise ValueError("threshold_step must be positive.")
        values = np.arange(self.threshold_start, self.threshold_stop + self.threshold_step / 2, self.threshold_step)
        return [round(float(value), 8) for value in values if value <= 1]


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def process_memory_bytes() -> dict[str, int] | None:
    """Return current/peak process working set on Windows without tracing allocations."""

    if os.name != "nt":
        return None
    import ctypes
    from ctypes import wintypes

    class ProcessMemoryCountersEx(ctypes.Structure):
        _fields_ = [
            ("cb", wintypes.DWORD),
            ("PageFaultCount", wintypes.DWORD),
            ("PeakWorkingSetSize", ctypes.c_size_t),
            ("WorkingSetSize", ctypes.c_size_t),
            ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
            ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
            ("PagefileUsage", ctypes.c_size_t),
            ("PeakPagefileUsage", ctypes.c_size_t),
            ("PrivateUsage", ctypes.c_size_t),
        ]

    counters = ProcessMemoryCountersEx(cb=ctypes.sizeof(ProcessMemoryCountersEx))
    # Declare the Win32 signatures explicitly.  Without them ctypes can
    # truncate 64-bit handles/pointers and silently return no measurement.
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    psapi = ctypes.WinDLL("psapi", use_last_error=True)
    kernel32.GetCurrentProcess.restype = wintypes.HANDLE
    psapi.GetProcessMemoryInfo.argtypes = [wintypes.HANDLE, ctypes.POINTER(ProcessMemoryCountersEx), wintypes.DWORD]
    psapi.GetProcessMemoryInfo.restype = wintypes.BOOL
    success = psapi.GetProcessMemoryInfo(kernel32.GetCurrentProcess(), ctypes.byref(counters), counters.cb)
    if not success:
        return None
    return {
        "working_set_bytes": int(counters.WorkingSetSize),
        "peak_working_set_bytes": int(counters.PeakWorkingSetSize),
        "private_usage_bytes": int(counters.PrivateUsage),
    }


def read_source1_ids(path: str | Path) -> pd.Series:
    path = Path(path)
    frame = pd.read_csv(path, sep="," if path.suffix.lower() == ".csv" else "\t", dtype="string")
    column = "source1_entity_id" if "source1_entity_id" in frame.columns else "entity_id"
    if column not in frame.columns:
        raise ValueError(f"{path} must contain source1_entity_id or entity_id.")
    return validate_complete_source1_ids(frame[column])


def validate_frozen_split_manifest(
    manifest_path: str | Path, train_ids_path: str | Path, validation_ids_path: str | Path
) -> dict[str, object]:
    """Verify the immutable Source-1 ID artifacts before a reported benchmark."""

    manifest_path = Path(manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    required = {
        "train_source1_count",
        "validation_source1_count",
        "train_id_file_sha256",
        "validation_id_file_sha256",
    }
    if missing := sorted(required - set(manifest)):
        raise ValueError(f"Frozen split manifest missing fields: {missing}.")
    train_ids = read_source1_ids(train_ids_path)
    validation_ids = read_source1_ids(validation_ids_path)
    validate_entity_disjoint(train_ids, validation_ids)
    actual = {
        "manifest_sha256": sha256_file(manifest_path),
        "train_id_sha256": sha256_file(train_ids_path),
        "validation_id_sha256": sha256_file(validation_ids_path),
        "train_source1_count": len(train_ids),
        "validation_source1_count": len(validation_ids),
    }
    expected = {
        "train_id_sha256": manifest["train_id_file_sha256"],
        "validation_id_sha256": manifest["validation_id_file_sha256"],
        "train_source1_count": manifest["train_source1_count"],
        "validation_source1_count": manifest["validation_source1_count"],
    }
    mismatches = {key: {"expected": expected[key], "actual": actual[key]} for key in expected if expected[key] != actual[key]}
    if mismatches:
        raise ValueError(f"Frozen split artifacts do not match manifest: {mismatches}.")
    return {**manifest, **actual}


def verify_repository_evaluator_known_answer() -> dict[str, float | int]:
    """Check shared evaluator behavior on a fixed, hand-computable one-to-many case."""

    ground_truth = pd.DataFrame(
        {
            "source1_entity_id": ["S1-known-1", "S1-known-2", "S1-known-3"],
            "matched_entity_ids": ["S2-known-1", "S2-known-2,S3-known-2", ""],
        }
    )
    predictions = pd.DataFrame(
        {
            "source1_entity_id": ["S1-known-1", "S1-known-2", "S1-known-3"],
            "matched_entity_ids": ["S2-known-1", "S2-known-2", ""],
        }
    )
    actual = Evaluator().evaluate(ground_truth, predictions)
    expected = {
        "macro_f05": 17 / 18,
        "macro_precision": 1.0,
        "macro_recall": 5 / 6,
        "total_entities": 3,
        "true_singletons": 1,
        "correct_singletons": 1,
        "singleton_accuracy": 1.0,
    }
    mismatches = {
        key: {"expected": value, "actual": actual.get(key)}
        for key, value in expected.items()
        if not np.isclose(actual.get(key), value)
    }
    if mismatches:
        raise AssertionError(f"Shared evaluator failed its known-answer check: {mismatches}.")
    return actual


def read_candidate_groups(
    path: str | Path,
    source1_ids: pd.Series,
    max_pairs: int,
    chunksize: int = 100_000,
) -> pd.DataFrame:
    """Read complete selected Source-1 candidate groups without all-pair loading.

    A safety cap rejects an overlarge selection instead of truncating an entity's
    group, so recall and one-to-many decisions remain meaningful.
    """

    if max_pairs <= 0 or chunksize <= 0:
        raise ValueError("max_pairs and chunksize must be positive.")
    allowed = set(source1_ids.astype(str))
    selected: list[pd.DataFrame] = []
    total = 0
    for chunk in pd.read_csv(path, sep="\t", dtype="string", chunksize=chunksize):
        candidate_input = validate_candidate_pairs(chunk)
        subset = chunk.loc[candidate_input.pairs["source1_entity_id"].astype(str).isin(allowed)].copy()
        total += len(subset)
        if total > max_pairs:
            raise ValueError(
                f"Selected candidate groups exceed max_pairs={max_pairs:,}; increase the bound or reduce Source-1 entities."
            )
        if not subset.empty:
            selected.append(subset)
    if not selected:
        return pd.DataFrame(columns=["source1_entity_id", "candidate_entity_id"])
    result = pd.concat(selected, ignore_index=True)
    validate_candidate_pairs(result)
    return result


def read_source1_records(path: str | Path, source1_ids: pd.Series, chunksize: int = 100_000) -> pd.DataFrame:
    required = ["entity_id", "country"]
    allowed = set(source1_ids.astype(str))
    selected: list[pd.DataFrame] = []
    for chunk in pd.read_csv(path, sep="\t", dtype="string", usecols=required, chunksize=chunksize):
        subset = chunk.loc[chunk["entity_id"].astype(str).isin(allowed)]
        if not subset.empty:
            selected.append(subset)
    records = pd.concat(selected, ignore_index=True) if selected else pd.DataFrame(columns=required)
    if records["entity_id"].duplicated().any() or set(records["entity_id"].astype(str)) != allowed:
        raise ValueError("Source-1 record file does not contain each selected ID exactly once.")
    return records


def _parse_matched_ids(value: object) -> set[str]:
    if pd.isna(value) or not str(value).strip():
        return set()
    values = {part.strip() for part in str(value).split(",") if part.strip()}
    if not all(value.startswith(("S2-", "S3-")) for value in values):
        raise ValueError("Ground-truth matched_entity_ids must contain only S2-/S3- IDs.")
    return values


def _match_count_group(value: object) -> tuple[str, int]:
    count = len(_parse_matched_ids(value))
    return ("0" if count == 0 else "1" if count == 1 else "2+"), count


def select_stable_stratified_train_ids(
    *,
    frozen_train_ids: pd.Series,
    ground_truth: pd.DataFrame,
    source1_path: str | Path,
    sample_size: int,
    seed: int,
    chunksize: int = 100_000,
) -> tuple[pd.Series, dict[str, object]]:
    """Select complete-group training entities by stable country/match-count strata.

    This is a sample-selection utility only: it never modifies frozen IDs or
    candidates.  A SHA-256 rank of ``seed + NUL + source1_id`` makes selection
    independent of source-file row order while proportional quotas preserve
    coverage across populated country x {0, 1, 2+} strata.
    """

    if sample_size <= 0 or chunksize <= 0:
        raise ValueError("sample_size and chunksize must be positive.")
    frozen_ids = validate_complete_source1_ids(frozen_train_ids)
    if sample_size > len(frozen_ids):
        raise ValueError("sample_size cannot exceed the frozen training population.")
    train_truth = select_ground_truth(ground_truth, frozen_ids)
    match_info = {
        str(row.source1_entity_id): _match_count_group(row.matched_entity_ids)
        for row in train_truth.itertuples(index=False)
    }
    population: dict[tuple[str, str], int] = {}
    seen: set[str] = set()
    for chunk in pd.read_csv(source1_path, sep="\t", dtype="string", usecols=["entity_id", "country"], chunksize=chunksize):
        for entity_id, country in chunk.itertuples(index=False):
            source = str(entity_id).strip()
            info = match_info.get(source)
            if info is None:
                continue
            if source in seen:
                raise ValueError(f"Duplicate Source-1 record encountered: {source!r}.")
            seen.add(source)
            country_key = str(country).strip() or "<missing>"
            stratum = (country_key, info[0])
            population[stratum] = population.get(stratum, 0) + 1
    if len(seen) != len(frozen_ids):
        missing = sorted(set(frozen_ids.astype(str)) - seen)[:5]
        raise ValueError(f"Source-1 records are missing frozen train IDs, e.g. {missing}.")
    for group in ("0", "1", "2+"):
        if not any(stratum_group == group for _country, stratum_group in population):
            raise ValueError(f"Frozen training population has no {group}-match entities; cannot satisfy preflight coverage.")
    strata = sorted(population)
    if sample_size < len(strata):
        raise ValueError(
            f"sample_size={sample_size} cannot cover all {len(strata)} populated country/match-count strata; increase it."
        )
    # One entity per populated stratum guarantees singleton, one-match, and
    # multi-match coverage; remaining quota is proportional to the population.
    quotas = {stratum: 1 for stratum in strata}
    remaining = sample_size - len(strata)
    total_population = sum(population.values())
    raw_extra = {stratum: remaining * population[stratum] / total_population for stratum in strata}
    for stratum in strata:
        quotas[stratum] += min(population[stratum] - quotas[stratum], int(raw_extra[stratum]))
    unassigned = sample_size - sum(quotas.values())
    ranked_strata = sorted(strata, key=lambda item: (raw_extra[item] % 1, population[item], item), reverse=True)
    while unassigned:
        progressed = False
        for stratum in ranked_strata:
            if quotas[stratum] < population[stratum]:
                quotas[stratum] += 1
                unassigned -= 1
                progressed = True
                if unassigned == 0:
                    break
        if not progressed:
            break
    if unassigned:
        raise AssertionError("Unable to allocate all requested stratified training entities.")

    heaps: dict[tuple[str, str], list[tuple[int, str]]] = {stratum: [] for stratum in strata}
    for chunk in pd.read_csv(source1_path, sep="\t", dtype="string", usecols=["entity_id", "country"], chunksize=chunksize):
        for entity_id, country in chunk.itertuples(index=False):
            source = str(entity_id).strip()
            info = match_info.get(source)
            if info is None:
                continue
            stratum = (str(country).strip() or "<missing>", info[0])
            rank = int.from_bytes(hashlib.sha256(f"{seed}\0{source}".encode("utf-8")).digest()[:8], "big")
            heap = heaps[stratum]
            entry = (-rank, source)
            if len(heap) < quotas[stratum]:
                heapq.heappush(heap, entry)
            elif rank < -heap[0][0]:
                heapq.heapreplace(heap, entry)
    selected_with_strata = [(source, stratum) for stratum, heap in heaps.items() for _rank, source in heap]
    selected = sorted(source for source, _stratum in selected_with_strata)
    if len(selected) != sample_size or len(set(selected)) != sample_size:
        raise AssertionError("Stable stratified sample did not contain the requested number of unique Source-1 IDs.")

    def distribution(rows: list[tuple[str, str]]) -> list[dict[str, object]]:
        counts: dict[tuple[str, str], int] = {}
        for country, group in rows:
            counts[(country, group)] = counts.get((country, group), 0) + 1
        total = sum(counts.values())
        return [
            {"country": country, "match_count_group": group, "entity_count": count, "entity_rate": count / total}
            for (country, group), count in sorted(counts.items())
        ]

    selected_strata = [stratum for _source, stratum in selected_with_strata]
    selected_match_pairs = sum(match_info[source][1] for source in selected)
    # The compact counter implementation avoids persisting a population-sized
    # DataFrame; only the reporting list is small (number of strata).
    population_distribution = [
        {"country": country, "match_count_group": group, "entity_count": count, "entity_rate": count / len(frozen_ids)}
        for (country, group), count in sorted(population.items())
    ]
    selected_hash = hashlib.sha256(("\n".join(selected) + "\n").encode("utf-8")).hexdigest()
    sample_group_counts = {group: sum(1 for _country, selected_group in selected_strata if selected_group == group) for group in ("0", "1", "2+")}
    return pd.Series(selected, dtype="string"), {
        "selection_method": "stable_sha256_stratified_country_match_count_v1",
        "seed": seed,
        "frozen_train_id_count": len(frozen_ids),
        "selected_source1_entity_count": len(selected),
        "selected_source1_ids_sha256": selected_hash,
        "stratification": "country x true_match_count_group(0,1,2+); one minimum entity per populated stratum; remainder proportional",
        "population_distribution": population_distribution,
        "selected_distribution": distribution(selected_strata),
        "selected_match_count_groups": sample_group_counts,
        "selected_true_match_pair_count": selected_match_pairs,
        "selected_zero_candidate_entity_count": "pending_candidate_artifact",
        "candidate_pair_count": "pending_candidate_artifact",
        "positive_negative_class_balance": "pending_candidate_artifact_labels",
    }


def select_ground_truth(ground_truth: pd.DataFrame, source1_ids: pd.Series) -> pd.DataFrame:
    required = {"source1_entity_id", "matched_entity_ids"}
    if missing := sorted(required - set(ground_truth.columns)):
        raise ValueError(f"Ground truth missing columns: {missing}.")
    truth = ground_truth.loc[:, ["source1_entity_id", "matched_entity_ids"]].copy()
    truth["source1_entity_id"] = truth["source1_entity_id"].astype("string").str.strip()
    # The supplied TSV represents true singletons as blank fields.  Pandas'
    # nullable-string parser reads those blanks as <NA>; convert them before
    # reindexing so only an actually absent Source-1 row is treated as missing.
    truth["matched_entity_ids"] = truth["matched_entity_ids"].fillna("").astype("string")
    if truth["source1_entity_id"].duplicated().any():
        raise ValueError("Ground truth contains duplicate Source-1 rows.")
    selected = truth.set_index("source1_entity_id").reindex(source1_ids.astype(str)).reset_index()
    selected.columns = ["source1_entity_id", "matched_entity_ids"]
    if selected["matched_entity_ids"].isna().any():
        raise ValueError("Ground truth is missing one or more selected Source-1 IDs.")
    selected["matched_entity_ids"] = selected["matched_entity_ids"].astype("string")
    for value in selected["matched_entity_ids"]:
        _parse_matched_ids(value)
    return selected


def validate_entity_disjoint(train_ids: pd.Series, validation_ids: pd.Series) -> None:
    overlap = set(train_ids.astype(str)) & set(validation_ids.astype(str))
    if overlap:
        raise ValueError(f"Training and validation Source-1 IDs overlap, e.g. {sorted(overlap)[:5]}.")


def labels_for_candidates(candidates: pd.DataFrame, ground_truth: pd.DataFrame) -> pd.Series:
    true_matches = {
        str(row.source1_entity_id): _parse_matched_ids(row.matched_entity_ids)
        for row in ground_truth.itertuples(index=False)
    }
    return pd.Series(
        [int(str(candidate) in true_matches[str(source)]) for source, candidate in candidates.loc[:, ["source1_entity_id", "candidate_entity_id"]].itertuples(index=False)],
        index=candidates.index,
        dtype="int8",
        name="is_true_match",
    )


def candidate_recall(candidates: pd.DataFrame, ground_truth: pd.DataFrame) -> dict[str, int | float]:
    available = candidates.groupby("source1_entity_id", sort=False)["candidate_entity_id"].agg(lambda ids: set(ids.astype(str)))
    total_true = 0
    retrieved_true = 0
    missing_rows: list[dict[str, str]] = []
    for row in ground_truth.itertuples(index=False):
        source = str(row.source1_entity_id)
        truths = _parse_matched_ids(row.matched_entity_ids)
        total_true += len(truths)
        present = available.get(source, set())
        retrieved_true += len(truths & present)
        for candidate in sorted(truths - present):
            missing_rows.append({"source1_entity_id": source, "candidate_entity_id": candidate})
    return {
        "total_true_pairs": total_true,
        "retrieved_true_pairs": retrieved_true,
        "candidate_recall": retrieved_true / total_true if total_true else 1.0,
        "missing_true_pairs": total_true - retrieved_true,
        "missing_rows": missing_rows,
    }


def _decision_metrics(ground_truth: pd.DataFrame, predictions: pd.DataFrame) -> dict[str, float | int]:
    local = Evaluator().evaluate(ground_truth, predictions)
    truth_map = {str(row.source1_entity_id): _parse_matched_ids(row.matched_entity_ids) for row in ground_truth.itertuples(index=False)}
    prediction_map = {str(row.source1_entity_id): _parse_matched_ids(row.matched_entity_ids) for row in predictions.itertuples(index=False)}
    tp = fp = fn = 0
    non_singletons = 0
    non_singleton_empty_predictions = 0
    for source, truth in truth_map.items():
        predicted = prediction_map.get(source, set())
        tp += len(truth & predicted)
        fp += len(predicted - truth)
        fn += len(truth - predicted)
        if truth:
            non_singletons += 1
            non_singleton_empty_predictions += int(not predicted)
    local.update(
        {
            "pairwise_precision": tp / (tp + fp) if tp + fp else 0.0,
            "pairwise_recall": tp / (tp + fn) if tp + fn else 0.0,
            "pairwise_tp": tp,
            "pairwise_fp": fp,
            "pairwise_fn": fn,
            "non_singleton_empty_prediction_rate": (
                non_singleton_empty_predictions / non_singletons if non_singletons else 0.0
            ),
        }
    )
    return local


def threshold_sweep(
    scores: pd.DataFrame, ground_truth: pd.DataFrame, source1_ids: pd.Series, score_column: str, config: BoundedExperimentConfig
) -> pd.DataFrame:
    rows = []
    for threshold in config.thresholds():
        decisions = ThresholdDecisionLayer(threshold=threshold, score_column=score_column).decide(scores, source1_ids)
        rows.append({"threshold": threshold, **_decision_metrics(ground_truth, decisions)})
    return pd.DataFrame(rows)


def grouped_metrics(
    scores: pd.DataFrame,
    ground_truth: pd.DataFrame,
    source1_records: pd.DataFrame,
    score_column: str,
    threshold: float,
) -> dict[str, list[dict[str, object]]]:
    decisions = ThresholdDecisionLayer(threshold=threshold, score_column=score_column).decide(scores, ground_truth["source1_entity_id"])
    groups = ground_truth.copy()
    groups["true_match_count"] = groups["matched_entity_ids"].map(lambda value: len(_parse_matched_ids(value)))
    groups["match_count_group"] = np.select(
        [groups["true_match_count"].eq(0), groups["true_match_count"].eq(1)], ["0", "1"], default="2+"
    )
    countries = source1_records.loc[:, ["entity_id", "country"]].rename(columns={"entity_id": "source1_entity_id"})
    groups = groups.merge(countries, on="source1_entity_id", how="left", validate="one_to_one")
    if groups["country"].isna().any():
        raise ValueError("Country is missing for one or more selected Source-1 records.")

    def summarize(column: str) -> list[dict[str, object]]:
        output = []
        for value, subset in groups.groupby(column, dropna=False, sort=True):
            source_ids = subset["source1_entity_id"]
            prediction_subset = decisions.loc[decisions["source1_entity_id"].isin(source_ids)]
            output.append({column: str(value), **_decision_metrics(subset[["source1_entity_id", "matched_entity_ids"]], prediction_subset)})
        return output

    return {"by_country": summarize("country"), "by_true_match_count": summarize("match_count_group")}


def _joined(adapter: RecordAdapter, candidates: pd.DataFrame, batch_size: int) -> pd.DataFrame:
    batches = [records for records, _metadata in adapter.iter_joined_batches(candidates, batch_size=batch_size)]
    if not batches:
        raise ValueError("At least one candidate pair is required to fit or score a bounded experiment.")
    return pd.concat(batches, ignore_index=True)


def _write_tsv(frame: pd.DataFrame, path: Path) -> None:
    frame.to_csv(path, sep="\t", index=False)


def run_bounded_experiment(
    *,
    train_ids: pd.Series,
    validation_ids: pd.Series,
    train_candidates: pd.DataFrame,
    validation_candidates: pd.DataFrame,
    ground_truth: pd.DataFrame,
    source1_records: pd.DataFrame,
    record_adapter: RecordAdapter,
    run_dir: str | Path,
    config: BoundedExperimentConfig | None = None,
    input_hashes: dict[str, str] | None = None,
    experiment_id: str = "UNASSIGNED",
    split_version: str | None = None,
    candidate_version: str | None = None,
    code_version: str | None = None,
    evaluator_invocation: str | None = None,
    evaluation_status: str = "local_preflight_not_official_until_mudit_confirms_evaluator",
) -> dict[str, object]:
    """Fit train-only features/models and evaluate a supplied validation universe.

    Metrics use the repository's current evaluator as a *preflight* only.  The
    caller is responsible for selecting Mudit's confirmed invocation before any
    result is treated as an official validation score.
    """

    config = config or BoundedExperimentConfig()
    run_started = time.perf_counter()
    memory_start = process_memory_bytes()
    train_ids = validate_complete_source1_ids(train_ids)
    validation_ids = validate_complete_source1_ids(validation_ids)
    validate_entity_disjoint(train_ids, validation_ids)
    train_input = validate_candidate_pairs(train_candidates).pairs
    validation_input = validate_candidate_pairs(validation_candidates).pairs
    if not set(train_input["source1_entity_id"].astype(str)) <= set(train_ids.astype(str)):
        raise ValueError("Training candidates contain Source-1 IDs outside the supplied training universe.")
    if not set(validation_input["source1_entity_id"].astype(str)) <= set(validation_ids.astype(str)):
        raise ValueError("Validation candidates contain Source-1 IDs outside the supplied validation universe.")
    if len(train_input) > config.max_train_pairs or len(validation_input) > config.max_validation_pairs:
        raise ValueError("Input exceeds configured bounded pair limit.")

    train_truth = select_ground_truth(ground_truth, train_ids)
    validation_truth = select_ground_truth(ground_truth, validation_ids)
    train_labels = labels_for_candidates(train_input, train_truth)
    validation_labels = labels_for_candidates(validation_input, validation_truth)
    if train_labels.nunique() < 2:
        raise ValueError("Training candidates require both positive and negative pairs for Logistic Regression.")

    started = time.perf_counter()
    train_joined = _joined(record_adapter, train_input, config.batch_size)
    extractor = PairwiseFeatureExtractor().fit(train_joined)
    train_features = extractor.transform(train_joined)
    feature_columns = extractor.feature_order
    validation_features = (
        extractor.transform(_joined(record_adapter, validation_input, config.batch_size))
        if not validation_input.empty
        else pd.DataFrame(columns=["source1_entity_id", "candidate_entity_id", *feature_columns])
    )
    feature_seconds = time.perf_counter() - started

    model_started = time.perf_counter()
    model = fit_logistic_regression(
        train_features,
        train_labels,
        feature_columns,
        LogisticRegressionConfig(
            c=config.logistic_c,
            max_iter=config.logistic_max_iter,
            class_weight=config.class_weight,
        ),
    )
    logistic_probability = (
        predict_match_probabilities(model, validation_features, feature_columns)
        if not validation_features.empty
        else np.array([], dtype=float)
    )
    model_seconds = time.perf_counter() - model_started

    rule_scores = (
        DeterministicScorer().score(validation_features)
        if not validation_features.empty
        else pd.DataFrame(columns=RAW_SCORE_COLUMNS)
    )
    logistic_scores = pd.DataFrame(
        {
            "source1_entity_id": validation_features["source1_entity_id"].astype("string"),
            "candidate_entity_id": validation_features["candidate_entity_id"].astype("string"),
            "match_probability": logistic_probability,
            "scoring_method": "logistic_regression_l2_v1",
            "is_true_match": validation_labels.to_numpy(),
        }
    )
    rule_scores["is_true_match"] = validation_labels.to_numpy()
    rule_sweep = threshold_sweep(rule_scores, validation_truth, validation_ids, "raw_match_score", config)
    logistic_sweep = threshold_sweep(logistic_scores, validation_truth, validation_ids, "match_probability", config)
    rule_best = rule_sweep.loc[rule_sweep["macro_f05"].idxmax()].to_dict()
    logistic_best = logistic_sweep.loc[logistic_sweep["macro_f05"].idxmax()].to_dict()
    rule_groups = grouped_metrics(rule_scores, validation_truth, source1_records, "raw_match_score", float(rule_best["threshold"]))
    logistic_groups = grouped_metrics(logistic_scores, validation_truth, source1_records, "match_probability", float(logistic_best["threshold"]))
    candidate_stats = candidate_recall(validation_input, validation_truth)

    output = Path(run_dir)
    output.mkdir(parents=True, exist_ok=True)
    extractor.save(output / "features.pkl")
    with (output / "logistic_regression.pkl").open("wb") as handle:
        pickle.dump(model, handle)
    # Integration-facing Logistic output is exactly the three-column probability
    # contract.  Labels and method details stay in a separate local diagnostic.
    _write_tsv(rule_scores.loc[:, RAW_SCORE_COLUMNS], output / "validation_rule_scores.tsv")
    _write_tsv(logistic_scores.loc[:, PROBABILITY_SCORE_COLUMNS], output / "validation_logistic_scores.tsv")
    _write_tsv(logistic_scores, output / "validation_logistic_labeled_scores.tsv")
    _write_tsv(rule_sweep, output / "rule_threshold_sweep.tsv")
    _write_tsv(logistic_sweep, output / "logistic_threshold_sweep.tsv")
    _write_tsv(pd.DataFrame(candidate_stats.pop("missing_rows")), output / "validation_blocking_misses.tsv")

    errors = []
    for method, frame, score_column, best in (
        ("rule", rule_scores, "raw_match_score", rule_best),
        ("logistic", logistic_scores, "match_probability", logistic_best),
    ):
        selected = frame.loc[:, ["source1_entity_id", "candidate_entity_id", score_column, "is_true_match"]].copy()
        selected["prediction"] = (selected[score_column] >= float(best["threshold"])).astype(int)
        selected["error_type"] = np.select(
            [(selected["prediction"] == 1) & (selected["is_true_match"] == 0), (selected["prediction"] == 0) & (selected["is_true_match"] == 1)],
            ["false_positive", "false_negative"],
            default="",
        )
        selected = selected.loc[selected["error_type"] != ""].sort_values(score_column, ascending=False).head(config.error_limit)
        selected.insert(0, "method", method)
        errors.append(selected)
    _write_tsv(pd.concat(errors, ignore_index=True), output / "representative_pair_errors.tsv")

    report = {
        "experiment_id": experiment_id,
        "status": evaluation_status,
        "evaluator": "src.entity_resolution.evaluation.evaluator.Evaluator",
        "evaluator_invocation": evaluator_invocation or "UNCONFIRMED_BY_MUDIT",
        "split_version": split_version or "UNSPECIFIED",
        "candidate_version": candidate_version or "UNSPECIFIED",
        "code_version": code_version or "UNSPECIFIED",
        "config": asdict(config),
        "input_hashes": input_hashes or {},
        "feature_schema": feature_columns,
        "train": {
            "source1_entities": len(train_ids),
            "candidate_pairs": len(train_input),
            "positive_pairs": int(train_labels.sum()),
            "negative_pairs": int((1 - train_labels).sum()),
            "positive_pair_rate": float(train_labels.mean()),
        },
        "validation": {
            "source1_entities": len(validation_ids),
            "candidate_pairs": len(validation_input),
            "positive_pairs_in_candidates": int(validation_labels.sum()),
            **candidate_stats,
        },
        "runtime_seconds": {
            "feature_fit_and_transform": feature_seconds,
            "logistic_fit_and_probability": model_seconds,
            "total_before_report_write": time.perf_counter() - run_started,
        },
        "process_memory_bytes": {"start": memory_start, "before_report_write": process_memory_bytes()},
        "rule": {"best_tuning_result": rule_best, **rule_groups},
        "logistic_regression": {
            "best_tuning_result": logistic_best,
            "calibration": "predict_proba only; empirical calibration was not measured",
            **logistic_groups,
        },
    }
    (output / "run_report.json").write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    return report
