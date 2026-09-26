"""Run Ashank's bounded rule-baseline and Logistic Regression preflight.

This command requires frozen, entity-disjoint Source-1 ID files.  It never
generates a split, candidate set, submission, or full-test inference.  Store
all generated paths under ignored `.tmp/` until Mudit integrates a run.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.entity_resolution.data.config import resolve_data_dir
from src.entity_resolution.models.experiments import (
    BoundedExperimentConfig,
    read_candidate_groups,
    read_source1_ids,
    read_source1_records,
    run_bounded_experiment,
    sha256_file,
    select_stable_stratified_train_ids,
    validate_frozen_split_manifest,
    verify_repository_evaluator_known_answer,
)
from src.entity_resolution.matching.records import SQLiteRecordStore

FROZEN_SPLIT_DIR = REPO_ROOT / "artifacts" / "validation_split"


def parser() -> argparse.ArgumentParser:
    command = argparse.ArgumentParser(description=__doc__)
    command.add_argument("--train-source1-ids", type=Path, default=FROZEN_SPLIT_DIR / "train_ids.csv")
    command.add_argument("--validation-source1-ids", type=Path, default=FROZEN_SPLIT_DIR / "val_ids.csv")
    command.add_argument(
        "--split-manifest", type=Path, default=FROZEN_SPLIT_DIR / "manifest.json", help="Mudit's immutable JSON split manifest."
    )
    command.add_argument("--train-candidates", type=Path, required=True)
    command.add_argument("--validation-candidates", type=Path, required=True)
    command.add_argument("--ground-truth", type=Path, required=True)
    command.add_argument("--store", type=Path, required=True, help="Existing train-record SQLite store.")
    command.add_argument("--run-dir", type=Path, required=True)
    command.add_argument("--experiment-id", required=True, help="Mudit's experiment-registry ID; never auto-generated.")
    command.add_argument("--split-version", help="Optional human-readable frozen split version.")
    command.add_argument("--candidate-version", help="Aayush candidate artifact/version identifier.")
    command.add_argument(
        "--evaluator-invocation",
        default="src.entity_resolution.evaluation.evaluator.Evaluator.evaluate",
        help="Mudit-confirmed evaluator callable reference.",
    )
    command.add_argument("--data-dir", type=Path, help="Supplied dataset root; defaults to AMAZON_ML_DATA_DIR.")
    command.add_argument("--max-train-entities", type=int, default=1_000)
    command.add_argument(
        "--training-selection-seed",
        type=int,
        default=20260926,
        help="Seed for stable country x true-match-count stratified frozen-train sampling.",
    )
    command.add_argument("--max-validation-entities", type=int, default=1_000)
    command.add_argument("--max-train-pairs", type=int, default=50_000)
    command.add_argument("--max-validation-pairs", type=int, default=50_000)
    command.add_argument("--batch-size", type=int, default=5_000)
    command.add_argument("--logistic-c", type=float, default=1.0)
    command.add_argument("--class-weight", choices=("balanced",), default=None)
    return command


def _take_bounded(ids: pd.Series, maximum: int, label: str) -> pd.Series:
    if maximum <= 0:
        raise ValueError(f"{label} must be positive.")
    return ids.iloc[:maximum].reset_index(drop=True)


def main() -> None:
    args = parser().parse_args()
    required = [args.split_manifest, args.train_source1_ids, args.validation_source1_ids, args.train_candidates, args.validation_candidates, args.ground_truth, args.store]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Required team artifact(s) missing: {missing}")
    frozen_manifest = validate_frozen_split_manifest(
        args.split_manifest, args.train_source1_ids, args.validation_source1_ids
    )
    evaluator_known_answer = verify_repository_evaluator_known_answer()
    data_dir = resolve_data_dir(args.data_dir)
    ground_truth = pd.read_csv(args.ground_truth, sep="\t", dtype="string")
    train_ids, training_selection = select_stable_stratified_train_ids(
        frozen_train_ids=read_source1_ids(args.train_source1_ids),
        ground_truth=ground_truth,
        source1_path=data_dir / "train" / "train_source1.tsv",
        sample_size=args.max_train_entities,
        seed=args.training_selection_seed,
    )
    validation_ids = _take_bounded(read_source1_ids(args.validation_source1_ids), args.max_validation_entities, "max_validation_entities")
    config = BoundedExperimentConfig(
        batch_size=args.batch_size,
        max_train_pairs=args.max_train_pairs,
        max_validation_pairs=args.max_validation_pairs,
        logistic_c=args.logistic_c,
        class_weight=args.class_weight,
    )
    train_candidates = read_candidate_groups(args.train_candidates, train_ids, config.max_train_pairs)
    validation_candidates = read_candidate_groups(args.validation_candidates, validation_ids, config.max_validation_pairs)
    training_selection["candidate_pair_count"] = len(train_candidates)
    training_selection["selected_zero_candidate_entity_count"] = int(
        len(train_ids) - train_candidates["source1_entity_id"].nunique()
    )
    args.run_dir.mkdir(parents=True, exist_ok=True)
    (args.run_dir / "training_selection.json").write_text(
        json.dumps(training_selection, indent=2, sort_keys=True), encoding="utf-8"
    )
    source1_records = read_source1_records(data_dir / "train" / "train_source1.tsv", validation_ids)
    hashes = {
        "train_source1_ids": sha256_file(args.train_source1_ids),
        "validation_source1_ids": sha256_file(args.validation_source1_ids),
        "train_candidates": sha256_file(args.train_candidates),
        "validation_candidates": sha256_file(args.validation_candidates),
        "ground_truth": sha256_file(args.ground_truth),
        "split_manifest": sha256_file(args.split_manifest),
        "selected_train_source1_ids_sha256": str(training_selection["selected_source1_ids_sha256"]),
    }
    report = run_bounded_experiment(
        train_ids=train_ids,
        validation_ids=validation_ids,
        train_candidates=train_candidates,
        validation_candidates=validation_candidates,
        ground_truth=ground_truth,
        source1_records=source1_records,
        record_adapter=SQLiteRecordStore(args.store),
        run_dir=args.run_dir,
        config=config,
        input_hashes=hashes,
        experiment_id=args.experiment_id,
        split_version=args.split_version or frozen_manifest["split_creation_date"],
        candidate_version=args.candidate_version,
        code_version=subprocess.check_output(["git", "describe", "--always", "--dirty"], cwd=REPO_ROOT, text=True).strip(),
        evaluator_invocation=args.evaluator_invocation,
        evaluation_status="official_frozen_evaluation",
    )
    report["frozen_split_manifest"] = frozen_manifest
    report["official_evaluator_known_answer"] = evaluator_known_answer
    training_selection["positive_negative_class_balance"] = {
        "positive_pairs": report["train"]["positive_pairs"],
        "negative_pairs": report["train"]["negative_pairs"],
        "positive_pair_rate": report["train"]["positive_pair_rate"],
    }
    (args.run_dir / "training_selection.json").write_text(
        json.dumps(training_selection, indent=2, sort_keys=True), encoding="utf-8"
    )
    report["training_selection"] = training_selection
    (args.run_dir / "run_report.json").write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({"status": report["status"], "run_dir": str(args.run_dir), "rule": report["rule"]["best_tuning_result"], "logistic": report["logistic_regression"]["best_tuning_result"]}, indent=2))


if __name__ == "__main__":
    main()
