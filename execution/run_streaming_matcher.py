"""Score a large plain-TSV or gzip-compressed candidate artifact using saved Ashank feature/model artifacts.

This is a scoring/evaluation adapter, not a split generator or a trainer.  A
full D2 validation artifact can be scored with a previously fitted model, but
its result is a transfer experiment until a same-generator training artifact
is supplied for a fair fit and threshold selection.
"""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path

from src.entity_resolution.matching.records import SQLiteRecordStore
from src.entity_resolution.models.streaming import (
    CandidatePairSpool,
    StreamingConfig,
    assemble_score_output,
    build_threshold_decisions,
    compute_disk_backed_error_counts,
    evaluate_with_shared_evaluator,
    score_logistic_stream,
    write_source1_prefix,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidates", required=True, type=Path)
    parser.add_argument("--source1-ids", required=True, type=Path)
    parser.add_argument("--store", required=True, type=Path)
    parser.add_argument("--feature-artifact", required=True, type=Path)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--pair-batch-size", type=int, default=5_000)
    parser.add_argument("--max-rank", type=int, help="Keep only input candidate rows with rank <= this value; requires a rank column.")
    parser.add_argument("--max-source1-entities", type=int, help="Diagnostic prefix only; never a full benchmark.")
    parser.add_argument("--no-rule-raw-scores", action="store_true")
    parser.add_argument("--threshold", type=float)
    parser.add_argument("--ground-truth", type=Path)
    args = parser.parse_args()

    if args.ground_truth and args.threshold is None:
        parser.error("--ground-truth requires a fixed --threshold; this command does not tune on validation labels.")
    selected_ids = args.source1_ids
    selection: dict[str, object] | None = None
    if args.max_source1_entities is not None:
        selected_ids = args.run_dir / "selected_source1_ids.csv"
        selection = write_source1_prefix(args.source1_ids, selected_ids, args.max_source1_entities)
    spool = CandidatePairSpool(args.run_dir / "candidate_spool.sqlite")
    metadata = spool.build(args.candidates, selected_ids, max_rank=args.max_rank)
    with args.feature_artifact.open("rb") as handle:
        extractor = pickle.load(handle)
    with args.model.open("rb") as handle:
        model = pickle.load(handle)
    state = score_logistic_stream(
        spool=spool,
        record_store=SQLiteRecordStore(args.store),
        extractor=extractor,
        model=model,
        run_dir=args.run_dir,
        config=StreamingConfig(args.pair_batch_size, not args.no_rule_raw_scores),
        model_path=args.model,
        feature_artifact_path=args.feature_artifact,
    )
    logistic = assemble_score_output(args.run_dir, args.run_dir / "validation_logistic_scores.tsv")
    result: dict[str, object] = {"candidate_spool": metadata, "score_state": state, "logistic_scores": logistic}
    if selection:
        result["source1_selection"] = selection
    if not args.no_rule_raw_scores:
        result["rule_raw_scores"] = assemble_score_output(args.run_dir, args.run_dir / "validation_rule_raw_scores.tsv", raw=True)
    if args.threshold is not None:
        decisions = build_threshold_decisions(
            score_path=args.run_dir / "validation_logistic_scores.tsv",
            source1_ids_path=selected_ids,
            output_path=args.run_dir / "validation_logistic_decisions.tsv",
            threshold=args.threshold,
            working_database=args.run_dir / "decisions.sqlite",
        )
        result["decisions"] = decisions
        if args.ground_truth:
            result["shared_evaluator"] = evaluate_with_shared_evaluator(
                ground_truth_path=args.ground_truth,
                source1_ids_path=selected_ids,
                decisions_path=args.run_dir / "validation_logistic_decisions.tsv",
            )
            result["error_counts"] = compute_disk_backed_error_counts(
                spool=spool,
                ground_truth_path=args.ground_truth,
                decision_database=args.run_dir / "decisions.sqlite",
            )
    import json

    (args.run_dir / "streaming_run_report.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
