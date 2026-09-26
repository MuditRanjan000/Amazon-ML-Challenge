"""CLI for Ashank's deterministic matcher components.

Use fit only with training pairs. The commands deliberately produce raw baseline
scores, not calibrated probabilities, until Mudit's frozen validation split is
available for calibration.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.entity_resolution.data.config import resolve_data_dir
from src.entity_resolution.features import PairwiseFeatureExtractor
from src.entity_resolution.models import DeterministicScorer, ThresholdDecisionLayer
from src.entity_resolution.matching.records import SQLiteRecordStore


def _read_candidates(path: Path, max_pairs: int | None) -> pd.DataFrame:
    return pd.read_csv(path, sep="\t", dtype="string", nrows=max_pairs)


def _joined_records(store: SQLiteRecordStore, candidates: pd.DataFrame, batch_size: int) -> pd.DataFrame:
    return pd.concat(
        [joined for joined, _metadata in store.iter_joined_batches(candidates, batch_size=batch_size)],
        ignore_index=True,
    )


def build_store(args: argparse.Namespace) -> None:
    data_dir = resolve_data_dir(args.data_dir)
    source_paths = [data_dir / args.split / f"{args.split}_source{source}.tsv" for source in (1, 2, 3)]
    store = SQLiteRecordStore(args.store)
    store.build(source_paths, overwrite=args.overwrite)
    print(f"Built SQLite record store: {args.store}")


def fit_features(args: argparse.Namespace) -> None:
    if args.fit_split != "train":
        raise ValueError("TF-IDF fitting is allowed only with --fit-split train.")
    candidates = _read_candidates(args.candidates, args.max_fit_pairs)
    joined = _joined_records(SQLiteRecordStore(args.store), candidates, args.batch_size)
    extractor = PairwiseFeatureExtractor().fit(joined)
    extractor.save(args.artifact)
    print(f"Fitted feature artifact on {len(joined):,} training candidate pairs: {args.artifact}")


def extract_features(args: argparse.Namespace) -> None:
    candidates = _read_candidates(args.candidates, args.max_pairs)
    store = SQLiteRecordStore(args.store)
    extractor = PairwiseFeatureExtractor.load(args.artifact)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    wrote_header = False
    total = 0
    for joined, _metadata in store.iter_joined_batches(candidates, batch_size=args.batch_size):
        features = extractor.transform(joined)
        features.to_csv(output_path, sep="\t", index=False, mode="a" if wrote_header else "w", header=not wrote_header)
        wrote_header = True
        total += len(features)
    print(f"Extracted {total:,} pair feature rows: {output_path}")


def score_features(args: argparse.Namespace) -> None:
    features = pd.read_csv(args.features, sep="\t")
    scores = DeterministicScorer().score(features)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    scores.to_csv(output_path, sep="\t", index=False)
    print(f"Wrote {len(scores):,} raw deterministic scores: {output_path}")


def aggregate_decisions(args: argparse.Namespace) -> None:
    scores = pd.read_csv(args.scores, sep="\t")
    source1 = pd.read_csv(args.source1_ids, sep="\t", dtype="string")
    source_column = "entity_id" if "entity_id" in source1.columns else "source1_entity_id"
    if source_column not in source1.columns:
        raise ValueError("Source-1 TSV must contain entity_id or source1_entity_id.")
    decisions = ThresholdDecisionLayer(threshold=args.threshold, score_column=args.score_column).decide(
        scores, source1[source_column]
    )
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    decisions.to_csv(output_path, sep="\t", index=False)
    print(f"Wrote {len(decisions):,} one-to-many decisions: {output_path}")


def parser() -> argparse.ArgumentParser:
    command_parser = argparse.ArgumentParser(description=__doc__)
    commands = command_parser.add_subparsers(dest="command", required=True)

    build = commands.add_parser("build-store", help="Build a reusable local SQLite record lookup.")
    build.add_argument("--split", choices=("train", "test"), required=True)
    build.add_argument("--store", type=Path, required=True)
    build.add_argument("--data-dir", type=Path)
    build.add_argument("--overwrite", action="store_true", help="Replace only the specified local store.")
    build.set_defaults(handler=build_store)

    fit = commands.add_parser("fit-features", help="Fit TF-IDF representations on training candidate pairs only.")
    fit.add_argument("--fit-split", required=True, help="Must be literal 'train'.")
    fit.add_argument("--candidates", type=Path, required=True)
    fit.add_argument("--store", type=Path, required=True)
    fit.add_argument("--artifact", type=Path, required=True)
    fit.add_argument("--max-fit-pairs", type=int, default=100_000)
    fit.add_argument("--batch-size", type=int, default=10_000)
    fit.set_defaults(handler=fit_features)

    extract = commands.add_parser("extract-features", help="Transform candidates in bounded source-record join batches.")
    extract.add_argument("--candidates", type=Path, required=True)
    extract.add_argument("--store", type=Path, required=True)
    extract.add_argument("--artifact", type=Path, required=True)
    extract.add_argument("--output", type=Path, required=True)
    extract.add_argument("--max-pairs", type=int)
    extract.add_argument("--batch-size", type=int, default=10_000)
    extract.set_defaults(handler=extract_features)

    score = commands.add_parser("score", help="Produce explicitly raw deterministic scores from feature TSV.")
    score.add_argument("--features", type=Path, required=True)
    score.add_argument("--output", type=Path, required=True)
    score.set_defaults(handler=score_features)

    decide = commands.add_parser("decide", help="Apply a threshold while retaining every Source-1 row.")
    decide.add_argument("--scores", type=Path, required=True)
    decide.add_argument("--source1-ids", type=Path, required=True)
    decide.add_argument("--output", type=Path, required=True)
    decide.add_argument("--threshold", type=float, required=True)
    decide.add_argument("--score-column", default="raw_match_score")
    decide.set_defaults(handler=aggregate_decisions)
    return command_parser


def main() -> None:
    args = parser().parse_args()
    args.handler(args)


if __name__ == "__main__":
    main()
