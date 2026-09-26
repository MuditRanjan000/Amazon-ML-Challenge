"""Profile the pairwise matcher on a bounded, diagnostic workload only.

This is not candidate generation, a canonical validation split, or a quality
benchmark. It deterministically selects real training records to stress text
length, missing address fields, and non-Latin scripts while repeating each S1
across multiple S2/S3 candidates, similar to a blocking fan-out.
"""

from __future__ import annotations

import argparse
import io
import json
import sys
import tempfile
import time
import tracemalloc
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.entity_resolution.data.config import resolve_data_dir
from src.entity_resolution.features import PairwiseFeatureExtractor
from src.entity_resolution.matching.normalization import script_indicators
from src.entity_resolution.matching.records import DataFrameRecordAdapter, SQLiteRecordStore


def _timed(operation):
    start = time.perf_counter()
    result = operation()
    return result, time.perf_counter() - start


def _selected_indices(frame: pd.DataFrame, required: int) -> list[int]:
    name = frame["business_name"].fillna("")
    address = frame["business_address"].fillna("")
    script = name.map(lambda value: bool(script_indicators(value)["devanagari"] or script_indicators(value)["other"]))
    ordered = (
        list(frame.index[script])
        + list(frame.index[name.eq("") | address.eq("")])
        + list((name.str.len() + address.str.len()).sort_values(ascending=False).index)
        + list(frame.index)
    )
    selected, seen = [], set()
    for index in ordered:
        if index not in seen:
            selected.append(index)
            seen.add(index)
        if len(selected) == required:
            return selected
    raise ValueError(f"Sample has fewer than {required} usable records.")


def _sqlite_timings(source_frames: tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame], pairs: pd.DataFrame, batch_size: int) -> dict[str, float]:
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        source_paths = []
        for label, frame in zip(("s1", "s2", "s3"), source_frames):
            path = root / f"{label}.tsv"
            frame.to_csv(path, sep="\t", index=False)
            source_paths.append(path)
        store = SQLiteRecordStore(root / "records.sqlite")
        _, build_seconds = _timed(lambda: store.build(source_paths))
        _, cold_seconds = _timed(lambda: list(store.iter_joined_batches(pairs, batch_size=batch_size)))
        _, warm_seconds = _timed(lambda: list(store.iter_joined_batches(pairs, batch_size=batch_size)))
    return {
        "sqlite_build_seconds_for_sample": build_seconds,
        "sqlite_cold_join_seconds": cold_seconds,
        "sqlite_warm_join_seconds": warm_seconds,
    }


def run(args: argparse.Namespace) -> dict[str, object]:
    data_dir = resolve_data_dir(args.data_dir)
    source_frames = tuple(
        pd.read_csv(data_dir / "train" / f"train_source{source}.tsv", sep="\t", nrows=args.rows_per_source, dtype="string")
        for source in (1, 2, 3)
    )
    source1, source2, source3 = source_frames
    source1_count = max(1, args.pair_count // args.candidates_per_source1)
    source1_indices = _selected_indices(source1, source1_count)
    source2_indices = _selected_indices(source2, (args.pair_count + 1) // 2)
    source3_indices = _selected_indices(source3, args.pair_count // 2)
    pairs = pd.DataFrame(
        {
            "source1_entity_id": [source1.loc[source1_indices[index % source1_count], "entity_id"] for index in range(args.pair_count)],
            "candidate_entity_id": [
                source2.loc[source2_indices[index // 2], "entity_id"] if index % 2 == 0 else source3.loc[source3_indices[index // 2], "entity_id"]
                for index in range(args.pair_count)
            ],
            "rank": [(index % args.candidates_per_source1) + 1 for index in range(args.pair_count)],
        }
    )
    adapter, adapter_seconds = _timed(lambda: DataFrameRecordAdapter(source1, source2, source3))
    joined_batches, dataframe_join_seconds = _timed(
        lambda: list(adapter.iter_joined_batches(pairs, batch_size=args.batch_size))
    )
    joined = pd.concat([batch[0] for batch in joined_batches], ignore_index=True)
    fit_pairs = min(args.fit_pairs, len(joined))
    extractor, fit_seconds = _timed(lambda: PairwiseFeatureExtractor().fit(joined.iloc[:fit_pairs]))
    _, cold_seconds = _timed(lambda: extractor.transform(joined))
    features, warm_seconds = _timed(lambda: extractor.transform(joined))
    output_buffer = io.StringIO()
    _, output_seconds = _timed(lambda: features.to_csv(output_buffer, sep="\t", index=False))
    output_bytes = len(output_buffer.getvalue().encode("utf-8"))

    tracemalloc.start()
    _, traced_seconds = _timed(lambda: extractor.transform(joined))
    _, traced_peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    script_sides = sum(
        bool(script_indicators(value)["devanagari"] or script_indicators(value)["other"])
        for value in pd.concat([joined["source1_business_name"], joined["candidate_business_name"]]).fillna("")
    )
    results: dict[str, object] = {
        "workload": {
            "rows_per_source": args.rows_per_source,
            "pairs": len(pairs),
            "distinct_source1": int(pairs["source1_entity_id"].nunique()),
            "distinct_candidates": int(pairs["candidate_entity_id"].nunique()),
            "candidates_per_source1": args.candidates_per_source1,
            "batch_size": args.batch_size,
            "fit_pairs": fit_pairs,
            "missing_name_pairs": int((joined["source1_business_name"].fillna("").eq("") | joined["candidate_business_name"].fillna("").eq("")).sum()),
            "missing_address_pairs": int((joined["source1_business_address"].fillna("").eq("") | joined["candidate_business_address"].fillna("").eq("")).sum()),
            "script_bearing_record_sides": script_sides,
        },
        "seconds": {
            "dataframe_adapter_build": adapter_seconds,
            "dataframe_join": dataframe_join_seconds,
            "tfidf_fit": fit_seconds,
            "feature_cold": cold_seconds,
            "feature_warm": warm_seconds,
            "output_write": output_seconds,
            "feature_with_tracemalloc": traced_seconds,
        },
        "throughput_pairs_per_second": {
            "feature_warm": len(features) / warm_seconds,
            "feature_plus_write": len(features) / (warm_seconds + output_seconds),
        },
        "output": {"feature_columns": len(features.columns) - 2, "bytes": output_bytes, "bytes_per_pair": output_bytes / len(features)},
        "memory": {"tracemalloc_peak_mib": traced_peak / 1024**2},
        "notes": [
            "This workload uses real records but deliberately constructed pairs, not Aayush candidates or a validation split.",
            "tracemalloc materially changes timings and does not report all native allocations.",
            "SQLite timing is built from the bounded sample, not the full challenge corpus.",
        ],
    }
    if args.measure_sqlite:
        results["seconds"].update(_sqlite_timings(source_frames, pairs, args.batch_size))
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path)
    parser.add_argument("--rows-per-source", type=int, default=10_000)
    parser.add_argument("--pair-count", type=int, default=2_000)
    parser.add_argument("--candidates-per-source1", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=1_000)
    parser.add_argument("--fit-pairs", type=int, default=1_000)
    parser.add_argument("--measure-sqlite", action="store_true")
    parser.add_argument("--output", type=Path, help="Optional ignored JSON report path.")
    args = parser.parse_args()
    if args.pair_count <= 0 or args.candidates_per_source1 <= 0 or args.batch_size <= 0 or args.rows_per_source <= 0:
        parser.error("All count arguments must be positive.")
    results = run(args)
    report = json.dumps(results, indent=2)
    print(report)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(report + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
