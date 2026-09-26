"""Validated, bounded joins between candidate pairs and source records."""

from __future__ import annotations

import csv
import sqlite3
from pathlib import Path
from typing import Iterator

import pandas as pd

from .contracts import CandidateInput, validate_candidate_pairs

RECORD_COLUMNS = ("entity_id", "business_name", "business_address", "country")


def _validate_records(frame: pd.DataFrame, expected_prefix: str) -> pd.DataFrame:
    missing = [column for column in RECORD_COLUMNS if column not in frame.columns]
    if missing:
        raise ValueError(f"Record frame missing columns: {missing}.")
    records = frame.loc[:, RECORD_COLUMNS].copy()
    records["entity_id"] = records["entity_id"].astype("string").str.strip()
    if records["entity_id"].isna().any() or not records["entity_id"].str.startswith(expected_prefix).all():
        raise ValueError(f"Record IDs must use the {expected_prefix} prefix.")
    if records["entity_id"].duplicated().any():
        raise ValueError(f"Duplicate {expected_prefix} IDs in record frame.")
    return records


def _format_joined_rows(pairs: pd.DataFrame, lookup: dict[str, dict[str, object]]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    missing: list[str] = []
    for pair in pairs.to_dict("records"):
        source = lookup.get(pair["source1_entity_id"])
        candidate = lookup.get(pair["candidate_entity_id"])
        if source is None:
            missing.append(pair["source1_entity_id"])
        if candidate is None:
            missing.append(pair["candidate_entity_id"])
        if source is None or candidate is None:
            continue
        rows.append(
            {
                **pair,
                "source1_business_name": source["business_name"],
                "source1_business_address": source["business_address"],
                "source1_country": source["country"],
                "candidate_business_name": candidate["business_name"],
                "candidate_business_address": candidate["business_address"],
                "candidate_country": candidate["country"],
            }
        )
    if missing:
        raise KeyError(f"Candidate/source record IDs unavailable, e.g. {sorted(set(missing))[:5]}.")
    return pd.DataFrame(rows)


class DataFrameRecordAdapter:
    """Reusable in-memory adapter for fixtures and already-loaded small subsets."""

    def __init__(self, source1: pd.DataFrame, source2: pd.DataFrame, source3: pd.DataFrame):
        source_frames = [
            _validate_records(source1, "S1-"),
            _validate_records(source2, "S2-"),
            _validate_records(source3, "S3-"),
        ]
        records = pd.concat(source_frames, ignore_index=True)
        self._lookup = records.set_index("entity_id").to_dict("index")

    def iter_joined_batches(
        self, candidate_frame: pd.DataFrame, batch_size: int = 10_000
    ) -> Iterator[tuple[pd.DataFrame, pd.DataFrame]]:
        candidate_input = validate_candidate_pairs(candidate_frame)
        if batch_size <= 0:
            raise ValueError("batch_size must be positive.")
        for start in range(0, len(candidate_input.pairs), batch_size):
            stop = start + batch_size
            yield (
                _format_joined_rows(candidate_input.pairs.iloc[start:stop], self._lookup),
                candidate_input.retrieval_metadata.iloc[start:stop].reset_index(drop=True),
            )


class SQLiteRecordStore:
    """Disk-backed lookup built once, then reused for bounded candidate joins.

    The store avoids retaining all source records in RAM and avoids rescanning the
    TSVs for each candidate batch. Store files are local intermediates and should
    remain under the ignored `.tmp/` directory.
    """

    def __init__(self, database_path: str | Path):
        self.database_path = Path(database_path)

    def build(self, source_paths: list[str | Path], overwrite: bool = False) -> None:
        if self.database_path.exists() and not overwrite:
            raise FileExistsError(f"Record store already exists: {self.database_path}.")
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        connection = sqlite3.connect(self.database_path)
        try:
            connection.execute("DROP TABLE IF EXISTS records")
            connection.execute(
                "CREATE TABLE records (entity_id TEXT PRIMARY KEY, business_name TEXT, business_address TEXT, country TEXT)"
            )
            for source_path in source_paths:
                with Path(source_path).open(encoding="utf-8", newline="") as handle:
                    reader = csv.DictReader(handle, delimiter="\t")
                    required = set(RECORD_COLUMNS)
                    if not reader.fieldnames or not required.issubset(reader.fieldnames):
                        raise ValueError(f"Unexpected record schema in {source_path}.")
                    rows = (
                        (
                            (row.get("entity_id") or "").strip(),
                            row.get("business_name") or "",
                            row.get("business_address") or "",
                            row.get("country") or "",
                        )
                        for row in reader
                    )
                    connection.executemany("INSERT INTO records VALUES (?, ?, ?, ?)", rows)
            connection.commit()
        finally:
            connection.close()

    def _fetch(self, ids: list[str]) -> dict[str, dict[str, object]]:
        connection = sqlite3.connect(self.database_path)
        try:
            lookup: dict[str, dict[str, object]] = {}
            for start in range(0, len(ids), 900):
                chunk = ids[start : start + 900]
                placeholders = ",".join("?" for _ in chunk)
                for entity_id, name, address, country in connection.execute(
                    f"SELECT entity_id, business_name, business_address, country FROM records WHERE entity_id IN ({placeholders})",
                    chunk,
                ):
                    lookup[entity_id] = {
                        "business_name": name,
                        "business_address": address,
                        "country": country,
                    }
            return lookup
        finally:
            connection.close()

    def iter_joined_batches(
        self, candidate_frame: pd.DataFrame, batch_size: int = 10_000
    ) -> Iterator[tuple[pd.DataFrame, pd.DataFrame]]:
        candidate_input = validate_candidate_pairs(candidate_frame)
        if batch_size <= 0:
            raise ValueError("batch_size must be positive.")
        for start in range(0, len(candidate_input.pairs), batch_size):
            stop = start + batch_size
            pairs = candidate_input.pairs.iloc[start:stop]
            ids = pd.concat([pairs["source1_entity_id"], pairs["candidate_entity_id"]]).unique().tolist()
            yield (
                _format_joined_rows(pairs, self._fetch(ids)),
                candidate_input.retrieval_metadata.iloc[start:stop].reset_index(drop=True),
            )
