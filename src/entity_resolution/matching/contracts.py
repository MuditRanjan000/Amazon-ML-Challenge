"""Contracts shared by candidate adapters, feature extraction, and scoring."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

CANDIDATE_COLUMNS = ("source1_entity_id", "candidate_entity_id")
RAW_SCORE_COLUMNS = (
    "source1_entity_id",
    "candidate_entity_id",
    "raw_match_score",
    "scoring_method",
)
PROBABILITY_SCORE_COLUMNS = (
    "source1_entity_id",
    "candidate_entity_id",
    "match_probability",
)


@dataclass(frozen=True)
class CandidateInput:
    """Validated pair identifiers and optional blocker metadata kept separately."""

    pairs: pd.DataFrame
    retrieval_metadata: pd.DataFrame


def _require_columns(frame: pd.DataFrame, columns: tuple[str, ...]) -> None:
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise ValueError(f"Missing required column(s): {missing}.")


def validate_candidate_pairs(frame: pd.DataFrame) -> CandidateInput:
    """Validate pair IDs and return optional blocker fields outside the pair contract."""

    _require_columns(frame, CANDIDATE_COLUMNS)
    pairs = frame.loc[:, CANDIDATE_COLUMNS].copy()
    for column in CANDIDATE_COLUMNS:
        if pairs[column].isna().any():
            raise ValueError(f"{column} contains missing values.")
        pairs[column] = pairs[column].astype("string").str.strip()
        if (pairs[column] == "").any():
            raise ValueError(f"{column} contains empty values.")

    if not pairs["source1_entity_id"].str.startswith("S1-").all():
        raise ValueError("source1_entity_id values must use the S1- prefix.")
    if not pairs["candidate_entity_id"].str.startswith(("S2-", "S3-")).all():
        raise ValueError("candidate_entity_id values must use an S2- or S3- prefix.")
    if pairs.duplicated().any():
        examples = pairs.loc[pairs.duplicated(keep=False)].head(3).to_dict("records")
        raise ValueError(f"Duplicate candidate pairs are not allowed, e.g. {examples}.")

    metadata_columns = [column for column in frame.columns if column not in CANDIDATE_COLUMNS]
    return CandidateInput(
        pairs=pairs.reset_index(drop=True),
        retrieval_metadata=frame.loc[:, metadata_columns].reset_index(drop=True),
    )


def validate_complete_source1_ids(source1_ids: pd.Series | list[str]) -> pd.Series:
    """Validate the explicit S1 universe needed to retain no-candidate entities."""

    ids = pd.Series(source1_ids, dtype="string").str.strip()
    if ids.isna().any() or (ids == "").any() or not ids.str.startswith("S1-").all():
        raise ValueError("Complete source1 IDs must be non-empty S1- identifiers.")
    if ids.duplicated().any():
        raise ValueError("Complete source1 IDs must be unique.")
    return ids
