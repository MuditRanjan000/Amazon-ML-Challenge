"""Separate v2 features derived from Aayush's full-pool retrieval statistics.

These features are deliberately not part of :class:`PairwiseFeatureExtractor`.
The saved 43-feature BLK-020 baseline remains immutable and pair-local; a v2
model must be fitted and persisted with this additional schema.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


COMPETITION_STATS_COLUMNS = (
    "candidate_entity_id",
    "best_score",
    "second_score",
    "n_s1",
    "best_s1",
)

COMPETITION_FEATURE_COLUMNS = (
    "blocking_score",
    "best_score",
    "second_score",
    "n_s1",
    "is_best_s1",
    "margin_to_best",
    "owner_gap",
)


@dataclass(frozen=True)
class CompetitionFeatureConfig:
    """Versioned, explicit missing-runner-up convention for model v2."""

    version: str = "competition-v2"
    missing_second_score: float = -1.0
    require_complete_stats: bool = True


def add_competition_features(
    pair_features: pd.DataFrame,
    candidate_pairs: pd.DataFrame,
    competition_stats: pd.DataFrame,
    config: CompetitionFeatureConfig | None = None,
) -> tuple[pd.DataFrame, list[str]]:
    """Join complete full-pool stats and return a *new* v2 feature frame.

    ``best_s1`` is never used as a hard filter.  Its only model representation
    is the boolean ``is_best_s1``; predictions retain every candidate pair.
    The caller must supply candidate retrieval ``score`` for every pair and
    stats produced over the relevant complete Source-1 pool (train+validation
    for validation, all test Source-1 IDs for test).
    """

    config = config or CompetitionFeatureConfig()
    pair_key = ["source1_entity_id", "candidate_entity_id"]
    missing_features = [column for column in pair_key if column not in pair_features]
    missing_pairs = [column for column in (*pair_key, "score") if column not in candidate_pairs]
    missing_stats = [column for column in COMPETITION_STATS_COLUMNS if column not in competition_stats]
    if missing_features or missing_pairs or missing_stats:
        raise ValueError(
            "Competition feature inputs are incomplete: "
            f"pair_features={missing_features}, candidate_pairs={missing_pairs}, stats={missing_stats}."
        )
    if pair_features.duplicated(pair_key).any() or candidate_pairs.duplicated(pair_key).any():
        raise ValueError("Competition feature inputs contain duplicate candidate pairs.")
    if competition_stats["candidate_entity_id"].duplicated().any():
        raise ValueError("Competition statistics must have one row per candidate_entity_id.")

    metadata = candidate_pairs.loc[:, [*pair_key, "score"]].copy()
    metadata["source1_entity_id"] = metadata["source1_entity_id"].astype("string")
    metadata["candidate_entity_id"] = metadata["candidate_entity_id"].astype("string")
    metadata["blocking_score"] = pd.to_numeric(metadata.pop("score"), errors="coerce")
    if metadata["blocking_score"].isna().any() or not np.isfinite(metadata["blocking_score"]).all():
        raise ValueError("Candidate retrieval score must be finite for every v2 pair.")
    stats = competition_stats.loc[:, COMPETITION_STATS_COLUMNS].copy()
    stats["candidate_entity_id"] = stats["candidate_entity_id"].astype("string")
    for column in ("best_score", "second_score", "n_s1"):
        stats[column] = pd.to_numeric(stats[column], errors="coerce")
    if stats["best_score"].isna().any() or stats["n_s1"].isna().any() or not np.isfinite(stats[["best_score", "n_s1"]]).all().all():
        raise ValueError("Competition statistics require finite best_score and n_s1 values.")
    stats["best_s1"] = stats["best_s1"].astype("string")

    joined = pair_features.merge(metadata, on=pair_key, how="left", validate="one_to_one")
    joined = joined.merge(stats, on="candidate_entity_id", how="left", validate="many_to_one")
    missing_stats_rows = joined["best_score"].isna()
    if config.require_complete_stats and missing_stats_rows.any():
        examples = joined.loc[missing_stats_rows, "candidate_entity_id"].astype(str).drop_duplicates().head(5).tolist()
        raise ValueError(f"Competition statistics are missing for {missing_stats_rows.sum():,} pairs, e.g. {examples}.")
    if missing_stats_rows.any():
        joined.loc[missing_stats_rows, ["best_score", "n_s1"]] = 0.0
        joined.loc[missing_stats_rows, "best_s1"] = ""
    joined["second_score"] = joined["second_score"].fillna(config.missing_second_score)
    joined["is_best_s1"] = (joined["source1_entity_id"] == joined["best_s1"]).astype("int8")
    joined["margin_to_best"] = joined["best_score"] - joined["blocking_score"]
    joined["owner_gap"] = joined["best_score"] - joined["second_score"]
    if not np.isfinite(joined.loc[:, COMPETITION_FEATURE_COLUMNS].to_numpy(dtype=float)).all():
        raise ValueError("Competition feature construction produced non-finite values.")

    base_columns = [column for column in pair_features.columns if column not in pair_key]
    feature_columns = [*base_columns, *COMPETITION_FEATURE_COLUMNS]
    return joined.loc[:, [*pair_key, *feature_columns]], feature_columns
