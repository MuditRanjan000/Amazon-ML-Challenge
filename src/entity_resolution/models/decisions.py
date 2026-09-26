"""Threshold decisions that retain explicit singleton/no-candidate outputs."""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..matching.contracts import validate_complete_source1_ids


class ThresholdDecisionLayer:
    """Apply a configurable threshold without top-1 or one-to-one constraints."""

    def __init__(self, threshold: float = 0.75, score_column: str = "raw_match_score"):
        if not 0.0 <= threshold <= 1.0:
            raise ValueError("threshold must be in [0, 1].")
        self.threshold = threshold
        self.score_column = score_column

    def decide(self, scores: pd.DataFrame, complete_source1_ids: pd.Series | list[str]) -> pd.DataFrame:
        required = {"source1_entity_id", "candidate_entity_id", self.score_column}
        missing = sorted(required - set(scores.columns))
        if missing:
            raise ValueError(f"Score table missing columns: {missing}.")
        source1_ids = validate_complete_source1_ids(complete_source1_ids)
        pairs = scores.loc[:, ["source1_entity_id", "candidate_entity_id", self.score_column]].copy()
        if pairs.duplicated(["source1_entity_id", "candidate_entity_id"]).any():
            raise ValueError("Scores contain duplicate candidate pairs.")
        if not pairs["source1_entity_id"].astype("string").str.startswith("S1-").all():
            raise ValueError("Scores contain invalid Source-1 IDs.")
        if not pairs["candidate_entity_id"].astype("string").str.startswith(("S2-", "S3-")).all():
            raise ValueError("Scores contain invalid candidate IDs.")
        numeric_scores = pd.to_numeric(pairs[self.score_column], errors="coerce")
        if numeric_scores.isna().any() or not np.isfinite(numeric_scores).all():
            raise ValueError("Scores must be finite numeric values.")
        accepted = pairs.loc[numeric_scores >= self.threshold]
        grouped = accepted.groupby("source1_entity_id", sort=False)["candidate_entity_id"].agg(
            lambda ids: ",".join(ids.astype(str))
        )
        output = pd.DataFrame({"source1_entity_id": source1_ids})
        output["matched_entity_ids"] = output["source1_entity_id"].map(grouped).fillna("")
        return output
