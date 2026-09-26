"""Transparent deterministic baseline scoring; output is explicitly not calibrated."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from ..matching.contracts import RAW_SCORE_COLUMNS


@dataclass(frozen=True)
class DeterministicScoringConfig:
    """Provisional evidence weights to be calibrated only on Mudit's frozen split."""

    name_exact_weight: float = 0.30
    name_similarity_weight: float = 0.25
    name_tfidf_weight: float = 0.10
    address_exact_weight: float = 0.12
    address_similarity_weight: float = 0.12
    address_tfidf_weight: float = 0.05
    country_match_weight: float = 0.03
    postal_match_weight: float = 0.08
    numeric_disagreement_penalty: float = 0.18
    country_disagreement_penalty: float = 0.20
    postal_disagreement_penalty: float = 0.20
    missing_both_penalty: float = 0.20


class DeterministicScorer:
    method_name = "deterministic_lexical_v1_raw"

    def __init__(self, config: DeterministicScoringConfig | None = None):
        self.config = config or DeterministicScoringConfig()

    def score(self, features: pd.DataFrame) -> pd.DataFrame:
        required = {"source1_entity_id", "candidate_entity_id", "name_exact", "name_jaro_winkler", "name_tfidf_cosine", "address_exact", "address_char_ngram_similarity", "address_tfidf_cosine", "country_match", "postal_match", "numeric_disagreement", "country_disagreement", "postal_disagreement", "source1_name_missing", "candidate_name_missing", "source1_address_missing", "candidate_address_missing"}
        missing = sorted(required - set(features.columns))
        if missing:
            raise ValueError(f"Features missing deterministic baseline inputs: {missing}.")
        config = self.config
        raw_score = (
            config.name_exact_weight * features["name_exact"]
            + config.name_similarity_weight * features["name_jaro_winkler"]
            + config.name_tfidf_weight * features["name_tfidf_cosine"]
            + config.address_exact_weight * features["address_exact"]
            + config.address_similarity_weight * features["address_char_ngram_similarity"]
            + config.address_tfidf_weight * features["address_tfidf_cosine"]
            + config.country_match_weight * features["country_match"]
            + config.postal_match_weight * features["postal_match"]
            - config.numeric_disagreement_penalty * features["numeric_disagreement"]
            - config.country_disagreement_penalty * features["country_disagreement"]
            - config.postal_disagreement_penalty * features["postal_disagreement"]
        )
        both_blank = (
            features["source1_name_missing"]
            & features["candidate_name_missing"]
            & features["source1_address_missing"]
            & features["candidate_address_missing"]
        )
        raw_score = (raw_score - config.missing_both_penalty * both_blank).clip(lower=0.0, upper=1.0)
        return pd.DataFrame(
            {
                "source1_entity_id": features["source1_entity_id"].astype("string"),
                "candidate_entity_id": features["candidate_entity_id"].astype("string"),
                "raw_match_score": raw_score.astype(float),
                "scoring_method": self.method_name,
            }
        ).loc[:, RAW_SCORE_COLUMNS]
