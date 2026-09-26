"""Legacy compatibility surface for Ashank's moved matching interfaces.

The canonical public implementations are now in ``features`` and ``models``.
Lazy exports prevent importing matching-specific normalisation helpers from
creating a feature-package cycle.
"""

from .contracts import CANDIDATE_COLUMNS, RAW_SCORE_COLUMNS

__all__ = [
    "CANDIDATE_COLUMNS",
    "RAW_SCORE_COLUMNS",
    "FeatureConfig",
    "PairwiseFeatureExtractor",
    "DeterministicScorer",
    "DeterministicScoringConfig",
    "ThresholdDecisionLayer",
]


def __getattr__(name: str):
    if name in {"FeatureConfig", "PairwiseFeatureExtractor"}:
        from src.entity_resolution.features import FeatureConfig, PairwiseFeatureExtractor

        return {"FeatureConfig": FeatureConfig, "PairwiseFeatureExtractor": PairwiseFeatureExtractor}[name]
    if name in {"DeterministicScorer", "DeterministicScoringConfig", "ThresholdDecisionLayer"}:
        from src.entity_resolution.models import DeterministicScorer, DeterministicScoringConfig, ThresholdDecisionLayer

        return {
            "DeterministicScorer": DeterministicScorer,
            "DeterministicScoringConfig": DeterministicScoringConfig,
            "ThresholdDecisionLayer": ThresholdDecisionLayer,
        }[name]
    raise AttributeError(name)
