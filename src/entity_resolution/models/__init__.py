"""Ashank-owned matching score and decision interfaces."""

from .decisions import ThresholdDecisionLayer
from .logistic import LogisticRegressionConfig, fit_logistic_regression, predict_match_probabilities
from .rule import DeterministicScorer, DeterministicScoringConfig
from .experiments import validate_frozen_split_manifest, verify_repository_evaluator_known_answer

__all__ = [
    "DeterministicScorer",
    "DeterministicScoringConfig",
    "LogisticRegressionConfig",
    "ThresholdDecisionLayer",
    "fit_logistic_regression",
    "predict_match_probabilities",
    "validate_frozen_split_manifest",
    "verify_repository_evaluator_known_answer",
]
