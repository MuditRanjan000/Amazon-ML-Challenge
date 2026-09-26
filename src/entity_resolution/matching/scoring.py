"""Compatibility import for Ashank's rule baseline model."""

from src.entity_resolution.models.rule import DeterministicScorer, DeterministicScoringConfig

__all__ = ["DeterministicScorer", "DeterministicScoringConfig"]
