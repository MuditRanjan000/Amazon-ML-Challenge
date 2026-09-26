"""Compatibility import for the pre-ownership pairwise feature module.

New code must import from :mod:`src.entity_resolution.features`.  Keeping this
module preserves existing CLI/test imports and lets trusted local feature
pickles created before the ownership refactor resolve their class path.
"""

from src.entity_resolution.features.pairwise import FeatureConfig, PairwiseFeatureExtractor

__all__ = ["FeatureConfig", "PairwiseFeatureExtractor"]
