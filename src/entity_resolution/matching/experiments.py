"""Compatibility import for bounded experiments now owned by ``models``."""

from entity_resolution.models.experiments import (
    BoundedExperimentConfig,
    candidate_recall,
    labels_for_candidates,
    read_candidate_groups,
    read_source1_ids,
    read_source1_records,
    run_bounded_experiment,
    select_ground_truth,
    sha256_file,
    validate_entity_disjoint,
    verify_repository_evaluator_known_answer,
)

__all__ = [
    "BoundedExperimentConfig",
    "candidate_recall",
    "labels_for_candidates",
    "read_candidate_groups",
    "read_source1_ids",
    "read_source1_records",
    "run_bounded_experiment",
    "select_ground_truth",
    "sha256_file",
    "validate_entity_disjoint",
    "verify_repository_evaluator_known_answer",
]
