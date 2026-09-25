# Validation Split Manifest

This document serves as the immutable ground-truth reference for the validation split used in all Entity Resolution experiments for the Amazon ML Challenge.

## Rationale
To ensure perfectly comparable metrics across models (TF-IDF, embeddings, logical rules), it is critical that all experiments are evaluated on the exact same hold-out set of Source 1 queries. If splits fluctuate, improvements in metrics could be hallucinated by data variance.

## Split Details
- **Test Size**: 20%
- **Random State (Seed)**: 42
- **Method**: The split was strictly executed on `source1_entity_id` values to completely prevent data leakage between Train and Validation sets.

## Frozen Artifacts
The specific sets of Source 1 entity IDs have been serialized and frozen to disk. These files are frozen experiment-control artifacts and must not change, ensuring all teammates have identical validation splits when checking out the repository.
- **Train IDs**: `artifacts/validation_split/train_ids.csv`
- **Validation IDs**: `artifacts/validation_split/val_ids.csv`

## Usage Policy
All subsequent blocking and matching experiments MUST use this frozen split by leveraging the `create_validation_split` pipeline, which will automatically load these exact artifacts.

A strict JSON manifest representing these exact files and their SHA-256 checksums is located at `artifacts/validation_split/manifest.json`.
