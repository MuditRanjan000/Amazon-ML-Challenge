# Directive: Baseline Matching (Experiment 1)

## Goal
Establish a naive baseline for the Amazon ML Challenge 2026 by matching Source 2 (S2) and Source 3 (S3) records to Source 1 (S1) using exact string matching on the `business_name` field, partitioned by `country`.

## Inputs
- `6ab10eb3b23ba_student_resource/student_resource/dataset/train/train_source1.tsv`
- `6ab10eb3b23ba_student_resource/student_resource/dataset/train/train_source2.tsv`
- `6ab10eb3b23ba_student_resource/student_resource/dataset/train/train_source3.tsv`
- `6ab10eb3b23ba_student_resource/student_resource/dataset/train/train_ground_truth.tsv`

## Outputs
- `--split train|val`: local F0.5 printed and logged to `experiments/results/experiments.jsonl`. The regression target on full train is **0.19372** (reproduced 2026-09-25 by the refactored pipeline: 0.1937173).
- `--split test`: `output/matching_results.tsv` + `output/candidate_pairs.tsv` (candidates == matches), validated with the official validator (PASS, including `--check-ids`).

Run: `python execution/run_baseline.py --split {train,val,test}`. Data paths come from `ER_DATA_DIR`; nothing is hardcoded.

## Script to Use / Create
- `execution/run_baseline.py`

## Instructions
1. **Load Data**: Read the training data using Pandas (sep='\t'). Treat `NaN` or `"null"` values in `business_name` as empty strings.
2. **Text Normalization**: Lowercase and strip leading/trailing whitespaces on `business_name`.
3. **Partition by Country**: Only compare records that share the exact same `country` string.
4. **Exact Match**: Join S2 and S3 onto S1 where `normalized_business_name` and `country` are exactly equal.
5. **Format Output**: For each S1 `entity_id`, group the matched S2 and S3 `entity_id`s into a comma-separated list. Ensure singletons (S1 entities with no matches) are preserved with an empty string.
6. **Local Evaluation**: 
   - Calculate Precision and Recall per S1 entity. 
   - Calculate F0.5 = (1.25 * Precision * Recall) / (0.25 * Precision + Recall).
   - Macro-average the F0.5 score across all S1 entities (including singletons, which score 1.0 if correctly predicted empty, and 0.0 otherwise).
7. **Write Files**: Export the two TSV files to `.tmp/`.
