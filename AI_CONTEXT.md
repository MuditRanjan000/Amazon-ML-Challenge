# Project Overview
Amazon ML Challenge 2026 - Business Entity Resolution. Goal is to map Source 2 and Source 3 entities to a reference set of Source 1 entities across large, noisy, multilingual datasets.

# Challenge Understanding
- **Input:** 3 TSV sources with `entity_id`, `business_name`, `business_address`, `country`.
- **Output:** Two TSV files: `matching_results.tsv` (scored on leaderboard) and `candidate_pairs.tsv` (used for auditing).
- **Metric:** Macro-averaged F0.5 per Source 1 entity. Precision is weighted 2x over recall. Singletons must be correctly predicted as empty.
- **Constraints:** Max 8B parameter model, MIT/Apache 2.0 license, no external data/APIs.

# Dataset Discoveries
- Train set: S1 (~2.2M), S2 (~5M), S3 (~5.2M). Test set: S1 (~1.7M), S2 (~4.8M), S3 (~5.0M).
- S1 entities map to zero, one, or many S2/S3 entities (1-to-many relationship).
- About 5.5% of train S1 entities are singletons (0 matches).
- Noise includes transliterations (English/Hindi/Tamil), missing addresses, typos, and formatting differences.

# Current Architecture
- Modular pipeline split into: Data Loading -> Validation Split -> Candidate Generation (Blocking) -> Pairwise Feature Engineering -> Model Inference -> Threshold Logic.
- Current codebase relies on deterministic exact matching (Baseline).

# Team Responsibilities
- **Mudit:** Project lead, validation framework, F0.5 evaluator, experiment tracking, integration, final submission.
- **Aayush:** Candidate generation / blocking specialist, retrieval strategies, candidate recall optimization.
- **Ashank:** Matching model specialist, feature engineering, ML models, threshold optimization, model-side error analysis.
# Frozen Validation Split
A 20% validation split on `source1_entity_id` is strictly enforced and frozen to disk. All experiments must use `artifacts/validation_split/val_ids.csv` to ensure comparability. These files are frozen experiment-control artifacts and must not change, ensuring all teammates evaluate against identical data. Details are logged in `docs/validation_manifest.md`.

# Completed Experiments
- EXP-001: Baseline Exact Match (Failed/Unusable, capped at ~32% recall).
- EXP-002: Advanced TF-IDF Blocking (Aborted due to inefficient global retrieval and OOM errors).
- System validation framework established.
- Submission generation layer and validator implemented to guarantee 100% portal compliance.

# Active Experiments
- **EXP-002B**: Optimized Partitioned TF-IDF Blocking. Redesigned to strictly enforce country partitions, disk-caching of vectorizers/sparse matrices, and multi-threaded batch inference for scalability. Testing on a 10k query sample.

# Experiment Results
- **EXP-001:** F0.5 = 0.19372. Baseline confirms need for fuzzy matching and blocking.
- **EXP-002:** Aborted. Computing non-partitioned sparse-matrix dot products for 10.3M rows proved computationally non-viable for rapid iteration without an AWS cluster.

# Current Best Pipeline
- Exact string matching on normalized business names partitioned by country. (Baseline)

# Known Issues
- Baseline misses all typo, transliteration, and abbreviation variations, resulting in extremely poor recall.

# Future Experiments
- **Current Goal:** Establish a robust candidate generation (blocking) framework. We are starting with an exact normalized name baseline to measure the strict recall floor before testing token, character n-gram, and TF-IDF blocking strategies.
- Pairwise string distance features + LightGBM matching (Ashank).

# Submission History
- None yet.

# Final Pipeline Architecture
1. Data Loading
2. Preprocessing
3. Blocking (Candidate Generation)
4. Feature Generation
5. Matching Model
6. Decision Layer
7. Submission Generator -> `output/matching_results.tsv`

# Important Decisions
- Team Playbook adopted as the core execution strategy.
- Repositiory migrated to a strict, modular framework under `src/entity_resolution`.
