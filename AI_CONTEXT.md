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

# Completed Experiments
- EXP-001: Baseline Exact Match

# Experiment Results
- **EXP-001:** F0.5 = 0.19372. Baseline confirms need for fuzzy matching and blocking.

# Current Best Pipeline
- Exact string matching on normalized business names partitioned by country. (Baseline)

# Known Issues
- Baseline misses all typo, transliteration, and abbreviation variations, resulting in extremely poor recall.

# Future Experiments
- Implement robust cross-validation split (Mudit).
- TF-IDF character n-gram blocking to improve candidate recall (Aayush).
- Pairwise string distance features + LightGBM matching (Ashank).

# Submission History
- None yet.

# Important Decisions
- Team Playbook adopted as the core execution strategy.
- Repositiory migrated to a strict, modular framework under `src/entity_resolution`.
