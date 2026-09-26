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
- **EXP-002B**: Optimized Partitioned TF-IDF Blocking.
    - **Stage 1 (Completed)**: Out-of-core memory-safe pipeline created. Evaluated on 1k queries. D2 (`business_name` + `business_address`, char ngrams 3,5) achieved a remarkable **96.29% Recall@200**.
    - **Stage 2 (Running)**: Actively running Variant D2 against the entire frozen validation split (35k queries) to generate `validation_candidate_pairs.tsv` and measure full-scale metrics.
    - **2026-09-26 (Aayush): Stage 2 code could not produce D2.**
      - It ran **word** (3,5)-grams (the `analyzer` was never passed).
      - It would OOM on train.
      - It fails the split SHA check on Linux.
      - Val is 441,365 S1, not 35k.
      - Fixed on `experiment/aayush-blocking-hybrid-v2`. Any Stage 2 output produced from the old code is invalid.
    - **BLK-018 (fixed D2, 5k val):** R@10 0.900 / R@50 0.940 / R@200 0.957, ceiling F0.5@200 0.984. **About 10 S1/s on the laptop**, so train+val is about 60 h and test about 48 h.
    - **BLK-019 fair comparison (same 20k val S1, same harness):**
      - D2: R@10/50/200 0.902/0.942/0.959, 9.6 S1/s.
      - word unigram: 0.925/0.963/0.977, 99.8 S1/s.
      - word + max_df 0.02: 0.920/0.959/0.974, 446 S1/s.
      - Word@50 beats D2@200. Mudit to decide the final blocker.
    - **Decision needed (Mudit):**
      - (a) D2 on AWS, fanned out across instances;
      - (b) D2 on a train-S1 subsample for Ashank (val + test stay full);
      - (c) switch to word-unigram name+address (R@200 0.972, 10–30x faster; re-measure with the new class first).

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
- AGENTS.md rule 5 (push to main + tradebot redeploy) replaced with branch + PR policy; CLAUDE.md/GEMINI.md mirror AGENTS.md via `execution/sync_agent_docs.py` (PR from `feature/aayush-agent-docs`).
- Blocking contract + oracle-ceiling F0.5 metric + hand-off gates proposed in `directives/blocking.md` (pending Ashank/Mudit sign-off).
- 2026-09-25 pipeline refactor (`feature/aayush-pipeline-refactor`): installable package, pinned deps, env-driven config, raw-text parquet loader, Unicode-safe normalizer (fixes the Devanagari shredding bug), sparse top-k TF-IDF (no OOM on 15.6 GB), vectorized evaluators, official-validator wrapper, JSONL experiment log, 25 tests.
- **Current best blocking (BLK-011, 20k val):** word-unigram TF-IDF on name+address, forward top-K, `max_df` 0.02: R@10 0.912 / R@50 0.956 / R@200 0.972; ceiling F0.5 ≥ 0.96 in every bucket. The next experiments are reverse + union and a full-val run to freeze K / max_df.
