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
    - **Stage 1 & 2 (Completed)**: Evaluated D2 vs word-unigram on frozen validation.
    - **BLK-019 fair comparison (same 20k val S1, same harness):**
      - D2: R@10/50/200 0.902/0.942/0.959, 9.6 S1/s.
      - word unigram: 0.925/0.963/0.977, 99.8 S1/s.
      - word + max_df 0.02: 0.920/0.959/0.974, 446 S1/s.
      - Word@50 beats D2@200.
    - **Decision (Mudit)**: The D2 comparison is complete (metrics source: BLK-019). The final selected blocker is **word unigram** (name+address, K=200). `char_wb` (D2) has been retired.
    - **BLK-020 Handoff**: Aayush delivered the final blocker artifacts (`train_candidate_pairs.tsv.gz` and `validation_candidate_pairs.tsv.gz`) and metadata. Redundant AWS generation scripts have been purged, and a verification handoff script (`scripts/verify_blk020_handoff.py`) has been added to the repository. The low-memory streaming verification pipeline confirmed exactly 353,091,200 train pairs and 88,273,000 validation pairs with exactly 0 duplicates. The validation set perfectly matches the 441,365 unique S1 IDs required by the frozen manifest. Handoff to Ashank is complete.

# Blocking — FINAL (BLK-020, 2026-09-26)
- Frozen blocker: word unigram, name+address, country partition, min_df 2, K=200 (config_sha 657f59868e58, commit 69a91f1).
- Train: 353,091,200 pairs, R@200 0.97795. Validation: 88,273,000 pairs, R@200 0.9779, ceiling F0.5 0.99213.
- Files in `s3://amazon-ml-2026-blocking-716522590518/run-69a91f1/final/`; see `artifacts/blocking/blocking_report.md`. Test candidates pending the IDF-on-test ruling.

# Night of 26 Sep (Aayush)
- Test candidates done (346.5M pairs, France partition OK).
- RULE-001 baseline: val F0.5 0.7505, submission files validated.
- Blocking v2 rejected at the gate (best ceiling 0.9959); BLK-020 final.
- Competition stats for train+val ready.
- AWS paused: key revoked, new IAM keys needed.

# Experiment Results
- **EXP-001:** F0.5 = 0.19372. Baseline confirms need for fuzzy matching and blocking.
- **EXP-002:** Aborted. Computing non-partitioned sparse-matrix dot products for 10.3M rows proved computationally non-viable for rapid iteration without an AWS cluster.

# Current Best Pipeline
- Exact string matching on normalized business names partitioned by country. (Baseline)

## Ashank matcher: BLK-020 full frozen-validation baseline (AWS, 2026-09-27)

- The pair-local 43-feature L2 Logistic model trained on the verified 2,500-S1 / 500,000-pair BLK-020 training preflight was scored over all 88,273,000 frozen-validation candidate pairs (441,365 Source-1 IDs). The pinned AWS runner archive SHA-256 is `7e0d7cf3f0bd93542856d65752f3d3c31d01e8e90fbc8c2cd83d124bbd1438d9`.
- Official shared-evaluator tuning result: K=200 at probability threshold 0.60, macro F0.5 `0.8554358243`, macro precision `0.9046184359`, macro recall `0.7707768545`, pairwise TP/FP/FN `1,180,659 / 79,972 / 347,884`, and singleton accuracy `0.7714413281` (19,006/24,637). K=100 at the same threshold was `0.8553176338`; this is a tuning comparison, not a final submission threshold.
- Both complete score files, rank-join report, decision artifacts, threshold sweep, fitted feature artifact, Logistic artifact, and record store are durable in the private challenge S3 bucket. The two temporary c7i.2xlarge workers were confirmed terminated after preservation. No reusable feature/model artifact remains only on EBS.
- The temporary rank-sidecar SQLite database itself was not synced before termination; its hash and zero-missing/orphan report are durable, and it is deterministically rebuildable from the preserved BLK-020 gzip and pair-ID score files without re-extracting features.
- Integration work is isolated in `D:\Amazon-ML-Challenge-integration` atop `origin/feature/mudit-submission` `385b36d`; the five reported shared-file conflicts were resolved using Mudit's branch as the base. Package imports now use `entity_resolution.*` and shared `ER_DATA_DIR` configuration.
- Model v2 is a separate, unfit feature schema. It adds BLK-020 full-pool retrieval statistics without changing the saved 43-feature artifact: locking_score, est_score, second_score, 
_s1, is_best_s1, margin_to_best, and owner_gap. The required 10.2M-row train+validation statistics parquet, its SHA-256, and matching full train candidate artifact are not present in the checkout or private challenge S3; obtain fresh paths/hashes from Aayush/Mudit before fitting v2.
- The Model V2 Diagnostic was run successfully on AWS: Evaluated against an internal split of the 500k sample candidate preflight dataset, the HistGradientBoostingClassifier achieved a macro F0.5 of 0.899 (Precision: 0.934, Recall: 0.833) at a threshold of 0.55, vastly outperforming the Logistic Regression baseline which achieved 0.864 at a threshold of 0.50.

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
