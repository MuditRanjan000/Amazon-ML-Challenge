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

## [2026-09-25] Ashank Workstream Reconciliation
- Local matching setup remains preserved outside this checkout at `D:\Amazon_ML`; supplied data and PDFs remain there and are not copied into Git.
- Ashank's local-only branch `feature/ashank-model` was created from `origin/main` at `e8f2517833414444a1509f25322404a609eb31df`; no push, merge, submission, or deployment was performed.
- Execution scripts now resolve the supplied dataset via ignored `AMAZON_ML_DATA_DIR` configuration rather than a machine-specific hard-coded path. `.env.example` documents the contract.
- Reusable shared components verified: `DataLoader`, lightweight text normalisation, deterministic Source-1 split helper, macro-F0.5 evaluator, exact-name blocker, and blocking evaluator. The reported EXP-001 score remains unrerun on this checkout.
- Still missing for the matching workstream: saved canonical split metadata, a versioned candidate artifact from Aayush, calibrated model training/selection, and integration-facing score contracts from Mudit. See `docs/ashank_matching.md`.

## [2026-09-25] Ashank Pairwise Infrastructure
- Added unmeasured matching infrastructure under `src/entity_resolution/matching`: candidate contracts, bounded record adapters, Unicode-preserving lexical features with training-only fit/transform TF-IDF, deterministic raw scoring, and one-to-many threshold decisions with explicit singleton rows.
- Added `execution/run_matcher_baseline.py` and fixture tests. The deterministic score is intentionally named `raw_match_score`, not a probability; its weights/threshold are provisional until Mudit's frozen split is available.
- A 1,000-pair synthetic transform smoke check ran at 126 pairs/s (7.937 s; 4.67 MiB Python-tracked peak). It excludes fit/I/O/native allocations and is not a full-scale estimate.
- No canonical split was generated, no candidate artifact was consumed, and no full training/inference or submission work was run.

## [2026-09-25] Ashank Matcher Performance Investigation
- No Aayush candidates, Mudit frozen split/evaluator invocation, registry schema, or score-integration contract appeared in the checkout.
- Added a bounded record-representation cache, sparse unique-string TF-IDF transforms, and the pinned MIT `rapidfuzz==3.14.3` compiled distance path while preserving feature semantics and order.
- The reproducible 2,000-pair real-record diagnostic improved from 607 to 1,124 warm feature pairs/s (588 to 1,068 including TSV serialization). It is not a competition score or full-scale estimate; see `docs/ashank_matching.md` and `execution/profile_matcher.py`.

## [2026-09-25] Ashank Bounded Rule/LR Preflight
- Mudit clarified that benchmark comparisons must use a pre-existing frozen Source-1 split and official evaluator. Neither the actual split artifact nor the evaluator invocation is present in this checkout; the local random-split helper must not be used.
- Reconciled ownership without changing feature semantics: canonical pairwise features are under `src/entity_resolution/features/`; rule/LR/decision/experiment interfaces are under `src/entity_resolution/models/`; `matching/` keeps compatibility adapters for current imports and saved local feature artifacts.
- Added `execution/run_bounded_matching_experiment.py`. It requires explicit frozen train/validation ID files, candidates, labels, a train SQLite record store, and Mudit's experiment ID; it verifies disjoint IDs/deduplicated candidate pairs, retains complete candidate groups, hashes inputs, and writes full three-column Logistic probability scores separately from threshold decisions and diagnostic labels.
- The runner is fixture-tested only. It uses training-fitted deduplicated TF-IDF documents, `StandardScaler` plus L2 Logistic Regression, and a 0.50--0.95 threshold sweep. `predict_proba` is not claimed calibrated. No official benchmark result, split creation, full inference, commit, push, merge, deployment, cloud use, or paid compute was performed.

## [2026-09-25] Frozen Manifest and Temporary-Candidate Readiness
- Fetched `origin/main` to `93f69ec` and incorporated Mudit's manifest, partitioned blocking/submission support, and frozen-split loader changes into the working tree without a merge or commit. Local matching work remains preserved.
- The earlier `experiments/results` / `output` paths are superseded by `artifacts/validation_split/manifest.json`; do not invoke the fallback split generator.
- Added manifest hash/count/disjointness verification and a fixed shared-evaluator known-answer check to the bounded runner. The local evaluator check passes; Mudit's supplied known-answer fixture/invocation is still needed before a reported official score.
- Added disk-backed `execution/generate_temp_exact_candidates.py` for `TEMP-EXACT-NAME-v1`, using all training S2/S3 records for lookup while retaining complete candidate groups for bounded frozen entities. It has fixture coverage only; no candidates, feature artifacts, model, or benchmark score were generated because the frozen ID CSVs are missing.

## [2026-09-25] EXP-003A Execution Blocker Audit
- Fetched `origin/feature/mudit-submission` at `925ebb7`. It moves the manifest to `artifacts/validation_split/manifest.json` and declares the frozen CSV paths there, but `git ls-tree` confirms neither frozen CSV was committed; the branch ignores `*.csv`.
- Updated Ashank's bounded-run defaults and reproducible commands to the new `artifacts/validation_split/` path. The manifest is preserved locally, but no split was regenerated and no temporary candidates or matcher score were produced.
- EXP-003A remains blocked only on the two immutable CSV blobs (with manifest hashes) and Mudit's registry convention. Once supplied, run the documented bounded exact-name preflight before considering a larger frozen validation run.
- Created `experiments/results/experiment_registry.csv` with an explicit `EXP-003A-PENDING` blocked row; no metrics were fabricated.

## [2026-09-26] EXP-003A Temporary Exact-Name Diagnostic
- Fetched frozen CSVs from Mudit commit `9c0e8ae`; manifest SHA-256, both full CSV SHA-256 values, declared counts (1,765,456 train / 441,365 validation), and split disjointness verified. No split file was regenerated or edited.
- Fixed a label-ingestion defect: blank ground-truth cells are valid singleton labels, not missing Source-1 rows. Added regression coverage; all tests pass.
- Built the full S2/S3 temporary exact-name index (1.201 GiB) and full train record store (1.515 GiB), then ran a 1,000-train/1,000-validation frozen-subset diagnostic. TEMP-EXACT-NAME-v1 validation recall was 0.3139 (1,117/3,559); 200 selected validation entities had zero candidates.
- At threshold 0.55, rule F0.5 was 0.3600 and Logistic Regression F0.5 was 0.5124. Logistic had 32 FP / 2,530 FN and 0.96 singleton accuracy; 2,442 FNs were blocking misses and 88 were matcher rejections among retrieved positives. This is not a full benchmark or final model selection result.
- Full temporary-exact execution is deferred: projected 27.3M training and 8.52M validation pairs exceed the current bounded materialization limits and would take multi-hour feature extraction before candidate-generation overhead. Swap in Aayush EXP-002B paths/version using the documented command.

## [2026-09-26] D2 Scalable Matcher Readiness
- Mudit requires the supplied D2 country-partitioned name+address character-(3,5), top-200 blocker on both frozen train and frozen validation IDs. The stated 96.29% Recall@200 is Stage 1 only. D2 artifacts and Stage 2 diagnostics are not present locally.
- Added a SQLite candidate spool, complete-group bounded scoring, run identity/checkpoint hashes, resumable score parts, exact Logistic probability assembly, separate rule raw scores, and disk-backed one-to-many decisions. The shared evaluator is called only after complete frozen-ID coverage is checked. No frozen split, evaluator, or D2 blocker changed.
- The 50,000-pair limits remain diagnostic-only. Streaming validation has no silent pair cap. EXP-003A parity passed on 19,315 temporary-exact pairs / 1,000 validation entities: same IDs and 43 features, max Logistic delta `1.11e-16`, identical threshold decisions and macro F0.5 `0.5123914614` / singleton accuracy `0.96`.
- Measured 5,000-pair streaming batches at 1,402.3 pairs/s; working set 164.9 MiB start / 231.8 MiB peak; retained spool/parts/assembled outputs about 356.4 bytes/pair in this diagnostic. Full-scale feasibility remains uncertain pending actual D2 distribution.
- Added disk-backed candidate-recall / blocking-miss / matcher-rejection / TP-FP-FN counting. On EXP-003A it exactly reproduced 2,442 blocking misses, 88 matcher rejections, 1,029 TP, 32 FP, and 2,530 FN.
- Needed next: a D2 validation TSV for all frozen validation IDs and a same-generator D2 frozen-train artifact or reproducible complete-group train subset, with schema, hashes, corpus identity, counts/distribution/zero groups/recall metadata, plus Mudit’s EXP-003B-D2 registry convention. The first fit now uses stable country x training-label match-count stratification across the entire frozen train population (`stable_sha256_stratified_country_match_count_v1`, seed `20260926`), records population/sample distributions and selected-ID hash, then records retrieved candidate counts and class balance. Call that first LR model subset-trained, never all-train-equivalent. Once D2 validation size/distribution arrives, calculate full-run local time/disk from a D2-trained preflight before launch. A validation-only artifact remains transfer scoring only.
- Actual population check (labels + S1 country only): 1,765,456 frozen train IDs produced seed-`20260926` selected-ID hash `aa4b38718c00571feabe3b075ef69e63c0bb5aeed853edec6784571bb3778945`. The 1,000-entity sample contains 57 singletons, 56 one-match, and 887 multi-match entities (3,480 true match pairs), matching the two-country population strata closely. Candidate-derived retrieved-positive/difficult-negative/class-balance checks await D2 training candidates.

## [2026-09-26] BLK-019 Word-Unigram Supersession
- BLK-019 is now the final blocker. Earlier character-n-gram/D2 candidates are superseded and must not train a reported matcher. Required configuration: word unigram over business name + address, country partitioning, K=200, generator commit `187d955`, configuration hash `657f59868e58`.
- Mudit reported a 2,500-frozen-train-entity / 500,000-pair training preflight artifact at `artifacts/blocking/train_sample_candidate_pairs.tsv`, with blocker Recall@200 `0.97807` and ceiling F0.5@200 `0.99318`. These are unverified blocker diagnostics, not matcher performance.
- Local and all fetched remote references were checked: the TSV is absent; `origin/main` still exposes only the older D2 report. No model fit, threshold tuning, or validation score was run. Need Mudit to supply the TSV plus the exact 2,500 Source-1 ID manifest/hash and artifact metadata; without that ID universe, zero-candidate group accounting cannot be verified from pair rows alone.
- The pending preflight must use the supplied 2,500 candidate groups, not the local 1,000-ID seed sample. It needs a disk-backed/batched training path because 500,000 pairs exceed the old 50,000 diagnostic cap. The remaining required official input is BLK-019 validation candidates for all frozen validation IDs with the same generator/config and its count/distribution/recall metadata.
- Added `execution/run_streaming_training_preflight.py` in readiness for the physical artifact. It is training-only: SQLite pair spool/deduplication, SQLite unique normalized TF-IDF corpora, 43-column feature/label memory maps, L2 Logistic fit, and exact three-column training probability output. It has no validation IDs/evaluator/threshold path and no default 50,000-pair cap. Fixture coverage confirms disk-style corpus fitting preserves feature values.
- BLK-019 handoff arrived and passed integrity checks. ZIP remains preserved in Downloads; extracted local inputs are under ignored `.tmp/incoming/BLK-019_Ashank_Handoff`. Both declared SHA-256 hashes passed; all 2,500 supplied IDs are frozen-train members; 500,000 pairs are unique and reference valid records; every group has K=200; zero-candidate count is 0. Training recall is 8,429/8,618 = `0.9780691576`, rounding to metadata `0.97807`; 189 true pairs are blocking misses.
- `BLK-019-TRAIN-PREFLIGHT-v1` completed locally with the supplied Word-Unigram artifact: 43-feature L2 Logistic (`C=1.0`, train-only scaling, no calibration), 8,429 positive / 491,571 negative pairs, 305.77 s total, 1.275 GiB peak working set, 1.651 GiB peak private commit, and 223.6 MiB generated run artifacts. The exact three-column training score file has 500,000 finite probabilities in `[1.42e-09, 0.999999997]`. No validation F0.5 or threshold was produced.
- Next required artifact: same-version BLK-019 Word-Unigram candidates for every frozen validation Source-1 ID, with ID manifest, zero-candidate accounting, file/generator hashes, pair-count distribution, and candidate recall. Use it for the first official frozen evaluation only after calculating measured local time/disk from its actual distribution.

## [2026-09-26] BLK-019 Full-Scale Execution Preparation
- Preserved the verified BLK-019 inputs, fitted model, train-only TF-IDF artifact, 43-feature schema, and preflight report. A fresh-process 12-pair software fixture loaded the saved feature/model artifacts, emitted the exact three-column score contract, and verified finite `[0,1]` probabilities. It has no labels, threshold, or evaluation result.
- The preflight measured 6,427.3 transform/predict/write pairs/s (77.79 s for 500,000), with one-time TF-IDF fit 133.23 s, feature materialisation 82.58 s, Logistic fit 1.28 s, and unseparated spool/setup residual 10.89 s. Provisional 25%-contingency sequential budgets are 5.41 h / 20.28 GiB transient for roughly 88M validation pairs and 21.36 h / 79.96 GiB for roughly 347M test pairs; recompute them from the verified artifacts.
- `docs/BLK019_EXECUTION_PROPOSAL.md` records local-first staging, removable/reproducible intermediates, test prerequisites, and an unapproved AWS contingency. No AWS resource, credentials, upload, credit, paid service, or cloud data transfer was used. Before any cloud action, obtain Mudit's existing region, private storage identity, IAM role, and data-owner approval to avoid duplicate candidate-generation infrastructure.

## [2026-09-26] BLK-020 Matcher Preflight and Streaming Readiness
- BLK-020 is now active: Word-Unigram name+address, country partitioning, K=200, generator `69a91f1`, configuration hash `657f59868e58`. Its scoped archive members are Git-excluded. Sample and validation-gzip hashes match `SHA256SUMS`.
- `BLK-020-TRAIN-PREFLIGHT-v1` completed locally: 2,500 frozen-train entities, 500,000 unique K=200 pairs, 8,429 positive / 491,571 negative pairs, and training candidate recall `0.9780691576` (8,429/8,618). The 43-feature train-only TF-IDF/L2 Logistic run took 963.64 s with 1.274 GiB peak working set; it produced 500,000 finite exact-schema probabilities. No threshold or validation metric exists.
- `BLK-020-VAL-PERF-50K-v2` selected the first 250 complete frozen-validation groups from the compressed source in row order, without labels/evaluation. It scored 50,000 pairs at 1,406.6 score-stage pairs/s (1,222.7 end-to-end) with 245.6 MiB peak working set. Gzip input plus explicit rank cutoff support is test-covered; full validation remains unstarted.
- Future requirements recorded in `docs/TEAM_HANDOFF.md`: retain rule baseline; K=50/K=100/K=200 sweeps; slice processing below about 6 GiB; no country one-hot/hard-coded country rule; ground-truth verification before any uniqueness assignment; await Aayush's documented competition features.
