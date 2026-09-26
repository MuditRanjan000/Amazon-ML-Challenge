# Session Changelog

## [2026-09-25] Project Initialization & Architecture Enforcement
- **Architectural Changes:** 
  - Restructured project to follow the strict 3-layer architecture defined in `AGENTS.md`.
  - Created `directives/` for markdown SOPs (Layer 1).
  - Created `execution/` for deterministic python tools (Layer 3).
  - Created `.tmp/` for scratch files and intermediate outputs (e.g., exploratory analysis scripts).
  - Initialized `.env` and this `AI_ONBOARDING.md` log.
- **Data Movement:** Moved the initial exploratory script (`analyze_datasets.py`) into `.tmp/` as it was a scratch task to understand the data.
- **Next Steps:** Begin converting the competition strategy (from `AI_CONTEXT.md` / `competition_analysis_report.md`) into formalized Markdown SOPs within `directives/` and write the corresponding deterministic tools in `execution/` starting with the data validation and baseline generation.

## [2026-09-25] Experiment 1: Baseline Matching
- **Implementation:** Created `directives/baseline_matching.md` (Layer 1) and `execution/run_baseline.py` (Layer 3).
- **Strategy:** Exact string match on normalized `business_name`, partitioned by country.
- **Results:** Achieved a local F0.5 score of 0.19372. This confirms that simple exact matching is insufficient due to typos, transliteration, and missing data.
- **Next Steps:** Implement Experiment 2 (Blocking) using TF-IDF character n-grams to drastically improve the recall ceiling before training an ML model.

## [2026-09-25] Ashank Matching Workstream Reconciliation
- **Checkout:** Created local-only `feature/ashank-model` from `origin/main` commit `e8f2517833414444a1509f25322404a609eb31df`; no push, merge, submission, or deployment.
- **Portable data setup:** Added ignored `AMAZON_ML_DATA_DIR` configuration with a tracked `.env.example`; updated existing execution scripts to use it instead of a developer-specific absolute path.
- **Repository hygiene:** Narrowed ignore rules so raw data remains excluded while conventional tests and CSV experiment metadata are no longer hidden.
- **Handoff:** Documented matching interfaces, reusable modules, and missing team artifacts in `docs/ashank_matching.md`.

## [2026-09-25] Ashank Pairwise Feature Infrastructure
- **Implementation:** Added matching contracts, DataFrame/SQLite candidate-record adapters, Unicode-preserving lexical features, deterministic raw baseline scoring, and one-to-many threshold decisions under `src/entity_resolution/matching/`.
- **Execution:** Added `execution/run_matcher_baseline.py` for local record-store creation, train-only feature fitting, batch feature extraction, raw scoring, and decision aggregation.
- **Verification:** Fixture tests cover missing fields, Unicode/Indian script, accents, abbreviations/reordered tokens, conflicting numeric tokens, invalid/duplicate IDs, batch consistency, fitted-artifact persistence, multiple matches, and no-candidate entities.
- **Boundary:** No candidate artifact, canonical split, training, calibration, threshold tuning, final inference, commit, push, merge, or deployment was performed.

## [2026-09-25] Ashank Matcher Performance Investigation
- **Profile:** Replaced the tracing-distorted synthetic 126-pairs/s result with a reproducible 2,000-pair bounded real-record diagnostic that repeats each Source 1 across ten candidates.
- **Optimisation:** Added batch-bounded record representations, sparse unique-string TF-IDF transforms, and pinned MIT `rapidfuzz==3.14.3` compiled Levenshtein/Jaro-Winkler with Python fallback equivalence checks.
- **Result:** Warm feature throughput improved from 607 to 1,124 pairs/s; feature plus TSV serialization improved from 588 to 1,068 pairs/s. TF-IDF remains the main bottleneck; no feature/model-quality claims were made.
- **Reproduction:** `execution/profile_matcher.py` emits a JSON report under ignored `.tmp/`; full-scale runs remain deferred pending Aayush/Mudit artifacts.

## [2026-09-26] Ashank D2 Scalable Matcher Preparation
- **Implementation:** Added disk-backed candidate spooling, complete Source-1 group batches, resumable/checksummed Logistic score parts, separate heuristic raw scores, and complete-universe threshold decisions in `models.streaming`; added `execution/run_streaming_matcher.py`.
- **Verification:** Existing EXP-003A 1,000-entity diagnostic reproduced candidate IDs, all 43 feature semantics, threshold decisions, and shared-evaluator metrics; max probability delta was `1.11e-16`.
- **Boundary:** D2 must be supplied by Mudit/Aayush for both frozen partitions using the exact same generator/config. No D2 candidate generation, frozen-split/evaluator modification, full benchmark, cloud use, commit, push, or deployment was performed.

## [2026-09-26] D2 First-Fit Handoff Refinement
- **Training scope:** A reproducible subset of complete frozen-train D2 candidate groups is authorised for the first Logistic fit when it uses the exact validation D2 generator/configuration. Record the frozen-ID selection method, entity/pair/zero-candidate counts, and positive/negative class balance; label the model subset-trained.
- **Run gate:** Wait for Mudit's full-validation D2 pair count and distribution, then derive local time/disk estimates from a D2-trained bounded preflight before starting full validation. Full validation retains every frozen Source-1 entity, including zero-candidate entities.

## [2026-09-26] Stratified D2 Preflight Selection
- **Implementation:** Replaced train CSV-prefix selection in the bounded runner with `stable_sha256_stratified_country_match_count_v1`. It scans frozen training entities across country and training-label match-count strata (0/1/2+), uses seed `20260926` by default, preserves complete candidate groups downstream, and writes `training_selection.json` with the procedure, distributions, selected-ID hash, group counts, and final class balance.
- **Boundary:** This selects a reproducible first-fit subset only. It does not modify Mudit's frozen validation split/evaluator or Aayush's D2 generator, and its model is explicitly subset-trained.
- **Population check:** Seed `20260926` produced a 1,000-entity sample with 57 singleton, 56 one-match, and 887 multi-match training entities across the full frozen-train country/multiplicity strata. No D2 candidate or model run was performed.

## [2026-09-26] BLK-019 Intake Blocker
- **Supersession:** The final blocker is BLK-019 Word Unigram (name + address, country partitioning, K=200, commit `187d955`, config `657f59868e58`); prior char-ngram/D2 artifacts are not eligible for a reported matcher fit.
- **State:** The reported 500,000-pair training preflight TSV is absent locally and from fetched origin refs, so no fitting occurred. Await the physical file, the exact 2,500-ID manifest needed for zero-candidate verification, and matching BLK-019 full-validation candidates.
- **Readiness:** Added a training-only disk-backed preflight runner for the 500,000-pair BLK-019 sample; it preserves all pairs, uses memory-mapped features/labels, and cannot evaluate or tune on validation.
- **Completed:** Verified and ran `BLK-019-TRAIN-PREFLIGHT-v1`: 2,500 frozen-train IDs, 500,000 unique K=200 pairs, Recall@200 `0.9780691576` (rounded metadata match), and 43-feature L2 Logistic fit. Output is training-only; no validation threshold/F0.5. Await matching full frozen-validation BLK-019 candidates.

## [2026-09-26] Ashank BLK-020 Local Readiness
- BLK-020 Word-Unigram (generator `69a91f1`, config `657f59868e58`, K=200) supersedes BLK-019 for the active matcher. Its large artifact remains Git-excluded.
- The 500,000-pair training-only preflight passed with the supplied exact 2,500-ID manifest: 43 features, L2 Logistic, 8,429/491,571 positive/negative pairs, candidate recall `0.9780691576`, and exact probability output. No validation threshold or F0.5 was produced.
- Candidate spooling now accepts gzip input and an explicit rank cutoff, with fixture coverage. The 50,000-pair validation performance diagnostic preserved complete groups and measured 1,406.6 score-stage pairs/s / 245.6 MiB peak working set. Full validation and AWS remain unstarted.
