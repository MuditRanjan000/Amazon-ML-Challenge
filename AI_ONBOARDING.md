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

## [2026-09-25] Agent instruction files + blocking directive (Aayush, `feature/aayush-agent-docs`)
- **AGENTS.md fixed:** Rule 5 ("push to main + `scripts/redeploy.py tradebot`") was copied from an unrelated project. It is replaced with the branch/PR policy from `docs/github_automation_policy.md`: never push to main, Mudit merges. The RTK and graphify/llm-council rules are now conditional on those tools being installed (`execution/bin/rtk.exe` isn't in the repo). Added project sections: challenge facts, ownership, data contracts, frozen-split SHA check, experiment protocol, engineering rules, and end-of-task duties.
- **Mirrors:** `CLAUDE.md` and `GEMINI.md` are now byte-identical copies of AGENTS.md, maintained by `execution/sync_agent_docs.py` (the newest edit wins; overwritten versions go to `.tmp/agent_docs_backup/`; `--check` for CI; `--selftest`).
- **Personal layers:** `CLAUDE.local.md` / `GEMINI.local.md` (gitignored) hold per-operator role and state. The shared docs tell agents to read them if present, so teammates are unaffected.
- **New directive:** `directives/blocking.md` proposes the Blocking → Matching parquet contract, the oracle-ceiling F0.5 metric, and hand-off gates G1–G5. **Needs sign-off from Ashank and Mudit.**
- **Found (not fixed; owner decision needed):**
  - `tfidf_blocking.py` still builds dense 500 × N_country similarity chunks. That's about 20 GB for India, under all-thread parallelism, so it will run out of memory at full scale.
  - Queries with unseen countries are skipped.
  - `.gitignore` blocks `*.csv` (the registry can't be committed) and `test_*.py`.
  - `execution/*.py` hardcode `c:\Users\dell\...`.
  - `requirements.txt` is empty.

## [2026-09-25] Pipeline refactor: modular, installable, resource-bounded (Aayush, `feature/aayush-pipeline-refactor`)
- **Packaging:** `pyproject.toml` (src layout, `pip install -e .`, import `entity_resolution.*`), a pinned `requirements.txt` (pandas 3 / pyarrow / sklearn / sparse_dot_topn, all permissive licences), `.env.example`, `Dockerfile` (not yet built: Docker daemon was off), and a `tests/` suite (25 tests).
- **config.py:** every path comes from env vars / `.env`. All hardcoded `c:\Users\dell\...` paths are gone.
- **Loader:** a pyarrow raw-text parser (no quote processing, no NA coercion, CRLF safe) cached as parquet. The old default-pandas path rewrote about 800 test fields containing `"`.
- **Normalizer (shared with matching):** removes punctuation by Unicode category and strips accents from Latin letters only. **Bug fixed:** the old `[^\w\s]` regex shredded every Devanagari name.
- **Validation split:** fixed a crash on pandas 3, uses repo-root paths, and verifies the SHA-256 against the manifest on every load. Regenerating reproduces Mudit's frozen files byte for byte.
- **Evaluator:** vectorized, same API and keys, plus `per_entity_scores()` for error analysis. Tested against the original loop implementation.
- **Blocking:**
  - `TfidfBlocker` rewritten with hashed TF-IDF and `sparse_dot_topn` (memory-bounded top-k), open-set countries, `max_df`, forward and `reverse_candidates`.
  - `union_candidates()` added.
  - `BlockingEvaluator` now reports ceiling F0.5, coverage and buckets on integer codes.
- **Submission:** vectorized writers for both official files. The validator wraps the official `utils/validate_submission.py`.
- **Tracking:** `tracking.log_run()` writes to `experiments/results/experiments.jsonl` (commit, config, metrics, runtime, peak RSS).
- **Removed:** placeholder blockers, unused metric helpers, `run_tfidf_blocking.py` (superseded by the parameterized `run_blocking_eval.py`).
- **Verified on real data:**
  - The baseline reproduces EXP-001 (0.1937173).
  - The baseline test submission passes the official validator, including `--check-ids`.
  - Forward blocking against the full 10.3M index peaks at 5.9–7.4 GB on the laptop.
- **Key findings:**
  - Many-to-one GT: no S2/S3 ID belongs to more than one S1.
  - Name+address word TF-IDF raises R@50 from 0.706 to 0.958 and the ceiling F0.5 from 0.829 to 0.985.
  - Sparse top-k is memory-bandwidth bound: no gain beyond about 4 threads.
  - Details in `directives/blocking.md`.

## [2026-09-26] Hybrid blocking v1 beats the D2 baseline (Aayush, `experiment/aayush-blocking-hybrid`)
- **Target (Mudit):** beat D2 = char_wb (3,5) name+address TF-IDF, R@200 96.29% (measured on 1,000 val queries, `feature/mudit-submission`).
- **Result:** word TF-IDF name+address (max_df 0.02) top-100 ∪ new **name_key** channel top-25 → **R 0.9740, ceiling F0.5 0.9906, 120 candidates/S1** on 100k val S1 (BLK-017; 0.9739 on 20k, BLK-016). Beats 96.29% by about 1.1pp with 40% fewer candidates. Forward channel runs at about 340 q/s.
- **name_key** (`blocking/keys.py`): anyascii (ISC) transliteration → normalize with legal suffixes stripped → drop web tokens and spaces → char 3-grams. It targets the measured misses: cross-script Indic names (~40%), typo'd names with an empty address (~21%), and domain-style names.
- **Measured and rejected:** score-ratio/absolute adaptive cutoffs (no better than fixed K); address canonicalization (+0.00pp in the union); vowel skeleton (−0.05pp, kept as a speed flag). Reverse channel skipped: the 6+ bucket is not under-recalled.
- **Method notes:** a 1k-query sample has a ±0.6pp 95% band on R@200, too noisy to rank configs; compare on `--sample 20000`. `run_blocking_eval.py`'s evaluator gives R@K identical to `origin/main`'s `BlockingEvaluator` on the same candidates (checked).
- **Found on `feature/mudit-submission` (owner: Mudit):** `data/validation_split.py` has an unterminated docstring (SyntaxError on import). The committed split CSVs are stored with LF, so their SHA-256 matches the manifest only on Windows checkouts (autocrlf) and will fail on Linux/AWS. The committed `tfidf_blocking.py` still does dense `toarray()` and skips unseen countries (France). The report says "35,000 queries = full val", but val has 441,365 S1.
- New dependency: `anyascii==0.3.3` (ISC).
