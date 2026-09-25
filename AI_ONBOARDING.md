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
