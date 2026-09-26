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

## [2026-09-26] D2 generator fixed + faithfulness check (Aayush, `experiment/aayush-blocking-hybrid-v2`, cut from `feature/mudit-submission` @426555c)
- **Why:** Mudit asked for D2 train candidates from the same generator as validation. The D2 script on `feature/mudit-submission` could not produce D2:
  - it ran **word** (3,5)-grams, because `analyzer` was never passed and the class default was `word`;
  - it would OOM on train (one Python dict per pair, 353M pairs);
  - it crashed on Linux at the split-manifest SHA check (LF vs CRLF).
- **`TfidfBlocker` rewritten** (same public API):
  - exact per-country IDF (sklearn TfidfVectorizer formula), with `min_df`/`max_df` masks that actually apply;
  - index row shards held in RAM or on disk (`cache_dir`, keyed by config + data hash);
  - vectorized top-k merge across shards.
  - Removed test-gaming mocks. 26/26 tests pass, including a new disk-shard == in-memory test.
- **`scripts/aws/run_d2_blocking.py` rewritten:**
  - pinned D2 config (char_wb (3,5), name+address, country partition, min_df 2, K=200);
  - `--split trainval` writes train + validation TSVs from one index/run; `--split test`; `--sample N` check;
  - streamed pyarrow TSV writing;
  - per-split metadata: config_sha, commit, tsv_sha256, pair distribution, recall and ceiling F0.5;
  - refuses to run from a dirty tree.
- `.gitattributes` keeps the split CSVs CRLF on every OS. Removed the dead `execution/run_tfidf_blocking.py` (hardcoded `dell` paths).
- **Result (BLK-018, 5k val):** R@10 0.900 / R@50 0.940 / R@200 0.957, ceiling F0.5@200 0.984 (original D2: 0.963 on 1k queries, ±0.6pp).
  - **Throughput is about 10 S1/s on the laptop**, so full train+val is about 60 h and test about 48 h locally. Full runs need AWS fan-out or a decision (see AI_CONTEXT).
- **Comparison, same val split and evaluator (earlier runs):** word-unigram name+address reached R@200 0.972 at 92–334 q/s (BLK-011). The hybrid reached R 0.974 at 120 cands/S1 (BLK-017).

## [2026-09-26] BLK-019: fair D2 vs word-unigram comparison (Aayush)
- PR #5 (the D2 fix + `--blocker` switch) was merged by Mudit into `feature/mudit-submission` (af44b7c). Before merging, I merged his 4a813f7 into the branch; the conflict was resolved by keeping the rewrite, which already has the same imports.
- **Setup:** same 20k frozen-val S1, same harness, evaluator, pool, fields, normalizer, min_df 2 and K=200.
- **Results:**
  - D2 char_wb (3,5): R@10/50/200 0.902/0.942/0.959, ceiling 0.984, 9.6 S1/s.
  - Word unigram: 0.925/0.963/0.977, ceiling 0.992, 99.8 S1/s.
  - Word + max_df 0.02: 0.920/0.959/0.974, ceiling 0.990, 446 S1/s.
- **Implication:** word beats D2 at every K, and word@50 already exceeds D2@200. Projected train+val runtime on the laptop: D2 ~64 h vs word ~6 h vs word+max_df ~1.4 h.

## [2026-09-26] BLK-020: frozen word-unigram blocker, full train + validation candidates on AWS (Aayush)
- **Decision (Mudit):** word unigram is the final blocker (BLK-019). `--blocker word` is now the default; config_sha `657f59868e58`.
- **Code (`0abe61a`, `69a91f1`):**
  - `--part i/n` + `--merge n` in `scripts/aws/run_d2_blocking.py`. Contiguous slices make the merge byte-identical to a single run (tested in `tests/test_blocking_parts.py`).
  - The merge also writes a seed-42 2,500-S1 train sample + ID manifest.
  - `ER_GIT_COMMIT` override in `tracking.git_commit()` for tarball-shipped code.
  - S1 is loaded with only the 4 columns blocking reads.
  - `scripts/aws/blocking_instance.sh` is the EC2 user-data. `.gitattributes` pins `*.sh` to LF.
- **Run:** 48 × m7i-flex.large (32 on-demand + 16 spot) in ap-south-1 plus 1 merge instance, ~25 min wall, ≈ $1.5 of free-plan credits. Everything self-terminated; nothing left running.
- **Results:**
  - train: 1,765,456 S1 / 353,091,200 pairs / R@200 0.97795 / ceiling 0.99225.
  - validation: 441,365 S1 / 88,273,000 pairs / R@200 0.9779 / ceiling 0.99213.
  - 0 S1 without candidates.
  - 2,500-S1 sample: R@200 0.97807, which matches the figure Ashank was given.
- **Artifacts:** `s3://amazon-ml-2026-blocking-716522590518/run-69a91f1/final/` (TSV + .gz, sample, manifest, metadata). Hashes in `artifacts/blocking/blocking_metadata.json`.
- **Gotchas:**
  - The free plan only allows free-tier instance types (2 vCPU / ≤ 8 GB), so the job needs swap.
  - The first launch OOMed because `mkswap -q` is invalid on AL2023.
  - Local commit `187d955` referenced by Ashank was never pushed. The BLK-020 sample reproduces its reported R@200 exactly.
- **Next:** test candidates (`--split test --blocker word`), pending Mudit's ruling on fitting IDF on test data. Choose the final K on validation via `rank <= K`.

## [2026-09-26] BLK-020 Handoff & Pipeline Verification (Mudit, `feature/mudit-submission`)
- **Artifact Validation:** Extracted Ayush's final `BLK-020` artifacts (`train_candidate_pairs.tsv.gz` 5.16GB, `validation_candidate_pairs.tsv.gz` 1.29GB).
- **OOM Prevention:** Rewrote `scripts/verify_blk020_handoff.py` from a Pandas loader into an aggressive low-memory streaming generator. 
- **Verification Results:** Successfully parsed 441M+ rows. 
  - Train: 353,091,200 rows, 1,765,456 unique S1, 0 duplicates.
  - Validation: 88,273,000 rows, 441,365 unique S1 (Matches frozen validation manifest exactly!), 0 duplicates.
- **Merge & Sync:** Merged Ayush's PR (#7) `experiment/aayush-blocking-hybrid-v2` into `feature/mudit-submission`, resolving metadata lineage conflicts to lock in `word` blocker (`657f59868e58`). Tests passed. Code is clean and handoff is fully prepared for Ashank.
