# Amazon ML Challenge 2026 — Business Entity Resolution

For every Source-1 (reference) business, find all Source-2/3 records describing the same real-world entity.
The metric is the macro F0.5 over S1 entities (precision-heavy; singletons count). The full rules are in `6ab10eb3b23ba_student_resource/student_resource/README.md`.

## Setup (≈2 min)
```bash
python -m venv .venv
.venv\Scripts\activate            # Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt   # exact pinned versions
pip install -e .                  # installs the `entity_resolution` package from src/
python -m pytest -q               # 25 tests, ~3 s
```
Data: unzip the challenge resource so that `6ab10eb3b23ba_student_resource/student_resource/dataset/{train,test}` exists at the repo root. Alternatively, point `ER_DATA_DIR` somewhere else (see `.env.example`). No path is hardcoded anywhere.

## Commands
| Task | Command |
|---|---|
| Baseline, validation split | `python execution/run_baseline.py --split val` |
| Baseline test submission + official validation | `python execution/run_baseline.py --split test` |
| Benchmark blocking (fast dev loop) | `python execution/run_blocking_eval.py --exp-id BLK-0xx --sample 20000 --analyzer word --fields business_name business_address` |
| Validate output files | `python scripts/validate_submission.py [--check-ids]` |
| Keep agent docs mirrored | `python execution/sync_agent_docs.py [--check]` |

Every run appends one JSON line to `experiments/results/experiments.jsonl`. The line records the git commit, config, metrics, runtime and peak RAM.

## Layout
```
src/entity_resolution/
  config.py              all paths/settings (env-overridable)
  tracking.py            experiment log (JSONL), peak RSS, git commit
  data/loader.py         exact TSV parsing -> parquet cache; explode_id_lists()
  data/preprocessing.py  shared normalizer (Unicode-safe; blocking AND matching use it)
  data/validation_split.py  frozen split, SHA-256 verified against the committed manifest
  blocking/              exact-name + TF-IDF (forward & reverse) blockers, union, blocking evaluator
  evaluation/evaluator.py   official macro F0.5 (+ per-entity scores for error analysis)
  submission/            official-format writers + wrapper around the organisers' validator
execution/   thin CLI entry points (Layer 3)      directives/  SOPs (Layer 1)
tests/       pytest suite                          experiments/results/  manifest + run log
```

## Resource behaviour (measured on a 15.6 GB / 20-thread laptop)
- **Loading:** the TSVs are parsed once with pyarrow and cached as parquet. The full train baseline (2.2M S1 against 10.3M S2+S3) runs in about 70 s with a 5.3 GB peak.
- **Blocking:** hashed TF-IDF plus `sparse_dot_topn` keeps only the top-k per query, so memory is bounded (the full 10.3M index peaks at about 6–11 GB). Candidate IDs are int32 categoricals.
- **Speed limit:** exhaustive sparse retrieval is bound by memory bandwidth and gains little beyond 4 threads. Throughput is config-dependent (see `directives/blocking.md`). Full test-scale runs belong on AWS (`docs/aws_strategy.md`, `Dockerfile`).

## Data-handling rules baked into the code
- TSVs are read as raw text: no quote processing, no `NA`→NaN, and CRLF is stripped. Default pandas would silently alter about 800 test fields that contain `"`.
- Text normalization removes punctuation by Unicode category. The old `[^\w\s]` regex shredded every Devanagari name.
- `country` is an open set. Test adds France, and unseen countries are searched against every partition.

## Team workflow
See `AGENTS.md` (mirrored to `CLAUDE.md` / `GEMINI.md`) and `docs/github_automation_policy.md`. In short: work on your own branch, open a PR, and Mudit merges. Owners: Mudit (validation, integration, submission), Aayush (blocking), Ashank (matching).
