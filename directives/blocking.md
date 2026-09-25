# Directive: Candidate Generation / Blocking

**Owner:** Aayush • **Consumers:** Ashank (matching), Mudit (integration, `candidate_pairs.tsv`)
**Status:** v0: contract proposed, pending sign-off. No blocking config is frozen yet.

## Goal
For every Source-1 entity, produce a candidate set of S2/S3 records that contains its true matches (high **oracle-ceiling F0.5**) while staying small enough for the matcher to score, and for the matcher to avoid false merges.

## Inputs
- `$ER_DATA_DIR/{train,test}/{split}_source{1,2,3}.tsv` and `$ER_DATA_DIR/train/train_ground_truth.tsv`
- Frozen split: `output/validation_split/{train,val}_ids.csv`. **Verify the SHA-256 against `experiments/results/validation_split_manifest.json` before every run.**

## Output contract (Blocking → Matching)
Parquet (never committed; stored under `output/candidates/` locally and in S3 for AWS runs), one file per split: `candidates_{train,val,test}_<config_id>.parquet`. Every split is produced with the **identical config**.

| Column | Type | Meaning |
|---|---|---|
| `source1_entity_id` | string | S1 query |
| `candidate_entity_id` | string | S2/S3 record |
| `candidate_source` | int8 | 2 or 3 |
| `channels` | int16 | bitmask of the channels that proposed the pair (C1=1, C2=2, C3=4, …) |
| `fwd_rank`, `fwd_score` | int16 / float32 | rank and score in S1→S2/S3 retrieval (null if not retrieved this way) |
| `rev_rank`, `rev_score` | int16 / float32 | rank and score in S2/S3→S1 retrieval (null if not retrieved this way) |
| `name_tfidf_cos` | float32 | name char-ngram cosine, filled for every pair |

A sidecar `<config_id>.json` holds the full config, git commit, runtime, peak RSS, row count, and content hash.
Mudit derives the official `candidate_pairs.tsv` from the test parquet. That TSV must equal the set the final model scored.

## Metrics (reported for the val split, every run)
1. **Oracle-ceiling F0.5**: if |T|>0, R=|T∩C|/|T| and F0.5=1.25R/(0.25+R); if |T|=0, 1.0. Mean over all val S1.
2. Micro pair recall and full-coverage rate at K = 10/25/50/100/200.
3. Breakdown by match bucket (0/1/2–5/6+), country, source (S2/S3), and script (Latin/Indic).
4. Candidates per S1: mean, p50, p95, p99, max. Total pairs, runtime, peak RSS.
Choose the **knee** of the ceiling-vs-pairs curve, not the maximum K.

## Hand-off gates (all must pass; `execution/check_candidates.py`)
- **G1 schema:** dtypes, no nulls in ID columns, no self-pairs, no duplicate `(s1, cand)` pairs, IDs belong to the correct split.
- **G2 reproducibility:** rerunning the same config and commit gives an identical content hash.
- **G3 metric:** val ceiling F0.5 and recall ≥ the registered values for this config.
- **G4 budget:** total pairs ≤ the budget agreed with Ashank; p99 per S1 within limits.
- **G5 test sanity:** test per-country candidate-count and score distributions resemble val. No country has zero candidates (France exists only in test).

## Scripts
| Script | Status |
|---|---|
| `execution/run_blocking_eval.py` | **ready**: any method, direction and config on the frozen val split (`--sample N` for dev). Logs to `experiments/results/experiments.jsonl`; `--save` writes the candidates parquet. |
| `execution/profile_blocking.py` | to build: country agreement, field survival (GT structure already measured, see below) |
| `execution/run_blocking.py` | to build: frozen config → train/val/test parquet + sidecar |
| `execution/check_candidates.py` | to build: gates G1–G5 |

Library: `entity_resolution.blocking` provides `TfidfBlocker` (fit/query, forward), `reverse_candidates`, `ExactNameBlocker`, `union_candidates`, and `BlockingEvaluator` (recall@K, ceiling F0.5, coverage, buckets).

## Measured (2026-09-25, 3k val S1 against the full 10.3M train S2+S3 index, laptop 15.6 GB / 20 threads)
| Config | q/s | R@10 | R@50 | R@200 | ceiling@50 | ceiling@200 |
|---|---|---|---|---|---|---|
| char_wb 3-gram, name | 41 | 0.580 | 0.706 | 0.778 | 0.829 | 0.881 |
| char_wb 4-gram, name | 77 | 0.565 | 0.683 | 0.750 | 0.822 | 0.868 |
| char_wb 5-gram, name | 112 | 0.552 | 0.658 | 0.724 | 0.811 | 0.858 |
| word unigram, name | 202 | 0.558 | 0.662 | 0.731 | 0.812 | 0.865 |
| char_wb 4-gram, name+address | 28 | 0.905 | 0.942 | 0.958 | 0.976 | 0.983 |
| **word unigram, name+address (default)** | **92** | **0.915** | **0.958** | **0.973** | **0.985** | **0.991** |
| ↳ + `--max-df 0.05` | 155 | 0.912 | 0.956 | 0.973 | 0.984 | 0.991 |
| ↳ + `--max-df 0.02` (recommended for dev loops) | 334 | 0.909 | 0.956 | 0.972 | 0.984 | 0.991 |
| ↳ + `--max-df 0.01` | 572 | 0.907 | 0.954 | 0.971 | 0.984 | 0.990 |

Word-level `max_df` is nearly free (common words like road/street/pvt carry no identity), unlike char n-gram pruning. **Decide the final max_df on the full val split before freezing.**

## Edge cases & learnings
- **GT structure (train):** 7,638,365 true pairs. **Every S2/S3 ID belongs to at most one S1** (many-to-one), so reverse retrieval and a one-S1-per-record constraint in the matcher are both valid. 74% of S2+S3 records match some S1; 26% are distractors. Matches per S1: 0 → 123,247 | 1 → 119,157 | 2–5 → 1.71M | 6–11 → 252k.
- **The address is the biggest recall lever.** Name-only retrieval tops out around R@200 0.78; adding the address reaches R@50 0.96.
- **Char trigrams on names are a trap at this scale:** each India query touches ~2.1M of 4.1M index rows. `max_df` pruning of char n-grams destroys recall (0.02 → R@200 −4 pts; 0.005 → −22 pts). Short names need their common trigrams.
- **Sparse top-k is memory-bandwidth bound:** `sp_matmul_topn` gives 18 q/s on 1 thread, 42 on 4, and 46 on 19. Buy memory bandwidth, not cores. 1.73M test S1 at about 90 q/s is about 5.5 h on the laptop, so run full scale on AWS or in parallel processes per country.
- The loader must read raw text: default pandas quoting rewrites about 800 test fields that contain `"`.
- The normalizer must not use `[^\w\s]`: it shreds Devanagari (vowel signs are not `\w`).
- Dense `(queries × index)` similarity is not viable: 500 × 5M float64 ≈ 20 GB per chunk. Use sparse top-k or ANN.
- Query countries missing from the index must still get candidates. Never skip them.
- Empty/null names: fall back to address channels. Don't drop the S1 row.
- Library licences: MIT/Apache/BSD/ISC only (no GPL, e.g. `unidecode`); everything must run offline.
