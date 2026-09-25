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
| `execution/profile_blocking.py` | to build: GT structure (many-to-one?), country agreement, field survival |
| `execution/run_blocking.py` | to build: channels → union → parquet + sidecar |
| `execution/check_candidates.py` | to build: gates G1–G5 |
| `execution/run_blocking_eval.py`, `run_tfidf_blocking.py` | legacy: hardcoded paths, dense similarity (OOM at full scale) |

## Edge cases & learnings
- Dense `(queries × index)` similarity is not viable: 500 × 5M float64 ≈ 20 GB per chunk. Use sparse top-k or ANN.
- Query countries missing from the index must still get candidates. Never skip them.
- Empty/null names: fall back to address channels. Don't drop the S1 row.
- Library licences: MIT/Apache/BSD/ISC only (no GPL, e.g. `unidecode`); everything must run offline.
