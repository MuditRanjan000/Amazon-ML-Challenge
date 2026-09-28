# Blocking Report — BLK-020 (final, frozen)

**Status:** DONE (2026-09-26). Train + validation candidates generated with the frozen blocker.

## Frozen blocker (Mudit's decision after BLK-019)
- `scripts/aws/run_d2_blocking.py --blocker word` (the default); config_sha **657f59868e58**.
- Country-partitioned TF-IDF, `business_name + business_address`, **word unigrams**, min_df 2, exact per-country IDF, top **K=200**.
- Generator commit **69a91f1**. Run as 48 contiguous parts (`--part i/48`) on AWS, then `--merge 48`. Merged output is byte-identical to a single run (tested).

## Why word unigram (BLK-019: same 20k val S1, same harness, K=200)
| Blocker | R@10 | R@50 | R@200 | Ceiling F0.5@200 | S1/s |
|---|---|---|---|---|---|
| D2 char_wb (3,5) | 0.902 | 0.942 | 0.959 | 0.984 | 9.6 |
| **word unigram** | **0.925** | **0.963** | **0.977** | **0.992** | **99.8** |
| word + max_df 0.02 | 0.920 | 0.959 | 0.974 | 0.990 | 446 |

## BLK-020 results (full frozen split)
| | train | validation |
|---|---|---|
| S1 | 1,765,456 | 441,365 |
| Candidate pairs | 353,091,200 | 88,273,000 |
| Avg / p50 / p99 candidates per S1 | 200 / 200 / 200 | 200 / 200 / 200 |
| S1 with no candidates | 0 | 0 |
| Recall@10 / @50 / @100 / @200 | 0.9250 / 0.9630 / 0.9713 / 0.9780 | 0.9248 / 0.9628 / 0.9712 / 0.9779 |
| Oracle-ceiling F0.5@200 | 0.9923 | 0.9921 |
| Full coverage@200 | 0.9340 | 0.9340 |
| TSV sha256 | `e3bd1d76…4330e2` | `906fc126…dddf87c` |

Train and validation agree to within 0.001, so there's no sign of split leakage or drift. Full hashes, per-country breakdown and timings are in `blocking_metadata.json`.

**2,500-S1 train sample** (seed 42, for quick preflight checks): 500,000 pairs, R@200 **0.97807**, ceiling 0.99318, all IDs in frozen train, 0 duplicates.
`train_sample_candidate_pairs.tsv` sha256 `7a2232bd…b2d8f108`; `train_sample_s1_ids.csv` (includes `n_candidates` per S1) sha256 `77583451…a9972b24`.

## Where the files are
Private S3 bucket `amazon-ml-2026-blocking-716522590518`, prefix `run-69a91f1/final/`:
`train_candidate_pairs.tsv` (13.1 GB) + `.gz` (4.8 GB), `validation_candidate_pairs.tsv` (3.3 GB) + `.gz` (1.2 GB), the sample, the ID manifest, and `blocking_metadata.json`. TSV schema: `source1_entity_id, candidate_entity_id, score, rank` (tab-separated, unquoted).

## Notes for the matcher
- K=200 gives 353M training pairs. `rank` lets you train on `rank <= K` or downsample negatives without regenerating. Recall is already 0.963 at K=50.
- `candidate_pairs.tsv` size is graded. Pick the final K on validation via `rank <= K` and apply the same K to test.
- Test candidates are not generated yet. They need Mudit's ruling on fitting TF-IDF IDF on test S2+S3 (unsupervised, provided data). Then run `--split test --blocker word` with the same config.
