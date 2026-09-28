# Methodology Document

## Methodology
*(To be populated prior to final submission)*

## Candidate Generation / Blocking Strategy
*Owner: Aayush. Maps to section 3 of `Documentation_template.md`. All numbers were measured on the frozen validation split (441,365 S1, seed 42, hash-verified) with the repo's evaluators.*

### Problem framing
Blocking cannot raise the score; it only sets the **ceiling**. A true pair that blocking drops can never be matched. We therefore judged every blocking variant by the **oracle-ceiling F0.5**: the macro F0.5 a perfect matcher would reach using only our candidates. For an S1 with true set T and candidate set C, F0.5 = 1.25R/(0.25+R) with R = |T∩C|/|T|; singletons score 1.0. We read the ceiling alongside pair recall, full coverage (the share of S1 with T ⊆ C), and candidates per S1.

### Final method (BLK-020, frozen)
- **Index:** Source 2 + Source 3 records, **partitioned by country**. The country label is an open set: France exists only in test and gets its own partition, and a query whose country has no partition would be searched against all partitions.
- **Representation:** TF-IDF over **word unigrams** of `business_name + business_address`, after the shared Unicode-safe normalizer (NFKC, casefold, `&`→and, Latin accents stripped, punctuation removed by Unicode category so Indic vowel signs survive). HashingVectorizer (2^22 buckets), exact per-country IDF (smooth, l2 rows), `min_df = 2`.
- **Retrieval:** exact cosine **top-K = 200 per S1** via sparse top-k matrix multiplication (`sparse_dot_topn`). We never materialize a dense query × index matrix.
- **Output:** `source1_entity_id, candidate_entity_id, score, rank`. The same pinned config (sha `657f59868e58`) is used for train, validation and test. For test, the index and IDF are fitted on the provided test S2+S3 only; no external data is used.
- **Scale:** one run fans out as 48 contiguous S1 slices on AWS (2 vCPU / 8 GB instances). Merging the slices reproduces the single-run output byte for byte (tested). Train+val took ~25 min, test ~22 min.

### Candidate pairs generated
| Split | S1 | Pairs | Pair recall@200 | Full coverage@200 | Ceiling F0.5@200 |
|---|---|---|---|---|---|
| train | 1,765,456 | 353,091,200 | 0.9780 | 0.9340 | 0.9923 |
| validation | 441,365 | 88,273,000 | 0.9779 | 0.9340 | 0.9921 |
| test | 1,732,544 (US 663k / India 810k / France 259k) | 346,508,800 | — | — | — |

Every S1 in every split received candidates (0 empty), France included. The final `candidate_pairs.tsv` is the subset the final model actually scores (`rank <= K`, K chosen end-to-end on validation):

| K | 10 | 25 | 50 | 100 | 200 |
|---|---|---|---|---|---|
| Val pair recall | 0.9248 | 0.9513 | 0.9628 | 0.9712 | 0.9779 |
| Val ceiling F0.5 | 0.9731 | 0.9824 | 0.9866 | 0.9897 | 0.9921 |
| Test pairs | 17.3M | 43.3M | 86.6M | 173.3M | 346.5M |

### How we ensured true matches were not lost
1. **Measured, not assumed.** Recall@10/25/50/100/200, coverage and ceiling on the full frozen validation split for every variant (registry `BLK-011`…`BLK-022`), broken down by match bucket and country (`experiments/results/diagnostics/BLK-020_val_buckets.md`):

   | Val bucket | S1 share | Pair recall@200 | Ceiling F0.5 |
   |---|---|---|---|
   | 0 matches (singletons) | 5.6% | n/a | 1.000 |
   | 1 match | 5.4% | 0.9752 | 0.9752 |
   | 2–5 matches | 77.5% | 0.9777 | 0.9924 |
   | 6+ matches | 11.5% | 0.9787 | 0.9943 |
   | India | 40.1% | 0.9590 | 0.9848 |
   | US | 59.9% | 0.9906 | 0.9971 |
2. **Fair comparisons.** The final representation won a same-harness comparison (BLK-019: identical 20k val S1, pool, fields, normalizer, `min_df`, K):

   | Variant | R@200 | Ceiling | Speed |
   |---|---|---|---|
   | char_wb (3,5) name+address | 0.959 | 0.984 | 9.6 S1/s |
   | **word unigram (chosen)** | **0.977** | **0.992** | **99.8 S1/s** |
   | word + max_df 0.02 | 0.974 | 0.990 | 446 S1/s |

   Earlier, name-only retrieval capped R@200 at ~0.78; adding the address was the single biggest recall lever.
3. **Miss analysis (BLK-021).** Of the 33,778 missed val pairs (2.21%), only 3 share no usable text. 99.9% share at least one word with their S1 but are crowded out of the top-200 by near-identical businesses. Causes: name typos ("Fonudlion"), domain-style names ("anglinharvin.com"), cross-script names (Devanagari/Kannada/Bengali vs Latin), and abbreviated or empty addresses.
4. **Extra channels, measured and rejected (BLK-022).**
   - Tested: forward transliterated-name retrieval (anyascii, ISC licence; character 3-grams), reverse retrieval (every S2/S3 record queries an S1 index for its top-k), and their unions.
   - The best union recovered 49% of misses (ceiling 0.9921 → 0.9959) at up to +184 candidates per S1.
   - It failed our pre-registered adoption gate (ceiling ≥ 0.997 with reasonable candidate growth), so we kept the smaller candidate set.
5. **Structural signal passed to the matcher (not a filter).**
   - In the training ground truth every S2/S3 record belongs to at most one S1, while 96% of candidate records are retrieved by several S1 (median 14).
   - Per-record competition statistics (best/second score, number of competing S1, best S1) are exported as matcher features.
   - We don't hard-filter on them: the true S1 is the record's top-scoring S1 for only 93.9% of true pairs, so a hard blocking-score filter would cap recall.

### Rejected along the way (logged, not deleted)
- Exact normalized name (EXP-001, F0.5 0.19).
- Global (non-partitioned) TF-IDF (OOM).
- Dense cosine (OOM at scale).
- A D2 char-(3,5) run that silently used word n-grams (bug found and fixed before any artifact was used).
- Word `max_df` pruning (faster, −0.3pp recall).
- Adaptive score cutoffs (no better than a fixed K).
- The channel unions above.

### Reproduce
- **Setup:** `pip install -r requirements.txt && pip install -e .`, set `ER_DATA_DIR`.
- **Train + val:** `python scripts/aws/run_d2_blocking.py --split trainval --blocker word`.
- **Test:** the same command with `--split test`.
- **Fan-out:** add `--part i/n` per worker, then `--merge n`.
- **Outputs:** `artifacts/blocking/*_candidate_pairs.tsv` + `blocking_metadata.json` (config, commit, sha256 per file).

## Model Architecture
*(To be populated by Ashank)*

## Feature Engineering
*(To be populated by Ashank)*

## Experiments
*(To be detailed from our experiment_registry.csv)*

## Conclusions
*(To be finalized by Mudit)*
