# Amazon ML Challenge 2026 - AI Context

## Current Understanding of the Problem
The task is a Business Entity Resolution challenge. We need to match business entities across three independent noisy data sources. Source 1 is a deduplicated reference source. Our goal is to find all matching records from Source 2 and Source 3 for each Source 1 entity. 

- **Evaluation Metric:** F0.5 score (macro-averaged per Source 1 entity). This metric heavily penalizes false merges (precision is weighted 2x over recall). Singletons are included in the score calculation.
- **Submissions:** Output format requires two files: `matching_results.tsv` (final matches) and `candidate_pairs.tsv` (blocking candidates before final inference).
- **Constraints:** Max model size: 8 Billion parameters. External Data lookup is STRICTLY PROHIBITED.

## Dataset Discoveries
- **Volume:** ~2.2M Source 1 entities in train, ~1.7M in test. ~10M S2/S3 candidates in train, ~10M in test.
- **Missing Data:** Names are mostly present. Addresses are missing in ~3% of S2/S3 records.
- **Singletons:** ~5.5% of S1 entities have no matches.
- **Matches:** This is a one-to-many mapping. An S1 entity maps to 1 to 11 records in S2/S3 (mode is 3 matches). No many-to-one mappings exist.
- **Noise:** Transliteration (English to Hindi/Tamil), typos, truncated names, missing address components, and abbreviation variations.

## Experiments Completed
- Dataset Analysis & Profiling (Completed via script)
- Experiment 1 (Baseline): Exact string match on normalized `business_name` partitioned by country.

## Scores Achieved
- Local F0.5: 0.19372 (Baseline - Exact Match)
- Public Leaderboard: N/A

## Failed Approaches
- None yet.

## Current Best Pipeline
- **Proposed:** TF-IDF character n-gram blocking (partitioned by country) -> Pair-wise Feature Engineering -> LightGBM Classifier -> High probability thresholding (optimized for F0.5).

## Next Recommended Action
- Await user approval on the proposed strategy.
- Set up local validation split and F0.5 evaluator script.
- Build the baseline (Exact match) to establish a baseline score.
