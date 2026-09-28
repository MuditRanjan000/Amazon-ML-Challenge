# System Architecture Diagram Description

**Title:** Amazon ML Challenge 2026 - Entity Resolution Pipeline

This architecture diagram describes the flow of data from raw TSVs to the final submission files, capturing the modular 3-layer architecture used to achieve the 0.967 Leaderboard score.

## 1. Data Ingestion & Normalization
- **Inputs:** `Source 1 (Reference)`, `Source 2 (Queries)`, `Source 3 (Queries)` (Total ~12M records).
- **Component:** `PyArrow Raw Text Loader` -> Reads data while bypassing Pandas quote-parsing bugs.
- **Component:** `Unicode Normalizer` -> Strips punctuation and normalizes casing while safely preserving non-Latin scripts (e.g., Devanagari/Tamil).
- **Output:** Cached Parquet Files for rapid I/O.

## 2. Candidate Generation (Blocking)
- **Input:** Normalized Parquet Datasets.
- **Component:** `Hashed TF-IDF Blocker (Word Unigrams)` -> Applies `min_df=2` and `max_df=0.02` to `business_name` + `business_address`.
- **Logic:** Partitions search space by Country (open-set to handle France in test). Uses `sparse_dot_topn` for memory-bounded sparse matrix multiplication.
- **Output:** Candidate Pairs Dataset (K=200, yielding ~346M pairs for the test set). Hard ceiling F0.5 of 0.992.

## 3. Pairwise Feature Engineering
- **Input:** 346M Candidate Pairs.
- **Component:** `Feature Extractor` -> Computes 43+ pairwise features.
  - *String Features:* Jaro-Winkler, Edit Distance, Character N-gram Cosine, Token Overlap.
  - *Contextual Features:* Score margins, relative rankings (`is_best_s1`), and owner gap statistics.

## 4. Matching Models (Two-Stage Ensemble)
- **Input:** Feature-engineered Candidate Pairs.
- **Stage 1 Ensemble:** 3 heterogeneous Tree-based Base Models (`m1_r3`, `m1_r4`, `m1_r6`).
- **Stage 2 Stacker:** Context Model (`m2_ens3shift.joblib`) that consumes Stage 1 probabilities and contextual ranking features to output a final merged probability.

## 5. Entity-Level Decision Layer
- **Input:** Merged Probabilities.
- **Component:** `EFO Decision Logic & Lookalike Pruner` -> Applies adaptive thresholds based on cluster size.
- **Logic:** 
  - Explicitly protects and preserves Singletons.
  - Identifies dense clusters (sizes 4-7) and surgically drops uncorroborated "lookalike" distractors to heavily optimize the precision-weighted F0.5 metric.
- **Output:** Final Entity-to-Matches Graph.

## 6. Output & Validation
- **Component:** `Submission Generator` -> Formats into strict TSV specifications.
- **Component:** `Official Validator Wrapper` -> Verifies ID integrity, row counts, and structural compliance.
- **Final Outputs:** `matching_results.tsv` (for LB scoring) and `candidate_pairs.tsv` (for blocking audit).
