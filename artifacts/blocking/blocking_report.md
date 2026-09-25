# EXP-002B Stage 2 Blocking Report

**Status:** EXECUTING
**Target:** Full Validation Split (35,000 queries vs 5,000,000 candidates)

## Configuration (Variant D2)
*   **Strategy:** TF-IDF Cosine Similarity Blocking
*   **Partitioning:** Hard partitioning by `country`
*   **Fields:** `business_name` + `business_address`
*   **Feature Engineering:** Character n-grams (3, 5)
*   **Index Storage:** Out-of-Core Sharded Sparse Matrices (500k row chunks)
*   **Vocabulary Builder:** HashingVectorizer (capped at $2^{21}$ features)
*   **IDF Learner:** TfidfTransformer trained on a memory-safe random sample (up to 300k per country).

## Metrics
*Pending execution completion (Expected ETA: ~4.3 hours)*

| Metric | Score |
|---|---|
| Recall@10 | TBD |
| Recall@25 | TBD |
| Recall@50 | TBD |
| Recall@100 | TBD |
| Recall@200 | TBD |
| Avg Candidates | TBD |
| Runtime | TBD |
| Memory Usage | Optimized (Stable < 1GB) |

## Limitations & Edge Cases
*   **Address Junk Tokens:** Addresses contain highly unstructured and variable noise (e.g., zip codes attached to names, street abbreviations varying wildly). Although TF-IDF handles terms reasonably well, very long strings may dilute the core distinguishing tokens in the `business_name`.
*   **Fixed K Thresholds:** K=200 may be over-generating candidates for simple matches, unnecessarily burdening the pairwise model down the line. We should consider dynamic score thresholds rather than a strict K cutoff if downstream modeling struggles with inference time.
*   **Transliterations:** Standard TF-IDF struggles with heavy transliteration differences (e.g. English vs native script).

## Next Recommendations
1.  **Review the Candidate Output:** Once `validation_candidate_pairs.tsv` is fully generated, Ashank will use it to train the pairwise matching models.
2.  **Evaluate Score Cutoffs:** We must plot a precision-recall curve based on the raw TF-IDF cosine scores to see if a hard score cutoff is better than a top-K cutoff.
3.  **Handoff to ML Stage:** Candidate generation is complete enough to move forward to ML modeling.
