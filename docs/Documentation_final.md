# ML Challenge 2026: Business Entity Resolution — Solution

**Team Name:** [Team name]
**Team Members:** Mudit, Aayush, Ashank
**Submission Date:** 27 September 2026

---

## 1. Executive Summary
We retrieve, for every Source-1 entity, its 200 nearest S2/S3 records under a country-partitioned word TF-IDF over name + address. A two-stage gradient-boosted matcher then scores these pairs.
- **Stage 1** scores each pair on fuzzy name, address, number, legal-form and name-rarity features.
- **Stage 2** re-scores each pair using its context: how the record's other S1 compete for it, and whether the S1's other candidates agree with it ("sibling consensus").
- **Decisions** enforce the many-to-one structure: each record goes to at most one S1. Each S1 then keeps the candidates that maximize expected F0.5.
- **Score:** frozen validation F0.5 is **[FINAL_VAL_F05]** (official evaluator), against 0.855 for our logistic-regression baseline.

---

## 2. Methodology

### 2.1 Problem Analysis
- **Structure:**
  - 5.6% of S1 have no match (singletons).
  - 26% of S2/S3 records are distractors that match no S1.
  - In the training ground truth, **every S2/S3 record belongs to at most one S1**, while 96% of candidate records are retrieved by several S1 (median 14).
- **Name noise:**
  - typos and digit substitutions (`Crysta1 Lending`, `HELI0S`);
  - domain and handle forms (`crystallending.com`, `@helios`);
  - legal-form variants (`LLC`, `[L.L.C.]`, `Private Ltd`);
  - word swaps;
  - whole names transliterated to Devanagari, Kannada or Gujarati;
  - gibberish replacement names (`Kelonyla`) at the exact address.
- **Address noise:**
  - components reordered;
  - state names vs codes (`Texas`/`TX`), and states in native script;
  - house and unit numbers perturbed (`2370` → `237`, `032/2`);
  - city replaced;
  - address emptied.
- **Distractors are twins of real entities:**
  - the same street with a different number;
  - an added descriptive word (`Exports`, `Jewellers`, `North`);
  - a swapped legal form (`LLC` → `Co`, `Private` → `Public`).
- **Test** adds France, a country unseen in training.

### 2.2 Solution Strategy
**Approach Type:** Blocking + two-stage GBDT classifier + record-level assignment.
**Core Innovation:** Two ideas taken from how the data is generated:
- **Sibling consensus.** The true copies of an entity agree with *each other* even where the S1 itself was perturbed. Distractor twins disagree with the cluster.
- **One owner per record.** The ground truth is many-to-one, so each record is assigned to its single most probable S1, followed by per-S1 expected-F0.5 set selection.

---

## 3. Candidate Generation (Blocking)
- **Blocking keys used:**
  - TF-IDF over word unigrams of `business_name + business_address`, after a Unicode-safe normalizer;
  - HashingVectorizer (2^22) with exact per-country IDF, `min_df=2`;
  - one index per country, with open-set countries (France gets its own index);
  - exact cosine top-200 per S1 via sparse top-k matrix multiplication (`sparse_dot_topn`).
- **Candidate pairs generated:**
  - test: 346,508,800 (1,732,544 S1 × 200) from blocking;
  - `candidate_pairs.tsv` = the rank ≤ **[FINAL_K]** subset the matcher scored.
- **How we ensured true matches were not lost:**
  - The oracle-ceiling F0.5 was measured on the frozen validation split:
    - K=100: 0.9897;
    - K=200: 0.9921 (pair recall 0.978).
  - We compared representations under the same harness. Word unigrams beat character 3–5-grams (0.992 vs 0.984 ceiling, 10× faster).
  - We diagnosed the 2.2% of missed pairs: typos, domain names and cross-script names crowded out of the top-200 by near-duplicates.
  - We measured extra channels (reverse and transliterated retrieval). They lift the ceiling to at most 0.996 at +184 candidates per S1, which failed our pre-registered cost gate.

---

## 4. Matching Model

**Features used:** 49 in stage 1, all vectorized (RapidFuzz `cpdist`, C++, multi-threaded).
- **Name features:**
  - ratio, token-set, token-sort, partial, Jaro-Winkler and exact key, on a normalized name (transliterated with anyascii, legal words removed, digit-to-letter typo repair);
  - no-space ratio and partial ratio (domains);
  - name lengths and non-Latin script flags;
  - legal-form class sets (equal / conflict / extra / missing);
  - typo-tolerant extra and missing name tokens;
  - name-token rarity (log document frequency: generated gibberish names are unique tokens).
- **Address features:**
  - token-set, token-sort, partial and ratio on a normalized address (abbreviations and state names mapped to codes);
  - number-set overlap and coverage;
  - digit-string similarity and first-number similarity;
  - empty-field flags.
- **Other:**
  - blocking score, rank, ratio to and gap from the S1's top score;
  - record competition over all S1: number of retrieving S1, whether this S1 is the best, margin to the best.
  - **Stage 2 adds:**
    - the record's best competing probability, rank and sum;
    - S1 context: count, rank, mass and max;
    - sibling consensus: the probability mass, count and share of the S1's other candidates sharing this record's house number, name key, address key or number set, plus equality with the S1 itself.

**Model type:**
- scikit-learn HistGradientBoosting (BSD-3), two stages; stage 1 is an average of **[N_MODELS]** models.
- Trained on train-split S1 only (seed 42); stage 2 is trained on train S1 unseen by stage 1.
- Scoring runs as an AWS fan-out over contiguous S1 slices; a full train+val+test pass takes about 15 minutes.

**Threshold selection method:**
- Rules were grid-searched on the full frozen validation split with the official macro-F0.5 formula. The winner was confirmed with the official evaluator.
- Candidate rules: a probability threshold, or per-S1 expected-F0.5 selection with a bias term, each with or without one-owner assignment.
- Validation S1 compete with train S1 for the same records, exactly as all test S1 do.

---

## 5. Results & Error Analysis

| Run | Val F0.5 | Precision | Recall | Singleton acc. |
|---|---|---|---|---|
| Rule baseline (RULE-001) | 0.7505 | 0.802 | 0.679 | 0.169 |
| Logistic regression, 43 features (LR-43) | 0.8554 | 0.905 | 0.771 | 0.771 |
| V3 round 1 (36 features, K=100) | 0.9634 | 0.980 | 0.925 | 0.932 |
| **Final ([FINAL_RUN])** | **[FINAL_VAL_F05]** | [P] | [R] | [S] |

- **F_0.5 Score (macro):** **[FINAL_VAL_F05]** on the full frozen validation split (441,365 S1). The ceiling at K=200 is 0.9921.
- **Common false positives (wrong merges):**
  - distractor twins at the same address that differ only by a small typo or a dropped legal suffix (`Ollanelle Micro` vs `Olanelle Micro LLC`);
  - name-only records (empty address) with an exact name that belong to a same-named S1 elsewhere.
- **Common false negatives (missed matches):**
  - about half are name-only copies with empty addresses, which are genuinely ambiguous between same-named S1;
  - true copies whose house number was replaced (`4507` → `380`);
  - gibberish-name copies with a partial address;
  - pairs outside the top-200 candidates (the blocking ceiling).

---

## 6. Conclusion
Most of the gain came from modelling how the data was generated, not from model capacity: legal-form and extra-word features, sibling consensus, and one-owner assignment together lifted validation F0.5 from 0.855 to **[FINAL_VAL_F05]**. The remaining gap to the 0.992 ceiling is dominated by records that carry too little information to be matched confidently (empty address, same-named entities).

---

## Appendix

### A. Code Artefacts
`code/business_entity_resolution/` (see its `README.md`):
- `src/entity_resolution/`:
  - `blocking/tfidf_blocking.py`: the blocker;
  - `v3_match.py`: normalization and pair features;
  - `evaluation/evaluator.py`: the official metric;
  - `submission/`: output writers and the validator wrapper.
- Entry points:
  - `scripts/aws/run_d2_blocking.py --split test --blocker word`: candidates;
  - `execution/v3_pipeline.py normalize | train | score | decide`: matcher, decisions, and `output/matching_results.tsv` + `output/candidate_pairs.tsv`;
  - `scripts/aws/launch_fanout.sh` and `scripts/aws/v3_fanout.sh`: the same steps fanned out on AWS.
- Everything is seeded (42). The frozen validation split is hash-verified. Every run is logged to `experiments/results/experiments.jsonl`.

### B. Additional Results
Blocking ceiling by K on validation: K=10 0.9731, K=25 0.9824, K=50 0.9866, K=100 0.9897, K=200 0.9921. The full experiment registry is in `experiments/results/experiment_registry.csv`.
