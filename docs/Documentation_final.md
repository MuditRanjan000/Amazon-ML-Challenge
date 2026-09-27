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
- **Score:** frozen validation F0.5 **0.9779** (official evaluator; logistic-regression baseline 0.855); public leaderboard **0.959**.

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
  - `candidate_pairs.tsv` = the rank ≤ **200** subset (346,508,800 pairs) the matcher scored.
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
- scikit-learn HistGradientBoosting (BSD-3), two stages; stage 1 is an average of **2** models (r3: 200k train S1, top-100 candidates, 45 features; r4: 150k train S1, top-200, 49 features, rank > 100 negatives subsampled to 25% and reweighted).
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
| V3 round 1 (36 features, 100k S1, K=100) | 0.9634 | 0.980 | 0.925 | 0.932 |
| V3 r3 (49 features, 200k S1, K=100) | 0.9732 | 0.987 | 0.941 | 0.962 |
| V3 r3, K=200 | 0.9751 | 0.988 | 0.946 | 0.962 |
| V3 ensemble r3 + r4 (stage 1 averaged), K=100 | 0.9757 | 0.989 | 0.945 | 0.970 |
| V3 ensemble r3 + r4, K=200 | 0.9776 | 0.990 | 0.949 | 0.970 |
| V3 ensemble + twin-contrast / name-ambiguity stage-2 features (r4ensX), K=100 | 0.9760 | 0.989 | 0.945 | 0.971 |
| V3 r4ensX, K=200 | 0.9779 | 0.990 | 0.950 | 0.971 |
| **Final: r4ensX, K=200** | **0.9779** | **0.990** | **0.950** | **0.971** |

- **Public leaderboard history:**

  | Submission | Val F0.5 | Public LB |
  |---|---|---|
  | V3_r1 (K=100) | 0.9634 | 0.889 |
  | Ensemble r3 + r4, K=200, per-S1 cap 5 (France 3) | 0.9776 before the cap | 0.952 |
  | **r4ensX, K=200, uncapped (final)** | **0.9779** | **0.959** |

  The val→LB gap shrank from 0.074 (V3_r1) to 0.019 (r4ensX): the stronger stage 1, the ensemble and the twin-contrast / name-ambiguity features transfer to test. A hard per-S1 cap on the number of matches was **rejected by the leaderboard** (0.952 < 0.959): the extra test predictions are not simply the lowest-ranked ones, and the cap removes true matches of S1 with many copies. Section 5.1 analyses the remaining gap.

- **F_0.5 Score (macro):** **0.9779** on the full frozen validation split (441,365 S1). The ceiling at K=200 is 0.9921.
- **Common false positives (wrong merges):**
  - distractor twins at the same address that differ only by a small typo or a dropped legal suffix (`Ollanelle Micro` vs `Olanelle Micro LLC`);
  - name-only records (empty address) with an exact name that belong to a same-named S1 elsewhere.
- **Common false negatives (missed matches):**
  - about half are name-only copies with empty addresses, which are genuinely ambiguous between same-named S1;
  - true copies whose house number was replaced (`4507` → `380`);
  - gibberish-name copies with a partial address;
  - pairs outside the top-200 candidates (the blocking ceiling).

### 5.1 Test distribution shift and the orphan simulation
- **The shift.** Test has more S2/S3 records per S1 than train, uniformly across countries (no labels needed to see it):

  | Split | Country | S1 | S2+S3 | Records per S1 |
  |---|---|---|---|---|
  | train | US | 1,323,633 | 6,186,873 | 4.67 |
  | train | India | 883,188 | 4,133,346 | 4.68 |
  | test | US | 663,106 | 3,817,031 | 5.76 |
  | test | India | 809,986 | 4,717,565 | 5.82 |
  | test | France | 259,452 | 1,434,993 | 5.53 |

  If test S1 have as many true matches as train S1 (~3.46), about 40% of test records are distractors, against 26% in train.
- **Why it matters for this matcher.** Our strongest signals are competitive: a distractor twin loses its record because the twin's own S1 is in the reference set and claims it (record competition in stage 1, one owner per record in the decision). A twin whose S1 is *absent* is an "orphan": nothing claims its records, so a similar S1 can win them. Validation cannot show this, because validation S1 compete with the complete train S1 set.
- **The simulation.** We treat a fixed 20% of train+val S1 (seed 11, `output/v3/dropped_s1.txt`, 441,364 S1) as absent, which turns their records into owner-less distractors and brings records per S1 to ~5.8, as in test. Blocking needs no change (retrieval is per S1, so the remaining S1 keep their exact candidates). We recompute the record competition statistics without them (`candidate_record_stats.py --drop-s1`), train stage 1 on the remaining S1 (`train --exclude-s1`), and fit/tune stage 2 on the remaining train S1 and evaluate on the remaining 352,903 val S1 (`decide --drop-s1`).
- **Results.**

  | Check | Val F0.5 |
  |---|---|
  | V3_r1 unchanged (stage-1 stats unshifted, stage-2 context shifted) | 0.9626 (vs 0.9634) |
  | [r3 K=200 stage 2 on the fully shifted val] | [SHIFT_A] |
  | [stage 1 + stage 2 trained under the shift] | [SHIFT_B] |

  Shifting only the stage-2 context barely moves the score; the stage-1 competition features carry the effect, which is why the full simulation rescores stage 1 with shifted statistics.
- **Test-time calibration.** From one stage-2 pass we also write variants with per-country logit shifts of the stage-2 probability before the tuned rule (`variants`). On validation, V3_r1's F0.5 by shift: +0.5 0.9611, 0 0.9626, −0.5 0.9623, −1.0 0.9609, −1.5 0.9581, −2.0 0.9534, −3.0 0.9388. Mild conservative shifts cost almost nothing on validation and protect precision on the denser test distribution. The final submission applies the validation-tuned rule without a shift or a cap.

---

## 6. Conclusion
Most of the gain came from modelling how the data was generated, not from model capacity: legal-form and extra-word features, sibling consensus, and one-owner assignment together lifted validation F0.5 from 0.855 to **0.9779**. The remaining gap to the 0.992 ceiling is dominated by records that carry too little information to be matched confidently (empty address, same-named entities).

---

## Appendix

### A. Code Artefacts
`code/business_entity_resolution/` (its `README.md` = `docs/submission_README.md`: package layout, environment and every command in order):
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
