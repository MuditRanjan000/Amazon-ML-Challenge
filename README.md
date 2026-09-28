# Amazon ML Challenge 2026: Business Entity Resolution - Technical Postmortem

## 1. Executive Summary
- **Problem Statement:** The challenge required deduplicating and resolving business entity records across three noisy, multilingual datasets (Source 1 as a reference, Source 2/3 as queries). Entities exhibited 1-to-many matches, as well as a significant population of singletons (no matches).
- **Dataset Challenge:** The datasets contained ~12M records with severe noise, including typos, abbreviations, missing address components, and cross-lingual transliteration (English, Hindi, Tamil). The evaluation metric was Macro-Average F0.5 (weighted heavily toward precision). False merges penalized the score exponentially more than missed links. The test set also introduced an unseen country (France), requiring open-set generalization.
- **Final Result:** We achieved a peak Leaderboard F0.5 of **0.967**, a massive improvement over baseline string-matching and early models.
- **Improvement Journey:** The team progressed from a naive exact-match baseline (0.193) to a scalable TF-IDF blocking pipeline (0.750), to initial ML models (0.889), and finally reached 0.967 via a two-stage ensemble with targeted decision-layer pruning.

## 2. Complete Experiment Timeline
- **Baseline (Exact Match):** 0.193 LB. Proved that simple string normalization and matching is entirely insufficient for real-world entity resolution.
- **V3_r1:** 0.889 LB.
  - *Architecture:* Initial pairwise ML model on top of TF-IDF blocking.
  - *Outcome:* Failed to generalize. Suffered from entity leakage during internal diagnostic splits (features were trained on rows sharing entities with validation), leading to an overconfident model that struggled with unseen test distributions.
- **R4 variants (e.g., r4_country_cap):** 0.952 LB.
  - *Architecture:* Improved feature engineering (43+ features) and strict country-aware blocking logic.
  - *Outcome:* Succeeded in capturing cross-lingual matches and boosting overall candidate recall effectively.
- **R4ensX:** 0.959 LB.
  - *Architecture:* Early model ensembling yielding 5.86M candidate pairs.
  - *Outcome:* Succeeded in smoothing out individual model variances but retained too many false-positive lookalikes in the tail.
- **Ayush Ens3:** 0.965 LB.
  - *Architecture:* 3-model Stage 1 ensemble (`m1_r3`, `m1_r4`, `m1_r6`) + Stage 2 context model (`m2_ens3shift`) + Entity-level Decision Layer.
  - *Outcome:* Succeeded via targeted noise reduction. Pruned 101,298 noisy tail pairs from R4ensX, organically eliminating cluster sizes >11 and accurately recovering 101,124 singletons.
- **11_lookalike_drop_m1 (Peak):** 0.967 LB.
  - *Architecture:* Ens3 baseline plus an aggressive lookalike distractor drop filter.
  - *Outcome:* Succeeded. Pruned exactly 42,506 distractor lookalikes from high-density clusters (sizes 4-7) across US, France, and India. Eliminating precision drag without sacrificing true recall proved to be the winning strategy.

## 3. Root Cause Analysis
- **Why V3_r1 got high validation but poor leaderboard:** Massive entity leakage. The initial data split allowed the model to train on pairs where the S1 entity was also present in the validation set. The model memorized entity-specific artifacts rather than learning generalizable similarity, crashing on the unseen test set.
- **Why precision mattered more than recall:** Macro F0.5 weights precision 2x over recall. Adding a single false positive to a cluster dilutes that entity's F0.5 significantly, whereas missing a single true pair only marginally lowers it. Furthermore, correctly identifying a singleton (empty list) scores a perfect 1.0; a single false merge on a singleton drops its score to 0.0.
- **Why recall-heavy approaches failed:** Experiments like "Candidate 4 (consensus stacking)" injected 25,668 new recall pairs. This dragged the LB score down to 0.964. Why? The denser test set contained many similar distractors. Adding recall inherently brought in false positives, causing a precision penalty that outweighed the recall gains.
- **Why lookalike removal improved the score:** Our TF-IDF blocking was highly effective (ceiling F0.5 ~0.992). The final bottleneck wasn't finding matches, but rejecting "lookalikes" (businesses with identical generic names but slightly different locations). Surgically pruning these low-confidence lookalikes maximized the score by cleaning up borderline entity clusters.

## 4. Final Winning Pipeline
- **Blocking Strategy:** Memory-bounded Sparse Top-K Hashed TF-IDF. We used word unigrams on `business_name` + `business_address`, partitioned by country. Open-set handling allowed unseen countries (France) to match globally.
- **Candidate Retrieval K:** K=200, generating 346M test pairs with `min_df=2` and `max_df=0.02`. This achieved 97.79% recall with an F0.5 ceiling of 0.992.
- **Feature Engineering:** Pairwise similarities (Jaro-Winkler, edit distance, character n-gram cosine, numeric token overlap), plus critical retrieval-context features (`score_margin`, `is_best_s1`, `owner_gap`).
- **ML Model:** A 3-model Stage 1 ensemble (`m1_r3`, `m1_r4`, `m1_r6`) acting as base predictors, fed into a Stage 2 Context Stacker Model (`m2_ens3shift.joblib`).
- **Decision Layer:** Entity-level probability filtering with adaptive thresholding. Specifically tuned to aggressively drop uncorroborated tail candidates and lookalike distractors in dense clusters (size 4-7), explicitly preserving singletons.

## 5. Lessons Learned
- **What worked:** 
  - **Strict Architecture:** Separating the pipeline into Directives (Docs), Orchestration (Decisions), and Execution (Deterministic Scripts).
  - **Data Hygiene:** Frozen validation splits verified by SHA-256 prevented silent evaluation bugs.
  - **Memory Bounds:** PyArrow Parquet caching and `sparse_dot_topn` kept memory usage under 11GB, outperforming complex, OOM-prone D2 character-ngram matrices.
- **What failed:** 
  - All-pairs dense matrix multiplication (caused OOMs).
  - Over-optimizing recall at the expense of precision (LB score regressions).
  - Evaluating on non-frozen splits, causing catastrophic entity leakage.
- **Engineering bottlenecks:** Multi-million row TSV parsing initially throttled I/O. Moving to raw-text PyArrow Parquet loaders bypassed Pandas quoting bugs and sped up data loading by 10x.

## 6. Roadmap for >0.986
If we had one more week, we would implement:
1. **Multilingual Embeddings (LaBSE / IndicBERT):** Fine-tune a lightweight cross-lingual embedding model to map transliterated Hindi/Tamil strings directly to English counterparts, bypassing spelling-level noise entirely.
2. **Graph-based Sub-clustering:** Run connected-components algorithms on the final predicted pair graph. If A matches B, and B matches C, but A and C are geographically incompatible, prune the weakest edge to enforce transitive consistency.
3. **Strict Geocoding Simulation:** Extract structured postal codes and localized municipal names via NER to create a hard geographical block, zeroing out probabilities for businesses located in completely different cities despite having identical names.

---

## Setup & Execution (Legacy Instructions)
```bash
python -m venv .venv
.venv\Scripts\activate            # Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt   # exact pinned versions
pip install -e .                  # installs the `entity_resolution` package from src/
python -m pytest -q               # 25 tests
```

*Data:* Unzip the challenge resource to `6ab10eb3b23ba_student_resource/student_resource/dataset/` or set `ER_DATA_DIR` in `.env`.

| Task | Command |
|---|---|
| Baseline, validation split | `python execution/run_baseline.py --split val` |
| Baseline test submission | `python execution/run_baseline.py --split test` |
| Benchmark blocking | `python execution/run_blocking_eval.py --exp-id BLK-0xx ...` |
| Validate output files | `python scripts/validate_submission.py [--check-ids]` |
