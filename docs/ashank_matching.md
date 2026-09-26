# Ashank Matching Workstream

## Scope

This workstream consumes Aayush's candidate pairs and returns pairwise scores plus conservative decision guidance to Mudit. It owns feature engineering, matching classifiers, threshold calibration, singleton/no-match treatment, multilingual/transliteration experiments, and model-side error analysis. It does not replace blocking, the canonical split/evaluator, integration, or submission ownership.

## Local development setup

- Checkout: `feature/ashank-model`, created locally from `origin/main` commit `e8f2517833414444a1509f25322404a609eb31df`. No upstream/push was created during setup.
- Supplied inputs remain in `D:\Amazon_ML\student_resource\dataset` and root PDFs remain in `D:\Amazon_ML`; no copies belong in this checkout.
- Set `AMAZON_ML_DATA_DIR` in ignored `.env` to the directory containing `train/` and `test/`. `src.entity_resolution.data.config.resolve_data_dir()` is the shared path entry point for execution scripts.
- Local profile from the preserved setup: i5-13500H; 15.73 GiB RAM; Intel Iris Xe graphics; 146.77 GiB free on D:. Use streamed/chunked CPU-friendly experiments.

## Verified shared interfaces

| Component | Verified capability | Matching implication |
| --- | --- | --- |
| `DataLoader` | Loads full source/ground-truth TSVs from a supplied dataset root using pandas. | Reuse for small/baseline runs; its "memory-efficient" claim is not verified for full-scale work. |
| `create_validation_split` | Deterministic random Source-1 ID split, default 20%/seed 42. | Do not call it as a canonical split until Mudit publishes saved IDs/metadata. |
| `Evaluator.evaluate` | In-memory macro per-entity F0.5, with explicit singleton empty-list semantics. | Reuse as the current evaluator, while Mudit confirms the official frozen invocation. |
| `BaseBlocker` | Candidate frame contract: `source1_entity_id`, `candidate_entity_id`, `score`, `rank`. | Matcher must require the first two fields and preserve optional retrieval score/rank; do not force top-1 decisions. |
| `ExactNameBlocker` | Country + normalised-name exact candidate baseline. | Existing blocking baseline only; Aayush owns replacements. |
| `BlockingEvaluator` | Candidate recall and average candidate count at K. | It lacks documented multiplicity slices and reduction ratio; obtain Aayush's diagnostics separately. |

## Pairwise interchange

The matcher requires at least one UTF-8, tab-separated row per candidate pair:

```text
source1_entity_id<TAB>candidate_entity_id
S1-...<TAB>S2-...
```

When available, keep Aayush's retrieval `score` and `rank` as optional columns. Return all received pairs with:

```text
source1_entity_id<TAB>candidate_entity_id<TAB>match_probability
```

`match_probability` must be finite in [0, 1]. Decision logic must permit zero, one, or many accepted S2/S3 IDs for each Source 1 entity. Mudit aggregates the exact final scored candidates into official one-row-per-Source-1 `candidate_pairs.tsv`, and selected matches must be its subset.

## Frozen-validation status and ownership

Mudit's immutable manifest is now present at `artifacts/validation_split/manifest.json`; it declares 1,765,456 train and 441,365 validation Source-1 IDs, seed 42, and SHA-256 checksums for `artifacts/validation_split/train_ids.csv` and `artifacts/validation_split/val_ids.csv`. The two CSV blobs are **not present in the fetched remote tree or this checkout** (the remote `.gitignore` ignores `*.csv`), so their published checksums cannot yet be verified and no frozen-split score can be run or reported. The only local split helper (`data.validation_split.create_validation_split`) would create a new random split if the files are absent; it is prohibited for benchmark comparison and must not be called for this workstream.

The shared evaluator callable is `src.entity_resolution.evaluation.evaluator.Evaluator.evaluate`. Ashank's fixed known-answer test (one complete match, one one-of-two match, and one correct singleton) returns macro F0.5 `17/18`, macro precision `1.0`, macro recall `5/6`, and singleton accuracy `1.0`. This check passes, but the corresponding Mudit-provided known-answer fixture/invocation has not been found.

- Canonical 43-feature code now lives in `src/entity_resolution/features/pairwise.py`.
- Rule scoring, Logistic Regression, threshold decisions, and bounded experiment orchestration now live in `src/entity_resolution/models/`.
- `src/entity_resolution/matching/` retains import-compatible adapters only. Existing CLI imports and trusted local feature artifacts remain usable; previously generated feature artifacts do not need regeneration solely because of this ownership refactor.
- Thresholded one-to-many decisions are local evaluation artifacts. The integration output is always one probability row for every supplied candidate: `source1_entity_id`, `candidate_entity_id`, `match_probability`.

## Implemented infrastructure (not a measured experiment)

- `matching.contracts`: validates required pair columns, S1/S2/S3 prefixes, empty IDs, and duplicate pairs. Optional blocker fields such as `score` and `rank` remain separate retrieval metadata.
- `matching.records`: provides a fixture/small-subset DataFrame adapter and a reusable SQLite record store. The SQLite store scans each selected source TSV once when built, then fetches only IDs needed for each joined candidate batch; missing source records fail fast.
- `matching.normalization`: adds matching-specific NFKC/casefold/token representations without changing shared `normalize_text`. It preserves digits, accents, and Unicode scripts; it does not transliterate.
- `features.pairwise`: provides fitted name/address TF-IDF cosine features plus exact, Jaro-Winkler, Levenshtein, token, character n-gram, numeric, postal, country, source-pair, missingness, combined-evidence, and script-indicator features. Two missing fields produce zero positive lexical evidence.
- `models.rule`: produces `raw_match_score` using provisional transparent lexical weights. This is **not** a calibrated probability and must not be exported as `match_probability`.
- `models.decisions`: thresholds supplied scores without top-1, one-to-one, or margin constraints, and requires the full Source-1 ID universe so no-candidate entities receive empty predictions.

### Address evidence limits

Postal matching is attempted only for country-labelled US (ZIP), India (PIN), and France (five-digit) patterns. Locality is emitted only for a conservative comma-separated address with a final short region token; it is intentionally unavailable for many records, including many French addresses. Neither parser is a geocoder or an external lookup.

## Local commands

Run from the repository root. All generated stores, artifacts, feature TSVs, raw scores, and decisions belong under ignored `.tmp/` until Mudit integrates them.

```powershell
# Build once per train/test source collection; this does not train a model.
python execution/run_matcher_baseline.py build-store `
  --split train --store .tmp/train_records.sqlite

# Fit TF-IDF only on Aayush's *training* candidates. The default 100,000-pair cap
# is a controlled local fit; pass a representative training shard, never validation.
python execution/run_matcher_baseline.py fit-features `
  --fit-split train --candidates <train_candidates.tsv> `
  --store .tmp/train_records.sqlite --artifact .tmp/lexical_features.pkl

# Transform a candidate shard with the fitted artifact, preserving bounded record joins.
python execution/run_matcher_baseline.py extract-features `
  --candidates <validation_candidates.tsv> --store .tmp/train_records.sqlite `
  --artifact .tmp/lexical_features.pkl --output .tmp/validation_features.tsv `
  --batch-size 10000

# Produce raw heuristic scores, then aggregate a provisional threshold decision.
python execution/run_matcher_baseline.py score `
  --features .tmp/validation_features.tsv --output .tmp/validation_raw_scores.tsv
python execution/run_matcher_baseline.py decide `
  --scores .tmp/validation_raw_scores.tsv `
  --source1-ids <validation_source1.tsv> --threshold 0.75 `
  --output .tmp/validation_decisions.tsv
```

The candidate TSV is currently loaded one shard at a time for contract validation; source-record joins are batch-bounded and disk-backed. On this 16 GiB CPU-only host, shard candidates before extraction and do not run a full candidate set until its size/runtime has been measured. Fitted pickle artifacts are local trusted artifacts only; persist the adjacent JSON manifest containing feature order and configuration.

### Bounded local throughput check

The earlier 126-pairs/s synthetic check was run under `tracemalloc` and is not a throughput baseline: tracing materially slows this Python-heavy workload. Use the following reproducible diagnostic instead:

```powershell
python execution/profile_matcher.py `
  --rows-per-source 10000 --pair-count 2000 `
  --candidates-per-source1 10 --batch-size 1000 --fit-pairs 1000 `
  --measure-sqlite --output .tmp/matcher_profile_2000.json
```

It reads only the first 10,000 rows of each supplied training source. It selects script-bearing, missing-field, and long-text records first, then fills deterministically; 200 distinct S1 records are each paired with ten unique alternating S2/S3 records. These are real records but constructed pairs, not Aayush blocking output, ground-truth positives, a canonical split, or a quality measurement.

| Measurement | Before optimisation | After optimisation |
| --- | ---: | ---: |
| Same 2,000-pair diagnostic, warm feature transform | 607 pairs/s | 1,124 pairs/s |
| Same diagnostic, feature transform + TSV serialization | 588 pairs/s | 1,068 pairs/s |
| Feature columns | 43 | 43 |
| Reference-value delta on 200 fixed supplied-record pairs | N/A | 0.0 for checked lexical features |

The final profiler run had 2,000 pairs, 200 distinct S1 IDs, 2,000 distinct candidate IDs, 1,441 script-bearing record sides, and 458 pairs with a missing address. Measured stages were: DataFrame adapter build 0.443 s; two 1,000-pair DataFrame joins 0.045 s; sample SQLite-store build for 30,000 records 0.716 s; sample SQLite cold/warm joins 0.111/0.116 s; TF-IDF fit on 1,000 diagnostic pairs 1.099 s; cold/warm feature transform 1.911/1.780 s; and TSV serialization 0.094 s. The approximate warm SQLite end-to-end path is therefore 1,005 pairs/s, excluding candidate-file ingestion and full-store construction.

`cProfile` inflated the final transform from 1.87 s to 4.07 s; `tracemalloc` inflated it to 11.33 s (6.4x) and reported 37.17 MiB Python-tracked peak. A separate process-RSS check on the same workload showed roughly a 30 MiB increase, but neither method completely attributes native NumPy/scikit-learn allocations. Do not use profiler-instrumented timings for capacity planning.

### Optimisations and remaining bottleneck

- Replaced repeated per-feature normalization with a transient, batch-bounded record representation: normalized name/address, token sets, n-grams, numeric/postal/locality evidence, and script flags are computed once per distinct record text within a transform batch.
- Added `rapidfuzz==3.14.3` (MIT according to installed package metadata) for compiled Levenshtein and Jaro-Winkler. The original Python implementation remains a fallback and is checked for numerical equivalence.
- Changed TF-IDF cosine to transform each unique normalized string once per vectorizer and gather aligned sparse rows. It remains sparse; no dense all-record similarity matrix is created.
- Preserved all 43 reference features and feature order. No feature was removed or reweighted.

After these changes, sparse TF-IDF tokenization/transformation is the largest measured cost (about 42% of profiled cumulative transform time); transient record representation construction is next (about 40%, with overlapping cumulative accounting). Joins and TSV serialization are small at this diagnostic size. A reduced feature set, multiprocessing, or model-side changes are deliberately not proposed as performance/quality improvements until real candidate distributions and canonical validation exist.

At the approximately 1,005-pairs/s warm SQLite diagnostic rate and 236 bytes/pair feature TSV density, illustrative lower-bound estimates are: 1M pairs ≈17 minutes and 236 MB; 10M ≈2.8 hours and 2.36 GB; 50M ≈13.8 hours and 11.8 GB. These exclude full SQLite-store build, candidate ingestion/validation, different text and fan-out distributions, larger TF-IDF vocabularies, native memory pressure, and output compression; full-scale performance remains uncertain.

## Required handoffs before model work

### From Mudit

1. Canonical saved Source-1 train/validation IDs or a manifest with checksum, seed/rule, and repository location.
2. Official evaluator command/function, prediction schema, and a known-answer test.
3. Experiment registry location/schema and integration convention for pairwise scores and decision parameters.

### From Aayush

1. Versioned, deduplicated candidate artifact for the same frozen split, with source-data/candidate-generator identifiers.
2. Candidate recall at K=10/25/50/100/200, candidate-size distribution, runtime, memory, and multiplicity slices.
3. Exact final test candidate set only when integration is ready; it must be the set actually scored, not an early blocking superset.

## Bounded rule/LR preflight runner

`execution/run_bounded_matching_experiment.py` never creates a split, drops a selected candidate group, or runs test inference. It first verifies the frozen ID CSV hashes/counts and disjointness against Mudit's JSON manifest, then runs the evaluator known-answer check. It reads the first requested IDs from the supplied frozen ID files, validates candidate ID references/deduplication, records SHA-256 input hashes, fits TF-IDF on unique training documents only, and fits an L2 Logistic Regression with a training-fitted `StandardScaler`. Default bounds are 1,000 Source-1 entities and 50,000 complete candidate pairs per split; exceeding the pair bound fails rather than truncating a group.

Its report includes tuning sweeps from 0.50 to 0.95, macro entity-level F0.5 from the current local evaluator, macro precision/recall from that evaluator, global pairwise precision/recall, FP/FN counts, singleton accuracy, non-singleton empty-prediction rate, country and true-match-count slices, candidate recall, input hashes, runtime, errors, feature schema, and model artifacts. Threshold results are explicitly tuning results. Logistic `predict_proba` is a probability estimate; no empirical calibration is applied. The runner labels all output as `local_preflight_not_official_until_mudit_confirms_evaluator` until the official invocation is supplied.

```powershell
python execution/run_bounded_matching_experiment.py `
  --train-source1-ids artifacts/validation_split/train_ids.csv `
  --validation-source1-ids artifacts/validation_split/val_ids.csv `
  --split-manifest artifacts/validation_split/manifest.json `
  --train-candidates <aayush_train_candidates.tsv> `
  --validation-candidates <aayush_validation_candidates.tsv> `
  --ground-truth D:\Amazon_ML\student_resource\dataset\train\train_ground_truth.tsv `
  --store .tmp\train_records.sqlite `
  --run-dir .tmp\runs\EXP-XXX `
  --experiment-id EXP-XXX `
  --split-version <manifest-or-version> `
  --candidate-version <aayush-artifact-version> `
  --evaluator-invocation <mudits-confirmed-command-or-callable>
```

### Temporary exact-name candidates while EXP-002B is pending

`execution/generate_temp_exact_candidates.py` creates the explicitly temporary `TEMP-EXACT-NAME-v1` source. It streams all supplied Source 2 and Source 3 training records into a disk-backed SQLite exact-name index, then streams selected frozen Source-1 records. It uses the existing `ExactNameBlocker` lower-case, punctuation-removal, suffix-removal, and whitespace-normalisation semantics; it never consults labels to generate candidates. Ground truth is read only afterwards to report candidate recall. It retains every generated exact-name candidate for each selected entity, fails rather than truncating a group, and records zero-candidate selected entities in its JSON metadata.

After Mudit supplies both frozen CSVs, first run the 1,000-entity bounded flow below. This indexes the full S2/S3 lookup corpus once; it does not run test inference.

```powershell
$trainIds = 'artifacts/validation_split/train_ids.csv'
$valIds = 'artifacts/validation_split/val_ids.csv'
$manifest = 'artifacts/validation_split/manifest.json'
$truth = 'D:\Amazon_ML\student_resource\dataset\train\train_ground_truth.tsv'

python execution/generate_temp_exact_candidates.py --split train `
  --source1-ids $trainIds --other-source1-ids $valIds --split-manifest $manifest `
  --ground-truth $truth --index .tmp\temp_exact_name.sqlite --rebuild-index `
  --output .tmp\candidates\TEMP-EXACT-NAME-v1_train.tsv
python execution/generate_temp_exact_candidates.py --split validation `
  --source1-ids $valIds --other-source1-ids $trainIds --split-manifest $manifest `
  --ground-truth $truth --index .tmp\temp_exact_name.sqlite `
  --output .tmp\candidates\TEMP-EXACT-NAME-v1_validation.tsv

python execution/run_matcher_baseline.py build-store --split train --store .tmp\train_records.sqlite
python execution/run_bounded_matching_experiment.py `
  --train-source1-ids $trainIds --validation-source1-ids $valIds --split-manifest $manifest `
  --train-candidates .tmp\candidates\TEMP-EXACT-NAME-v1_train.tsv `
  --validation-candidates .tmp\candidates\TEMP-EXACT-NAME-v1_validation.tsv `
  --ground-truth $truth --store .tmp\train_records.sqlite --run-dir .tmp\runs\EXP-003A `
  --experiment-id EXP-003A --candidate-version TEMP-EXACT-NAME-v1
```

For EXP-002B, replace only the two `--*-candidates` paths and `--candidate-version` with Aayush's versioned frozen-split TSVs. Preserve the manifest, ID files, store, feature configuration, pair limits, evaluator, and command otherwise. The exact-name results measure joint blocker-plus-matcher behavior and must not select a final model alone.

### EXP-003A frozen-subset diagnostic (2026-09-26)

The frozen manifest and both CSVs passed their declared SHA-256 hashes, counts, and disjointness check. The shared evaluator known-answer check also passed. `EXP-003A` then used the first 1,000 frozen train IDs and first 1,000 frozen validation IDs, full S2/S3 lookup corpora, and the same `TEMP-EXACT-NAME-v1` candidate generator on both sides. This is an evaluator-valid frozen **subset diagnostic**, not a full frozen benchmark or final model comparison.

| Model | Tuned threshold | Macro entity F0.5 | Macro P / R | Pairwise P / R | FP / FN | Singleton accuracy |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Rule (`raw_match_score`) | 0.55 | 0.3600 | 0.4784 / 0.2302 | 0.4147 / 0.2119 | 1,064 / 2,805 | 0.64 |
| L2 Logistic Regression | 0.55 | 0.5124 | 0.6567 / 0.3237 | 0.9698 / 0.2891 | 32 / 2,530 | 0.96 |

The validation candidate artifact contains 19,315 pairs (19.315 per selected entity); 800 entities have candidates and 200 retain explicit zero-candidate predictions. Candidate recall is 1,117/3,559 = 0.3139, so 2,442 true pairs are blocking misses. At the Logistic threshold, only 88 of the 1,117 retrieved true pairs are rejected; most false negatives therefore originate in temporary exact-name blocking. High-score Logistic false positives are near-name/address collisions (often a small street/house-number discrepancy), not generic low-similarity pairs. `predict_proba` is not empirically calibrated.

Feature fit/transform for the shared 34,776 pairs took 11.91 s (about 1,621.5 pairs/s); Logistic fit/probability took 0.07 s; total pre-report run time was 17.81 s. One-time local setup built a 1.201 GiB exact-name SQLite index and a 1.515 GiB full-record SQLite store. Extrapolating the temporary candidate density yields about 27.30M train pairs (4.68 feature hours) and 8.52M validation pairs (1.46 feature hours), excluding candidate generation and in-memory candidate-frame overhead. The current generator/runner intentionally caps materialized groups at 50,000 pairs; a full temporary-exact run is therefore not attempted on this 16 GiB host. EXP-002B candidates should replace this blocker before scalable full validation work.

Before a reported benchmark run, obtain from Mudit: the actual frozen ID CSV artifacts, the official evaluator command/callable plus its supplied known-answer fixture, the experiment-registry schema, and the exact score-integration convention. No cloud resource or paid service is needed for this runner.

## Constraints carried into every experiment

- Use only supplied data: no external business lookup, geocoding, commercial ER APIs, or internet augmentation.
- Final models must be MIT/Apache-2.0 licensed and at most 8B parameters.
- Macro F0.5 includes singletons: correct empty prediction earns 1.0; a false merge earns 0.0.
- Candidate sets must scale without exhaustive all-pairs comparison. The organiser says smaller candidate sets per Source 1 affect final rankings; no formula is known.

## Next implementation step

Obtain Mudit's actual frozen ID CSV artifacts, verify their SHA-256 checksums against the committed manifest, then run `TEMP-EXACT-NAME-v1` on the bounded frozen subset. Record its official evaluator result only after Mudit confirms the evaluator's supplied known-answer fixture. When Aayush publishes EXP-002B candidates, replace the candidate paths/version only, rerun the same bounded flow, and inspect saved false positives, false negatives, blocking misses, country slices, and multiplicity slices before choosing a follow-up model experiment. Do not calibrate on canonical validation labels or run final inference.

## D2 scalable-ingestion preparation (2026-09-26)

> Historical only: BLK-019 Word Unigram supersedes this D2 character-n-gram section and every D2 candidate artifact. Do not use its commands or artifacts for current model training/evaluation.

### BLK-019 training-only preflight

`execution/run_streaming_training_preflight.py` is the current path for the reported 500,000-pair BLK-019 training sample. It requires Mudit's physical pair TSV and exact 2,500 Source-1 ID manifest, proves that the manifest is a subset of frozen train IDs, uses a SQLite pair spool to reject duplicates and retain complete groups, fits training-only TF-IDF from SQLite-deduplicated normalized corpora, stores 43 feature columns and labels in memory-mapped arrays, and fits L2 Logistic Regression locally. Its optional `--max-pairs` is an explicit hard guard only; no default pair limit truncates 500,000 rows.

It writes a training probability for every supplied candidate with exactly `source1_entity_id`, `candidate_entity_id`, `match_probability`, plus feature/model artifacts and a hash/resource report. It has no validation ID input, no evaluator call, no threshold sweep, and no validation F0.5 output. This is intentional: blocker Recall@200 `0.97807` and ceiling F0.5@200 `0.99318` are not matcher performance.

#### BLK-019-TRAIN-PREFLIGHT-v1 result (2026-09-26)

The handoff ZIP was preserved at `C:\Users\Ashank\Downloads\BLK_019_Ashank_Handoff.zip` and extracted only to ignored `.tmp/incoming/BLK-019_Ashank_Handoff`. Metadata hashes passed: candidate TSV `7a2232bda229508a4bf9219f147c6bc07c29419d3e27460710576cb3d2b8f108`; source-ID CSV `819516d26742cee17b8fbedf635470ca8f490ee9aae17e2dfc5a147229ee3dd8`. Metadata also matched Word Unigram name+address, country partitioning, K=200, commit `187d955`, and config hash `657f59868e58`.

All 2,500 supplied Source-1 IDs belong to Mudit's frozen train split; all candidate/source record references resolved; the TSV schema is `source1_entity_id`, `candidate_entity_id`, `score`, `rank`; and it contains exactly 500,000 unique pairs. Every source entity has exactly 200 candidates, hence zero zero-candidate groups in this preflight. Training-label candidate recall is 8,429/8,618 = `0.9780691576`, which rounds to the reported `0.97807`; 189 true pairs are blocking misses. The reported blocker ceiling F0.5 is not a matcher result.

The local training-only run fitted 43 features and L2 Logistic Regression (`C=1.0`, train-only `StandardScaler`, no class weighting or calibration). It has 8,429 positive / 491,571 negative candidate pairs (positive rate `0.016858`). TF-IDF corpus fitting took 133.23 s, feature extraction 82.58 s, Logistic fitting 1.28 s, and probability writing 77.79 s; total 305.77 s. Peak working set was 1.275 GiB and peak private commit 1.651 GiB; generated run artifacts used 223.6 MiB in addition to the existing 1.515 GiB record store. The 500,000-row score output has exactly the required three columns, all finite probabilities, and range `[1.42e-09, 0.999999997]`.

Saved local run: `.tmp/runs/BLK-019-TRAIN-PREFLIGHT-v1` (`features.pkl`, schema JSON, `logistic_regression.pkl`, memory-mapped feature/label intermediates, score TSV, and report). Reproduce with:

```powershell
$env:PYTHONPATH = '.'
python -B execution/run_streaming_training_preflight.py `
  --candidates .tmp/incoming/BLK-019_Ashank_Handoff/train_sample_candidate_pairs.tsv `
  --sample-source1-ids .tmp/incoming/BLK-019_Ashank_Handoff/train_sample_source_ids.csv `
  --frozen-train-source1-ids artifacts/validation_split/train_ids.csv `
  --ground-truth D:\Amazon_ML\student_resource\dataset\train\train_ground_truth.tsv `
  --store .tmp/train_records.sqlite `
  --run-dir .tmp/runs/BLK-019-TRAIN-PREFLIGHT-v1 `
  --candidate-version BLK-019-WORD-UNIGRAM-K200 `
  --generator-commit 187d955 --config-hash 657f59868e58 --batch-size 5000
```

This is not a validation comparison and has no tuned decision threshold. The remaining requirement is a same-version BLK-019 Word-Unigram candidate artifact covering every frozen validation Source-1 ID, including its source-ID manifest/zero-candidate accounting and pair-count/distribution/recall metadata.

### BLK-020 active training preflight and bounded validation performance diagnostic

BLK-020 supersedes BLK-019 for current matcher work: Word-Unigram TF-IDF over name plus address, country partitioning, K=200, generator commit `69a91f1`, configuration hash `657f59868e58`. Its candidate archive remains Git-excluded; metadata/sample/validation-gzip hashes are verified locally.

`BLK-020-TRAIN-PREFLIGHT-v1` is training-only and uses its supplied exact 2,500-ID frozen-train manifest. It verified 500,000 unique record-resolving pairs, complete K=200 groups, and zero zero-candidate entities. Candidate recall is 8,429/8,618 = `0.9780691576`; the fitted 43-feature L2 Logistic model has 8,429 positive and 491,571 negative pairs. Runtime was 203.29 s corpus+TF-IDF, 370.00 s features, 3.99 s fit, 353.91 s score write, 963.64 s total; peak working set was 1.274 GiB. Its 500,000 score rows are finite and exactly `source1_entity_id`, `candidate_entity_id`, `match_probability`. It has no threshold or validation F0.5.

`BLK-020-VAL-PERF-50K-v2` selected the first 250 complete K=200 frozen-validation groups in canonical gzip row order (50,000 pairs; no labels, threshold, or evaluator). Ten 5,000-pair group-safe batches scored in 35.547 s (1,406.6 pairs/s); end-to-end spool/score/assembly was 40.892 s (1,222.7 pairs/s), with 245.6 MiB peak working set. The scorer now accepts `.tsv.gz` and offers `--max-rank`, which requires a rank column and retains only `rank <= K`; this enables later K=50/K=100/K=200 sweeps without unpacking the full candidate TSV.

Future full runs remain gated on a fresh capacity calculation and must keep the rule baseline, avoid country one-hot/hard-coded country logic, remain slice-based under about 6 GiB, and verify any one-S1-per-candidate assignment premise against ground truth before use. No full validation is authorised by this diagnostic.

Mudit’s current contract supersedes the temporary-blocker next step above. The real benchmark uses supplied D2 candidates for both frozen partitions: frozen train Source-1 IDs feed D2 training candidates and train-only feature/Logistic fitting; frozen validation Source-1 IDs feed the **same D2 version/configuration** and Mudit’s evaluator. `TEMP-EXACT-NAME-v1` remains EXP-003A pipeline-diagnostic material only. The reported D2 Stage 1 Recall@200 of **96.29%** is a Stage 1 blocking result, not Stage 2 validation candidate recall or matching quality.

The D2 configuration to consume—not reimplement here—is country partitioning, business name plus address, character n-grams `(3,5)`, and its published top-200 policy. D2 candidates must be generated against the full appropriate S2/S3 corpus with no validation-label supplementation.

### Disk-backed scoring path

`src/entity_resolution/models/streaming.py` and `execution/run_streaming_matcher.py` add a scoring-only path for a supplied large candidate TSV. `CandidatePairSpool` stores pairs in SQLite, validates IDs and cross-file duplicates, checks every candidate Source-1 ID belongs to the provided frozen universe, and emits complete Source-1 groups in bounded batches even if input chunks interleave groups. A separate Source-1 table preserves explicit empty decisions for zero-candidate entities.

The prior 50,000-pair settings stay as explicit limits in the bounded diagnostic runner. The streaming path has no pair-count cap and will fail rather than split an oversized Source-1 group; it never silently truncates validation. Each run records input SHA-256 values, feature/model/store hashes, batch configuration, and per-part hashes in `score_state.json`. Checked completed parts can resume only with the same identity.

It produces two distinct outputs:

```text
# integration score output (exact schema)
source1_entity_id    candidate_entity_id    match_probability

# separate, uncalibrated diagnostic output
source1_entity_id    candidate_entity_id    raw_match_score    scoring_method
```

`assemble_score_output` concatenates score parts without loading all pairs. `build_threshold_decisions` uses SQLite and emits one row per frozen Source-1 ID while supporting zero, one, or many accepted matches. `evaluate_with_shared_evaluator` verifies that complete universe before invoking Mudit’s shared `Evaluator`; it does not replace or modify macro F0.5. Entity-level truth/decisions are materialised only for that evaluator invocation, not all pair features or scores.

`compute_disk_backed_error_counts` uses the candidate spool and threshold-decision database to keep the bounded-run definitions on large artifacts: a **blocking miss** is a true pair absent from candidates; a **matcher rejection** is a retrieved true pair below threshold; their sum is pairwise FN. It also reports TP/FP/FN, pairwise precision/recall, candidate recall, and empty singleton accuracy without materialising all pair labels.

### EXP-003A scalable-path parity diagnostic

This implementation was checked against the preserved 1,000-entity EXP-003A validation diagnostic only—not against D2. For 19,315 `TEMP-EXACT-NAME-v1` pairs, including 200 zero-candidate entities, it produced four group-safe 5,000-pair batches. Candidate IDs were identical, 43 feature columns were retained, maximum absolute Logistic-probability difference was `1.11e-16`, and threshold-0.55 decisions were identical. Shared-evaluator macro F0.5 remained `0.5123914614`; macro precision/recall remained `0.6566666667` / `0.3236630952`; singleton accuracy remained `0.96`.

The score stage (SQLite lookup, features, Logistic probability, rule raw score, and part writes) took 13.774 s = **1,402.3 pairs/s**. Windows working set was 164.9 MiB at start, 187.5 MiB at end, and 231.8 MiB peak. This is an OS process measurement—not Python-only tracing—and native-library allocation attribution remains incomplete. The final probability output used 941,209 bytes (48.7 bytes/pair), raw-score output 1,447,718 bytes (74.9 bytes/pair), score parts 2,389,310 bytes, and the SQLite spool 2,084,864 bytes. Keeping all resumability/assembled artifacts used 6,884,075 bytes (356.4 bytes/pair) on this small diagnostic.

The disk-backed count adapter reproduced the prior diagnostic accounting: 1,117/3,559 retrieved true pairs (candidate recall `0.3138522057`), 2,442 blocking misses, 88 matcher rejections, 1,029 TP, 32 FP, 2,530 FN, and 48/50 correct singleton decisions.

Illustratively at this measured score-stage throughput: 1M/5M/10M/25M pairs take about 0.20/0.99/1.98/4.95 hours. This excludes D2 spool construction and differs with text length, country mix, fan-out, filesystem behavior, and output retention; it is not a full-run promise. Start D2 at 5,000 pairs/batch on the 16 GiB host, inspect the first checkpoint/disk growth, then adjust only from measurements.

### Required D2 handoff and exact first command

Mudit/Aayush need to provide:

1. A deduplicated D2 validation TSV for **all** frozen validation IDs and either a D2 train TSV for all frozen train IDs or a declared reproducible subset of complete frozen-train groups. The train subset must be generated with the identical D2 version/configuration as validation and include at least `source1_entity_id`, `candidate_entity_id`. Optional retrieval rank/score fields are accepted separately.
2. One common generator commit/version and settings, source corpus hashes, candidate TSV checksums, row counts, candidate-count distribution, zero-candidate counts, and candidate recall/missing-true-pair reporting. The actual full validation D2 Stage 2 diagnostics are still pending.
3. For a first Logistic fit, a complete-group train subset is acceptable. Record its frozen-ID selection method, selected entity count, pair count, zero-candidate entity count, positive/negative pair counts, and positive-pair rate. It is a subset-trained model and must not be described as equivalent to all-frozen-train fitting. A validation-only file can only score an existing model as a labelled transfer experiment; the exact-name-trained LR must not be reported as the D2 model.
4. Mudit’s `EXP-003B-D2` registry/run ID and evaluator invocation. Select a fixed threshold from a documented entity-disjoint train-side procedure (or record any validation threshold tuning explicitly); never use validation labels for fitting/calibration.

After a fresh D2 train-only feature/model artifact is available, score all supplied D2 validation pairs as follows. This command generates no candidates and changes neither split nor evaluator.

First, run this bounded **diagnostic** fit after both files arrive. Training IDs use `stable_sha256_stratified_country_match_count_v1`: scan the entire frozen train population, form `country x true_match_count_group(0,1,2+)` strata using training labels, reserve one entity per populated stratum, then allocate the remaining quota proportionally and choose each stratum's lowest SHA-256 ranks of `seed + NUL + source1_entity_id`. Default seed is `20260926`; selection is independent of source-file order. The generated `training_selection.json` records seed, procedure, selected-ID SHA-256, population versus sample distributions, selected group counts, true-match-pair count, candidate pair/zero-candidate counts, and positive/negative pair class balance.

Every D2 candidate group for a selected training entity is retained and the runner fails if selected train or validation pairs exceed the explicit 50,000-pair guard. This includes zero-candidate train entities and all retrieved D2 non-matches (the available difficult lexical negatives). Confirm both positive and negative training pairs before fitting. Start at 1,000 train entities and increase the sample only if class balance, country/multiplicity coverage, or error coverage is inadequate. This first model is `subset-trained`, not equivalent to an all-frozen-train fit.

The selector was checked against the actual 1,765,456 frozen train entities using only supplied training labels and Source-1 country fields; no D2 candidates or models were created. Population strata were India: 39,554 / 37,910 / 628,945 and US: 59,056 / 57,311 / 942,680 for match-count groups 0 / 1 / 2+. With seed `20260926`, the selected-ID hash was `aa4b38718c00571feabe3b075ef69e63c0bb5aeed853edec6784571bb3778945`; the 1,000-entity sample was India 23 / 23 / 355 and US 34 / 33 / 532, totaling 57 singleton, 56 one-match, and 887 multi-match entities (3,480 true match pairs). Retrieved-positive count, difficult-negative count, and LR class balance remain pending the exact D2 training candidate artifact.

```powershell
$env:PYTHONPATH = '.'
python execution/run_bounded_matching_experiment.py `
  --train-candidates <d2_training_candidate_pairs.tsv> `
  --validation-candidates <d2_validation_candidate_pairs.tsv> `
  --ground-truth D:\Amazon_ML\student_resource\dataset\train\train_ground_truth.tsv `
  --store .tmp/train_records.sqlite `
  --run-dir .tmp/runs/EXP-003B-D2-PREFLIGHT `
  --experiment-id EXP-003B-D2-PREFLIGHT `
  --candidate-version <d2_generator_version> `
  --training-selection-seed 20260926 `
  --max-train-entities 1000 --max-validation-entities 1000 `
  --max-train-pairs 50000 --max-validation-pairs 50000 --batch-size 5000
```

Its validation output is a diagnostic only. Do not promote the prefix-trained model or any threshold tuned there to a full D2 comparison. After Mudit shares full-validation pair count and candidate distribution, calculate full-run time/disk from the measured D2 preflight rate and bytes/pair before launch. If feasible, use a documented entity-disjoint train-side fit/threshold procedure from the same D2 generator, then score all validation candidates—including zero-candidate Source-1 entities—with the full streaming command below.

```powershell
$env:PYTHONPATH = '.'
python execution/run_streaming_matcher.py `
  --candidates <d2_validation_candidate_pairs.tsv> `
  --source1-ids artifacts/validation_split/val_ids.csv `
  --store .tmp/train_records.sqlite `
  --feature-artifact <d2_train_only_features.pkl> `
  --model <d2_train_only_logistic.pkl> `
  --run-dir .tmp/runs/EXP-003B-D2 `
  --pair-batch-size 5000 `
  --threshold <train_selected_threshold> `
  --ground-truth D:\Amazon_ML\student_resource\dataset\train\train_ground_truth.tsv
```
