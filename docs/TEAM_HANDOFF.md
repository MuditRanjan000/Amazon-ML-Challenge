# Ashank Matching Handoff

## Inputs required from Mudit and Aayush

## Current blocker contract: BLK-020 Word Unigram

BLK-020 supersedes BLK-019 and all earlier D2/exact-name artifacts for the active matcher work. Its immutable provenance is Word-Unigram TF-IDF on `business_name + business_address`, country-partitioned, `K=200`, configuration hash `657f59868e58`, and generator commit `69a91f1`. The pair input schema remains `source1_entity_id`, `candidate_entity_id`, `score`, `rank`; retrieval score/rank are not model probabilities.

Mudit's frozen split and evaluator remain authoritative. Do not change the split, evaluator, or submission formatting. Report a full frozen-validation result only with complete validation-Source-1 coverage; all zero-candidate entities must remain present in decision evaluation.

### Next full-sweep requirements (recorded, not started)

- Run matching sweeps at `rank <= 100` and `rank <= 200` in addition to any `rank <= 50` diagnostic. Preserve the deterministic rule baseline alongside L2 Logistic Regression.
- Keep processing slice-based, bounded, and measured below approximately 6 GiB RAM. Do not add country one-hot features or country-specific hard rules: France is test-only and country must remain open-set.
- Do not enable a global one-Source-1-per-S2/S3 assignment unless a direct ground-truth audit verifies that constraint. Source 1 may still have multiple valid matches.
- Await Aayush's documented competition-feature columns/specification. The matcher must accept them as optional inputs without replacing BLK-020 or hard-coding country logic.
- For every future full result, record candidate artifact/hash, rank cutoff, feature/model version, threshold provenance, macro entity F0.5, precision/recall definition, singleton results, blocking misses, matcher rejections, FP/FN categories, candidate density, runtime, and memory.

## Historical BLK-019 Word-Unigram contract

BLK-019 supersedes every earlier character-n-gram/D2 candidate plan. Do not train or evaluate a reported matcher on those older artifacts. The only accepted preflight/future-validation candidate source is the same BLK-019 configuration:

- word unigram over `business_name + business_address`
- country partitioning
- `K=200`
- generator commit `187d955`
- configuration hash `657f59868e58`

The reported training-sample diagnostics are Recall@200 `0.97807` and ceiling F0.5@200 `0.99318`; these are blocker diagnostics, not achieved matcher results.

For the first local preflight, provide the immutable training TSV at `artifacts/blocking/train_sample_candidate_pairs.tsv` plus its source-ID manifest. For official evaluation, provide the matching full frozen-validation TSV generated from the same BLK-019 version/configuration:

- declared frozen-train sample Source-1 IDs -> `train_sample_candidate_pairs.tsv`
- all frozen validation Source-1 IDs -> `<blk019_validation_candidate_pairs.tsv>`

Each TSV must contain at least these tab-separated columns, with one unique row per pair:

```text
source1_entity_id    candidate_entity_id
```

Optional retrieval rank/similarity fields may follow those columns. They are preserved as blocker metadata; they are not matcher probabilities.

Attach one manifest or metadata record covering both files:

- BLK-019 generator commit/version and exact configuration above.
- Full S2/S3 corpus identity and input hashes.
- Candidate file SHA-256 hashes, Source-1 entity counts, pair counts, candidate-count distribution, and zero-candidate count.
- An explicit train-sample Source-1 ID file/hash. Pair TSVs cannot represent entities with zero candidates, so this is required to verify the reported 2,500-entity universe and zero-candidate count.
- Candidate recall / missing-true-pair report and its evaluation scope. The supplied BLK-019 train-sample Recall@200 is `0.97807`; it must be recomputed from supplied training labels before fitting.
- Confirmation that validation labels were not used to add or rescue validation candidates.
- Mudit's frozen-split manifest identifier, official evaluator invocation, and BLK-019 matcher-preflight registry/run identifier.

A complete-group, reproducibly selected frozen-train subset is sufficient for the **first** BLK-019 Logistic fit, provided it uses the identical generator version/configuration as validation. The provided 2,500-ID universe—not Ashank's previous 1,000-ID seed sample—must be used. Record its exact IDs/hash, entity/pair count, zero-candidate count, and training positive/negative pair counts/rate. Label its model `subset-trained`; it is not equivalent to fitting all frozen train entities. A validation-only BLK-019 artifact can score an existing model only as a transfer diagnostic; it cannot support the reported matcher comparison.

## Outputs supplied by Ashank's subsystem

- Saved train-only 43-feature preprocessing artifact and feature schema.
- Saved regularized Logistic Regression artifact and configuration; `predict_proba` is a model probability estimate, with calibration status stated explicitly.
- Pairwise output for every supplied candidate, exactly:

```text
source1_entity_id    candidate_entity_id    match_probability
```

- Separate deterministic `raw_match_score` artifact, never relabelled as a probability.
- A fixed-threshold decision artifact with every frozen Source-1 entity present, including empty predictions for zero-candidate entities. It permits zero, one, or many matches.
- Hashes/configuration/checkpoints, candidate recall and blocking-miss accounting, model rejection accounting, shared-evaluator result, and representative errors.

## BLK-019 preflight and validation gate

First execute the guarded local BLK-019 training preflight once the TSV and its 2,500-ID manifest are physically present. Its 500,000 pairs exceed the old bounded-run 50,000-pair guard, so the preflight must use the disk-backed/batched training path and may not silently truncate. Verify schema, split membership, duplicate pairs, candidate distribution, candidate recall, retrieved positives, difficult negatives, feature validity, and class balance before fitting. Do not tune a threshold or report validation F0.5 on these training entities. Once Mudit provides the BLK-019 validation pair count and candidate distribution, calculate measured local full-run time/disk before launching all frozen validation entities.

Do not invoke the earlier D2 bounded or scoring commands: their candidate version/configuration is superseded. After physical BLK-019 artifacts arrive, use the training-only preflight below. It spools and de-duplicates pairs on disk, fits TF-IDF from disk-backed unique normalized training texts, stores 43 features/labels in memory-mapped arrays, fits L2 Logistic Regression, and writes pairwise training probabilities. No validation threshold or metric is part of this command.

```powershell
$env:PYTHONPATH = '.'
python execution/run_streaming_training_preflight.py `
  --candidates artifacts/blocking/train_sample_candidate_pairs.tsv `
  --sample-source1-ids <mudits_exact_2500_train_source1_ids.csv> `
  --frozen-train-source1-ids artifacts/validation_split/train_ids.csv `
  --ground-truth D:\Amazon_ML\student_resource\dataset\train\train_ground_truth.tsv `
  --store .tmp/train_records.sqlite `
  --run-dir .tmp/runs/BLK-019-TRAIN-PREFLIGHT `
  --candidate-version BLK-019-WORD-UNIGRAM `
  --generator-commit 187d955 --config-hash 657f59868e58 `
  --batch-size 5000
```

The resulting `training_logistic_scores.tsv` has exactly `source1_entity_id`, `candidate_entity_id`, and `match_probability`; `training_preflight_report.json` records file hashes, 2,500-ID membership, candidate recall, retrieved positives, difficult negatives, feature count, model setting, runtime, and process memory.
