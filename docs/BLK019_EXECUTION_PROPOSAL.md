# BLK-019 Full-Scale Execution Proposal

Status: proposal only. No AWS resource has been provisioned, no challenge data has been uploaded, and no credits have been consumed.

## Measured basis

The verified 500,000-pair `BLK-019-TRAIN-PREFLIGHT-v1` measured the following local stages. The saved train-only feature artifact and L2 Logistic Regression mean the first three are already complete for this model.

| Stage | Time | Use |
| --- | ---: | --- |
| TF-IDF corpus fit | 133.23 s | one-time training fit |
| Training feature materialisation | 82.58 s | one-time training preflight |
| Logistic Regression fit | 1.28 s | one-time training fit |
| Transform, `predict_proba`, exact probability write | 77.79 s | per-pair score rate: 6,427.3 pairs/s |
| Residual spool/setup/report overhead | 10.89 s | not separately instrumented |

Observed preflight densities: candidate TSV 39.90 bytes/pair, SQLite candidate spool 109.73 bytes/pair, and three-column probability score TSV 48.90 bytes/pair. They are provisional and must be replaced by the actual validation/test file measurements.

## Provisional sequential capacity

| Run | Pairs | Score-only | Score plus measured residual | 25% runtime budget | Transient peak disk | Retained disk after verified assembly |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Frozen validation | about 88M | 3.80 h | 4.33 h | 5.41 h | 20.28 GiB | 16.27 GiB |
| Test inference | about 347M | 15.00 h | 17.09 h | 21.36 h | 79.96 GiB | 64.16 GiB |

Transient disk is candidate TSV + SQLite spool + checked score parts + assembled score TSV, with `--no-rule-raw-scores`. Retained disk is candidate TSV + spool + assembled score after part hashes and final SHA-256 are verified. Add the existing 1.515-GiB record store and retain 5 GiB of journal/OS headroom. `D:` currently has 126.65 GiB free: validation is feasible locally on this estimate; test fits only when staged sequentially and only if the measured final density passes this gate.

| Artifact | Validation | Test |
| --- | ---: | ---: |
| Candidate input TSV | 3.51 GB | 13.84 GB |
| Pairwise probability TSV | 4.30 GB | 16.97 GB |
| SQLite spool | 9.66 GB | 38.08 GB |

## Full-validation sequence

1. Receive same-version BLK-019 validation pairs, a complete frozen-validation Source-1 manifest, metadata, file SHA-256, pair count/distribution, zero-candidate accounting, and candidate recall. Verify all of them before scoring.
2. Recalculate this table from the verified input. Use a new run directory, `--pair-batch-size 5000`, and `--no-rule-raw-scores`. Preserve every frozen validation Source-1 entity, including zero-candidate entities. The pairwise output is exactly:

   ```text
   source1_entity_id    candidate_entity_id    match_probability
   ```

3. Verify part hashes, assembled-score hash/row count/schema/probability range, and resume-state identity. Only then perform permitted train-side threshold selection and invoke Mudit's frozen evaluator. Never fit or calibrate on validation labels.
4. Do not delete files during scoring. After Mudit verifies downstream completion and the final score SHA-256 is recorded, score parts may be deleted; the spool may then be deleted because it is rebuilt from the immutable candidate TSV and Source-1 manifest. Keep the candidate set for the required audit/package. Archive inactive TSVs with gzip or zstd only after retaining their uncompressed hashes and manifests; compression ratio is not estimated here.

The training preflight corpus/memory maps/training score are not needed to load the saved model, but remain preserved for now. Keep the original ZIP, verified inputs, feature artifact/schema, model, and preflight report.

## Test prerequisites

Test scoring additionally needs a test-only SQLite record store from supplied test S1/S2/S3 records, final BLK-019 test candidates plus manifest, and a threshold selected without test labels. Mudit owns final candidate-package and submission aggregation. This proposal does not authorise test inference.

## Local versus AWS contingency

Local is preferred for validation after the real-file capacity gate: the machine has 16 GiB RAM and already fitted the model at 1.275 GiB peak working set. Retain a 5,000-pair batch and log actual inference working set.

To avoid duplicate infrastructure, use Mudit's existing private candidate-generation region, storage, and IAM setup if available. If that setup is in `us-east-1`, the concrete fallback is one Linux On-Demand `c7i.2xlarge` (8 vCPU, 16 GiB RAM) with a 150-GB gp3 EBS volume. Planning rates: USD 0.357/hour for compute and USD 0.08/GB-month for gp3. AWS speed is unbenchmarked, so the local runtime budget is retained.

| AWS operation | Runtime budget | EC2 estimate | 150-GB gp3 estimate | Planning ceiling before tax/transfer |
| --- | ---: | ---: | ---: | ---: |
| Validation only | 5.41 h | USD 1.94 | about USD 0.09 | USD 2.03 |
| Test only | 21.36 h | USD 7.63 | about USD 0.35 | USD 7.98 |
| Both, sequential | 26.77 h | USD 9.56 | about USD 0.44 | USD 10.00 |

These are planning estimates, not quotes. Confirm the selected-region price, tax, S3 request, and egress terms at approval time. Do not use Spot for the first full checkpointed run. If candidate files and record stores already exist in Mudit's private S3 location, use that in-region copy and return only agreed outputs; otherwise do not upload the provisional 18.9 GB of validation+test candidates plus a 1.515-GiB record store without a separate transfer approval.

Before any cloud action, obtain from Mudit: region, object versions/URIs or instance/volume identity, IAM role, data-owner approval, and retention location. Shutdown plan: hash and hand off agreed outputs, terminate the instance, delete its non-shared EBS volume, and verify no paid resource remains. Set a USD 15 budget alert as a guardrail.

No approval is requested yet. If local validation fails the real-file gate, request separate approval for this exact validation run: same Mudit region/storage, one `c7i.2xlarge`, 150-GB gp3, maximum 6 hours, and USD 3 planning cap. A separate test approval would be required later: maximum 24 hours and USD 12 planning cap.

## Independent saved-artifact check

In a fresh process, the saved `features.pkl` and `logistic_regression.pkl` loaded with the SQLite record store and scored a 12-pair held-out software fixture. Its output has exactly the required three columns, 12 finite probabilities in `[0,1]`, and SHA-256 `6adab0d75bc5b66b9d25a4296d93135ebd473af77138a995465b77141cdc1874`. It used no labels and yielded no threshold, metric, or validation claim.
