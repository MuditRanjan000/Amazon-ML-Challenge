# Experiment History

Append-only matching-workstream notes. Official comparisons require Mudit's frozen split and evaluator; diagnostic checks are labelled accordingly.

## 2026-09-26 — EXP-003A streaming parity diagnostic

- Hypothesis: disk-backed, complete-group batch scoring preserves the existing 43-feature Logistic output and decisions while removing the 50,000-pair full-materialisation constraint from validation scoring.
- Data: preserved `TEMP-EXACT-NAME-v1` 1,000-validation-entity EXP-003A diagnostic, 19,315 pairs, 200 zero-candidate entities. This is not D2 and not a full benchmark.
- Configuration: SQLite record lookup, `pair_batch_size=5000`, saved EXP-003A train-only feature artifact and L2 Logistic artifact, rule raw score retained separately, threshold 0.55 only for parity.
- Result: candidate IDs equal; 43 features retained; max probability delta `1.11e-16`; decisions equal; shared evaluator macro F0.5 `0.5123914614`, macro P/R `0.6566666667` / `0.3236630952`, singleton accuracy `0.96`.
- Disk-backed accounting reproduced 1,117/3,559 retrieved true pairs, 2,442 blocking misses, 88 matcher rejections, 1,029 TP, 32 FP, and 2,530 FN.
- Runtime/capacity: 13.774 s score stage = 1,402.3 pairs/s; Windows peak working set 231.8 MiB; retained spool/parts/assembled outputs 6,884,075 bytes (356.4 bytes/pair).
- Interpretation: software parity passed. No model-quality conclusion and no D2 score follows from this temporary-blocker diagnostic.
- Next: receive same-version D2 train and validation candidates plus generator/corpus/checksum/count/recall metadata; fit a new D2 train-only model before any real D2 comparison.

## 2026-09-26 — D2 preflight selection design (no candidate run)

- Procedure: `stable_sha256_stratified_country_match_count_v1`, default seed `20260926`; strata are training-record country x training ground-truth match-count group (0/1/2+), with one per populated stratum then proportional stable-hash allocation.
- Required record: selection method/seed, selected Source-1 ID hash, frozen-population and sample distributions, selected entity/group/true-pair counts, then D2 candidate-pair/zero-candidate/class-balance counts after Aayush/Mudit supply same-version D2 candidates.
- Boundary: intended only for the first complete-group subset-trained Logistic preflight. It is not an all-frozen-train model or a D2 benchmark result.
- Population verification (no candidates/model): actual frozen train sample with seed `20260926` has selected-ID SHA-256 `aa4b38718c00571feabe3b075ef69e63c0bb5aeed853edec6784571bb3778945`, 57/56/887 entities in match-count groups 0/1/2+, and 3,480 true match pairs. D2 retrieval/class balance is still pending.

## 2026-09-26 — BLK-019 training preflight intake (blocked before fit)

- Supersession: BLK-019 Word Unigram replaces the prior character-n-gram/D2 candidate plan. Configuration reported by Mudit: business name + address word unigrams, country partitioning, K=200, generator commit `187d955`, config hash `657f59868e58`.
- Reported artifact: 2,500 frozen-train entities, 500,000 pairs, Recall@200 `0.97807`, blocker ceiling F0.5@200 `0.99318`. These figures are not matcher metrics.
- Verification status: `artifacts/blocking/train_sample_candidate_pairs.tsv` is absent locally and in all fetched origin references; no schema/coverage/recall/model check can run. Required next input is the physical TSV plus its exact 2,500-ID manifest and metadata. The old 50,000-pair runner must not be used to silently truncate it.

## 2026-09-26 — BLK-019-TRAIN-PREFLIGHT-v1

- Inputs: verified handoff hashes candidate TSV `7a2232bda229508a4bf9219f147c6bc07c29419d3e27460710576cb3d2b8f108`, 2,500-ID CSV `819516d26742cee17b8fbedf635470ca8f490ee9aae17e2dfc5a147229ee3dd8`; Word Unigram, country partitioning, K=200, commit `187d955`, config `657f59868e58`.
- Integrity: 500,000 unique valid-record pairs; all supplied IDs within frozen train; 2,500 complete K=200 groups; zero zero-candidate groups. Training-label recall 8,429/8,618 = `0.9780691576` (reported rounded `0.97807`); 189 blocking misses.
- Model: 43-feature train-only TF-IDF + L2 Logistic Regression `C=1.0`, `StandardScaler`, no class weighting/calibration. Positive/negative pairs: 8,429 / 491,571 (1.6858% positive).
- Runtime/resources: corpus/feature/model/score 133.23 / 82.58 / 1.28 / 77.79 s; 305.77 s total; peak working/private 1.275 / 1.651 GiB; generated run artifacts 223.6 MiB plus existing 1.515 GiB record store.
- Output: 500,000 exact-schema probabilities, finite and in `[1.42e-09, 0.999999997]`.
- Boundary: training preflight only—no threshold, validation F0.5, calibration claim, or matcher-quality conclusion. Await same-version full frozen-validation BLK-019 candidates.

## 2026-09-26 — BLK-020-TRAIN-PREFLIGHT-v1

- Inputs/provenance: verified Word-Unigram name+address, country-partitioned K=200 sample from generator `69a91f1`, config `657f59868e58`. Sample TSV SHA-256 `7a2232bda229508a4bf9219f147c6bc07c29419d3e27460710576cb3d2b8f108`; exact 2,500-ID CSV SHA-256 `77583451c11f48558b5ba31c32248d563f036ac95da481d46d3fda38a9972b24`.
- Integrity/model: 500,000 unique valid-record pairs, 2,500 frozen-train IDs, complete K=200 groups, zero zero-candidate groups; 8,429/8,618 retrieved truth pairs (`0.9780691576` recall), 189 blocking misses. 43-feature train-only TF-IDF plus L2 Logistic Regression (`C=1.0`, no class weighting/calibration) has 8,429 positive / 491,571 negative pairs (`0.016858` positive rate).
- Runtime/resources: corpus+TF-IDF 203.29 s; features 370.00 s; LR fit 3.99 s; probability write 353.91 s; total 963.64 s. Peak working/private 1.274 / 1.649 GiB. Generated run artifacts 223.6 MiB plus the 1.515 GiB record store.
- Output/boundary: 500,000 finite probabilities in `[1.420573075258873e-09, 0.9999999970214757]`, exactly `source1_entity_id`, `candidate_entity_id`, `match_probability`. Training-only: no threshold, validation evaluator, or F0.5.

## 2026-09-26 — BLK-020-VAL-PERF-50K-v2

- Scope: diagnostic only, no labels/evaluator/threshold. Selected the first 250 contiguous complete K=200 Source-1 groups in canonical validation-gzip row order: 50,000 pairs, all frozen-validation members. Source gzip SHA-256 `b902a6d8c14ec8e7521b0820c916c0803499fe66c78170f3edbc99e40590b21a`; selected-ID SHA-256 `8e77af856352e196eab221192e1ecf6fd09a5339f67907c6ffacd97954c6bf41`.
- Result: ten complete-group 5,000-pair batches; score stage 35.547 s = 1,406.6 pairs/s; end-to-end including spool/assembly 40.892 s = 1,222.7 pairs/s. Peak working set 245.6 MiB and private commit 1.136 GiB. Rule raw scores remained separate.
- Output/readiness: 50,000 exact-schema finite Logistic probabilities in `[2.7849132716955035e-09, 0.9999999966109385]`. Candidate spooling now streams `.tsv.gz` and applies an explicit rank cutoff only when a rank column exists; this is test-covered and enables later K=50/K=100/K=200 sweeps without giant TSV unpacking.
