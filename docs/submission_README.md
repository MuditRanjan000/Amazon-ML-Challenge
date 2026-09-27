# Business Entity Resolution — reproduction guide

> This file becomes `code/business_entity_resolution/README.md` in the submission zip.
> Final run: **ensemble r3 + r4, K=200, per-S1 cap 5 (France 3)** (validation F0.5 0.9776 before the cap; leaderboard [LB_CAP]). Code snapshot: commit `c62a969`. Provenance: stage-1 models trained at `c0149fb` (r3) and `2b824df` (r4), test scored at `780ad8c`, decide run from the `v3run3` code snapshot, cap applied by `variants` at `d2db812`; later commits only add options (seeds, ensembles, orphan simulation) and leave these steps unchanged.
> The trained models are included in `models/` (`SHA256SUMS` there), so steps 5–6 reproduce the output without retraining. Each model file stores its feature list; retraining at this snapshot reproduces the method, not bit-identical models.

## 1. Submission package (official README, "Final Submission Package")

```
<team_name>_submission.zip
├── output/
│   ├── matching_results.tsv        # final matches (the file uploaded to the leaderboard)
│   └── candidate_pairs.tsv         # the exact candidate set the final model scored (rank <= K)
├── code/
│   └── business_entity_resolution/
│       ├── src/                    # all source: entity_resolution/ package + execution/ + scripts/ entry points
│       ├── README.md               # this file
│       └── requirements.txt        # pinned versions (copy of the repo's requirements.txt)
└── Documentation_template.md       # filled from docs/Documentation_final.md (keep the template's file name)
```

Rules the two TSVs must satisfy (checked by `utils/validate_submission.py` from the student resource):
- tab-separated, headers `source1_entity_id	matched_entity_ids` and `source1_entity_id	candidate_entity_ids`;
- exactly one row per test S1 (1,732,544), France included; empty list for singletons / no candidates;
- only S2-/S3- IDs that exist in test, no duplicates within a list;
- every matched ID must also be a candidate of that S1.

`candidate_pairs.tsv` at K=200 is 2.2 GB+ and too big for the official validator on a 16 GB machine. We run the official validator with `--check-ids` on `matching_results.tsv`, and stream-check the candidate file (`execution/v3_pipeline.py: check_candidate_file`: every S1 once, no duplicates, matches ⊆ candidates).

Assembling `src/`: copy `src/entity_resolution/`, `execution/`, `scripts/`, `tests/`, `pyproject.toml`, `requirements.txt` from the repo at the final commit (`git archive c62a969`), plus the trained models in `models/`. No data, no `output/`.

## 2. Environment

- Python 3.13, `python -m venv .venv`, then `pip install -r requirements.txt && pip install -e .`.
- All dependencies are BSD/MIT/Apache-2.0/ISC (numpy, pandas, pyarrow, scipy, scikit-learn, sparse-dot-topn, RapidFuzz, anyascii, joblib, psutil). No pretrained model, no network access, no external data: every statistic (IDF, record competition, token rarity) is computed from the provided files of the same split.
- Set `ER_DATA_DIR` to the `dataset/` folder (holding `train/` and `test/`); outputs go to `ER_OUTPUT_DIR` (default `<repo>/output`). All paths below are relative to the repo root.
- The frozen validation split (20% of train S1, seed 42) is created by `entity_resolution.data.validation_split.create_validation_split`; verify it against `experiments/results/validation_split_manifest.json` (train 1,765,456 / val 441,365 IDs).
- Every run logs commit, config, metrics, runtime and peak RSS to `experiments/results/experiments.jsonl`.

## 3. Pipeline, end to end

```
S1+S2+S3 ──► 1. blocking (BLK-020) ──► 2. record competition stats ──► 3. normalize
        ──► 4. stage-1 train ──► 5. stage-1 score (train / val / test) ──► 6. decide (stage 2 + rule, val-tuned)
        ──► [7. variants] ──► output/matching_results.tsv + output/candidate_pairs.tsv ──► 8. validate
```

Laptop (16 GB) is enough for every step run per part; the full runs were fanned out on AWS (section 4) because of wall time.

### 1. Blocking — country-partitioned word TF-IDF, top-200 per S1 (config sha `657f59868e58`)
```bash
python scripts/aws/run_d2_blocking.py --split trainval --blocker word --in-memory   # train + validation candidates
python scripts/aws/run_d2_blocking.py --split test     --blocker word --in-memory   # test candidates
# in slices: add --part i/n per worker, then: python scripts/aws/run_d2_blocking.py --merge n
```
Outputs `artifacts/blocking/{train,validation,test}_candidate_pairs.tsv` (`source1_entity_id, candidate_entity_id, score, rank`, 200 rows per S1, S1-contiguous) and `blocking_metadata.json`. Expected test sha256 `6e11c1ec60a8f73f47b214211002928f66eac08f19c60a3be73bc682bced5925` (byte-identical across two independent AWS runs).

### 2. Record competition stats (per S2/S3 record: best / second score, number of S1 retrieving it, best S1)
```bash
python execution/candidate_record_stats.py --split train \
  --inputs artifacts/blocking/train_candidate_pairs.tsv artifacts/blocking/validation_candidate_pairs.tsv \
  --out artifacts/blocking/record_stats_trainval.parquet
python execution/candidate_record_stats.py --split test \
  --inputs artifacts/blocking/test_candidate_pairs.tsv --out artifacts/blocking/record_stats_test.parquet
```
Train and validation S1 share one pool, so their stats are computed together, exactly as all test S1 share the test pool.

### 3. Normalize (names, addresses, keys, token rarity)
```bash
python execution/v3_pipeline.py normalize --split train
python execution/v3_pipeline.py normalize --split test
```
→ `output/v3/norm_{train,test}.parquet`, `output/v3/tokdf_{train,test}.parquet`.

### 4. Stage-1 pair models (HistGradientBoosting, trained on train-split S1 only)
Let `TR=artifacts/blocking/train_candidate_pairs.tsv.gz`, `VA=artifacts/blocking/validation_candidate_pairs.tsv.gz`, `ST=artifacts/blocking/record_stats_trainval.parquet`.
```bash
# r3: 200k S1, K=100, 500 iterations, seed 42
python execution/v3_pipeline.py train --train-cands $TR --val-cands $VA --stats $ST --n-s1 200000 --K 100 --iters 500 --tag _r3
# r4: 150k S1, K=200, rank>100 negatives subsampled to 25% (reweighted), 500 iterations, seed 42
python execution/v3_pipeline.py train --train-cands $TR --val-cands $VA --stats $ST --n-s1 150000 --K 200 --iters 500 \
  --deep-neg-keep 0.25 --tag _r4

```
→ `output/v3/m1_r3.joblib`, `output/v3/m1_r4.joblib` (each stores its training S1 sample; stage 2 never trains on them).

### 5. Stage-1 scoring (several models → averaged `p`, members kept as `p_m0, p_m1, …`)
Candidate files are S1-contiguous with 200 rows per S1, so `--part i/n` is a row range. Run every part `i = 1..n`:
```bash
M=output/v3/m1_r3.joblib,output/v3/m1_r4.joblib
python execution/v3_pipeline.py score --cands $TR --split train --name train --n-s1 1765456 --part i/24 --K 200 --model $M --stats $ST --out output/v3/scored
python execution/v3_pipeline.py score --cands $VA --split train --name val   --n-s1 441365  --part i/8  --K 200 --model $M --stats $ST --out output/v3/scored
python execution/v3_pipeline.py score --cands artifacts/blocking/test_candidate_pairs.tsv.gz --split test --name test --n-s1 1732544 \
  --part i/32 --K 200 --model $M --stats artifacts/blocking/record_stats_test.parquet --out output/v3/scored
```
Only pairs with stage-1 `p >= 0.01` are kept for stage 2 and the decision rule (all rank ≤ K pairs are still the scored candidate set).

### 6. Decide: stage-2 context model + decision rule tuned on the frozen validation split, then applied to test
```bash
python execution/v3_pipeline.py decide --scored output/v3/scored --m1 $M --K 200 --tag _r4ensk200 \
  --s2-n 1500000 --s2-iters 400 \
  --test-cands artifacts/blocking/test_candidate_pairs.tsv.gz --out output/submissions/V3_r4ens_k200
```
Stage 2 is fit on train S1 outside the stage-1 samples; the rule (probability threshold or per-S1 expected-F0.5 selection, with or without one owner per record) is grid-searched on validation and confirmed with the official evaluator. Writes `matching_results.tsv` and `candidate_pairs.tsv` (all rank ≤ K pairs = the scored set) and validates them.

### 7. Test-time variants: per-country match cap (final) and logit shifts of the stage-2 probability
```bash
python execution/v3_pipeline.py variants --scored output/v3/scored --m2 output/v3/m2_r4ensk200.joblib --K 200 \
  --out output/submissions/variants --variants '{"cap53": {"shift": 0, "cap": {"US": 5, "India": 5, "France": 3}}}'   # final: output/submissions/variants/cap53/matching_results.tsv
```
Shift-robustness check on the shifted validation (section 5 of the documentation):
```bash
python execution/v3_pipeline.py decide --scored output/v3/scored --m1 $M --K 200 --drop-s1 output/v3/dropped_s1.txt \
  --eval-m2 output/v3/m2_r4ensk200.joblib --tag _shifteval
```

### 8. Validate
```bash
python scripts/validate_submission.py --submission output/submissions/variants/cap53/matching_results.tsv \
  --candidate output/submissions/V3_r4ens_k200/candidate_pairs.tsv --check-ids   # official validator (needs > 16 GB at K=200)
```
Expected sha256: `matching_results.tsv` `653ac79d418b576121456cf758c7af3e18ed6479f99656d615207fd11eb05113` (after the cap), `candidate_pairs.tsv` `6e8e7a19baea8cd1663fab8c688211cd5460f097f297a79c190ce7322f6236c2`.

## 4. The same steps on AWS (what we actually ran)

Credits-only account, `ap-south-1`, every instance self-terminating; bucket / IAM role / security group are made once by `scripts/aws/launch_fanout.sh setup`.
```bash
scripts/aws/launch_fanout.sh run --bucket $B --split test --workers 32            # step 1 (+ step 2) for test: 32 workers + merge
scripts/aws/v3_fanout.sh prep   --bucket $B --run $RUN                             # code (git HEAD) + norm tables + m1 -> s3://$B/$RUN/
scripts/aws/v3_fanout.sh launch --bucket $B --run $RUN --name train --n 24 --K 200 # step 5, likewise --name val --n 8 / test --n 32
scripts/aws/v3_fanout.sh job    --bucket $B --run $RUN --name decide --cmd-file job.sh --type r7i.2xlarge   # steps 4 / 6 / 7 on one box
scripts/aws/v3_fanout.sh status --bucket $B --run $RUN
```
`job.sh` holds the exact commands of steps 4/6/7 (as above, with paths under `/opt/w`). The box downloads code at the launched commit, so push first.

## 5. Tests
`python -m pytest -q` (normalizer, blocking parts/merge byte-identity, record stats, context/owner rule, expected-F0.5 rule, orphan drop, evaluator, submission writer).
