# AWS Blocking Runbook (BLK-020, frozen word-unigram blocker)

Use this to (re)generate blocking candidates and per-record competition stats on **any** AWS account. It covers test, or train+val if they ever need regenerating. One command runs a whole split: N self-terminating workers + 1 merge instance. The worker code is the commit you launch from.

## What a run produces
`s3://<bucket>/run-<commit>-<split>/final/`:
| File | Content |
|---|---|
| `<split>_candidate_pairs.tsv` (+ `.gz`) | `source1_entity_id, candidate_entity_id, score, rank`, top-200 per S1 (train+val runs write both `train_` and `validation_`) |
| `record_stats_test.parquet` / `record_stats_trainval.parquet` | Per S2/S3 record: `best_score, second_score, n_s1, best_s1`, over **all** S1 of the split (train+val together) |
| `blocking_metadata.json` | config + config_sha, generator commit, distributions, recall (train/val), per-country counts, tsv_sha256 |
| `SHA256SUMS` | sha256 of the TSVs and stats parquet |
| `train_sample_candidate_pairs.tsv`, `train_sample_s1_ids.csv` | trainval runs only: the seed-42 2,500-S1 sample |

**Reproducibility check:** the blocker code and config (`657f59868e58`) are frozen, so a regenerated TSV must be **byte-identical** to the original.
| File | Expected sha256 |
|---|---|
| test | `6e11c1ec60a8f73f47b214211002928f66eac08f19c60a3be73bc682bced5925` |
| train | `e3bd1d765910f66dee555a3c4f10c02ea77ec3b6efbfee64ee0ef34f134330e2` |
| validation | `906fc126ece79a5665f2037c2127855862648c87bdb5f1133b5d55ef7dddf87c` |
| `record_stats_trainval.parquet` | file `3ddf045b…`; content fingerprint `e78f0dd0…` = `sha256(pd.util.hash_pandas_object(df, index=False).to_numpy().tobytes())` |

## Prerequisites (your machine)
- **AWS CLI:** v1 (`pip install awscli`) or v2, with credentials for the target account (`aws configure`, never in chat/git). If the CLI isn't on PATH, set `AWS_CLI`.
- **Repo:** a clean, **pushed** commit containing this runbook. `pip install -r requirements.txt && pip install -e .`, and `ER_DATA_DIR` set to the dataset (needed only for `bundle`).
- **Shell:** Git Bash on Windows works; the script handles `file://` paths and never passes `/dev/...` arguments.

## Commands
```bash
B=<your-private-bucket-name>        # globally unique; the region defaults to ap-south-1 (--region to change)

scripts/aws/launch_fanout.sh setup  --bucket $B             # once: bucket (public access blocked), IAM role+profile, no-inbound SG
scripts/aws/launch_fanout.sh bundle --bucket $B --split test   # once: ~750 MB parquet bundle of test S1/S2/S3
scripts/aws/launch_fanout.sh run    --bucket $B --split test   # ~25 min: 48 workers + merge -> test candidates + record_stats_test
scripts/aws/launch_fanout.sh status --bucket $B --split test   # progress / outputs / instances still running
# train+val (only if needed): bundle --split train, then run --split trainval
```
- **Options:**
  - `--workers N` (48); `--spot-from I` (parts ≥ I launch as spot, default 33; use a number > N for all on-demand).
  - `--type T` (m7i-flex.large); `--region R`; `--dry-run` prints every AWS call and launches nothing.
- **Sizing:**
  - vCPU quota needed ≈ 2 × on-demand workers (on-demand quota) and 2 × spot workers (spot quota).
  - Any type with ≥ 8 GB RAM works (a worker needs ~7 GB + 8 GB swap).
  - Free-plan accounts can launch only free-tier-eligible types (`m7i-flex.large` is the best).
  - Shared team account (655285961749, 32 on-demand + 32 spot vCPU): `--workers 32 --spot-from 17`.
  - The merge uploads the candidate files (+ `status/candidates.DONE`) **before** it computes the record stats, so a stats failure never holds back the candidates.
- **Measured, 48 × m7i-flex.large in ap-south-1:**
  - Test: 22 min wall, ≈ $0.81.
  - Train+val: ~25 min, ≈ $1.5.
  - Workers take 10–12 min (3 min index build + queries at ~80 S1/s per 2 vCPU). The merge takes 5 min, plus an estimated ~7–10 min for the record stats (laptop-measured, not yet run on AWS).

## After a run
1. `status` shows `DONE` for all parts and the merge, and 0 running instances. Every instance self-terminates (on success, failure, or a 2 h cap). `run` retries failed or interrupted parts once, on-demand.
2. Compare `final/SHA256SUMS` with the expected hashes above.
3. Download: `aws s3 cp s3://$B/run-<commit>-test/final/ . --recursive --exclude "*.tsv" --include "*.tsv.gz" --include "*.parquet" --include "*.json" --include "SHA256SUMS"`
4. Share with teammates through the bucket (IAM access) or `aws s3 presign <s3-uri> --expires-in 604800`. Presigned links die if the signing key is rotated.

## Without AWS
- **Record stats**, if you already have the candidate files (~7–8 min on a laptop):
  - test: `python execution/candidate_record_stats.py --split test --inputs test_candidate_pairs.tsv.gz --out record_stats_test.parquet`
  - train+val: `--split train --inputs train_candidate_pairs.tsv.gz validation_candidate_pairs.tsv.gz --out record_stats_trainval.parquet`
- **Test candidates on laptops:**
  - `python scripts/aws/run_d2_blocking.py --split test --blocker word --in-memory --part i/n` on each machine, then `--merge n` after collecting `artifacts/blocking/parts/`.
  - At ~100 S1/s per laptop, n=4 laptops take ~1.3 h each.

## Troubleshooting
| Symptom | Cause / fix |
|---|---|
| `VcpuLimitExceeded` | Quota reached, or terminated instances still count for a minute. `run` retries automatically; reduce `--workers` or request a quota increase. |
| `not eligible for Free Tier` | Free-plan account: use `--type m7i-flex.large` (or another free-tier-eligible type). |
| Part log ends in `Killed` | Out of memory: use ≥ 8 GB RAM (swap is enabled by the user-data). |
| `does not match the frozen manifest` | Split CSVs checked out with LF. `.gitattributes` pins them to CRLF; re-clone. |
| Logs | `s3://$B/run-<commit>-<split>/logs/partNN.log`, `merge.log` (uploaded every 60 s) |

## Not in this kit
Matcher scoring (Ashank's pipeline) runs separately. `scripts/aws/blocking_instance.sh` is the template if it needs the same fan-out: swap the part command for the scorer on `--part i/n` slices.
