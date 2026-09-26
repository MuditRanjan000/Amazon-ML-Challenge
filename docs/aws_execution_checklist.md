# AWS D2 Execution Checklist

## 1. AWS Infrastructure
* **Recommended AMI**: Ubuntu 24.04 LTS (HVM) or Amazon Linux 2023 (x86_64)
* **Recommended Instance**: `r5.4xlarge` / `r6i.4xlarge` (128 GiB). `--in-memory` peaks at ~25 GB; the 32 GiB `c5.4xlarge` is too tight.
* **vCPU**: 16
* **RAM**: 128 GiB
* **EBS Storage**: 100 GB gp3 volume (to hold dataset, virtual environment, and cache outputs)
* **Security Group Requirements**: Allow SSH (Port 22) inbound from your specific IP only. All outbound traffic allowed.
* **IAM Permissions Required**: Instance Profile with `s3:GetObject` and `s3:PutObject` for the designated `amazon-ml-challenge-2026-data` bucket.

## 2. Repository Setup
* **Exact Git Branch**: `feature/mudit-submission`
* **Clone Commands**:
  ```bash
  git clone https://github.com/MuditRanjan000/Amazon-ML-Challenge.git
  cd Amazon-ML-Challenge
  git fetch origin
  git checkout feature/mudit-submission
  ```
* **Commit Verification**:
  ```bash
  git log -1
  # Verify you are on the latest commit before running
  ```
* **Environment Setup & Dependencies**:
  * **Python Version**: Python 3.10+
  ```bash
  sudo apt-get update && sudo apt-get install -y git python3-venv python3-pip unzip awscli build-essential python3-dev
  python3 -m venv .venv
  source .venv/bin/activate
  pip install -r requirements.txt
  pip install -e .
  ```

## 3. Dataset Setup
* **Storage Location on EC2**: The dataset should be stored outside the repository directory or inside an ignored directory. Example: `~/dataset`.
* **ER_DATA_DIR Configuration**: You must set `ER_DATA_DIR` to the path where the dataset is extracted. If omitted, `config.py` defaults to looking for `6ab10eb3b23ba_student_resource/student_resource/dataset` at the repository root.
* **S3 Requirement**: Yes, S3 is required to transfer the dataset securely to the EC2 instance without checking it into version control.
* **Recommended Transfer Method**:
  ```bash
  aws s3 cp s3://amazon-ml-challenge-2026-data/dataset.zip /tmp/dataset.zip
  unzip /tmp/dataset.zip -d ~/dataset
  export ER_DATA_DIR=~/dataset
  ```

## 4. Validation Split Control
The frozen split is a competition control artifact. Do NOT regenerate validation splits on AWS. Before running D2, you must verify the frozen split artifacts exist and match the manifest.
* **Ensure these files are explicitly transferred to the EC2 instance**:
  * `artifacts/validation_split/train_ids.csv`
  * `artifacts/validation_split/val_ids.csv`
  * `artifacts/validation_split/manifest.json` (also mapped via config)
* **Run the Preflight Verification Command**:
  ```bash
  python -m src.entity_resolution.data.validation_split --verify
  ```
  *This will verify file existence, hashes, counts, and overlap. Abort the run if this fails!*

## 5. Execution Procedure
* **Required Environment Variables**:
  ```bash
  export ER_DATA_DIR=~/dataset
  export ER_N_JOBS=16
  ```
* **Execution Command**:
  ```bash
  python scripts/aws/run_d2_blocking.py --split trainval --sample 5000 --in-memory   # 1. timing check, extrapolate
  python scripts/aws/run_d2_blocking.py --split trainval --in-memory                 # 2. train + validation TSVs
  python scripts/aws/run_d2_blocking.py --split test --in-memory                     # 3. test TSV
  ```
* **Expected Runtime**: index build ~7 min. Query throughput measured at **~10 S1/s on the laptop** (char (3,5) is expensive): train+val 2.2M S1 ~60 h, test 1.73M ~48 h at that rate. Measure AWS throughput with step 1 before committing to a full run. It is memory-bandwidth bound, so fan out S1 across instances rather than adding cores.

## 6. Monitoring
While the script runs, open a second SSH session to monitor:
* **CPU & RAM Usage**: `htop`
* **Disk Usage**: `watch -n 10 df -h` (Ensure the EBS volume isn't filling up from TFIDF caching)
* **Process Status**: `ps aux | grep python`
* **Logs**: The script outputs directly to standard output. Run via `tmux` or redirect to a file if you want to persist the stdout stream: `python scripts/aws/run_d2_blocking.py > blocking_run.log 2>&1 &` then `tail -f blocking_run.log`.

## 7. Output Verification
After completion, verify `artifacts/blocking/{train,validation,test}_candidate_pairs.tsv` and the per-split record (config_sha, generator_commit, tsv_sha256, metrics) in `artifacts/blocking/blocking_metadata.json`. config_sha must be identical across splits.
* **File Verification commands**:
  ```bash
  # File size
  ls -lh artifacts/blocking/validation_candidate_pairs.tsv
  # Row count
  wc -l artifacts/blocking/validation_candidate_pairs.tsv
  # Checksum
  sha256sum artifacts/blocking/validation_candidate_pairs.tsv
  ```
* **Blocking Metadata Updates**: 
  Ensure `artifacts/blocking/blocking_metadata.json` is accurately updated with:
  * `experiment_id`
  * `blocker` version
  * configuration parameters
  * `git commit hash` (generator_commit)
  * generation timestamp

## 8. Cost Tracking
Update `docs/aws_cost_tracking.md` immediately upon terminating the instance.
