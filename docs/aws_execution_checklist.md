# AWS D2 Execution Checklist

## 1. AWS Infrastructure
* **Recommended AMI**: Ubuntu 24.04 LTS (HVM) or Amazon Linux 2023 (x86_64)
* **Recommended Instance**: `c5.4xlarge` (or `c6a.4xlarge`)
* **vCPU**: 16
* **RAM**: 32 GiB
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
  sudo apt-get update && sudo apt-get install -y python3-venv python3-pip
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

## 4. Execution Procedure
* **Required Environment Variables**:
  ```bash
  export ER_DATA_DIR=~/dataset
  export ER_N_JOBS=16
  ```
* **Execution Command**:
  ```bash
  python scripts/aws/run_d2_blocking.py
  ```
* **Expected Runtime**: 10 - 15 minutes (with `c5.4xlarge`).

## 5. Monitoring
While the script runs, open a second SSH session to monitor:
* **CPU & RAM Usage**: `htop`
* **Disk Usage**: `watch -n 10 df -h` (Ensure the EBS volume isn't filling up from TFIDF caching)
* **Process Status**: `ps aux | grep python`
* **Logs**: The script outputs directly to standard output. Run via `tmux` or redirect to a file if you want to persist the stdout stream: `python scripts/aws/run_d2_blocking.py > blocking_run.log 2>&1 &` then `tail -f blocking_run.log`.

## 6. Output Verification
After completion, verify the expected artifact: `artifacts/blocking/validation_candidate_pairs.tsv`

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

## 7. Cost Tracking
Update `docs/aws_cost_tracking.md` immediately upon terminating the instance.
