#!/bin/bash
# EC2 user-data for full matcher training
BUCKET=__BUCKET__; RUN=__RUN__; COMMIT=__COMMIT__
S3=s3://$BUCKET/$RUN
LOG=/var/log/matcher_train.log
export HOME=/root AWS_DEFAULT_REGION=us-east-1

exec > >(tee -a $LOG) 2>&1

up() { aws s3 cp $LOG $S3/logs/train.log --only-show-errors || true; }
finish() {
  trap - ERR
  echo "== $1 $(date -u +%FT%TZ)"
  up
  echo "$1 $(date -u +%FT%TZ)" | aws s3 cp - $S3/status/train.$1 --only-show-errors || true
  shutdown -h now
  exit 0
}
( while sleep 60; do up; done ) &
set -exo pipefail
trap 'echo "FAILED at line $LINENO"; finish FAILED' ERR

echo "== start $(date -u +%FT%TZ) commit=$COMMIT"
mkdir -p /opt/repo /opt/data /opt/out


aws s3 cp $S3/code.tar.gz - | tar -xz -C /opt/repo
curl -LsSf https://astral.sh/uv/install.sh | env UV_INSTALL_DIR=/usr/local/bin UV_NO_MODIFY_PATH=1 sh
cd /opt/repo
uv venv -q -p 3.13 .venv
uv pip install -q -p .venv/bin/python -r requirements.txt
uv pip install -q -p .venv/bin/python -e .

export ER_DATA_DIR=/opt/data/dataset ER_OUTPUT_DIR=/opt/out ER_CACHE_DIR=/opt/out/cache ER_GIT_COMMIT=$COMMIT PYTHONUNBUFFERED=1

echo "Downloading artifacts..."
aws s3 cp s3://$BUCKET/artifacts/blocking/train_candidate_pairs.tsv.gz /opt/train_candidates.tsv.gz
aws s3 cp s3://$BUCKET/artifacts/validation_split/train_ids.csv /opt/train_ids.csv
aws s3 cp s3://$BUCKET/artifacts/records/train_records.sqlite /opt/train_records.sqlite
aws s3 cp s3://$BUCKET/data/train_ground_truth.tsv /opt/train_ground_truth.tsv
aws s3 cp s3://$BUCKET/artifacts/blocking/blocking_metadata.json /opt/blocking_metadata.json

CANDIDATE_VERSION="train_candidate_pairs"
CONFIG_HASH=$(jq -r '.train.config_sha' /opt/blocking_metadata.json)
GENERATOR_COMMIT=$(jq -r '.train.generator_commit' /opt/blocking_metadata.json)

df -h
echo "Starting training preflight..."
.venv/bin/python execution/run_streaming_training_preflight.py \
    --candidates /opt/train_candidates.tsv.gz \
    --sample-source1-ids /opt/train_ids.csv \
    --frozen-train-source1-ids /opt/train_ids.csv \
    --ground-truth /opt/train_ground_truth.tsv \
    --store /opt/train_records.sqlite \
    --run-dir /opt/out/train_run \
    --candidate-version "$CANDIDATE_VERSION" \
    --generator-commit "$GENERATOR_COMMIT" \
    --config-hash "$CONFIG_HASH"

echo "Uploading model artifacts..."
aws s3 cp /opt/out/train_run/features.pkl $S3/models/features.pkl
aws s3 cp /opt/out/train_run/logistic_regression.pkl $S3/models/logistic_regression.pkl
aws s3 cp /opt/out/train_run/training_preflight_report.json $S3/models/training_preflight_report.json

finish DONE
