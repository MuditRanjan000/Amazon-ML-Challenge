#!/bin/bash
# EC2 user-data for matcher fan-out (Amazon Linux 2023).
MODE=__MODE__; SPLIT=__SPLIT__; EXP=__EXP__; PART=__PART__; N=__N__; BUCKET=__BUCKET__; RUN=__RUN__; COMMIT=__COMMIT__
S3=s3://$BUCKET/$RUN
TAG=$([ "$MODE" = merge ] && echo merge || printf 'part%02d' "$PART")
LOG=/var/log/matcher.log
export HOME=/root AWS_DEFAULT_REGION=ap-south-1
exec > >(tee -a $LOG) 2>&1

up() { aws s3 cp $LOG $S3/logs/$TAG.log --only-show-errors || true; }
finish() {
  trap - ERR
  echo "== $1 $(date -u +%FT%TZ)"
  up
  echo "$1 $(date -u +%FT%TZ)" | aws s3 cp - $S3/status/$TAG.$1 --only-show-errors || true
  shutdown -h now
  exit 0
}
( while sleep 60; do up; done ) &
( sleep 14400; echo "HARD CAP (4h) reached"; finish TIMEOUT ) &
set -eo pipefail
trap 'echo "FAILED at line $LINENO"; finish FAILED' ERR

echo "== start $(date -u +%FT%TZ) mode=$MODE part=$PART/$N commit=$COMMIT"
mkdir -p /opt/repo /opt/data /opt/out
fallocate -l 16G /swapfile && chmod 600 /swapfile && mkswap /swapfile && swapon /swapfile

aws s3 cp $S3/code.tar.gz - | tar -xz -C /opt/repo
curl -LsSf https://astral.sh/uv/install.sh | env UV_INSTALL_DIR=/usr/local/bin UV_NO_MODIFY_PATH=1 sh
cd /opt/repo
uv venv -q -p 3.13 .venv
uv pip install -q -p .venv/bin/python -r requirements.txt
uv pip install -q -p .venv/bin/python -e .
export ER_DATA_DIR=/opt/data/dataset ER_OUTPUT_DIR=/opt/out ER_CACHE_DIR=/opt/out/cache ER_GIT_COMMIT=$COMMIT PYTHONUNBUFFERED=1

if [ "$MODE" = part ]; then
  # Download inputs
  aws s3 cp s3://$BUCKET/artifacts/blocking/${SPLIT}_candidate_pairs.tsv.gz /opt/candidates.tsv.gz
  aws s3 cp s3://$BUCKET/artifacts/validation_split/${SPLIT}_ids.csv /opt/source1_ids.csv
  aws s3 cp s3://$BUCKET/artifacts/records/train_records.sqlite /opt/train_records.sqlite
  aws s3 cp $S3/models/features.pkl /opt/features.pkl
  aws s3 cp $S3/models/logistic.pkl /opt/logistic.pkl
  
  .venv/bin/python scripts/aws/run_matcher_fanout.py --split "$SPLIT" --part "$PART/$N" \
      --candidates /opt/candidates.tsv.gz \
      --source1-ids /opt/source1_ids.csv \
      --store /opt/train_records.sqlite \
      --feature-artifact /opt/features.pkl \
      --model /opt/logistic.pkl \
      --run-dir /opt/out
      
  aws s3 cp /opt/out/parts/part$(printf '%02d' $PART)/validation_logistic_scores.tsv $S3/parts/part$(printf '%02d' $PART)/validation_logistic_scores.tsv --only-show-errors
else
  # Download inputs for merge
  aws s3 cp $S3/parts/ /opt/out/parts/ --recursive --only-show-errors
  aws s3 cp s3://$BUCKET/artifacts/validation_split/${SPLIT}_ids.csv /opt/source1_ids.csv
  if [ "$SPLIT" = "validation" ]; then
    aws s3 cp s3://$BUCKET/data/train_ground_truth.tsv /opt/train_ground_truth.tsv
  fi
  
  .venv/bin/python scripts/aws/run_matcher_fanout.py --split "$SPLIT" --merge "$N" \
      --source1-ids /opt/source1_ids.csv \
      --ground-truth /opt/train_ground_truth.tsv \
      --threshold 0.6 \
      --run-dir /opt/out \
      --exp-id "$EXP"
      
  aws s3 cp /opt/out/${SPLIT}_logistic_scores.tsv $S3/final/${SPLIT}_logistic_scores.tsv --only-show-errors
  aws s3 cp /opt/out/${SPLIT}_logistic_decisions.tsv $S3/final/${SPLIT}_logistic_decisions.tsv --only-show-errors
  aws s3 cp experiments/results/experiments.jsonl $S3/final/experiments.jsonl --only-show-errors
fi
finish DONE
