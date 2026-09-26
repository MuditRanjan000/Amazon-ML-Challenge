#!/bin/bash
# EC2 user-data for blocking fan-out (Amazon Linux 2023). Placeholders are filled at launch:
#   __MODE__ part|merge   __PART__ i   __N__ n   __BUCKET__   __RUN__ (s3 prefix)   __COMMIT__
# part : run_d2_blocking.py --split trainval --blocker word --part i/n, upload parts/, terminate.
# merge: download all parts, --merge n, upload final TSVs (+ .gz) + metadata + sample, terminate.
# The instance is launched with shutdown-behavior=terminate: every exit path ends in `shutdown`.
MODE=__MODE__; PART=__PART__; N=__N__; BUCKET=__BUCKET__; RUN=__RUN__; COMMIT=__COMMIT__
S3=s3://$BUCKET/$RUN
TAG=$([ "$MODE" = merge ] && echo merge || printf 'part%02d' "$PART")
LOG=/var/log/blocking.log
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
( sleep 7200; echo "HARD CAP (2h) reached"; finish TIMEOUT ) &
set -eo pipefail
trap 'echo "FAILED at line $LINENO"; finish FAILED' ERR

echo "== start $(date -u +%FT%TZ) mode=$MODE part=$PART/$N commit=$COMMIT"
mkdir -p /opt/repo /opt/data /opt/out
# free-plan instance types have 8 GB RAM; swap turns a borderline peak into slowness instead of an OOM kill
fallocate -l 8G /swapfile && chmod 600 /swapfile && mkswap /swapfile && swapon /swapfile
aws s3 cp $S3/code.tar.gz - | tar -xz -C /opt/repo
curl -LsSf https://astral.sh/uv/install.sh | env UV_INSTALL_DIR=/usr/local/bin UV_NO_MODIFY_PATH=1 sh
cd /opt/repo
uv venv -q -p 3.13 .venv
uv pip install -q -p .venv/bin/python -r requirements.txt
uv pip install -q -p .venv/bin/python -e .
export ER_DATA_DIR=/opt/data/dataset ER_OUTPUT_DIR=/opt/out ER_CACHE_DIR=/opt/out/cache \
       ER_N_JOBS=$(nproc) ER_GIT_COMMIT=$COMMIT PYTHONUNBUFFERED=1
echo "== env ready $(date -u +%FT%TZ)"

if [ "$MODE" = part ]; then
  aws s3 cp s3://$BUCKET/data/train_parquet.tar - | tar -x -C /opt  # parquet cache + mtime-old TSV placeholders: no TSV parse
  .venv/bin/python scripts/aws/run_d2_blocking.py --split trainval --blocker word --in-memory \
      --part "$PART/$N" --batch 50000 --exp-id BLK-020
  free -m
  aws s3 cp artifacts/blocking/parts/ $S3/parts/ --recursive --only-show-errors
else
  aws s3 cp $S3/parts/ artifacts/blocking/parts/ --recursive --only-show-errors
  .venv/bin/python scripts/aws/run_d2_blocking.py --merge "$N" --exp-id BLK-020
  dnf install -y -q pigz
  for f in train_candidate_pairs.tsv validation_candidate_pairs.tsv; do pigz -k -1 artifacts/blocking/$f; done
  for f in train_candidate_pairs.tsv train_candidate_pairs.tsv.gz validation_candidate_pairs.tsv \
           validation_candidate_pairs.tsv.gz train_sample_candidate_pairs.tsv train_sample_s1_ids.csv \
           blocking_metadata.json; do
    aws s3 cp artifacts/blocking/$f $S3/final/$f --only-show-errors
  done
  aws s3 cp experiments/results/experiments.jsonl $S3/final/experiments_blk020.jsonl --only-show-errors
fi
finish DONE
