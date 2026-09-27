#!/bin/bash
# EC2 user-data: run one v3 job (train / decide) next to the data. Placeholders: __BUCKET__ __RUN__ __NAME__ __CMD64__
# (base64 of the bash commands to run inside /opt/repo with the venv active) __COMMIT__ __REGION__.
# Layout on the box: /opt/data/dataset (+ /opt/data/utils validator), /opt/w = ER_OUTPUT_DIR (v3/ tables, cache/).
# Uploads /opt/w/job/ to s3://B/RUN/jobs/NAME/, log to .../jobs/NAME.log; status .../jobs/NAME.DONE|FAILED. 3 h cap.
BUCKET=__BUCKET__; S3=s3://$BUCKET; R=$S3/__RUN__; J=$R/jobs/__NAME__; LOG=/var/log/job.log
export HOME=/root AWS_DEFAULT_REGION=__REGION__
exec > >(tee -a $LOG) 2>&1
up() { aws s3 cp $LOG $J.log --only-show-errors || true; }
finish() { trap - ERR; echo "== $1 $(date -u +%FT%TZ)"; aws s3 cp /opt/w/job/ $J/ --recursive --only-show-errors || true; up
           echo "$1" | aws s3 cp - $J.$1 --only-show-errors || true; shutdown -h now; exit 0; }
( while sleep 60; do up; done ) &
( sleep 10800; echo "HARD CAP (3h)"; finish TIMEOUT ) &
set -eo pipefail
trap 'echo "FAILED at line $LINENO"; finish FAILED' ERR
mkdir -p /opt/repo /opt/w/v3 /opt/w/job /opt/data/utils
fallocate -l 32G /swapfile && chmod 600 /swapfile && mkswap /swapfile && swapon /swapfile
aws s3 cp $R/code.tar.gz - | tar -xz -C /opt/repo
curl -LsSf https://astral.sh/uv/install.sh | env UV_INSTALL_DIR=/usr/local/bin UV_NO_MODIFY_PATH=1 sh
cd /opt/repo && uv venv -q -p 3.13 .venv && uv pip install -q -p .venv/bin/python -r requirements.txt \
  && uv pip install -q -p .venv/bin/python -e .
aws s3 cp $S3/data/test_parquet.tar - | tar -x -C /opt/w/..  # -> /opt/data/dataset/test + /opt/out/cache
aws s3 cp $S3/data/train_meta.tar - | tar -x -C /opt/w/..    # -> /opt/data/dataset/train GT + /opt/out/cache
mkdir -p /opt/w/cache && cp -p /opt/out/cache/*.parquet /opt/w/cache/
aws s3 cp $S3/data/validate_submission.py /opt/data/utils/validate_submission.py --only-show-errors
export ER_DATA_DIR=/opt/data/dataset ER_OUTPUT_DIR=/opt/w ER_CACHE_DIR=/opt/w/cache ER_GIT_COMMIT=__COMMIT__ PYTHONUNBUFFERED=1
export PATH=/opt/repo/.venv/bin:$PATH R S3
echo "== env ready $(date -u +%FT%TZ)"; free -m
echo __CMD64__ | base64 -d > /opt/job.sh
bash -eo pipefail /opt/job.sh
finish DONE
