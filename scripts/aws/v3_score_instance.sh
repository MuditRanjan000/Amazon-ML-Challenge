#!/bin/bash
# EC2 user-data: v3 scoring of candidate slices (Amazon Linux 2023). One instance runs parts __FIRST__..__LAST__ of __N__
# in parallel (one process each). Placeholders: __BUCKET__ __RUN__ __NAME__ (train|val|test) __SPLIT__ (train|test)
# __CANDS__ (s3 key) __STATS__ (s3 key) __NS1__ __N__ __FIRST__ __LAST__ __K__ __COMMIT__ __REGION__
# Outputs: s3://B/RUN/scored/NAME.partNNofN.parquet ; status/NAME.pNN.DONE|FAILED ; logs/. Always ends in shutdown.
BUCKET=__BUCKET__; S3=s3://$BUCKET; R=$S3/__RUN__; TAG=__NAME__-__FIRST__-__LAST__; LOG=/var/log/v3.log
export HOME=/root AWS_DEFAULT_REGION=__REGION__
exec > >(tee -a $LOG) 2>&1
up() { aws s3 cp $LOG $R/logs/$TAG.log --only-show-errors || true; }
finish() { trap - ERR; echo "== $1 $(date -u +%FT%TZ)"; up
           echo "$1" | aws s3 cp - $R/status/$TAG.$1 --only-show-errors || true; shutdown -h now; exit 0; }
( while sleep 60; do up; done ) &
( sleep 7200; echo "HARD CAP (2h)"; finish TIMEOUT ) &
set -eo pipefail
trap 'echo "FAILED at line $LINENO"; finish FAILED' ERR
echo "== start $(date -u +%FT%TZ) $TAG commit=__COMMIT__"
mkdir -p /opt/repo /opt/w/out
fallocate -l 16G /swapfile && chmod 600 /swapfile && mkswap /swapfile && swapon /swapfile
aws s3 cp $R/code.tar.gz - | tar -xz -C /opt/repo
curl -LsSf https://astral.sh/uv/install.sh | env UV_INSTALL_DIR=/usr/local/bin UV_NO_MODIFY_PATH=1 sh
cd /opt/repo && uv venv -q -p 3.13 .venv && uv pip install -q -p .venv/bin/python -r requirements.txt \
  && uv pip install -q -p .venv/bin/python -e .
export ER_OUTPUT_DIR=/opt/w ER_CACHE_DIR=/opt/w/cache ER_GIT_COMMIT=__COMMIT__ PYTHONUNBUFFERED=1
mkdir -p /opt/w/v3
aws s3 cp $R/norm___SPLIT__.parquet /opt/w/v3/norm___SPLIT__.parquet --only-show-errors
aws s3 cp $R/tokdf___SPLIT__.parquet /opt/w/v3/tokdf___SPLIT__.parquet --only-show-errors || true
MODELS=""; for m in $(aws s3 ls $R/ | awk '{print $4}' | grep -E '^m1.*\.joblib$'); do
  aws s3 cp $R/$m /opt/w/$m --only-show-errors; MODELS="${MODELS:+$MODELS,}/opt/w/$m"; done
echo "models: $MODELS"
aws s3 cp $S3/__STATS__ /opt/w/stats.parquet --only-show-errors
aws s3 cp $S3/__CANDS__ /opt/w/cands.tsv.gz --only-show-errors
echo "== data ready $(date -u +%FT%TZ)"; free -m
NP=$((__LAST__ - __FIRST__ + 1)); MAXP=$(( $(nproc) < NP ? $(nproc) : NP ))  # at most one part per vCPU at a time
TH=$(( $(nproc) / MAXP )); [ $TH -ge 1 ] || TH=1
pids=()
for i in $(seq __FIRST__ __LAST__); do
  while [ "$(jobs -rp | wc -l)" -ge "$MAXP" ]; do sleep 5; done
  ( RAYON_NUM_THREADS=$TH OMP_NUM_THREADS=$TH .venv/bin/python execution/v3_pipeline.py score --cands /opt/w/cands.tsv.gz \
      --split __SPLIT__ --name __NAME__ --n-s1 __NS1__ --part $i/__N__ --K __K__ --model $MODELS \
      --stats /opt/w/stats.parquet --out /opt/w/out --threads $TH > /var/log/p$i.log 2>&1 \
    && aws s3 cp /opt/w/out/__NAME__.part$(printf %02d $i)of__N__.parquet $R/scored/ --only-show-errors \
    && echo ok | aws s3 cp - $R/status/__NAME__.p$(printf %02d $i).DONE --only-show-errors \
    || { echo fail | aws s3 cp - $R/status/__NAME__.p$(printf %02d $i).FAILED --only-show-errors; } ;
    aws s3 cp /var/log/p$i.log $R/logs/__NAME__.p$(printf %02d $i).log --only-show-errors ) &
  pids+=($!)
done
wait "${pids[@]}"
free -m
finish DONE
