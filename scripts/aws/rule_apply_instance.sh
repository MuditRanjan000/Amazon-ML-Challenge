#!/bin/bash
# EC2 user-data: apply the tuned rule baseline to the test candidates next to the data (Amazon Linux 2023).
# Placeholders: __BUCKET__ __CODE_RUN__ (prefix holding code.tar.gz) __CAND__ (s3 key of test candidates .tsv.gz)
#               __OUT__ (s3 prefix for outputs) __COMMIT__ __ARGS__ (rule args, e.g. "--R 10 --t 0.6 --a 0.8")
# Writes matching_results.tsv + candidate_pairs.tsv (gz) to __OUT__/ and terminates; 1 h hard cap.
BUCKET=__BUCKET__; S3=s3://$BUCKET; OUT=$S3/__OUT__; LOG=/var/log/rule.log
export HOME=/root AWS_DEFAULT_REGION=ap-south-1
exec > >(tee -a $LOG) 2>&1
finish() { trap - ERR; aws s3 cp $LOG $OUT/rule.log --only-show-errors || true
           echo "$1" | aws s3 cp - $OUT/status.$1 --only-show-errors || true; shutdown -h now; exit 0; }
( sleep 3600; finish TIMEOUT ) &
set -eo pipefail
trap 'echo "FAILED at line $LINENO"; finish FAILED' ERR
mkdir -p /opt/repo /opt/sub
aws s3 cp $S3/__CODE_RUN__/code.tar.gz - | tar -xz -C /opt/repo
aws s3 cp $S3/data/test_parquet.tar - | tar -x -C /opt
curl -LsSf https://astral.sh/uv/install.sh | env UV_INSTALL_DIR=/usr/local/bin UV_NO_MODIFY_PATH=1 sh
cd /opt/repo && uv venv -q -p 3.13 .venv && uv pip install -q -p .venv/bin/python -r requirements.txt && uv pip install -q -p .venv/bin/python -e .
aws s3 cp $S3/__CAND__ /opt/test.tsv.gz --only-show-errors
export ER_DATA_DIR=/opt/data/dataset ER_OUTPUT_DIR=/opt/out ER_CACHE_DIR=/opt/out/cache ER_GIT_COMMIT=__COMMIT__ PYTHONUNBUFFERED=1
.venv/bin/python execution/rule_baseline.py apply --test /opt/test.tsv.gz __ARGS__ --out /opt/sub --skip-validate
gzip -1 /opt/sub/matching_results.tsv /opt/sub/candidate_pairs.tsv
aws s3 cp /opt/sub/ $OUT/ --recursive --only-show-errors
aws s3 cp experiments/results/experiments.jsonl $OUT/experiments.jsonl --only-show-errors
finish DONE
