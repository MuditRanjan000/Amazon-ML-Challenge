#!/usr/bin/env bash
# Blocking fan-out on AWS: N workers + 1 merge instance, all self-terminating (2 h cap each).
#
#   scripts/aws/launch_fanout.sh setup  --bucket B                        # one-time: private bucket, IAM role/profile, no-inbound SG
#   scripts/aws/launch_fanout.sh bundle --bucket B --split test|train     # one-time per split: parquet data bundle -> s3://B/data/
#   scripts/aws/launch_fanout.sh run    --bucket B --split test|trainval  # candidates + record stats -> s3://B/run-<commit>-<split>/final/
#   scripts/aws/launch_fanout.sh status --bucket B --split test|trainval  # progress of the current commit's run
# Options: --region R (ap-south-1)  --workers N (48)  --spot-from I (33: parts >= I are spot; > N = none)
#          --type T (m7i-flex.large: 2 vCPU / 8 GB, free-tier eligible)  --dry-run (print AWS calls, launch nothing)
# Needs: AWS CLI with credentials for the target account, git, python + this repo installed (`pip install -e .`)
# for `bundle`. Runs from a clean, pushed commit so the workers' code is reproducible.
set -euo pipefail

CMD=${1:-}; shift || true
BUCKET=""; REGION=ap-south-1; SPLIT=""; WORKERS=48; SPOT_FROM=33; TYPE=m7i-flex.large; DRY=""
while [ $# -gt 0 ]; do
  case "$1" in
    --bucket) BUCKET=$2; shift 2 ;;       --region) REGION=$2; shift 2 ;;
    --split) SPLIT=$2; shift 2 ;;         --workers) WORKERS=$2; shift 2 ;;
    --spot-from) SPOT_FROM=$2; shift 2 ;; --type) TYPE=$2; shift 2 ;;
    --dry-run) DRY=1; shift ;;            *) echo "unknown option $1" >&2; exit 2 ;;
  esac
done
[ -n "$BUCKET" ] || { echo "--bucket is required" >&2; exit 2; }
AWS=${AWS_CLI:-aws}
ROLE=amazon-ml-blocking-ec2; SG_NAME=amazon-ml-blocking-noinbound
ROOT=$(git rev-parse --show-toplevel); cd "$ROOT"
TMP=$ROOT/.tmp/fanout; mkdir -p "$TMP"
log() { echo "[$(date +%H:%M:%S)] $*" >&2; }
aws_() { if [ -n "$DRY" ]; then echo "DRY: $AWS --region $REGION $*" >&2; else "$AWS" --region "$REGION" "$@"; fi; }
furi() { if command -v cygpath >/dev/null 2>&1; then echo "file://$(cygpath -w "$1")"; else echo "file://$1"; fi; }
account() { [ -n "$DRY" ] && echo 000000000000 || "$AWS" --region "$REGION" sts get-caller-identity --query Account --output text; }

setup() {
  local acct; acct=$(account)
  if [ "$REGION" = us-east-1 ]; then aws_ s3api create-bucket --bucket "$BUCKET" || true
  else aws_ s3api create-bucket --bucket "$BUCKET" --create-bucket-configuration "LocationConstraint=$REGION" || true; fi
  aws_ s3api put-public-access-block --bucket "$BUCKET" \
    --public-access-block-configuration BlockPublicAcls=true,IgnorePublicAcls=true,BlockPublicPolicy=true,RestrictPublicBuckets=true
  cat > "$TMP/trust.json" <<'EOF'
{"Version":"2012-10-17","Statement":[{"Effect":"Allow","Principal":{"Service":"ec2.amazonaws.com"},"Action":"sts:AssumeRole"}]}
EOF
  cat > "$TMP/s3policy.json" <<EOF
{"Version":"2012-10-17","Statement":[{"Effect":"Allow","Action":["s3:GetObject","s3:PutObject"],"Resource":"arn:aws:s3:::$BUCKET/*"},
 {"Effect":"Allow","Action":"s3:ListBucket","Resource":"arn:aws:s3:::$BUCKET"}]}
EOF
  aws_ iam create-role --role-name "$ROLE" --assume-role-policy-document "$(furi "$TMP/trust.json")" || true
  aws_ iam put-role-policy --role-name "$ROLE" --policy-name s3-one-bucket --policy-document "$(furi "$TMP/s3policy.json")"
  aws_ iam attach-role-policy --role-name "$ROLE" --policy-arn arn:aws:iam::aws:policy/AmazonSSMManagedInstanceCore
  aws_ iam create-instance-profile --instance-profile-name "$ROLE" || true
  aws_ iam add-role-to-instance-profile --instance-profile-name "$ROLE" --role-name "$ROLE" || true
  local vpc; vpc=$(aws_ ec2 describe-vpcs --filters Name=isDefault,Values=true --query 'Vpcs[0].VpcId' --output text)
  aws_ ec2 create-security-group --group-name "$SG_NAME" --description "blocking workers: no inbound" --vpc-id "${vpc:-vpc-dry}" \
    --query GroupId --output text || true
  log "setup done for account $acct (bucket $BUCKET, role/profile $ROLE, SG $SG_NAME). Wait ~15 s for IAM to propagate."
}

bundle() {  # parquet cache + placeholder TSVs (mtime 2000) so workers skip TSV parsing
  case "$SPLIT" in test) files="test_source1 test_source2 test_source3" ;;
                   train) files="train_source1 train_source2 train_source3 train_ground_truth" ;;
                   *) echo "bundle --split must be test|train" >&2; exit 2 ;; esac
  python - "$SPLIT" <<'EOF'
import sys
from entity_resolution.data.loader import DataLoader
L = DataLoader()
if sys.argv[1] == "train":
    L.load_ground_truth()
for s in (1, 2, 3):
    L.load_source(sys.argv[1], s, columns=["entity_id"])  # builds/refreshes output/cache/*.parquet
EOF
  local cache; cache=$(python -c "from entity_resolution import config; print(config.CACHE_DIR)")
  local stage="$TMP/stage_$SPLIT"; rm -rf "$stage"; mkdir -p "$stage/data/dataset/$SPLIT" "$stage/out/cache"
  for f in $files; do
    cp -p "$cache/$f.parquet" "$stage/out/cache/"
    : > "$stage/data/dataset/$SPLIT/$f.tsv"; touch -d "2000-01-01" "$stage/data/dataset/$SPLIT/$f.tsv"
  done
  (cd "$stage" && tar -cf "../${SPLIT}_parquet.tar" data out)
  aws_ s3 cp "$TMP/${SPLIT}_parquet.tar" "s3://$BUCKET/data/${SPLIT}_parquet.tar" --only-show-errors
  log "uploaded s3://$BUCKET/data/${SPLIT}_parquet.tar ($(du -h "$TMP/${SPLIT}_parquet.tar" | cut -f1))"
}

render() {  # $1 mode, $2 part, $3 out file
  sed -e "s#__MODE__#$1#; s#__SPLIT__#$SPLIT#; s#__EXP__#BLK-020#; s#__PART__#$2#; s#__N__#$WORKERS#" \
      -e "s#__BUCKET__#$BUCKET#; s#__RUN__#$RUN#; s#__COMMIT__#$COMMIT#; s#__REGION__#$REGION#" \
      scripts/aws/blocking_instance.sh > "$3"
  grep -q "__[A-Z]*__" "$3" && { echo "unfilled placeholder in $3" >&2; exit 1; } || true
}

launch() {  # $1 name, $2 user-data file, $3 bdm json, $4 spot|ondemand -> instance id
  local market=() id="" try
  [ "$4" = spot ] && market=(--instance-market-options 'MarketType=spot,SpotOptions={SpotInstanceType=one-time,InstanceInterruptionBehavior=terminate}')
  for try in 1 2 3 4 5 6; do
    id=$(aws_ ec2 run-instances --image-id "$AMI" --instance-type "$TYPE" --count 1 "${market[@]}" \
      --iam-instance-profile "Name=$ROLE" \
      --network-interfaces "DeviceIndex=0,SubnetId=${SUBNETS[$((RANDOM % ${#SUBNETS[@]}))]},Groups=$SG,AssociatePublicIpAddress=true" \
      --block-device-mappings "$(furi "$3")" --metadata-options HttpTokens=required,HttpEndpoint=enabled \
      --instance-initiated-shutdown-behavior terminate \
      --tag-specifications "ResourceType=instance,Tags=[{Key=Name,Value=$1},{Key=Project,Value=amazon-ml-blocking}]" \
      --user-data "$(furi "$2")" --query 'Instances[0].InstanceId' --output text 2>&1 | tail -1) || true
    [ -n "$DRY" ] && { log "$id"; echo "i-dryrun"; return; }
    [[ $id == i-* ]] && { echo "$id"; return; }
    log "  $1 launch attempt $try failed: $id"; sleep 15
  done
  echo "FAILED"
}

count_status() { aws_ s3 ls "s3://$BUCKET/$RUN/status/" 2>/dev/null | grep -cE "part[0-9]+\.(DONE|FAILED|TIMEOUT)" || true; }

wait_for() {  # $1 predicate command, $2 what, $3 max minutes
  local waited=0
  [ -n "$DRY" ] && return
  until eval "$1"; do
    sleep 30; waited=$((waited + 1))
    [ $((waited % 4)) -eq 0 ] && log "waiting for $2 ... ($((waited / 2)) min)"
    [ $waited -ge $(( $3 * 2 )) ] && { log "gave up waiting for $2 after $3 min"; return 1; }
  done
}

prepare_run() {
  case "$SPLIT" in test|trainval) ;; *) echo "--split must be test|trainval" >&2; exit 2 ;; esac
  COMMIT=$(git rev-parse --short HEAD); RUN="run-$COMMIT-$SPLIT"
}

run() {
  prepare_run
  local guard=exit; [ -n "$DRY" ] && guard=true
  if [ -n "$(git status --porcelain --untracked-files=no)" ]; then echo "commit your changes first (workers run HEAD)" >&2; $guard 1; fi
  [ -n "$(git branch -r --contains HEAD)" ] || { echo "push HEAD first (reproducibility)" >&2; $guard 1; }
  local data=$([ "$SPLIT" = test ] && echo test || echo train)
  [ -n "$DRY" ] || aws_ s3 ls "s3://$BUCKET/data/${data}_parquet.tar" >/dev/null || { echo "run: bundle --split $data first" >&2; exit 1; }
  if [ -z "$DRY" ] && "$AWS" --region "$REGION" freetier get-account-plan-state >/dev/null 2>&1; then
    log "account plan: $("$AWS" --region "$REGION" freetier get-account-plan-state --query '[accountPlanType,accountPlanRemainingCredits.amount]' --output text)"
  fi
  AMI=$(aws_ ec2 describe-images --owners amazon --filters "Name=name,Values=al2023-ami-2023.*-kernel-6.*-x86_64" "Name=state,Values=available" \
        --query 'sort_by(Images,&CreationDate)[-1].ImageId' --output text); AMI=${AMI:-ami-dry}
  local vpc; vpc=$(aws_ ec2 describe-vpcs --filters Name=isDefault,Values=true --query 'Vpcs[0].VpcId' --output text)
  read -r -a SUBNETS <<< "$(aws_ ec2 describe-subnets --filters "Name=vpc-id,Values=${vpc:-vpc-dry}" --query 'Subnets[].SubnetId' --output text)"
  [ ${#SUBNETS[@]} -gt 0 ] || SUBNETS=(subnet-dry)
  SG=$(aws_ ec2 describe-security-groups --filters "Name=group-name,Values=$SG_NAME" --query 'SecurityGroups[0].GroupId' --output text); SG=${SG:-sg-dry}
  echo '[{"DeviceName":"/dev/xvda","Ebs":{"VolumeSize":30,"VolumeType":"gp3","DeleteOnTermination":true}}]' > "$TMP/bdm_worker.json"
  echo '[{"DeviceName":"/dev/xvda","Ebs":{"VolumeSize":100,"VolumeType":"gp3","Iops":6000,"Throughput":500,"DeleteOnTermination":true}}]' > "$TMP/bdm_merge.json"
  git archive --format=tar.gz -o "$TMP/code.tar.gz" HEAD
  aws_ s3 cp "$TMP/code.tar.gz" "s3://$BUCKET/$RUN/code.tar.gz" --only-show-errors
  log "run $RUN: $WORKERS x $TYPE (spot from part $SPOT_FROM), AMI $AMI, ${#SUBNETS[@]} subnets, SG $SG"

  local i id failed=()
  for i in $(seq 1 "$WORKERS"); do
    render part "$i" "$TMP/ud_part$i.sh"
    id=$(launch "blk-$SPLIT-part$i" "$TMP/ud_part$i.sh" "$TMP/bdm_worker.json" "$([ "$i" -ge "$SPOT_FROM" ] && echo spot || echo ondemand)")
    [ "$id" = FAILED ] && failed+=("$i")
  done
  [ ${#failed[@]} -eq 0 ] || log "could not launch parts: ${failed[*]} (quota?) - rerun them with: status + relaunch"
  log "launched; waiting for $WORKERS part status markers"
  wait_for '[ "$(count_status)" -ge "$WORKERS" ]' "workers" 120 || true
  local bad; bad=$(aws_ s3 ls "s3://$BUCKET/$RUN/status/" 2>/dev/null | grep -oE "part[0-9]+\.(FAILED|TIMEOUT)" | grep -oE "[0-9]+" | sed 's/^0*//' || true)
  for i in $bad; do  # one on-demand retry per failed / interrupted part
    log "retrying part $i on-demand"; aws_ s3 rm "s3://$BUCKET/$RUN/status/" --recursive --exclude "*" --include "part$(printf %02d "$i").*" --only-show-errors
    render part "$i" "$TMP/ud_part$i.sh"; launch "blk-$SPLIT-part$i-retry" "$TMP/ud_part$i.sh" "$TMP/bdm_worker.json" ondemand >/dev/null
  done
  [ -z "$bad" ] || wait_for '[ "$(aws_ s3 ls s3://$BUCKET/$RUN/status/ 2>/dev/null | grep -c "part[0-9]*\.DONE" || true)" -ge "$WORKERS" ]' "retried workers" 60
  render merge 0 "$TMP/ud_merge.sh"
  launch "blk-$SPLIT-merge" "$TMP/ud_merge.sh" "$TMP/bdm_merge.json" ondemand >/dev/null
  log "merge launched; waiting"
  wait_for 'aws_ s3 ls "s3://$BUCKET/$RUN/status/" 2>/dev/null | grep -qE "merge\.(DONE|FAILED|TIMEOUT)"' "merge" 90 || true
  status
}

status() {
  prepare_run
  aws_ s3 ls "s3://$BUCKET/$RUN/status/" 2>/dev/null | awk '{print $4}' | sed 's/.*\.//' | sort | uniq -c || true
  aws_ s3 ls "s3://$BUCKET/$RUN/final/" --human-readable 2>/dev/null || true
  log "running project instances: $(aws_ ec2 describe-instances --filters Name=tag:Project,Values=amazon-ml-blocking \
       Name=instance-state-name,Values=pending,running --query 'length(Reservations[].Instances[])' --output text)"
  log "logs: s3://$BUCKET/$RUN/logs/ ; outputs: s3://$BUCKET/$RUN/final/"
}

case "$CMD" in
  setup) setup ;; bundle) bundle ;; run) run ;; status) status ;;
  *) sed -n '2,12p' "$0"; exit 2 ;;
esac
