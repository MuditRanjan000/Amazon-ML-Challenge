#!/usr/bin/env bash
# v3 scoring fan-out (reuses the bucket/role/SG made by launch_fanout.sh setup). Every instance self-terminates.
#   scripts/aws/v3_fanout.sh prep   --bucket B --run R            # code (HEAD) + norm tables + m1 model -> s3://B/R/
#   scripts/aws/v3_fanout.sh launch --bucket B --run R --name train|val|test --n PARTS --per P [--type T] [--spot]
#   scripts/aws/v3_fanout.sh job    --bucket B --run R --name J --cmd-file F [--type T] [--spot]  # one box runs F
#   scripts/aws/v3_fanout.sh status --bucket B --run R
# One instance runs P parts in parallel. PAID-plan account: credits only (see directives), check before launching.
set -euo pipefail
CMD=${1:-}; shift || true
BUCKET=""; RUN=""; NAME=""; N=0; PER=4; TYPE=m7i.2xlarge; SPOT=""; REGION=ap-south-1; K=100; FROM=1; TO=0
while [ $# -gt 0 ]; do
  case "$1" in
    --bucket) BUCKET=$2; shift 2 ;; --run) RUN=$2; shift 2 ;; --name) NAME=$2; shift 2 ;;
    --n) N=$2; shift 2 ;; --per) PER=$2; shift 2 ;; --type) TYPE=$2; shift 2 ;; --spot) SPOT=1; shift ;;
    --K) K=$2; shift 2 ;; --from) FROM=$2; shift 2 ;; --to) TO=$2; shift 2 ;; --cmd-file) CMDF=$2; shift 2 ;; --stats) STATS_OVERRIDE=$2; shift 2 ;;
    *) echo "unknown option $1" >&2; exit 2 ;;
  esac
done
AWS=${AWS_CLI:-aws}; ROLE=amazon-ml-blocking-ec2; SG_NAME=amazon-ml-blocking-noinbound
ROOT=$(git rev-parse --show-toplevel); cd "$ROOT"; TMP=$ROOT/.tmp/v3fan; mkdir -p "$TMP"
log() { echo "[$(date +%H:%M:%S)] $*" >&2; }
aws_() { "$AWS" --region "$REGION" "$@"; }
furi() { if command -v cygpath >/dev/null 2>&1; then echo "file://$(cygpath -w "$1")"; else echo "file://$1"; fi; }

case "$NAME" in
  train) SPLIT=train; CANDS=v3/data/train_candidate_pairs.tsv.gz; STATS=v3/data/record_stats_trainval.parquet; NS1=1765456 ;;
  val)   SPLIT=train; CANDS=v3/data/validation_candidate_pairs.tsv.gz; STATS=v3/data/record_stats_trainval.parquet; NS1=441365 ;;
  test)  SPLIT=test; CANDS=run-90b0553-test/final/test_candidate_pairs.tsv.gz; STATS=run-90b0553-test/final/record_stats_test.parquet; NS1=1732544 ;;
  *) NS1=0 ;;
esac
[ -n "${STATS_OVERRIDE:-}" ] && STATS=$STATS_OVERRIDE  # e.g. v3/data/record_stats_trainval_drop20.parquet (orphan simulation)

prep() {
  git archive --format=tar.gz -o "$TMP/code.tar.gz" HEAD
  aws_ s3 cp "$TMP/code.tar.gz" "s3://$BUCKET/$RUN/code.tar.gz" --only-show-errors
  local out; out=$(python -c "from entity_resolution import config; print(config.OUTPUT_DIR / 'v3')")
  for f in norm_train.parquet norm_test.parquet m1.joblib; do
    [ -f "$out/$f" ] && aws_ s3 cp "$out/$f" "s3://$BUCKET/$RUN/$f" --only-show-errors && log "uploaded $f"
  done
}

launch() {
  [ "$N" -gt 0 ] && [ "$NS1" -gt 0 ] || { echo "--name train|val|test and --n required" >&2; exit 2; }
  [ "$TO" -gt 0 ] || TO=$N
  local commit ami vpc sg subnets id first last market=() try
  commit=$(git rev-parse --short HEAD)
  ami=$(aws_ ec2 describe-images --owners amazon --filters "Name=name,Values=al2023-ami-2023.*-kernel-6.*-x86_64" \
        "Name=state,Values=available" --query 'sort_by(Images,&CreationDate)[-1].ImageId' --output text)
  vpc=$(aws_ ec2 describe-vpcs --filters Name=isDefault,Values=true --query 'Vpcs[0].VpcId' --output text)
  read -r -a subnets <<< "$(aws_ ec2 describe-subnets --filters "Name=vpc-id,Values=$vpc" --query 'Subnets[].SubnetId' --output text)"
  sg=$(aws_ ec2 describe-security-groups --filters "Name=group-name,Values=$SG_NAME" --query 'SecurityGroups[0].GroupId' --output text)
  echo '[{"DeviceName":"/dev/xvda","Ebs":{"VolumeSize":60,"VolumeType":"gp3","DeleteOnTermination":true}}]' > "$TMP/bdm.json"
  [ -n "$SPOT" ] && market=(--instance-market-options 'MarketType=spot,SpotOptions={SpotInstanceType=one-time,InstanceInterruptionBehavior=terminate}')
  for ((first = FROM; first <= TO; first += PER)); do
    last=$(( first + PER - 1 )); [ $last -le "$TO" ] || last=$TO
    sed -e "s#__BUCKET__#$BUCKET#g; s#__RUN__#$RUN#g; s#__NAME__#$NAME#g; s#__SPLIT__#$SPLIT#g; s#__CANDS__#$CANDS#g" \
        -e "s#__STATS__#$STATS#g; s#__NS1__#$NS1#g; s#__N__#$N#g; s#__FIRST__#$first#g; s#__LAST__#$last#g" \
        -e "s#__K__#$K#g; s#__COMMIT__#$commit#g; s#__REGION__#$REGION#g" scripts/aws/v3_score_instance.sh > "$TMP/ud.sh"
    ! grep -q "__[A-Z0-9]*__" "$TMP/ud.sh" || { echo "unfilled placeholder" >&2; exit 1; }
    for try in 1 2 3 4 5 6; do
      id=$(aws_ ec2 run-instances --image-id "$ami" --instance-type "$TYPE" --count 1 "${market[@]}" \
        --iam-instance-profile "Name=$ROLE" \
        --network-interfaces "DeviceIndex=0,SubnetId=${subnets[$((RANDOM % ${#subnets[@]}))]},Groups=$sg,AssociatePublicIpAddress=true" \
        --block-device-mappings "$(furi "$TMP/bdm.json")" --metadata-options HttpTokens=required,HttpEndpoint=enabled \
        --instance-initiated-shutdown-behavior terminate \
        --tag-specifications "ResourceType=instance,Tags=[{Key=Name,Value=v3-$NAME-$first-$last},{Key=Project,Value=amazon-ml-v3}]" \
        --user-data "$(furi "$TMP/ud.sh")" --query 'Instances[0].InstanceId' --output text 2>&1 | tail -1) || true
      [[ $id == i-* ]] && { log "$NAME parts $first-$last -> $id"; break; }
      log "  launch attempt $try failed: $id"; sleep 15
    done
  done
}

run_one() {  # $1 user-data file, $2 name tag
  local ami vpc sg subnets id try market=()
  ami=$(aws_ ec2 describe-images --owners amazon --filters "Name=name,Values=al2023-ami-2023.*-kernel-6.*-x86_64"         "Name=state,Values=available" --query 'sort_by(Images,&CreationDate)[-1].ImageId' --output text)
  vpc=$(aws_ ec2 describe-vpcs --filters Name=isDefault,Values=true --query 'Vpcs[0].VpcId' --output text)
  read -r -a subnets <<< "$(aws_ ec2 describe-subnets --filters "Name=vpc-id,Values=$vpc" --query 'Subnets[].SubnetId' --output text)"
  sg=$(aws_ ec2 describe-security-groups --filters "Name=group-name,Values=$SG_NAME" --query 'SecurityGroups[0].GroupId' --output text)
  echo '[{"DeviceName":"/dev/xvda","Ebs":{"VolumeSize":150,"VolumeType":"gp3","Iops":6000,"Throughput":500,"DeleteOnTermination":true}}]' > "$TMP/bdm_job.json"
  [ -n "$SPOT" ] && market=(--instance-market-options 'MarketType=spot,SpotOptions={SpotInstanceType=one-time,InstanceInterruptionBehavior=terminate}')
  for try in 1 2 3 4 5 6; do
    id=$(aws_ ec2 run-instances --image-id "$ami" --instance-type "$TYPE" --count 1 "${market[@]}"       --iam-instance-profile "Name=$ROLE"       --network-interfaces "DeviceIndex=0,SubnetId=${subnets[$((RANDOM % ${#subnets[@]}))]},Groups=$sg,AssociatePublicIpAddress=true"       --block-device-mappings "$(furi "$TMP/bdm_job.json")" --metadata-options HttpTokens=required,HttpEndpoint=enabled       --instance-initiated-shutdown-behavior terminate       --tag-specifications "ResourceType=instance,Tags=[{Key=Name,Value=v3-job-$2},{Key=Project,Value=amazon-ml-v3}]"       --user-data "$(furi "$1")" --query 'Instances[0].InstanceId' --output text 2>&1 | tail -1) || true
    [[ $id == i-* ]] && { log "job $2 -> $id ($TYPE)"; return; }
    log "  launch attempt $try failed: $id"; sleep 15
  done
}

job() {
  [ -f "${CMDF:-}" ] || { echo "--cmd-file required" >&2; exit 2; }
  local c64; c64=$(base64 -w0 < "$CMDF")
  sed -e "s#__BUCKET__#$BUCKET#g; s#__RUN__#$RUN#g; s#__NAME__#$NAME#g; s#__CMD64__#$c64#g"       -e "s#__COMMIT__#$(git rev-parse --short HEAD)#g; s#__REGION__#$REGION#g" scripts/aws/v3_job_instance.sh > "$TMP/ud_job.sh"
  ! grep -q "__[A-Z0-9]*__" "$TMP/ud_job.sh" || { echo "unfilled placeholder" >&2; exit 1; }
  run_one "$TMP/ud_job.sh" "$NAME"
}

status() {
  aws_ s3 ls "s3://$BUCKET/$RUN/status/" 2>/dev/null | awk '{print $4}' | sed -E 's/\.p[0-9]+\./ /; s/-[0-9]+-[0-9]+\./ instance-/' \
    | sort | uniq -c || true
  log "running v3 instances: $(aws_ ec2 describe-instances --filters Name=tag:Project,Values=amazon-ml-v3 \
       Name=instance-state-name,Values=pending,running --query 'length(Reservations[].Instances[])' --output text)"
}

case "$CMD" in prep) prep ;; launch) launch ;; job) job ;; status) status ;; *) sed -n '2,7p' "$0"; exit 2 ;; esac
