import argparse
import base64
import os
import subprocess
import time
from datetime import datetime

import boto3

def get_current_commit():
    return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"]).decode("utf-8").strip()

def create_tarball():
    print("Creating code.tar.gz...")
    subprocess.run(["git", "archive", "-o", "code.tar.gz", "HEAD"], check=True)
    return "code.tar.gz"

def launch_instance(bucket, run_id, commit, dry_run=False):
    ec2 = boto3.client("ec2", region_name="us-east-1")
    
    with open("scripts/aws/matcher_train_instance.sh", "r") as f:
        script = f.read()
    
    if dry_run:
        # Override to just do a quick pre-flight test then exit
        script = script.replace(".venv/bin/python execution/run_streaming_training_preflight.py",
                                "mkdir -p /opt/out/train_run && touch /opt/out/train_run/features.pkl /opt/out/train_run/logistic_regression.pkl /opt/out/train_run/training_preflight_report.json && echo 'DRY-RUN: Python command is valid!' ")
    
    import configparser
    from pathlib import Path
    
    aws_creds = configparser.ConfigParser()
    aws_creds.read(Path.home() / ".aws" / "credentials")
    access_key = aws_creds["default"]["aws_access_key_id"]
    secret_key = aws_creds["default"]["aws_secret_access_key"]
    
    script = script.replace("__BUCKET__", bucket).replace("__RUN__", run_id).replace("__COMMIT__", commit)
    script = script.replace("export HOME=/root AWS_DEFAULT_REGION=us-east-1", f"export HOME=/root AWS_DEFAULT_REGION=us-east-1 AWS_ACCESS_KEY_ID={access_key} AWS_SECRET_ACCESS_KEY={secret_key}")
    
    user_data = base64.b64encode(script.encode("utf-8")).decode("utf-8")
    
    # R6i.8xlarge has 256GB RAM (spot price ~$0.50/hr)
    # For dry-run, we use t3.small (spot price ~$0.006/hr)
    instance_type = "t3.small" if dry_run else "r6i.8xlarge"
    
    print(f"Requesting On-Demand Instance ({instance_type})...")
    
    try:
        response = ec2.run_instances(
            ImageId="ami-051f8a213df8bc089", # Amazon Linux 2023 in us-east-1
            InstanceType=instance_type,
            MinCount=1,
            MaxCount=1,
            # InstanceMarketOptions={"MarketType": "spot"},
            UserData=user_data,
            # We need to give it enough disk space for downloads and swap (200G swap + 50G data)
            BlockDeviceMappings=[{
                "DeviceName": "/dev/xvda",
                "Ebs": {"VolumeSize": 300, "VolumeType": "gp3"}
            }],
            # Default VPC, auto-assign public IP
            NetworkInterfaces=[{"DeviceIndex": 0, "AssociatePublicIpAddress": True}],
            TagSpecifications=[{
                "ResourceType": "instance",
                "Tags": [
                    {"Key": "Project", "Value": "mudit-final"},
                    {"Key": "Owner", "Value": "mudit"},
                    {"Key": "Name", "Value": f"mudit-train-{run_id.split('/')[-1]}"}
                ]
            }],
        )
        instance_id = response["Instances"][0]["InstanceId"]
        print(f"Launched instance: {instance_id}")
        return instance_id
    except Exception as e:
        print(f"Failed to launch instance: {e}")
        return None

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    
    bucket = "amazon-ml-challenge-2026-data-655285961749"
    commit = get_current_commit()
    run_id = f"mudit-final/runs/train_{commit}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
    
    tarball = create_tarball()
    print("Uploading code payload...")
    subprocess.run(f"aws s3 cp {tarball} s3://{bucket}/{run_id}/code.tar.gz", shell=True, check=True)
    
    launch_instance(bucket, run_id, commit, args.dry_run)
    print(f"Run ID: {run_id}")
    print(f"Monitor S3: s3://{bucket}/{run_id}/status/")

if __name__ == "__main__":
    main()
