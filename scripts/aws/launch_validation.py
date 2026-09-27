import argparse
import base64
import os
import subprocess
from datetime import datetime

import boto3

def get_current_commit():
    return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"]).decode("utf-8").strip()

def create_tarball():
    print("Creating code.tar.gz...")
    subprocess.run(["git", "archive", "-o", "code.tar.gz", "HEAD"], check=True)
    return "code.tar.gz"

def launch_instance(bucket, run_id, commit, split, part, n, dry_run=False):
    ec2 = boto3.client("ec2", region_name="us-east-1")
    
    with open("scripts/aws/matcher_validation_instance.sh", "r") as f:
        script = f.read()
    
    import configparser
    from pathlib import Path
    
    aws_creds = configparser.ConfigParser()
    aws_creds.read(Path.home() / ".aws" / "credentials")
    access_key = aws_creds["default"]["aws_access_key_id"]
    secret_key = aws_creds["default"]["aws_secret_access_key"]
    
    script = script.replace("__BUCKET__", bucket).replace("__RUN__", run_id).replace("__COMMIT__", commit)
    script = script.replace("__SPLIT__", split).replace("__PART__", str(part)).replace("__N__", str(n))
    script = script.replace("export HOME=/root AWS_DEFAULT_REGION=us-east-1", f"export HOME=/root AWS_DEFAULT_REGION=us-east-1 AWS_ACCESS_KEY_ID={access_key} AWS_SECRET_ACCESS_KEY={secret_key}")
    
    if dry_run:
        script = script.replace(".venv/bin/python scripts/aws/run_matcher_fanout.py", "echo 'DRY RUN'")
    
    user_data = base64.b64encode(script.encode("utf-8")).decode("utf-8")
    
    instance_type = "t3.small" if dry_run else "c6i.4xlarge"
    
    print(f"Requesting On-Demand Instance for part {part}/{n}...")
    try:
        response = ec2.run_instances(
            ImageId="ami-051f8a213df8bc089",
            InstanceType=instance_type,
            MinCount=1,
            MaxCount=1,
            UserData=user_data,
            BlockDeviceMappings=[{
                "DeviceName": "/dev/xvda",
                "Ebs": {"VolumeSize": 100, "VolumeType": "gp3"}
            }],
            NetworkInterfaces=[{"DeviceIndex": 0, "AssociatePublicIpAddress": True}],
            TagSpecifications=[{
                "ResourceType": "instance",
                "Tags": [
                    {"Key": "Project", "Value": "mudit-final-submission"},
                    {"Key": "Owner", "Value": "Mudit"},
                    {"Key": "Name", "Value": f"matcher-val-{part:02d}-{run_id.split('/')[-1]}"}
                ]
            }],
        )
        print(f"Launched instance {response['Instances'][0]['InstanceId']} for part {part}/{n}")
    except Exception as e:
        print(f"Failed to launch part {part}/{n}: {e}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", required=True, help="The S3 path of the training run (e.g. mudit-final/runs/train_...)")
    parser.add_argument("--split", choices=["validation", "test"], default="validation")
    parser.add_argument("--n-parts", type=int, default=10, help="Number of instances to fan out to")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    
    bucket = "amazon-ml-challenge-2026-data-655285961749"
    commit = get_current_commit()
    
    tarball = create_tarball()
    print("Uploading code payload...")
    subprocess.run(f"aws s3 cp {tarball} s3://{bucket}/{args.run_id}/code.tar.gz", shell=True, check=True)
    
    for i in range(1, args.n_parts + 1):
        launch_instance(bucket, args.run_id, commit, args.split, i, args.n_parts, args.dry_run)
        
    print("Fanout initiated.")
    print(f"Monitor S3: s3://{bucket}/{args.run_id}/status/")

if __name__ == "__main__":
    main()
