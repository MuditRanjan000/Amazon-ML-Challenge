import os, time, subprocess, sys, json

KEY = "ec2_ephemeral"
REPO_URL = "https://github.com/MuditRanjan000/Amazon-ML-Challenge.git"
BRANCH = "feature/mudit-submission"
SG_NAME = "val-production-sg"
INSTANCE_TYPE = "r5.4xlarge"

def log(msg):
    print(f"[{time.strftime('%X')}] {msg}", flush=True)

def run_cmd(cmd_list):
    res = subprocess.run(cmd_list, capture_output=True, text=True)
    if res.returncode != 0:
        log(f"CMD FAILED: {' '.join(cmd_list)}\nOutput: {res.stderr}")
    return res

def push_key(instance_id):
    run_cmd(["aws", "ec2-instance-connect", "send-ssh-public-key", "--instance-id", instance_id, "--instance-os-user", "ubuntu", "--ssh-public-key", f"file://{KEY}.pub"])

def run_ssh(cmd, ip, instance_id):
    push_key(instance_id)
    res = run_cmd(["ssh", "-i", KEY, "-o", "StrictHostKeyChecking=no", f"ubuntu@{ip}", cmd])
    return res.stdout, res.stderr

def main():
    log("Fetching Ubuntu 24.04 AMI...")
    ami_res = run_cmd(["aws", "ec2", "describe-images", "--owners", "amazon", "--filters", "Name=name,Values=ubuntu/images/hvm-ssd-gp3/ubuntu-noble-24.04-amd64-server-*", "Name=state,Values=available", "--query", "sort_by(Images, &CreationDate)[-1].ImageId", "--output", "text"])
    ami = ami_res.stdout.strip()
    log(f"AMI: {ami}")

    log("Creating SG...")
    # Clean up old SG if it exists
    run_cmd(["aws", "ec2", "delete-security-group", "--group-name", SG_NAME])
    
    sg_res = run_cmd(["aws", "ec2", "create-security-group", "--group-name", SG_NAME, "--description", "Validation run SG", "--query", "GroupId", "--output", "text"])
    sg_id = sg_res.stdout.strip()
    run_cmd(["aws", "ec2", "authorize-security-group-ingress", "--group-id", sg_id, "--protocol", "tcp", "--port", "22", "--cidr", "0.0.0.0/0"])
    
    log(f"Launching {INSTANCE_TYPE} Spot Instance...")
    bd_map = '[{"DeviceName":"/dev/sda1","Ebs":{"VolumeSize":100,"VolumeType":"gp3"}}]'
    res = run_cmd(["aws", "ec2", "run-instances", "--image-id", ami, "--instance-type", INSTANCE_TYPE, "--security-group-ids", sg_id, "--block-device-mappings", bd_map, "--instance-market-options", '{"MarketType":"spot"}', "--query", "Instances[0].InstanceId", "--output", "text"])
    instance_id = res.stdout.strip()
    log(f"Instance ID: {instance_id}")

    log("Waiting for instance to be running...")
    run_cmd(["aws", "ec2", "wait", "instance-running", "--instance-ids", instance_id])
    time.sleep(10)
    ip_res = run_cmd(["aws", "ec2", "describe-instances", "--instance-ids", instance_id, "--query", "Reservations[0].Instances[0].PublicIpAddress", "--output", "text"])
    ip = ip_res.stdout.strip()
    log(f"Instance IP: {ip}")
    
    time.sleep(30)
    
    log("Transferring dataset over SCP...")
    push_key(instance_id)
    scp_res = run_cmd(["scp", "-i", KEY, "-o", "StrictHostKeyChecking=no", "dataset.zip", f"ubuntu@{ip}:/home/ubuntu/dataset.zip"])
    if scp_res.returncode != 0:
        log("SCP failed!")
        sys.exit(1)

    log("Cloning repository and setting up environment...")
    run_ssh(f"git clone -b {BRANCH} {REPO_URL} repo", ip, instance_id)
    run_ssh("sudo apt update && sudo apt install -y build-essential python3-dev unzip", ip, instance_id)
    run_ssh("cd repo && python3 -m venv .venv && .venv/bin/pip install -r requirements.txt && .venv/bin/pip install -e .", ip, instance_id)

    log("Unzipping dataset...")
    run_ssh("mkdir -p /home/ubuntu/dataset && unzip -q dataset.zip -d /home/ubuntu/dataset", ip, instance_id)

    log("Triggering val generation script...")
    run_ssh("cd repo && export ER_DATA_DIR=/home/ubuntu/dataset && nohup .venv/bin/python scripts/aws/run_d2_blocking.py --split trainval --val-only --blocker word --in-memory > run.log 2>&1 &", ip, instance_id)

    log("Monitoring execution...")
    while True:
        time.sleep(30)
        out, _ = run_ssh("pgrep -f run_d2_blocking.py", ip, instance_id)
        if not out.strip():
            break

    log("Execution completed. Verifying artifacts...")
    out, _ = run_ssh("ls -s repo/artifacts/blocking/validation_candidate_pairs.tsv", ip, instance_id)
    size = int(out.strip().split()[0]) if out.strip() else 0

    out_meta, _ = run_ssh("ls repo/artifacts/blocking/blocking_metadata.json", ip, instance_id)

    if size > 0 and "blocking_metadata.json" in out_meta:
        log("Verification SUCCESS. Downloading artifacts...")
        os.makedirs("artifacts/blocking", exist_ok=True)
        push_key(instance_id)
        run_cmd(["scp", "-i", KEY, "-o", "StrictHostKeyChecking=no", f"ubuntu@{ip}:/home/ubuntu/repo/artifacts/blocking/*", "artifacts/blocking/"])
        
        log("Generating handoff package locally...")
        handoff_env = os.environ.copy()
        handoff_env["PYTHONPATH"] = "src"
        subprocess.run([sys.executable, "scripts/prepare_validation_handoff.py"], env=handoff_env)

        log("Cleaning up AWS resources...")
        run_cmd(["aws", "ec2", "terminate-instances", "--instance-ids", instance_id])
        
        log("Waiting for instance termination before deleting SG...")
        run_cmd(["aws", "ec2", "wait", "instance-terminated", "--instance-ids", instance_id])
        run_cmd(["aws", "ec2", "delete-security-group", "--group-name", SG_NAME])
        log("Cleanup complete. Run finished successfully.")
    else:
        log(f"Verification FAILED. Size: {size}, Meta: {out_meta}")
        log("EC2 instance has been PRESERVED for debugging.")

if __name__ == '__main__':
    main()
