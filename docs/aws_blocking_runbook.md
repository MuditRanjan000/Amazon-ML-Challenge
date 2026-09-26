# AWS D2 Blocking Runbook

## Current Blocker Configuration (Variant D2)
* **Partitioning**: Country partitioning enabled
* **Text Fields**: `business_name` + `business_address`
* **Features**: Character N-grams (3,5)
* **Output**: Top K=200 candidates per query

## 1. Required Instance Resources
Since the algorithm is primarily bottlenecked by parallel compute and memory (despite our optimizations, large sparse arrays still consume memory), the recommended instance for the full candidate generation is:
* **Instance Type**: `r5.4xlarge` (16 vCPU, 128 GiB RAM) or `c5.9xlarge` (36 vCPU, 72 GiB RAM)
* **Disk Space**: At least 100 GB EBS (gp3) volume (to hold raw data, 10-20GB index cache, and 3GB candidates file).
* **OS**: Ubuntu 22.04 / Amazon Linux 2023

## 2. Setup Steps
```bash
# 1. Update and install prerequisites
sudo apt-get update && sudo apt-get install -y git python3-venv python3-pip

# 2. Clone the repository
git clone https://github.com/MuditRanjan000/Amazon-ML-Challenge.git
cd Amazon-ML-Challenge
git checkout feature/mudit-submission

# 3. Setup Python Virtual Environment
python3 -m venv .venv
source .venv/bin/activate

# 4. Install dependencies
pip install -r requirements.txt
pip install -e .

# 5. Set Environment Variables
# Crucial: Define the location of the dataset
export ER_DATA_DIR=/path/to/dataset
export ER_OUTPUT_DIR=./output
```

## 3. Execution Command
```bash
# Run the AWS D2 Blocking Script
python scripts/aws/run_d2_blocking.py
```

## 4. Expected Outputs
1. **Experiment Registry:** Results will be appended to `experiments/results/blocking_results.csv` tracking Recall@10-200, average candidates, and runtime.
2. **Index Cache:** Generated TF-IDF vectorizers and sparse chunk matrices in `output/cache/tfidf_cache_exp002b_D2/`.
3. **Candidates TSV:** The final scored pairs will be saved to `artifacts/blocking/validation_candidate_pairs.tsv`.
*Note: Ensure to back up `validation_candidate_pairs.tsv` and `blocking_results.csv` to S3 or push to GitHub immediately after the run.*

## 5. Estimated Cost Tracking
* **Compute:** An `r5.4xlarge` Spot Instance costs approximately $0.30 - $0.40 per hour (On-Demand is ~$1.00/hour).
* **Time:** The script is optimized to run sparse dot products in memory chunks. Expected runtime is 10 - 25 minutes depending on the CPU core count.
* **Total Estimate:** < $1.00 per run.
* **Important:** Terminate the EC2 instance immediately after the outputs are successfully transferred to S3!
