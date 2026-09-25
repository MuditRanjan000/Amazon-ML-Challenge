# AWS Strategy & Resource Plan

## Context
Each team member has $200 in AWS credits ($600 total). This is a tight budget for a large-scale data science competition. We must use resources intelligently and turn them off when not actively experimenting.

## 1. Expected Computational Bottlenecks
* **Dataset Loading & Preprocessing:** Disk I/O bound and moderately memory-heavy.
* **Candidate Generation (Blocking):** Extremely memory-heavy (RAM) and CPU-bound. Indexing ~10 million records for TF-IDF requires high RAM to avoid out-of-memory (OOM) errors.
* **Feature Generation:** CPU-bound. Calculating string distances across millions of candidate pairs is parallelizable but compute-intensive.
* **Model Training:** Memory and Compute bound. Gradient boosting on millions of pairs requires high RAM and fast CPUs. (GPUs are optional since LightGBM/XGBoost perform exceptionally well on many-core CPUs for tabular data, but GPU instances speed up training).
* **Inference (Final Test Set):** I/O and CPU bound.

## 2. Environment Split Decision
### Run Locally (Free)
* Initial exploratory data analysis.
* Code writing, testing, and debugging on small subsets of data.
* Writing evaluation frameworks and pipelines.
* Small-scale baseline experiments (e.g., exact matching).

### Run on AWS (Credit Consumption)
* Full-scale TF-IDF indexing and candidate blocking (K=50, 100, etc.) on the entire dataset.
* Full-scale feature engineering over candidate pairs.
* Hyperparameter tuning and final model training (LightGBM/XGBoost).
* Final inference run to generate the submission TSVs.

## 3. Recommended AWS Architecture
* **S3 Usage:** 
  * Create a private bucket (`amazon-ml-challenge-2026-data`).
  * Store the raw `*.tsv` files, intermediate candidate datasets, and trained model artifacts.
  * Enable versioning on model artifacts.
* **EC2 Usage:**
  * Avoid always-on instances. Use EC2 Spot Instances where possible to save up to 70-90% on compute costs.
  * For model training and blocking, spin up a memory-optimized instance (e.g., `r5.4xlarge` or `r5.8xlarge`), run the script, export results to S3, and immediately terminate.
* **Experiment Artifact Storage:**
  * Push small metadata (CSV logs) to GitHub.
  * Push large artifacts (model weights, candidate pairs) to S3, linking the S3 URI in the `experiment_registry.csv`.

## 4. Resource Estimation & Instance Choices
* **Expected Data Size:** ~1-2 GB raw text. Extracted features for candidate pairs may balloon to 10-20 GB.
* **RAM Requirements:** At least 64GB - 128GB RAM to comfortably hold indices and pandas DataFrames without swapping.
* **CPU Requirements:** 16 - 32 vCPUs to parallelize string distance metrics.
* **Instance Recommendations:**
  * `r5.4xlarge` (16 vCPU, 128 GiB RAM): ~$1.00/hour on-demand (cheaper on Spot). Excellent for blocking and feature engineering.
  * `c5.9xlarge` (36 vCPU, 72 GiB RAM): ~$1.53/hour on-demand. Good for highly parallel CPU tasks if memory footprint is managed.

## 5. Security & Cost Rules
1. **Avoid Unnecessary Spending:** Terminate EC2 instances immediately after scripts finish.
2. **Do Not Upload Secrets:** AWS credentials (`.aws/credentials`, `.env`) MUST remain excluded via `.gitignore`.
3. **Data Privacy:** S3 buckets must block public access. The challenge data cannot be shared publicly.
4. **Reproducibility:** Code must be pushed to `main` before running an AWS job to ensure the exact commit is recorded.
