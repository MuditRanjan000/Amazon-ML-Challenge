# Amazon ML Challenge 2026 - Business Entity Resolution

## Project Objective
Determine which records across three independent data sources refer to the same real-world business entity despite partial and noisy data.

## Challenge Description
Source 1 is the deduplicated reference source. The goal is to find all matching records from Source 2 and Source 3 for each Source 1 entity. 

## Folder Structure
- `data/`: Contains dataset descriptions (datasets ignored via .gitignore).
- `docs/`: Methodology, playbooks, and reports.
- `experiments/`: Experiment registry and results.
- `models/`: Saved model weights (ignored via .gitignore).
- `notebooks/`: Jupyter notebooks for exploratory work.
- `output/`: Generated submission files (`matching_results.tsv`, `candidate_pairs.tsv`).
- `scripts/`: Standalone scripts for deployment and environment management.
- `src/`: Main source code containing data loaders, evaluators, blocking algorithms, features, and ML models.

## Setup Instructions
1. Unzip the challenge dataset into `6ab10eb3b23ba_student_resource/` or adjust the data paths in `.env` / configuration.
2. Install dependencies: `pip install -r requirements.txt`.
3. Run the validation split creation to setup the local environment.

## Experiment Workflow
1. Check `AI_CONTEXT.md` and the playbook.
2. Define a hypothesis in `experiments/results/experiment_registry.csv`.
3. Create logic in `src/` modularly.
4. Run locally against the frozen validation split.
5. Commit with the experiment ID.

## Team Workflow
- **Mudit:** Project lead, integration, validation, evaluation.
- **Aayush:** Candidate generation and blocking.
- **Ashank:** Matcher models, feature engineering, and thresholds.
