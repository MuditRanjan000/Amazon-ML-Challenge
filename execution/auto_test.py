import argparse
import subprocess
import time
import sys
import pandas as pd
from pathlib import Path
import json

def run_cmd(cmd, check=True):
    print(f"\n=> Running: {cmd}")
    subprocess.run(cmd, shell=True, check=check)

def poll_s3_status(s3_path, n_parts, interval=30):
    print(f"Polling {s3_path} for DONE signals...")
    while True:
        res = subprocess.run(f"aws s3 ls {s3_path}/status/", shell=True, capture_output=True, text=True)
        out = res.stdout
        
        if "FAILED" in out or "TIMEOUT" in out:
            print(f"ERROR: Found FAILED/TIMEOUT status in {s3_path}:\n{out}")
            return False
            
        dones = [line for line in out.splitlines() if "DONE" in line and "test_part" in line]
        if len(dones) == n_parts:
            print("All parts completed successfully!")
            return True
            
        print(f"Waiting... {len(dones)}/{n_parts} parts done.")
        time.sleep(interval)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", required=True, help="Run ID e.g. mudit-final/runs/train_XXX")
    parser.add_argument("--s3-input-dir", required=True, help="Path to Ayush's S3 test dir, e.g. s3://amazon-ml-2026-team-655285961749/run-90b0553-test/final")
    parser.add_argument("--bucket", default="amazon-ml-challenge-2026-data-655285961749")
    parser.add_argument("--n-parts", type=int, default=10)
    args = parser.parse_args()

    print(f"Starting test inference pipeline for {args.run_id} using {args.s3_input_dir}")

    # 0. Get best validation configuration
    best_config_path = Path(f".tmp/runs/{args.run_id.split('/')[-1]}/sweep_results/best_config.json")
    if not best_config_path.exists():
        print(f"Error: {best_config_path} not found. Did validation finish?")
        sys.exit(1)
        
    with open(best_config_path) as f:
        best_config = json.load(f)
        
    best_k = best_config["k"]
    print(f"Using BEST validation config: {best_config}")

    # 1. Build test_records.sqlite
    test_store_path = Path("artifacts/records/test_records.sqlite")
    if not test_store_path.exists():
        print("\n--- Building test_records.sqlite ---")
        run_cmd("""python -c "
import sys
from pathlib import Path
sys.path.insert(0, 'src')
from entity_resolution.matching.records import SQLiteRecordStore
from entity_resolution.config import DATA_DIR
store_path = Path('artifacts/records/test_records.sqlite')
store_path.parent.mkdir(parents=True, exist_ok=True)
store = SQLiteRecordStore(store_path)
sources = [DATA_DIR / 'test/test_source1.tsv', DATA_DIR / 'test/test_source2.tsv', DATA_DIR / 'test/test_source3.tsv']
print(f'Building {store_path} from {len(sources)} sources...')
store.build(sources, overwrite=True)
" """)
    run_cmd(f"aws s3 cp {test_store_path} s3://{args.bucket}/artifacts/records/test_records.sqlite")
    
    # 2. Extract test S1 IDs
    test_ids_path = Path("artifacts/validation_split/test_ids.csv")
    if not test_ids_path.exists():
        print("\n--- Extracting test S1 IDs ---")
        test_ids_path.parent.mkdir(parents=True, exist_ok=True)
        df = pd.read_csv(".tmp/artifacts_unzip/data/test/test_source1.tsv", sep="\t", usecols=["record_id"], dtype=str)
        df.rename(columns={"record_id": "source1_entity_id"}).to_csv(test_ids_path, index=False)
    run_cmd(f"aws s3 cp {test_ids_path} s3://{args.bucket}/artifacts/validation_split/test_ids.csv")

    # 3. Copy test candidates and record_stats to the final execution bucket
    # Aayush provided them in a different bucket/path. Our launch scripts expect them in our bucket.
    print(f"\n--- Copying test artifacts from {args.s3_input_dir} to {args.bucket} ---")
    run_cmd(f"aws s3 cp {args.s3_input_dir}/test_candidate_pairs.tsv.gz s3://{args.bucket}/artifacts/blocking/test_candidate_pairs.tsv.gz")
    
    # Optional: If the model ends up needing record stats for advanced features later
    # run_cmd(f"aws s3 cp {args.s3_input_dir}/record_stats_test.parquet s3://{args.bucket}/data/record_stats_test.parquet")

    # 4. Run AWS matcher fanout
    print("\n--- Running AWS matcher fanout ---")
    run_cmd(f"python scripts/aws/launch_validation.py --run-id {args.run_id} --split test --n-parts {args.n_parts}")

    # 5. Wait for fanout
    if not poll_s3_status(f"s3://{args.bucket}/{args.run_id}", args.n_parts):
        sys.exit(1)

    # 6. Download and merge scores
    run_dir = Path(f".tmp/runs/{args.run_id.split('/')[-1]}")
    run_dir.mkdir(parents=True, exist_ok=True)
    merged_scores = run_dir / "test_logistic_scores.tsv"
    
    print("\n--- Downloading and merging scores ---")
    with open(merged_scores, "w", encoding="utf-8") as out:
        for i in range(1, args.n_parts + 1):
            part_tag = f"test_part{i:02d}"
            part_file = run_dir / f"test_logistic_scores_{i:02d}.tsv"
            run_cmd(f"aws s3 cp s3://{args.bucket}/{args.run_id}/parts/{part_tag}/test_logistic_scores.tsv {part_file}")
            
            with open(part_file, "r", encoding="utf-8") as f:
                header = f.readline()
                if i == 1:
                    out.write(header)
                out.write(f.read())
            part_file.unlink()

    # 7. Download candidates locally to build K filter
    local_candidates = run_dir / "test_candidate_pairs.tsv.gz"
    if not local_candidates.exists() and best_k != 200:
        print("\n--- Downloading candidate pairs for K-filtering ---")
        run_cmd(f"aws s3 cp {args.s3_input_dir}/test_candidate_pairs.tsv.gz {local_candidates}")

    # 8. Apply K filter if needed
    scores_k_path = merged_scores
    if best_k != 200:
        print(f"\n--- Filtering scores for K={best_k} ---")
        scores_k_path = run_dir / f"test_logistic_scores_K{best_k}.tsv"
        df_cands = pd.read_csv(local_candidates, sep="\t", usecols=["source1_entity_id", "candidate_entity_id", "rank"])
        valid_k = set(df_cands[df_cands["rank"] <= best_k][["source1_entity_id", "candidate_entity_id"]].itertuples(index=False, name=None))
        del df_cands
        
        with open(merged_scores, "r", encoding="utf-8") as src, open(scores_k_path, "w", encoding="utf-8") as dst:
            header = src.readline()
            dst.write(header)
            cols = header.strip().split("\t")
            idx_s1 = cols.index("source1_entity_id")
            idx_c2 = cols.index("candidate_entity_id")
            for line in src:
                parts = line.strip("\r\n").split("\t")
                if (parts[idx_s1], parts[idx_c2]) in valid_k:
                    dst.write(line)

    # 9. Apply Best Decisions
    print("\n--- Generating matching_results.tsv ---")
    if best_config["type"] == "basic":
        best_t = best_config["threshold"]
        run_cmd(f"""python -c "
import sys
sys.path.insert(0, 'src')
from entity_resolution.models.streaming import build_threshold_decisions
from pathlib import Path
db_path = Path('{run_dir}/test_decisions.sqlite')
db_path.unlink(missing_ok=True)
build_threshold_decisions(
    score_path=Path('{scores_k_path}').resolve(),
    source1_ids_path=Path('{test_ids_path}').resolve(),
    output_path=Path('{run_dir}/matching_results.tsv').resolve(),
    threshold={best_t},
    working_database=db_path.resolve()
)
" """)
    else:
        adv_name = best_config["name"]
        print(f"Applying advanced logic: {adv_name}")
        run_cmd(f"""python -c "
import pandas as pd
import sqlite3
from pathlib import Path

scores = pd.read_csv('{scores_k_path}', sep='\t', dtype={{'source1_entity_id': str, 'candidate_entity_id': str}})
stats = pd.read_parquet('s3://amazon-ml-2026-team-655285961749/run-90b0553-test/final/record_stats_test.parquet')
df = scores.merge(stats, on='candidate_entity_id', how='left')

df['is_owner'] = df['source1_entity_id'] == df['best_s1']
prob_ranks = df.sort_values(['candidate_entity_id', 'match_probability'], ascending=[True, False])
prob_ranks['prob_rank'] = prob_ranks.groupby('candidate_entity_id').cumcount()
best_probs = prob_ranks[prob_ranks['prob_rank'] == 0].set_index('candidate_entity_id')['match_probability'].rename('best_prob')
second_probs = prob_ranks[prob_ranks['prob_rank'] == 1].set_index('candidate_entity_id')['match_probability'].rename('second_prob')

df = df.join(best_probs, on='candidate_entity_id', how='left')
df = df.join(second_probs, on='candidate_entity_id', how='left').fillna({{'second_prob': 0.0}})
df['is_prob_owner'] = df['match_probability'] == df['best_prob']
df['prob_margin'] = df['best_prob'] - df['second_prob']

adv_name = '{adv_name}'
if adv_name.startswith('raw_'):
    t = float(adv_name.split('_')[1])
    mask = df['match_probability'] >= t
elif adv_name.startswith('one_owner_'):
    t = float(adv_name.split('_')[2])
    mask = (df['match_probability'] >= t) & df['is_prob_owner']
elif adv_name.startswith('margin_'):
    parts = adv_name.split('_')
    t1 = float(parts[2])
    t2 = float(parts[4])
    margin = float(parts[6])
    mask = (df['match_probability'] >= t1) | ((df['match_probability'] >= t2) & df['is_prob_owner'] & (df['prob_margin'] >= margin))
else:
    raise ValueError(f'Unknown advanced config name: {{adv_name}}')

accepted = df[mask]

db_path = Path('{run_dir}/test_decisions.sqlite')
db_path.unlink(missing_ok=True)
conn = sqlite3.connect(db_path)

s1_ids = pd.read_csv('{test_ids_path}', sep=',', dtype=str)['source1_entity_id'].tolist()
conn.execute('CREATE TABLE source1_ids (ordinal INTEGER PRIMARY KEY, source1_entity_id TEXT UNIQUE NOT NULL)')
conn.execute('CREATE TABLE accepted (source1_entity_id TEXT NOT NULL, candidate_entity_id TEXT NOT NULL)')

conn.executemany('INSERT INTO source1_ids (ordinal, source1_entity_id) VALUES (?, ?)', enumerate(s1_ids))
conn.executemany('INSERT INTO accepted VALUES (?, ?)', accepted[['source1_entity_id', 'candidate_entity_id']].values.tolist())

out_tsv = Path('{run_dir}/matching_results.tsv')
with open(out_tsv, 'w', encoding='utf-8') as f:
    f.write('source1_entity_id\\tmatched_entity_ids\\n')
    for source, matches in conn.execute(
        \\"\\"\\"SELECT s.source1_entity_id, COALESCE(GROUP_CONCAT(a.candidate_entity_id, ','), '')
        FROM source1_ids AS s LEFT JOIN accepted AS a ON a.source1_entity_id = s.source1_entity_id
        GROUP BY s.ordinal, s.source1_entity_id ORDER BY s.ordinal\\"\\"\\"
    ):
        f.write(f'{{source}}\\t{{matches}}\\n')
conn.close()
" """)

    print("\n--- Extracting candidate_pairs.tsv ---")
    run_cmd(f"""python -c "
import pandas as pd
from pathlib import Path
df = pd.read_csv('{scores_k_path}', sep='\t', usecols=['source1_entity_id', 'candidate_entity_id'])
df.to_csv(Path('{run_dir}/candidate_pairs.tsv').resolve(), sep='\t', index=False)
" """)

    # 10. Run final submission checks
    print("\n--- Running check_final_submission.py ---")
    run_cmd(f"python execution/check_final_submission.py --submission {run_dir}/matching_results.tsv --test-dir .tmp/artifacts_unzip/data/test/")
    
    print("\nDone! Ready for S3 upload.")

if __name__ == "__main__":
    main()
