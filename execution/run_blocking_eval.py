import sys
import os
import time
import pandas as pd

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.entity_resolution.data.loader import DataLoader
from src.entity_resolution.data.validation_split import create_validation_split, apply_validation_split
from src.entity_resolution.blocking.exact_name import ExactNameBlocker
from src.entity_resolution.blocking.evaluator import BlockingEvaluator

def run_evaluation():
    data_dir = r"c:\Users\dell\Desktop\Projects\Amazon ML Challenge\6ab10eb3b23ba_student_resource\student_resource\dataset"
    
    print("Loading data...")
    loader = DataLoader(data_dir)
    gt = loader.load_ground_truth()
    s1 = loader.load_source('train', 1)
    s2 = loader.load_source('train', 2)
    s3 = loader.load_source('train', 3)
    
    print("Creating validation split...")
    train_ids, val_ids = create_validation_split(gt)
    
    # We only care about validation set for evaluating blocking
    # S2/S3 can be full set since they are our search index
    val_s1, _ = apply_validation_split(s1, 'entity_id', val_ids, train_ids)
    val_gt, _ = apply_validation_split(gt, 'source1_entity_id', val_ids, train_ids)
    
    print("Building blocking index...")
    blocker = ExactNameBlocker()
    start_time = time.time()
    blocker.build_index(s2, s3)
    
    print("Generating candidates...")
    candidates = blocker.generate_candidates(val_s1, k=200)
    end_time = time.time()
    
    runtime = end_time - start_time
    
    print("Evaluating...")
    evaluator = BlockingEvaluator(val_gt)
    results = evaluator.evaluate(candidates)
    
    print(f"\n--- Blocking Results (Exact Name) ---")
    print(f"Runtime: {runtime:.2f}s")
    for k in [10, 25, 50, 100, 200]:
        recall = results.get(f'recall_at_{k}', 0)
        avg_cand = results.get(f'avg_cand_at_{k}', 0)
        print(f"K={k} | Recall: {recall:.4f} | Avg Candidates: {avg_cand:.2f}")

    # Log to CSV
    log_path = r"c:\Users\dell\Desktop\Projects\Amazon ML Challenge\experiments\results\blocking_results.csv"
    log_df = pd.DataFrame([{
        'experiment_id': 'BLK-001',
        'method': 'exact_name',
        'parameters': 'lowercase, punct removal, suffixes',
        'candidate_k': 100, # log standard 100
        'candidate_recall': results.get('recall_at_100', 0),
        'avg_candidates': results.get('avg_cand_at_100', 0),
        'runtime': runtime,
        'memory': 'Low',
        'observations': 'Very fast but low recall due to strict matching'
    }])
    log_df.to_csv(log_path, mode='a', header=False, index=False)
    
if __name__ == "__main__":
    run_evaluation()
