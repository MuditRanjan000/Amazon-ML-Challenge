import sys
import os
import time
import pandas as pd
import gc

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.entity_resolution.data.loader import DataLoader
from src.entity_resolution.data.validation_split import create_validation_split, apply_validation_split
from src.entity_resolution.blocking.tfidf_blocking import TfidfBlocker
from src.entity_resolution.blocking.evaluator import BlockingEvaluator

def run_experiment():
    data_dir = r"c:\Users\dell\Desktop\Projects\Amazon ML Challenge\6ab10eb3b23ba_student_resource\student_resource\dataset"
    
    loader = DataLoader(data_dir)
    gt = loader.load_ground_truth()
    
    s1_full = loader.load_source('train', 1)
    s2_full = loader.load_source('train', 2)
    s3_full = loader.load_source('train', 3)
    
    train_ids, val_ids = create_validation_split(gt)
    
    # 10k Validation Sample
    import random
    random.seed(42)
    val_ids_sample = set(random.sample(list(val_ids), min(10000, len(val_ids))))
    
    val_s1, _ = apply_validation_split(s1_full, 'entity_id', val_ids_sample, train_ids)
    val_gt, _ = apply_validation_split(gt, 'source1_entity_id', val_ids_sample, train_ids)
    
    s2 = s2_full
    s3 = s3_full
    
    evaluator = BlockingEvaluator(val_gt)
    log_path = r"c:\Users\dell\Desktop\Projects\Amazon ML Challenge\experiments\results\blocking_results.csv"
    
    variants = [
        # D1: country + name character ngrams
        {'name': 'D1', 'fields': ['business_name'], 'ngrams': (3,5)},
        # D2: country + name + address character ngrams
        {'name': 'D2', 'fields': ['business_name', 'business_address'], 'ngrams': (3,5)},
        # D3: country + combined text with tuned ngram range
        {'name': 'D3', 'fields': ['business_name', 'business_address', 'city', 'state', 'zip_code'], 'ngrams': (3,6)},
    ]
    
    for v in variants:
        print(f"\nRunning Variant {v['name']} - Fields: {v['fields']}, Ngrams: {v['ngrams']}")
        
        blocker = TfidfBlocker(text_fields=v['fields'], ngram_range=v['ngrams'], cache_dir=f"output/tfidf_cache_exp002b_{v['name']}")
        
        start_time = time.time()
        print("Building Index (Partitioned by Country)...")
        blocker.build_index(s2, s3)
        index_time = time.time() - start_time
        print(f"Index built in {index_time:.2f}s")
        
        print(f"Generating Candidates for 10k queries...")
        gen_start = time.time()
        candidates = blocker.generate_candidates(val_s1, k=200)
        total_time = index_time + (time.time() - gen_start)
        
        print("Evaluating results...")
        results = evaluator.evaluate(candidates)
        
        for k in [10, 25, 50, 100, 200]:
            recall = results.get(f'recall_at_{k}', 0)
            avg_cand = results.get(f'avg_cand_at_{k}', 0)
            
            log_df = pd.DataFrame([{
                'experiment_id': 'EXP-002B',
                'method': f"tfidf_partitioned_{v['name']}",
                'text_fields': "+".join(v['fields']),
                'ngram_range': str(v['ngrams']),
                'candidate_k': k,
                'candidate_recall': recall,
                'avg_candidates': avg_cand,
                'runtime': total_time,
                'memory': 'Optimized',
                'notes': "Partitioned by Country"
            }])
            log_df.to_csv(log_path, mode='a', header=False, index=False)
            
        print(f"Total Time: {total_time:.2f}s | K=200 Recall: {results.get('recall_at_200', 0):.4f}")
        
        del blocker
        del candidates
        gc.collect()

if __name__ == "__main__":
    run_experiment()
