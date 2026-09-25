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
    
    # Sub-sample datasets to make benchmark finish in reasonable time for development iteration
    # If this is production, we would run on full val set.
    s1_full = loader.load_source('train', 1)
    s2_full = loader.load_source('train', 2)
    s3_full = loader.load_source('train', 3)
    
    train_ids, val_ids = create_validation_split(gt)
    
    # Sub-sample validation to 2,000 for quick benchmarking across variants
    # Real run would evaluate full val
    import random
    random.seed(42)
    val_ids_sample = set(random.sample(list(val_ids), min(2000, len(val_ids))))
    
    val_s1, _ = apply_validation_split(s1_full, 'entity_id', val_ids_sample, train_ids)
    val_gt, _ = apply_validation_split(gt, 'source1_entity_id', val_ids_sample, train_ids)
    
    # We can use all of S2 and S3 for realistic search index size
    s2 = s2_full
    s3 = s3_full
    
    evaluator = BlockingEvaluator(val_gt)
    log_path = r"c:\Users\dell\Desktop\Projects\Amazon ML Challenge\experiments\results\blocking_results.csv"
    
    variants = [
        # A: Name only, no partition
        {'name': 'A', 'fields': ['business_name'], 'country': False, 'ngrams': (3,5)},
        # B: Address only, no partition
        {'name': 'B', 'fields': ['business_address'], 'country': False, 'ngrams': (3,5)},
        # C: Name + Address, no partition
        {'name': 'C', 'fields': ['business_name', 'business_address'], 'country': False, 'ngrams': (3,5)},
        # D: Name + Address + Partition
        {'name': 'D1', 'fields': ['business_name', 'business_address'], 'country': True, 'ngrams': (2,5)},
        {'name': 'D2', 'fields': ['business_name', 'business_address'], 'country': True, 'ngrams': (3,5)},
        {'name': 'D3', 'fields': ['business_name', 'business_address'], 'country': True, 'ngrams': (3,6)},
    ]
    
    for v in variants:
        print(f"\nRunning Variant {v['name']} - Fields: {v['fields']}, Partition: {v['country']}, Ngrams: {v['ngrams']}")
        
        blocker = TfidfBlocker(text_fields=v['fields'], ngram_range=v['ngrams'], partition_by_country=v['country'])
        
        start_time = time.time()
        blocker.build_index(s2, s3)
        index_time = time.time() - start_time
        
        gen_start = time.time()
        candidates = blocker.generate_candidates(val_s1, k=200)
        total_time = index_time + (time.time() - gen_start)
        
        results = evaluator.evaluate(candidates)
        
        for k in [10, 25, 50, 100, 200]:
            recall = results.get(f'recall_at_{k}', 0)
            avg_cand = results.get(f'avg_cand_at_{k}', 0)
            
            log_df = pd.DataFrame([{
                'experiment_id': 'EXP-002',
                'method': f"tfidf_variant_{v['name']}",
                'text_fields': "+".join(v['fields']),
                'ngram_range': str(v['ngrams']),
                'candidate_k': k,
                'candidate_recall': recall,
                'avg_candidates': avg_cand,
                'runtime': total_time,
                'memory': 'High',
                'notes': f"Partitioned: {v['country']}"
            }])
            log_df.to_csv(log_path, mode='a', header=False, index=False)
            
        print(f"Total Time: {total_time:.2f}s | K=200 Recall: {results.get('recall_at_200', 0):.4f}")
        
        del blocker
        del candidates
        gc.collect()

if __name__ == "__main__":
    run_experiment()
