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
    
    # Stage 2: Full Validation Set
    _, val_s1 = apply_validation_split(s1_full, 'entity_id', train_ids, val_ids)
    _, val_gt = apply_validation_split(gt, 'source1_entity_id', train_ids, val_ids)
    
    s2 = s2_full
    s3 = s3_full
    
    evaluator = BlockingEvaluator(val_gt)
    log_path = r"c:\Users\dell\Desktop\Projects\Amazon ML Challenge\experiments\results\blocking_results.csv"
    
    variants = [
        # D2: country + name + address character ngrams
        {'name': 'D2', 'fields': ['business_name', 'business_address'], 'ngrams': (3,5)}
    ]
    
    for v in variants:
        print(f"\nRunning Variant {v['name']} - Fields: {v['fields']}, Ngrams: {v['ngrams']}")
        
        blocker = TfidfBlocker(text_fields=v['fields'], ngram_range=v['ngrams'], cache_dir=f"output/tfidf_cache_exp002b_{v['name']}")
        
        start_time = time.time()
        print("Checking for existing Index cache...")
        if not os.path.exists(blocker.cache_dir) or len(os.listdir(blocker.cache_dir)) == 0:
            print("Building Index (Partitioned by Country)...")
            blocker.build_index(s2, s3)
        else:
            print("Index cache found. Skipping build.")
        index_time = time.time() - start_time
        print(f"Index built in {index_time:.2f}s")
        
        print(f"Generating Candidates for 1k queries...")
        gen_start = time.time()
        candidates = blocker.generate_candidates(val_s1, k=200)
        total_time = index_time + (time.time() - gen_start)
        
        print("Evaluating results...")
        results = evaluator.evaluate(candidates)
        
        log_df = pd.DataFrame([{
            'experiment_id': 'EXP-002B-small',
            'method': f"tfidf_partitioned_{v['name']}",
            'text_fields': "+".join(v['fields']),
            'ngram_range': str(v['ngrams']),
            'recall_10': results.get('recall_at_10', 0),
            'recall_25': results.get('recall_at_25', 0),
            'recall_50': results.get('recall_at_50', 0),
            'recall_100': results.get('recall_at_100', 0),
            'recall_200': results.get('recall_at_200', 0),
            'avg_candidates': results.get('avg_cand_at_200', 0),
            'runtime': total_time,
            'memory': 'Optimized',
            'notes': "Full validation split benchmark"
        }])
        log_df.to_csv(log_path, mode='a', header=False, index=False)
        
        # Save candidates
        output_dir = r"c:\Users\dell\Desktop\Projects\Amazon ML Challenge\artifacts\blocking"
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        candidates.to_csv(os.path.join(output_dir, "validation_candidate_pairs.tsv"), sep="\t", index=False)
            
        print(f"Total Time: {total_time:.2f}s | K=200 Recall: {results.get('recall_at_200', 0):.4f}")
        
        del blocker
        del candidates
        gc.collect()

if __name__ == "__main__":
    run_experiment()
