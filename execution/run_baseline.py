import pandas as pd
import numpy as np
import os

def normalize_text(series):
    return series.fillna("").astype(str).str.lower().str.strip()

def calculate_f05_macro(ground_truth, predictions):
    """
    ground_truth: dict mapping source1_entity_id to list of matched source2/3 entity_ids
    predictions: dict mapping source1_entity_id to list of matched source2/3 entity_ids
    """
    f05_scores = []
    
    for s1_id, true_matches in ground_truth.items():
        pred_matches = predictions.get(s1_id, [])
        
        true_set = set(true_matches)
        pred_set = set(pred_matches)
        
        if len(true_set) == 0 and len(pred_set) == 0:
            # Correctly predicted singleton
            f05_scores.append(1.0)
            continue
        elif len(true_set) == 0 and len(pred_set) > 0:
            # False merges on a singleton
            f05_scores.append(0.0)
            continue
        elif len(true_set) > 0 and len(pred_set) == 0:
            # Missed all matches
            f05_scores.append(0.0)
            continue
            
        tp = len(true_set.intersection(pred_set))
        fp = len(pred_set - true_set)
        fn = len(true_set - pred_set)
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        
        if precision + recall == 0:
            f05_scores.append(0.0)
        else:
            f05 = (1.25 * precision * recall) / (0.25 * precision + recall)
            f05_scores.append(f05)
            
    return np.mean(f05_scores)

def run_baseline():
    base_dir = r"c:\Users\dell\Desktop\Projects\Amazon ML Challenge\6ab10eb3b23ba_student_resource\student_resource\dataset\train"
    
    print("Loading data...")
    df1 = pd.read_csv(os.path.join(base_dir, "train_source1.tsv"), sep='\t')
    df2 = pd.read_csv(os.path.join(base_dir, "train_source2.tsv"), sep='\t')
    df3 = pd.read_csv(os.path.join(base_dir, "train_source3.tsv"), sep='\t')
    gt = pd.read_csv(os.path.join(base_dir, "train_ground_truth.tsv"), sep='\t')
    
    print("Normalizing text...")
    df1['norm_name'] = normalize_text(df1['business_name'])
    df2['norm_name'] = normalize_text(df2['business_name'])
    df3['norm_name'] = normalize_text(df3['business_name'])
    
    # Drop empty names from S2 and S3 to avoid matching them to empty S1 names
    df2 = df2[df2['norm_name'] != ""]
    df3 = df3[df3['norm_name'] != ""]
    
    # Keep only relevant columns
    df1_key = df1[['entity_id', 'country', 'norm_name']].rename(columns={'entity_id': 's1_id'})
    df2_key = df2[['entity_id', 'country', 'norm_name']].rename(columns={'entity_id': 'match_id'})
    df3_key = df3[['entity_id', 'country', 'norm_name']].rename(columns={'entity_id': 'match_id'})
    
    df_candidates = pd.concat([df2_key, df3_key], ignore_index=True)
    
    print("Finding exact matches...")
    # Exact match on norm_name and country
    matches = pd.merge(df1_key, df_candidates, on=['country', 'norm_name'], how='left')
    
    # Group by S1 id
    print("Formatting output...")
    # Remove NaNs (singletons will have NaN in match_id)
    valid_matches = matches.dropna(subset=['match_id'])
    
    grouped = valid_matches.groupby('s1_id')['match_id'].apply(lambda x: ','.join(x.unique())).reset_index()
    grouped.rename(columns={'s1_id': 'source1_entity_id', 'match_id': 'matched_entity_ids'}, inplace=True)
    
    # Merge back to ensure all S1 entities are present
    final_output = pd.merge(df1[['entity_id']].rename(columns={'entity_id': 'source1_entity_id'}), 
                           grouped, 
                           on='source1_entity_id', 
                           how='left')
    
    # Fill NaN with empty string for singletons
    final_output['matched_entity_ids'] = final_output['matched_entity_ids'].fillna("")
    
    # Save files
    os.makedirs(r"c:\Users\dell\Desktop\Projects\Amazon ML Challenge\.tmp", exist_ok=True)
    out_path = r"c:\Users\dell\Desktop\Projects\Amazon ML Challenge\.tmp\baseline_matching_results.tsv"
    final_output.to_csv(out_path, sep='\t', index=False)
    
    # For baseline, candidates = matches
    cand_path = r"c:\Users\dell\Desktop\Projects\Amazon ML Challenge\.tmp\baseline_candidate_pairs.tsv"
    final_output.rename(columns={'matched_entity_ids': 'candidate_entity_ids'}).to_csv(cand_path, sep='\t', index=False)
    
    print("Evaluating...")
    # Parse GT
    gt_dict = {}
    for _, row in gt.iterrows():
        s1 = row['source1_entity_id']
        matches_str = row['matched_entity_ids']
        if pd.isna(matches_str) or matches_str.strip() == "":
            gt_dict[s1] = []
        else:
            gt_dict[s1] = matches_str.split(',')
            
    # Parse Preds
    pred_dict = {}
    for _, row in final_output.iterrows():
        s1 = row['source1_entity_id']
        matches_str = row['matched_entity_ids']
        if pd.isna(matches_str) or matches_str.strip() == "":
            pred_dict[s1] = []
        else:
            pred_dict[s1] = matches_str.split(',')
            
    score = calculate_f05_macro(gt_dict, pred_dict)
    print(f"\n--- Results ---")
    print(f"Local F0.5 Score: {score:.5f}")
    
if __name__ == "__main__":
    run_baseline()
