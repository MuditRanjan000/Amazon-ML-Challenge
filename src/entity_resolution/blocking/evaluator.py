import pandas as pd
import numpy as np
import time
import os

class BlockingEvaluator:
    def __init__(self, ground_truth: pd.DataFrame):
        self.ground_truth = ground_truth
        
        # Parse ground truth to dict for fast lookup
        self.gt_dict = {}
        for _, row in ground_truth.iterrows():
            s1 = row['source1_entity_id']
            matches_str = row['matched_entity_ids']
            if pd.isna(matches_str) or matches_str.strip() == "":
                self.gt_dict[s1] = set()
            else:
                self.gt_dict[s1] = set(matches_str.split(','))

    def evaluate(self, candidate_pairs: pd.DataFrame, k_values=[10, 25, 50, 100, 200]) -> dict:
        """
        Evaluate candidate recall and reduction ratio at various K cutoffs.
        candidate_pairs must have: ['source1_entity_id', 'candidate_entity_id', 'rank']
        """
        results = {}
        
        # Calculate reduction ratio variables
        total_s1 = len(self.gt_dict)
        
        # Evaluate for each K
        for k in k_values:
            # Filter to top K
            top_k_candidates = candidate_pairs[candidate_pairs['rank'] <= k]
            
            # Group by S1 to get sets of candidates
            cand_dict = top_k_candidates.groupby('source1_entity_id')['candidate_entity_id'].apply(set).to_dict()
            
            total_true_matches = 0
            recovered_matches = 0
            
            for s1_id, true_set in self.gt_dict.items():
                total_true_matches += len(true_set)
                if len(true_set) > 0:
                    pred_set = cand_dict.get(s1_id, set())
                    recovered_matches += len(true_set.intersection(pred_set))
            
            recall = recovered_matches / total_true_matches if total_true_matches > 0 else 0.0
            avg_candidates = len(top_k_candidates) / total_s1 if total_s1 > 0 else 0.0
            
            results[f'recall_at_{k}'] = recall
            results[f'avg_cand_at_{k}'] = avg_candidates
            
        return results
