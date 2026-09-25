import pandas as pd
import numpy as np

class Evaluator:
    def __init__(self):
        pass
        
    def _parse_id_list(self, ids_str) -> set:
        if pd.isna(ids_str) or str(ids_str).strip() == "":
            return set()
        return set(str(ids_str).split(','))

    def evaluate(self, ground_truth: pd.DataFrame, predictions: pd.DataFrame) -> dict:
        """
        Evaluate predictions against ground truth using the official competition F0.5 macro-average metric.
        ground_truth: DataFrame with columns ['source1_entity_id', 'matched_entity_ids']
        predictions: DataFrame with columns ['source1_entity_id', 'matched_entity_ids']
        """
        merged = pd.merge(ground_truth, predictions, on='source1_entity_id', how='left', suffixes=('_true', '_pred'))
        
        f05_scores = []
        precisions = []
        recalls = []
        
        # Singletons tracking
        true_singletons = 0
        correct_singletons = 0
        
        for _, row in merged.iterrows():
            true_set = self._parse_id_list(row['matched_entity_ids_true'])
            pred_set = self._parse_id_list(row['matched_entity_ids_pred'])
            
            is_true_singleton = (len(true_set) == 0)
            if is_true_singleton:
                true_singletons += 1
                
            if is_true_singleton and len(pred_set) == 0:
                f05_scores.append(1.0)
                precisions.append(1.0)
                recalls.append(1.0)
                correct_singletons += 1
                continue
            elif is_true_singleton and len(pred_set) > 0:
                f05_scores.append(0.0)
                precisions.append(0.0)
                recalls.append(0.0)
                continue
            elif not is_true_singleton and len(pred_set) == 0:
                f05_scores.append(0.0)
                precisions.append(0.0)
                recalls.append(0.0)
                continue
                
            tp = len(true_set.intersection(pred_set))
            fp = len(pred_set - true_set)
            fn = len(true_set - pred_set)
            
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            
            precisions.append(precision)
            recalls.append(recall)
            
            if precision + recall == 0:
                f05_scores.append(0.0)
            else:
                f05 = (1.25 * precision * recall) / (0.25 * precision + recall)
                f05_scores.append(f05)
                
        metrics = {
            "macro_f05": np.mean(f05_scores),
            "macro_precision": np.mean(precisions),
            "macro_recall": np.mean(recalls),
            "total_entities": len(f05_scores),
            "true_singletons": true_singletons,
            "correct_singletons": correct_singletons,
            "singleton_accuracy": correct_singletons / true_singletons if true_singletons > 0 else 0.0
        }
        
        return metrics
