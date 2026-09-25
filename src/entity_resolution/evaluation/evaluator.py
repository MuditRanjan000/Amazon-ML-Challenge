"""Official challenge metric: macro-average F0.5 over Source-1 entities (vectorized).

Per S1 entity with true set T and predicted set P:
  * T empty, P empty  -> 1.0  (correct singleton)
  * T empty, P not    -> 0.0  (false merge on a singleton)
  * otherwise         -> F0.5 = 1.25*p*r / (0.25*p + r), p=|T∩P|/|P|, r=|T∩P|/|T| (0 if |P|=0)
Averaged over every S1 row of the ground truth frame.
"""
import numpy as np
import pandas as pd

from ..data.loader import explode_id_lists


def f05(precision, recall):
    """Elementwise F0.5; 0 where precision + recall == 0. Works on scalars and arrays."""
    p, r = np.asarray(precision, dtype=float), np.asarray(recall, dtype=float)
    denom = 0.25 * p + r
    return np.where(denom > 0, 1.25 * p * r / np.where(denom > 0, denom, 1), 0.0)


def per_entity_scores(ground_truth: pd.DataFrame, predictions: pd.DataFrame,
                      pred_col: str = "matched_entity_ids") -> pd.DataFrame:
    """One row per GT S1 entity: n_true, n_pred, tp, precision, recall, f05 (for error analysis)."""
    s1 = pd.Index(ground_truth["source1_entity_id"].astype("string[pyarrow]"), name="source1_entity_id")
    true_pairs = explode_id_lists(ground_truth, "matched_entity_ids")
    pred_pairs = explode_id_lists(predictions, pred_col)
    pred_pairs = pred_pairs[pred_pairs["source1_entity_id"].isin(s1)]

    def count(frame):
        return frame.groupby("source1_entity_id").size().reindex(s1, fill_value=0).to_numpy()

    n_true, n_pred = count(true_pairs), count(pred_pairs)
    tp = count(pred_pairs.merge(true_pairs, on=["source1_entity_id", "candidate_entity_id"]))
    precision = np.divide(tp, n_pred, out=np.zeros(len(s1)), where=n_pred > 0)
    recall = np.divide(tp, n_true, out=np.zeros(len(s1)), where=n_true > 0)
    score = f05(precision, recall)
    correct_singleton = (n_true == 0) & (n_pred == 0)
    precision[correct_singleton] = recall[correct_singleton] = score[correct_singleton] = 1.0
    return pd.DataFrame({"n_true": n_true, "n_pred": n_pred, "tp": tp, "precision": precision,
                         "recall": recall, "f05": score}, index=s1)


class Evaluator:
    def evaluate(self, ground_truth: pd.DataFrame, predictions: pd.DataFrame) -> dict:
        """ground_truth / predictions: columns ['source1_entity_id', 'matched_entity_ids']."""
        e = per_entity_scores(ground_truth, predictions)
        singletons = e["n_true"] == 0
        correct = int((singletons & (e["n_pred"] == 0)).sum())
        return {
            "macro_f05": float(e["f05"].mean()),
            "macro_precision": float(e["precision"].mean()),
            "macro_recall": float(e["recall"].mean()),
            "total_entities": int(len(e)),
            "true_singletons": int(singletons.sum()),
            "correct_singletons": correct,
            "singleton_accuracy": correct / int(singletons.sum()) if singletons.any() else 0.0,
        }
