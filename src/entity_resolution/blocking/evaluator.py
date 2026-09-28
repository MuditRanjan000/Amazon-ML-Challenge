"""Blocking quality on a labelled split, computed on integer codes (no per-row Python).

Headline metric - oracle-ceiling F0.5: the score a perfect matcher would get if it
could only choose among these candidates. For an S1 entity with true set T and
candidates C: |T|>0 -> F0.5(p=1, r=|T∩C|/|T|); |T|=0 -> 1.0 (oracle predicts empty).
This is the exact upper bound blocking puts on the leaderboard metric.
"""
import numpy as np
import pandas as pd

from ..data.loader import explode_id_lists
from ..evaluation.evaluator import f05

BUCKETS = {"0": (0, 0), "1": (1, 1), "2-5": (2, 5), "6+": (6, 10 ** 9)}


def _codes(series: pd.Series, categories: pd.Index) -> np.ndarray:
    """Positions of `series` values in `categories` (-1 if absent), cheap for categoricals."""
    if isinstance(series.dtype, pd.CategoricalDtype):
        cat_pos = categories.get_indexer(series.cat.categories)
        codes = series.cat.codes.to_numpy()
        return np.where(codes >= 0, cat_pos[codes], -1)
    return categories.get_indexer(series.astype("string[pyarrow]"))


class BlockingEvaluator:
    def __init__(self, ground_truth: pd.DataFrame):
        """ground_truth: wide GT restricted to the evaluation S1 entities."""
        self.s1 = pd.Index(ground_truth["source1_entity_id"].astype("string[pyarrow]"))
        true = explode_id_lists(ground_truth, "matched_entity_ids")
        self.true_s1 = self.s1.get_indexer(true["source1_entity_id"])
        self.true_cand = pd.Index(true["candidate_entity_id"])
        self.n_true = np.bincount(self.true_s1, minlength=len(self.s1))

    def evaluate(self, candidate_pairs: pd.DataFrame, k_values=(10, 25, 50, 100, 200)) -> dict:
        """candidate_pairs: source1_entity_id, candidate_entity_id [, rank]. No rank -> evaluated as one set."""
        s1 = _codes(candidate_pairs["source1_entity_id"], self.s1)
        cand_col = candidate_pairs["candidate_entity_id"]
        if isinstance(cand_col.dtype, pd.CategoricalDtype):
            cats = cand_col.cat.categories
            cand = cand_col.cat.codes.to_numpy().astype(np.int64)
        else:
            cand, cats = pd.factorize(cand_col.astype("string[pyarrow]"))
            cats = pd.Index(cats)
        true_cand = cats.get_indexer(self.true_cand)  # -1: true match never proposed by this blocker
        width = len(cats) + 1
        true_keys = self.true_s1[true_cand >= 0].astype(np.int64) * width + true_cand[true_cand >= 0]
        valid = s1 >= 0  # candidates for S1 entities outside the evaluation set are ignored
        keys = s1.astype(np.int64) * width + cand
        hit = valid & np.isin(keys, true_keys)
        # Duplicate (s1, cand) rows would double count hits: count each true pair once.
        dup = pd.Series(keys).duplicated().to_numpy()

        ranks = candidate_pairs["rank"].to_numpy() if "rank" in candidate_pairs and k_values else None
        results = {}
        for k in (k_values if ranks is not None else ["all"]):
            within = valid & ~dup & ((ranks <= k) if ranks is not None else True)
            n_hit = np.bincount(s1[within & hit], minlength=len(self.s1))
            n_cand = np.bincount(s1[within], minlength=len(self.s1))
            results.update(self._summarise(k, n_hit, n_cand, detailed=(k == (k_values[-1] if ranks is not None else "all"))))
        return results

    def _summarise(self, k, n_hit, n_cand, detailed: bool) -> dict:
        pos = self.n_true > 0
        entity_recall = np.divide(n_hit, self.n_true, out=np.zeros(len(n_hit)), where=pos)
        ceiling = np.where(pos, f05(1.0, entity_recall), 1.0)
        out = {
            f"recall_at_{k}": n_hit.sum() / max(self.n_true.sum(), 1),
            f"ceiling_f05_at_{k}": float(ceiling.mean()),
            f"full_coverage_at_{k}": float((n_hit[pos] == self.n_true[pos]).mean()) if pos.any() else 1.0,
            f"avg_cand_at_{k}": float(n_cand.mean()),
            f"p95_cand_at_{k}": float(np.percentile(n_cand, 95)),
            f"max_cand_at_{k}": int(n_cand.max()) if len(n_cand) else 0,
            f"total_pairs_at_{k}": int(n_cand.sum()),
            f"no_candidates_at_{k}": int((n_cand == 0).sum()),
        }
        if detailed:
            for name, (lo, hi) in BUCKETS.items():
                b = (self.n_true >= lo) & (self.n_true <= hi)
                if b.any():
                    out[f"bucket_{name}_n"] = int(b.sum())
                    out[f"bucket_{name}_ceiling_f05"] = float(ceiling[b].mean())
                    if lo > 0:
                        out[f"bucket_{name}_recall"] = float(n_hit[b].sum() / self.n_true[b].sum())
        return out
