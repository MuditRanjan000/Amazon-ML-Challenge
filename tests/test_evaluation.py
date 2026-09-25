import random

import numpy as np
import pandas as pd
import pytest

from entity_resolution.evaluation.evaluator import Evaluator, f05, per_entity_scores


def _frame(rows, col="matched_entity_ids"):
    return pd.DataFrame(rows, columns=["source1_entity_id", col])


def _reference(gt, pred):
    """The original row-by-row implementation, kept as an oracle."""
    pred_map = {r.source1_entity_id: set(filter(None, (r.matched_entity_ids or "").split(",")))
                for r in pred.itertuples()}
    scores = []
    for r in gt.itertuples():
        t = set(filter(None, (r.matched_entity_ids or "").split(",")))
        p = pred_map.get(r.source1_entity_id, set())
        if not t:
            scores.append(1.0 if not p else 0.0)
        elif not p:
            scores.append(0.0)
        else:
            tp = len(t & p)
            pr, rc = tp / len(p), tp / len(t)
            scores.append(0.0 if pr + rc == 0 else 1.25 * pr * rc / (0.25 * pr + rc))
    return float(np.mean(scores))


def test_readme_worked_example():
    gt = _frame([("S1-1", "S2-47,S3-812")])
    pred = _frame([("S1-1", "S2-47,S2-193,S3-812")])
    assert Evaluator().evaluate(gt, pred)["macro_f05"] == pytest.approx(0.7142857, rel=1e-6)


def test_singleton_rules_and_missing_prediction_rows():
    gt = _frame([("S1-1", ""), ("S1-2", ""), ("S1-3", "S2-1"), ("S1-4", "S3-9")])
    pred = _frame([("S1-1", ""), ("S1-2", "S2-5"), ("S1-3", "S2-1")])  # S1-4 missing entirely
    m = Evaluator().evaluate(gt, pred)
    assert m["macro_f05"] == pytest.approx((1 + 0 + 1 + 0) / 4)
    assert (m["true_singletons"], m["correct_singletons"]) == (2, 1)


def test_duplicate_ids_in_prediction_list_count_once():
    gt = _frame([("S1-1", "S2-1")])
    pred = _frame([("S1-1", "S2-1,S2-1")])
    assert per_entity_scores(gt, pred)["f05"].iloc[0] == pytest.approx(1.0)


def test_matches_reference_on_random_data():
    rng = random.Random(0)
    pool = [f"S2-{i}" for i in range(40)]
    gt_rows, pred_rows = [], []
    for i in range(500):
        t = rng.sample(pool, rng.choice([0, 0, 1, 2, 3, 5]))
        p = rng.sample(pool, rng.choice([0, 1, 2, 4])) + (t[:1] if rng.random() < 0.6 else [])
        gt_rows.append((f"S1-{i}", ",".join(t)))
        pred_rows.append((f"S1-{i}", ",".join(dict.fromkeys(p))))
    gt, pred = _frame(gt_rows), _frame(pred_rows)
    assert Evaluator().evaluate(gt, pred)["macro_f05"] == pytest.approx(_reference(gt, pred), abs=1e-12)


def test_f05_scalar():
    assert f05(1.0, 1.0) == pytest.approx(1.0)
    assert f05(0.0, 0.0) == 0.0
