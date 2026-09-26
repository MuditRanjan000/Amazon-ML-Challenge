import importlib.util
from pathlib import Path

import pandas as pd

_spec = importlib.util.spec_from_file_location(
    "rule_baseline", Path(__file__).resolve().parents[1] / "execution" / "rule_baseline.py")
rb = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(rb)


def test_owner_is_best_scoring_s1_per_record_and_rules_filter():
    pairs = pd.DataFrame({
        "source1_entity_id": ["A", "A", "B", "B", "C"],
        "candidate_entity_id": ["x", "y", "x", "z", "z"],
        "score": [0.9, 0.5, 0.7, 0.8, 0.8],   # z: tie between B and C -> first seen (B) owns it
        "rank": [1, 2, 1, 2, 1],
    })
    p = rb.annotate(pairs, R=2).set_index(["source1_entity_id", "candidate_entity_id"])
    assert p["owner"].to_dict() == {("A", "x"): True, ("A", "y"): True, ("B", "x"): False,
                                    ("B", "z"): True, ("C", "z"): False}
    assert p.loc[("A", "y"), "ratio"] == pd.Series([0.5 / 0.9], dtype="float32")[0]
    got = rb.predict(p.reset_index(), t=0.6, a=0.0)
    assert set(zip(got["source1_entity_id"], got["candidate_entity_id"])) == {("A", "x"), ("B", "z")}
    assert len(rb.annotate(pairs, R=1)) == 3  # rank cutoff defines the scored (candidate) set
