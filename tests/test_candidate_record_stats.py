import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd

_spec = importlib.util.spec_from_file_location(
    "candidate_record_stats", Path(__file__).resolve().parents[1] / "execution" / "candidate_record_stats.py")
crs = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(crs)


def test_stats_match_brute_force_across_files_and_batches(tmp_path, monkeypatch):
    rng = np.random.default_rng(0)
    s1_ids = [f"S1-{i}" for i in range(40)]
    rec_ids = [f"S2-{i}" for i in range(60)]
    df = pd.DataFrame({"source1_entity_id": rng.choice(s1_ids, 500), "candidate_entity_id": rng.choice(rec_ids, 500),
                       "score": np.round(rng.random(500), 2).astype(np.float32), "rank": 1})
    df = df.drop_duplicates(["source1_entity_id", "candidate_entity_id"]).reset_index(drop=True)
    paths = []
    half = len(df) // 2
    for i, part in enumerate((df.iloc[:half], df.iloc[half:])):  # two input files, like train + val
        p = tmp_path / f"c{i}.tsv"
        part.to_csv(p, sep="\t", index=False)
        paths.append(p)
    # tiny batches so the cross-batch merge of top-2 / owner is exercised
    real = crs._batches
    monkeypatch.setattr(crs, "_batches", lambda path: (b.slice(i, 7) for b in real(path) for i in range(0, b.num_rows, 7)))
    got, rows = crs.record_stats(paths, rec_ids, s1_ids)
    got = got.set_index("candidate_entity_id")
    assert rows == len(df)
    for rec, g in df.groupby("candidate_entity_id"):
        g = g.sort_values("score", ascending=False, kind="stable")
        assert got.loc[rec, "n_s1"] == len(g)
        assert got.loc[rec, "best_score"] == g["score"].iloc[0]
        assert np.isnan(got.loc[rec, "second_score"]) if len(g) == 1 else got.loc[rec, "second_score"] == g["score"].iloc[1]
        assert got.loc[rec, "best_s1"] == g["source1_entity_id"].iloc[0]  # ties -> first seen in file order
