import importlib.util
import json
from pathlib import Path

import pandas as pd
import pytest

from entity_resolution.blocking.tfidf_blocking import TfidfBlocker

_spec = importlib.util.spec_from_file_location(
    "run_d2_blocking", Path(__file__).resolve().parents[1] / "scripts" / "aws" / "run_d2_blocking.py")
d2 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(d2)

NAMES = ["acme widgets", "blue river cafe", "zenith labs", "orelees barbershop", "sharma traders",
         "gupta sweets", "delta motors", "river side inn"]


def _frame(prefix, names, countries):
    return pd.DataFrame({"entity_id": [f"{prefix}{i}" for i in range(len(names))], "business_name": names,
                         "business_address": "1 main road", "country": countries}).astype("string[pyarrow]")


def test_parts_merge_is_byte_identical_to_single_run(tmp_path):
    pool = _frame("P", NAMES * 2, ["US"] * 8 + ["India"] * 8)
    s1 = _frame("S", NAMES + ["zzz qqq"], ["US", "India"] * 4 + ["US"])
    s1.loc[8, "business_address"] = ""  # last S1 shares no token with the pool -> zero candidates
    gt = pd.DataFrame({"source1_entity_id": s1["entity_id"],
                       "matched_entity_ids": [f"P{i},P{i + 8}" for i in range(8)] + [""]}).astype("string[pyarrow]")
    blocker = TfidfBlocker(**{**d2.BLOCKERS["word"], "min_df": 1}, n_jobs=1).fit(pool)
    base = {"blocker": "word", "config": blocker.config, "config_sha": "cfg", "generator_commit": "abc",
            "generated_at": "t", "load_s": 0.0, "index_s": 0.0}
    meta = {}
    for n in (1, 3):
        out = tmp_path / f"n{n}"
        (out / "parts").mkdir(parents=True)
        for i in range(1, n + 1):
            d2.run_part(blocker, "train", d2.part_slice(s1, i, n), gt, 2, out / "parts" / f"train.part{i:02d}of{n:02d}",
                        base)
        meta[n] = d2.merge_parts(out, n, "TEST", sample_s1=4, log=False)["splits"]["train"]

    assert (tmp_path / "n1" / "train_candidate_pairs.tsv").read_bytes() == \
        (tmp_path / "n3" / "train_candidate_pairs.tsv").read_bytes()
    one, three = meta[1], meta[3]
    assert one["tsv_sha256"] == three["tsv_sha256"] and three["parts"] == 3
    for key in ("n_s1", "total_pairs", "no_candidates", "p99_cand", "by_country"):
        assert one[key] == three[key]
    for key in ("recall_at_10", "recall_at_200", "ceiling_f05_at_200", "full_coverage_at_200"):
        assert one[key] == pytest.approx(three[key])
    assert three["no_candidates"] == 1 and three["n_s1"] == 9

    ids = pd.read_csv(tmp_path / "n3" / "train_sample_s1_ids.csv")
    sample = pd.read_csv(tmp_path / "n3" / "train_sample_candidate_pairs.tsv", sep="\t")
    assert len(ids) == 4 and three["sample"]["s1"] == 4
    assert len(sample) == ids["n_candidates"].sum() == three["sample"]["pairs"]
    assert set(sample["source1_entity_id"]) <= set(ids["source1_entity_id"])
    assert json.loads((tmp_path / "n3" / "blocking_metadata.json").read_text())["top_k"] == d2.K


def test_merge_refuses_missing_part(tmp_path):
    (tmp_path / "parts").mkdir()
    (tmp_path / "parts" / "train.part01of02.json").write_text("{}")
    with pytest.raises(SystemExit, match="missing parts"):
        d2.merge_parts(tmp_path, 2, "TEST", log=False)
