import numpy as np
import pandas as pd
import pytest
from sklearn.metrics.pairwise import cosine_similarity

from entity_resolution.blocking.evaluator import BlockingEvaluator
from entity_resolution.blocking.exact_name import ExactNameBlocker
from entity_resolution.blocking.tfidf_blocking import TfidfBlocker, _rank_within, reverse_candidates

NAMES = ["acme widgets", "acme widget co", "blue river cafe", "blue rivers café", "zenith labs", "zenith lab inc",
         "राम मार्केटिंग", "राम मार्केटिंग प्राइवेट", "orelees barbershop", "orelee's barber shop"]


# Tests pin a fuzzy char-trigram config explicitly so they don't depend on class defaults.
CHAR = dict(analyzer="char_wb", ngram_range=(3, 3), text_fields=["business_name"], n_jobs=1)


def _blocker(**kw):
    return TfidfBlocker(**{**CHAR, **kw})


def _records(prefix, names, country="US"):
    return pd.DataFrame({"entity_id": [f"{prefix}-{i}" for i in range(len(names))], "business_name": names,
                         "business_address": "", "country": country})


def test_rank_within():
    assert list(_rank_within(np.array([0, 0, 0, 3, 3, 7]))) == [1, 2, 3, 1, 2, 1]


def test_sparse_topk_matches_brute_force_cosine():
    index, queries = _records("S2", NAMES[1::2]), _records("S1", NAMES[0::2])
    blocker = _blocker(partition_by_country=False).fit(index)
    got = blocker.query(queries, k=3)
    # brute force with the same vectors
    iv, qv = blocker.vectors(index), blocker.vectors(queries)  # partition "" (no country split)
    sim = cosine_similarity(qv, iv)
    for qi, qid in enumerate(queries["entity_id"]):
        expect = [index["entity_id"][j] for j in np.argsort(-sim[qi])[:3] if sim[qi, j] > 0]
        rows = got[got["query_entity_id"] == qid]
        assert list(rows["index_entity_id"].astype(str)) == expect
        assert np.allclose(rows["score"], np.sort(sim[qi])[::-1][:len(expect)], atol=1e-5)
    # each query's best candidate is its own variant
    top1 = got[got["rank"] == 1]
    assert len(top1) == len(queries)
    assert (top1["query_entity_id"].astype(str).str[3:] == top1["index_entity_id"].astype(str).str[3:]).all()


def test_max_df_prunes_common_ngrams_but_keeps_best_match():
    index = _records("S2", NAMES[1::2] + ["acme"] * 20)  # make acme n-grams very common
    full = _blocker(partition_by_country=False).fit(index)
    pruned = _blocker(partition_by_country=False, max_df=0.3).fit(index)
    assert pruned.nnz < full.nnz
    top = pruned.query(_records("S1", ["zenith labs"]), k=1)
    assert list(top["index_entity_id"].astype(str)) == ["S2-2"]


def test_disk_shards_match_single_in_memory_shard(tmp_path):
    index = pd.concat([_records("S2", NAMES[1::2], "US"), _records("S3", NAMES[0::2], "India")], ignore_index=True)
    queries = pd.concat([_records("S1", NAMES[0::2], "US"), _records("Q", NAMES[1::2], "India")], ignore_index=True)
    want = _blocker().fit(index).query(queries, k=4)
    sharded = _blocker(shard_size=2, cache_dir=tmp_path, chunk_size=3)
    got = sharded.fit(index).query(queries, k=4)
    assert len(list(tmp_path.rglob("*.npz"))) > 2
    cols = ["query_entity_id", "index_entity_id"]
    assert got[cols].astype(str).equals(want[cols].astype(str))
    assert np.allclose(got["score"], want["score"]) and (got["rank"] == want["rank"]).all()
    again = _blocker(shard_size=2, cache_dir=tmp_path, chunk_size=3).fit(index)  # served from the cache
    assert again.query(queries, k=4)[cols].astype(str).equals(want[cols].astype(str))


def test_unseen_country_still_gets_candidates():
    index = pd.concat([_records("S2", NAMES[:4], "US"), _records("S3", NAMES[4:], "India")], ignore_index=True)
    queries = _records("S1", ["acme widgets", "zenith labs"], "France")  # France not in the index
    got = _blocker().fit(index).query(queries, k=2)
    assert set(got["query_entity_id"].astype(str)) == {"S1-0", "S1-1"}
    assert got.groupby("query_entity_id", observed=True).size().max() <= 2


def test_country_partition_blocks_cross_country():
    index = pd.concat([_records("S2", ["acme widgets"], "US"), _records("S3", ["acme widgets"], "India")],
                      ignore_index=True)
    got = _blocker().fit(index).query(_records("S1", ["acme widgets"], "India"), k=5)
    assert list(got["index_entity_id"].astype(str)) == ["S3-0"]


def test_reverse_candidates_columns():
    rev = reverse_candidates(_records("S1", NAMES[0::2]), _records("S2", NAMES[1::2]), k=1, **CHAR)
    assert {"source1_entity_id", "candidate_entity_id", "rev_rank", "rev_score"} <= set(rev.columns)
    assert (rev["rev_rank"] == 1).all()


def test_exact_name_blocker_ignores_legal_suffix():
    s1 = _records("S1", ["Zenith Labs Inc"])
    s2 = _records("S2", ["zenith labs", "zenith labs"], "US")
    s2.loc[1, "country"] = "India"
    got = ExactNameBlocker().build_index(s2, s2.iloc[:0]).generate_candidates(s1)
    assert list(got["candidate_entity_id"]) == ["S2-0"]


def test_blocking_evaluator_ceiling_and_buckets():
    gt = pd.DataFrame({"source1_entity_id": ["S1-a", "S1-b", "S1-c"],
                       "matched_entity_ids": ["S2-1,S2-2", "", "S3-9"]})
    cands = pd.DataFrame({"source1_entity_id": ["S1-a", "S1-a", "S1-a", "S1-b", "S1-x"],
                          "candidate_entity_id": ["S2-1", "S2-7", "S2-1", "S2-5", "S2-2"],
                          "rank": [1, 2, 3, 1, 1]})
    m = BlockingEvaluator(gt).evaluate(cands, k_values=(1, 3))
    # S1-a: recall 1/2 -> F0.5(1, .5) = 0.625/0.75; S1-b singleton -> 1; S1-c nothing -> 0
    assert m["ceiling_f05_at_3"] == pytest.approx((1.25 * 0.5 / 0.75 + 1 + 0) / 3)
    assert m["recall_at_3"] == pytest.approx(1 / 3)          # duplicate S2-1 row counted once
    assert m["recall_at_1"] == pytest.approx(1 / 3)
    assert m["full_coverage_at_3"] == pytest.approx(0.0)
    assert m["bucket_2-5_n"] == 1 and m["bucket_0_ceiling_f05"] == 1.0


def test_blocking_evaluator_accepts_categorical_output():
    gt = pd.DataFrame({"source1_entity_id": ["S1-0", "S1-2"], "matched_entity_ids": ["S2-0", "S2-2"]})
    got = _blocker().fit(_records("S2", NAMES[1::2])).generate_candidates(_records("S1", NAMES[0::2]), k=1)
    m = BlockingEvaluator(gt).evaluate(got, k_values=(1,))
    assert m["recall_at_1"] == pytest.approx(1.0)
