"""B2: which extra blocking channels recover BLK-020's missed val pairs, at what candidate cost?

Channels (all country-partitioned, same TfidfBlocker, min_df 2):
  fwd_nk@K : S1 name_key queries the pool name_key index (anyascii, legal/web-stripped, space-free, char 3-grams)
  rev_word@k: each pool record queries an index of ALL train S1 (word unigram name+address) -> its top-k S1
  rev_nk@k  : same, on name_key
Recovery is measured exactly on the B1 misses; the added-candidate cost on random samples
(new pairs not already in word@200). Reports the val ceiling for each channel and unions.
"""
import re
import sys
import time

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.csv as pacsv
from anyascii import anyascii

sys.path.insert(0, "src")
from entity_resolution.blocking.tfidf_blocking import TfidfBlocker  # noqa: E402
from entity_resolution.data.loader import DataLoader, explode_id_lists  # noqa: E402
from entity_resolution.data.preprocessing import normalize_series  # noqa: E402
from entity_resolution.data.validation_split import create_validation_split  # noqa: E402
from entity_resolution.evaluation.evaluator import f05  # noqa: E402

t0 = time.time()
log = lambda *x: print(f"[{time.time() - t0:5.0f}s]", *x, flush=True)  # noqa: E731
WORD = dict(text_fields=["business_name", "business_address"], analyzer="word", ngram_range=[1, 1], min_df=2)
NK = dict(text_fields=["name_key"], analyzer="char", ngram_range=[3, 3], min_df=2)
_WEB = r"\b(?:www|com|net|org|co|in|us|biz|info|http|https)\b"
_TL = r"\b(?:praivet|prayivet|praiveta|limitedd|limiteda|pvt|ltd|llc|inc|corp)\b"


def name_key(names):
    s = pd.Series([anyascii(x) for x in names.astype("string[pyarrow]").fillna("").to_numpy(dtype=object)],
                  index=names.index, dtype="string[pyarrow]")
    s = normalize_series(s, strip_legal=True).str.replace(_WEB, " ", regex=True)
    return s.str.replace(_TL, " ", regex=True).str.replace(r"[^a-z0-9]", "", regex=True)


L = DataLoader()
cols = ["entity_id", "business_name", "business_address", "country"]
gt = L.load_ground_truth()
_, val_ids = create_validation_split(gt)
val_gt = gt[gt["source1_entity_id"].isin(val_ids)].reset_index(drop=True)
true = explode_id_lists(val_gt, "matched_entity_ids")
miss = pd.read_parquet(".tmp/b1_misses.parquet", columns=["source1_entity_id", "candidate_entity_id"])
s1_all = L.load_source("train", 1, columns=cols)
pool = L.load_candidates_pool("train", columns=cols)
log("loaded; building name_key for", len(s1_all), "S1 and", len(pool), "pool records")
s1_all["name_key"] = name_key(s1_all["business_name"]).to_numpy()
pool["name_key"] = name_key(pool["business_name"]).to_numpy()
log("name_key done")

rng = np.random.default_rng(42)
val_s1 = s1_all[s1_all["entity_id"].isin(val_ids)]
cost_s1 = val_s1.sample(5000, random_state=42)                      # forward cost sample
cost_pool = pool.sample(50000, random_state=42)                      # reverse cost sample
miss_s1 = val_s1[val_s1["entity_id"].isin(set(miss["source1_entity_id"]))]
miss_rec = pool[pool["entity_id"].isin(set(miss["candidate_entity_id"]))]

# word@200 lists of the cost-sample S1 (to count only NEW pairs a channel adds)
keep = pa.array(cost_s1["entity_id"].tolist())
types = {"source1_entity_id": pa.string(), "candidate_entity_id": pa.string(), "score": pa.float32(), "rank": pa.int16()}
reader = pacsv.open_csv(pa.input_stream("artifacts/blocking/BLK-020_word_unigram_69a91f1/validation_candidate_pairs.tsv.gz",
                                        compression="gzip"),
                        parse_options=pacsv.ParseOptions(delimiter="\t", quote_char=False),
                        convert_options=pacsv.ConvertOptions(column_types=types))
word200 = pa.Table.from_batches([b.filter(pc.is_in(b.column(0), value_set=keep)) for b in reader]).to_pandas()
word_keys = set(zip(word200["source1_entity_id"], word200["candidate_entity_id"]))
log("word@200 of cost sample loaded:", len(word200))

miss_keys = list(zip(miss["source1_entity_id"], miss["candidate_entity_id"]))
recovered = {}   # channel -> boolean array over misses
added = {}       # channel -> new pairs per val S1 (estimate)
n_s1_total, n_pool_total = len(s1_all), len(pool)

# ---- forward name_key -------------------------------------------------------------
fwd = TfidfBlocker(**NK, n_jobs=8).fit(pool)
log("fwd name_key index built")
for K in (10, 25, 50, 100):
    got = fwd.query(miss_s1, K)
    keys = set(zip(got["query_entity_id"].astype(str), got["index_entity_id"].astype(str)))
    recovered[f"fwd_nk@{K}"] = np.array([k in keys for k in miss_keys])
    c = fwd.query(cost_s1, K)
    new = sum(1 for k in zip(c["query_entity_id"].astype(str), c["index_entity_id"].astype(str)) if k not in word_keys)
    added[f"fwd_nk@{K}"] = new / len(cost_s1)
    log(f"fwd_nk@{K}: recovers {recovered[f'fwd_nk@{K}'].mean():.2%} of misses, +{added[f'fwd_nk@{K}']:.1f} new cands/S1")
del fwd

# ---- reverse channels: index all train S1, query pool records ---------------------
for name, cfg in (("rev_word", WORD), ("rev_nk", NK)):
    rev = TfidfBlocker(**cfg, n_jobs=8).fit(s1_all)
    got = rev.query(miss_rec, 10)
    for k in (1, 3, 5, 10):
        g = got[got["rank"] <= k]
        keys = set(zip(g["index_entity_id"].astype(str), g["query_entity_id"].astype(str)))
        recovered[f"{name}@{k}"] = np.array([mk in keys for mk in miss_keys])
    c = rev.query(cost_pool, 10)
    c = c.assign(s1=c["index_entity_id"].astype(str), rec=c["query_entity_id"].astype(str))
    in_val = c["s1"].isin(val_ids)
    for k in (1, 3, 5, 10):
        ck = c[(c["rank"] <= k)]
        # pairs per S1 = (pairs from sampled records scaled to the full pool) / all S1; minus pairs already in word@200
        # overlap measured on the val part of the sample against val word@200 of the cost sample is too sparse,
        # so report the raw rate (upper bound on new pairs)
        added[f"{name}@{k}"] = len(ck) / len(cost_pool) * n_pool_total / n_s1_total
        log(f"{name}@{k}: recovers {recovered[f'{name}@{k}'].mean():.2%} of misses, <= +{added[f'{name}@{k}']:.1f} cands/S1")
    del rev

# ---- ceilings ----------------------------------------------------------------------
idx = pd.Index(val_gt["source1_entity_id"])
n_true = np.bincount(idx.get_indexer(true["source1_entity_id"]), minlength=len(idx))
miss_code = idx.get_indexer(miss["source1_entity_id"])
n_hit0 = n_true - np.bincount(miss_code, minlength=len(idx))


def ceiling(mask):
    hit = n_hit0 + np.bincount(miss_code[mask], minlength=len(idx))
    r = np.divide(hit, n_true, out=np.zeros(len(idx)), where=n_true > 0)
    return float(np.where(n_true > 0, f05(1.0, r), 1.0).mean())


print(f"\nBASE word@200 ceiling {ceiling(np.zeros(len(miss), bool)):.5f}")
rows = [(ch, m.mean(), ceiling(m), added[ch]) for ch, m in recovered.items()]
combos = [("fwd_nk@25", "rev_word@3"), ("fwd_nk@50", "rev_word@3"), ("fwd_nk@25", "rev_word@3", "rev_nk@3"),
          ("fwd_nk@50", "rev_word@5", "rev_nk@5"), ("fwd_nk@100", "rev_word@5", "rev_nk@5"),
          ("fwd_nk@100", "rev_word@10", "rev_nk@10")]
for combo in combos:
    m = np.logical_or.reduce([recovered[c] for c in combo])
    rows.append((" + ".join(combo), m.mean(), ceiling(m), sum(added[c] for c in combo)))
rep = pd.DataFrame(rows, columns=["channel(s) added to word@200", "misses recovered", "val ceiling", "~new cands/S1 (<=)"])
print(rep.to_string(index=False, float_format=lambda v: f"{v:.5f}" if v < 1.5 else f"{v:.1f}"))
rep.to_csv(".tmp/b2_report.csv", index=False)
pd.DataFrame(recovered).assign(source1_entity_id=miss["source1_entity_id"], candidate_entity_id=miss["candidate_entity_id"]
                               ).to_parquet(".tmp/b2_recovered.parquet", index=False)
log("done")
