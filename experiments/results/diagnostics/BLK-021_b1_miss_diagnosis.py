"""B1: why does BLK-020 miss true validation pairs, and how much ceiling is each cause worth?

Finds every true val pair absent from the K=200 candidates, joins both records, classifies the miss,
and reports the val ceiling F0.5 if each category (and their union) were recovered.
Output: printed report + .tmp/b1_misses.parquet (for B2).
"""
import re
import sys
import time

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.csv as pacsv
from rapidfuzz import fuzz, process

sys.path.insert(0, "src")
from entity_resolution.data.loader import DataLoader, explode_id_lists  # noqa: E402
from entity_resolution.data.preprocessing import normalize_series  # noqa: E402
from entity_resolution.data.validation_split import create_validation_split  # noqa: E402
from entity_resolution.evaluation.evaluator import f05  # noqa: E402

t0 = time.time()
VAL = "artifacts/blocking/BLK-020_word_unigram_69a91f1/validation_candidate_pairs.tsv.gz"
L = DataLoader()
gt = L.load_ground_truth()
_, val_ids = create_validation_split(gt)
val_gt = gt[gt["source1_entity_id"].isin(val_ids)].reset_index(drop=True)
true = explode_id_lists(val_gt, "matched_entity_ids").astype("string[pyarrow]")
print(f"val S1 {len(val_gt)}, true pairs {len(true)}")

# rows of the candidate file whose candidate is some true match: small, joinable
keep = pa.array(true["candidate_entity_id"].unique().tolist())
types = {"source1_entity_id": pa.string(), "candidate_entity_id": pa.string(), "score": pa.float32(), "rank": pa.int16()}
reader = pacsv.open_csv(pa.input_stream(VAL, compression="gzip"),
                        parse_options=pacsv.ParseOptions(delimiter="\t", quote_char=False),
                        convert_options=pacsv.ConvertOptions(column_types=types),
                        read_options=pacsv.ReadOptions(block_size=1 << 26))
found = pa.Table.from_batches([b.filter(pc.is_in(b.column(1), value_set=keep)) for b in reader]).to_pandas()
tp = true.merge(found, on=["source1_entity_id", "candidate_entity_id"], how="left")
miss = tp[tp["rank"].isna()][["source1_entity_id", "candidate_entity_id"]].reset_index(drop=True)
print(f"found {tp['rank'].notna().sum()} / {len(tp)} true pairs; missed {len(miss)} "
      f"({len(miss) / len(tp):.4%}) in {time.time() - t0:.0f}s")

cols = ["entity_id", "business_name", "business_address", "country"]
s1 = L.load_source("train", 1, columns=cols).set_index("entity_id")
pool = L.load_candidates_pool("train", columns=cols)
pool = pool[pool["entity_id"].isin(set(miss["candidate_entity_id"]))].set_index("entity_id")
a = s1.reindex(miss["source1_entity_id"]).reset_index(drop=True)
b = pool.reindex(miss["candidate_entity_id"]).reset_index(drop=True)
m = miss.copy()
for side, f in (("s1", a), ("rec", b)):
    m[f"{side}_name"] = normalize_series(f["business_name"]).to_numpy()
    m[f"{side}_addr"] = normalize_series(f["business_address"]).to_numpy()
    m[f"{side}_country"] = f["country"].fillna("").to_numpy()
    m[f"{side}_raw_name"] = f["business_name"].fillna("").to_numpy()

deva = re.compile(r"[ऀ-ॿ]")
nonascii = lambda s: bool(re.search(r"[^\x00-\x7f]", s))  # noqa: E731
m["country_mismatch"] = m["s1_country"] != m["rec_country"]
m["cross_script"] = [bool(deva.search(x)) != bool(deva.search(y)) for x, y in zip(m["s1_raw_name"], m["rec_raw_name"])]
m["empty_addr"] = (m["s1_addr"] == "") | (m["rec_addr"] == "")


def toks(s):
    return set(s.split())


m["name_tok_jacc"] = [len(toks(x) & toks(y)) / max(len(toks(x) | toks(y)), 1) for x, y in zip(m["s1_name"], m["rec_name"])]
m["shared_word"] = [bool((toks(x + " " + xa)) & toks(y + " " + ya)) for x, xa, y, ya in
                    zip(m["s1_name"], m["s1_addr"], m["rec_name"], m["rec_addr"])]
m["name_fuzz"] = process.cpdist(m["s1_name"].tolist(), m["rec_name"].tolist(), scorer=fuzz.token_set_ratio, workers=-1)
m["addr_fuzz"] = process.cpdist(m["s1_addr"].tolist(), m["rec_addr"].tolist(), scorer=fuzz.token_set_ratio, workers=-1)
from anyascii import anyascii  # noqa: E402
nk = lambda s: re.sub(r"[^a-z0-9]", "", anyascii(s).lower())  # noqa: E731
m["nk_fuzz"] = process.cpdist([nk(x) for x in m["s1_raw_name"]], [nk(x) for x in m["rec_raw_name"]],
                              scorer=fuzz.ratio, workers=-1)
# "no usable text": nothing a string channel could key on, even after transliteration
m["unreachable"] = (~m["shared_word"]) & (m["nk_fuzz"] < 50) & (m["addr_fuzz"] < 50)

cats = {
    "country_mismatch": m["country_mismatch"],
    "cross_script": m["cross_script"] & ~m["country_mismatch"],
    "no_shared_word (word TF-IDF score 0)": ~m["shared_word"] & ~m["country_mismatch"],
    "shared_word -> crowded out of top-200": m["shared_word"] & ~m["country_mismatch"],
    "empty_addr (either side)": m["empty_addr"],
    "translit name similar (nk_fuzz>=80)": m["nk_fuzz"] >= 80,
    "UNREACHABLE (no shared word, nk<50, addr<50)": m["unreachable"],
}

# ceiling if a subset of misses were recovered (exact challenge formula, all val S1)
idx = pd.Index(val_gt["source1_entity_id"].astype("string[pyarrow]"))
n_true = np.bincount(idx.get_indexer(true["source1_entity_id"]), minlength=len(idx))
n_hit0 = np.bincount(idx.get_indexer(tp.loc[tp["rank"].notna(), "source1_entity_id"]), minlength=len(idx))


def ceiling(extra_mask):
    hit = n_hit0 + np.bincount(idx.get_indexer(m.loc[extra_mask, "source1_entity_id"]), minlength=len(idx))
    r = np.divide(hit, n_true, out=np.zeros(len(idx)), where=n_true > 0)
    return float(np.where(n_true > 0, f05(1.0, r), 1.0).mean())


base = ceiling(np.zeros(len(m), bool))
print(f"\nBASE ceiling@200 {base:.5f}   (perfect recall -> 1.0)")
print(f"{'category':48s} {'misses':>8s} {'share':>7s}  ceiling if recovered")
for name, mask in cats.items():
    print(f"{name:48s} {int(mask.sum()):8d} {mask.mean():7.2%}  {ceiling(mask.to_numpy()):.5f}")
print(f"{'ALL except UNREACHABLE':48s} {int((~m['unreachable']).sum()):8d} {(~m['unreachable']).mean():7.2%}  "
      f"{ceiling((~m['unreachable']).to_numpy()):.5f}   <- max achievable by any text channel")
print("\nmedians: name_fuzz", m["name_fuzz"].median(), "addr_fuzz", m["addr_fuzz"].median(), "nk_fuzz", m["nk_fuzz"].median())
print("\nsample misses:")
print(m.sample(12, random_state=42)[["s1_raw_name", "rec_raw_name", "s1_addr", "rec_addr", "s1_country", "rec_country"]]
      .to_string(max_colwidth=38))
m.drop(columns=["s1_raw_name", "rec_raw_name"]).to_parquet(".tmp/b1_misses.parquet", index=False)
print(f"\ndone in {time.time() - t0:.0f}s")
