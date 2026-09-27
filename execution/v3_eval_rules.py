"""Evaluate final decision rules for a saved stage 2 on the (optionally orphan-shifted) validation split.
Rules: logit shifts of p2 (as in `variants`), and 'twin drop': inside the band of pairs a shift of -s would remove,
drop only pairs whose house numbers disagree with the S1 (n1 not in record, number coverage < 1) and whose address
is non-empty - the pattern the leaderboard rewarded removing.
  python execution/v3_eval_rules.py --scored D --m2 m2.joblib --drop-s1 dropped_s1.txt --K 200
"""
import argparse, json, logging, sys, time
from pathlib import Path
import joblib, numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent))
import v3_pipeline as v
from entity_resolution.data.loader import DataLoader, explode_id_lists
from entity_resolution.data.validation_split import create_validation_split


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scored", required=True); ap.add_argument("--m2", required=True)
    ap.add_argument("--drop-s1"); ap.add_argument("--K", type=int, default=200)
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    t0 = time.time()
    svs = [joblib.load(m) for m in a.m2.split(",")]  # several stage-2 models -> average p2
    best = svs[0]["best"]
    gt = DataLoader().load_ground_truth(); _, val_ids = create_validation_split(gt)
    tv = pd.concat([v.read_scored(a.scored, "train", a.K), v.read_scored(a.scored, "val", a.K)], ignore_index=True)
    if a.drop_s1:
        gone = set(open(a.drop_s1, encoding="utf-8").read().split())
        tv = tv[~tv["s1"].isin(gone)].reset_index(drop=True); val_ids = val_ids - gone
    tv = v.siblings(v.context(tv), v.norm_keys("train", set(tv["s1"]) | set(tv["cand"])))
    tv["p2"] = np.mean([sv["m2"].predict_proba(tv[sv["features"]])[:, 1] for sv in svs], axis=0)
    logging.info("p2 ready %.0fs", time.time() - t0)
    val_gt = gt[gt["source1_entity_id"].isin(val_ids)].reset_index(drop=True)
    idx = pd.Index(val_gt["source1_entity_id"]); true = explode_id_lists(val_gt, "matched_entity_ids")
    n_true = np.bincount(idx.get_indexer(true["source1_entity_id"]), minlength=len(idx))
    tk = v.truth_keys(val_gt)
    is_val = tv["s1"].isin(val_ids).to_numpy()
    y = np.fromiter((k in tk for k in zip(tv["s1"], tv["cand"])), bool, len(tv))
    codes = idx.get_indexer(tv.loc[is_val, "s1"]); hit = y[is_val]
    z = np.log(np.clip(tv["p2"].to_numpy(), 1e-7, 1 - 1e-7) / np.clip(1 - tv["p2"].to_numpy(), 1e-7, 1))
    twin = ((tv["n1_in_b"] == 0) & (tv["num_cov_a"] < 1) & (tv["num_cov_a"] >= 0) & (tv["b_alen"] > 0)).to_numpy()
    def mask(shift):
        tv["q"] = 1 / (1 + np.exp(-(z + shift)))
        return v.select(tv, dict(best, prob="q"))
    base = mask(0.0)
    res = {}
    for s in (0.5, 0.0, -0.5, -1.0, -1.5, -2.0):
        m = mask(s); res[f"shift{s:+.1f}"] = v.fast_f05(codes, n_true, hit, m[is_val])
        if s < 0:
            tm = base & ~(base & ~m & twin)  # drop only the twin-pattern pairs of the band
            res[f"twin{s:+.1f}"] = v.fast_f05(codes, n_true, hit, tm[is_val])
            band = base & ~m
            logging.info("band %.1f: %d val pairs, precision %.3f; twin part %d precision %.3f", s, (band & is_val).sum(),
                         y[band & is_val].mean(), (band & twin & is_val).sum(), y[band & twin & is_val].mean())
    print(json.dumps({k: round(x, 5) for k, x in res.items()}, indent=1))


if __name__ == "__main__":
    main()
