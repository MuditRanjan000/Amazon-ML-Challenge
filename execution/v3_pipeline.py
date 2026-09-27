"""Matcher v3 on the frozen BLK-020 candidates: GBDT on vectorized pair features + one-owner decisions.

  normalize --split train|test                      -> output/v3/norm_<split>.parquet (S1+S2+S3, once per split)
  train     --n-s1 150000 --K 100                   -> output/v3/m1.joblib (+ quick val-sample check)
  score     --cands X.tsv.gz --split train|test --part i/n --K 100 --model m1.joblib --stats S.parquet --out D
                                                    -> D/<name>.partNN.parquet: s1, cand, score, rank, p (p >= P_KEEP)
  decide    --trainval D1 --test D2 --K 100         -> stage-2 context model, val tuning (official evaluator), test files
Candidate files hold exactly 200 rows per S1, S1-contiguous, so part i/n is a row range (i-1)*S/n .. i*S/n S1s.
"""
import argparse
import json
import logging
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.csv as pacsv
import pyarrow.parquet as pq
from sklearn.ensemble import HistGradientBoostingClassifier

from entity_resolution import config
from entity_resolution.data.loader import DataLoader, explode_id_lists
from entity_resolution.data.validation_split import create_validation_split
from entity_resolution.evaluation.evaluator import Evaluator
from entity_resolution.submission.generator import SubmissionGenerator
from entity_resolution.submission.validator import SubmissionValidator
from entity_resolution.tracking import log_run
from entity_resolution.v3_match import iter_normalized, pair_features

V3 = config.OUTPUT_DIR / "v3"
ROWS_PER_S1, P_KEEP, CHUNK = 200, 0.01, 2_000_000
TYPES = {"source1_entity_id": pa.string(), "candidate_entity_id": pa.string(), "score": pa.float32(), "rank": pa.int16()}


def stream_pairs(path, K, s1_keep=None, row_range=None):
    """Stream a candidate TSV(.gz): rank <= K, optionally only S1 in s1_keep or rows in [lo, hi)."""
    stream = pa.input_stream(str(path), compression="gzip" if str(path).endswith(".gz") else None)
    reader = pacsv.open_csv(stream, parse_options=pacsv.ParseOptions(delimiter="\t", quote_char=False),
                            convert_options=pacsv.ConvertOptions(column_types=TYPES),
                            read_options=pacsv.ReadOptions(block_size=1 << 26))
    out, seen = [], 0
    keep = pa.array(sorted(s1_keep)) if s1_keep is not None else None
    for b in reader:
        n = b.num_rows
        if row_range is not None:
            lo, hi = row_range
            if seen + n <= lo or seen >= hi:
                seen += n
                if seen >= hi:
                    break
                continue
            b = b.slice(max(lo - seen, 0), min(hi, seen + n) - max(lo, seen))
        seen += n
        m = pc.less_equal(b.column(3), K)
        if keep is not None:
            m = pc.and_(m, pc.is_in(b.column(0), value_set=keep))
        out.append(b.filter(m))
    df = pa.Table.from_batches(out).to_pandas()
    df.columns = ["s1", "cand", "score", "rank"]
    return df


def load_norm(split, ids):
    t = pq.read_table(V3 / f"norm_{split}.parquet", filters=[("entity_id", "in", list(ids))])
    norm = t.to_pandas().set_index("entity_id")
    tok = V3 / f"tokdf_{split}.parquet"
    if tok.exists():  # name-token rarity: generated gibberish names are unique tokens, real words are shared
        norm = norm.join(pq.read_table(tok, filters=[("entity_id", "in", list(ids))]).to_pandas().set_index("entity_id"))
    return norm


def token_rarity(nn: pd.Series) -> pd.DataFrame:
    """Per entity: log document frequency (over all names of the split) of its rarest name token, and the mean."""
    toks = nn.str.split().explode()
    df = toks.map(toks.value_counts()).astype("float64")
    ldf = np.log1p(df)
    g = ldf.groupby(level=0)
    return pd.DataFrame({"tok_min_ldf": g.min().fillna(-1), "tok_mean_ldf": g.mean().fillna(-1)}).astype("float32")


def load_stats(path, ids):
    t = pq.read_table(path, filters=[("candidate_entity_id", "in", list(ids))])
    return t.to_pandas().set_index("candidate_entity_id")


def featurize(pairs, norm, stats, workers=-1):
    idx = norm.index
    ia, ib = idx.get_indexer(pairs["s1"]), idx.get_indexer(pairs["cand"])
    assert (ia >= 0).all() and (ib >= 0).all(), "entity missing from the normalized table"
    st = stats.reindex(pairs["cand"])
    s1_top = pairs.groupby("s1", sort=False)["score"].transform("max").to_numpy()
    feats = []
    for lo in range(0, len(pairs), CHUNK):
        sl = slice(lo, lo + CHUNK)
        feats.append(pair_features(
            ia[sl], ib[sl], norm, norm, pairs["score"].to_numpy()[sl], pairs["rank"].to_numpy()[sl], s1_top[sl],
            pairs["cand"].str.startswith("S3").to_numpy()[sl], st["best_score"].to_numpy()[sl],
            st["second_score"].to_numpy()[sl], st["n_s1"].to_numpy()[sl],
            (st["best_s1"].to_numpy()[sl] == pairs["s1"].to_numpy()[sl]), workers=workers))
    return pd.concat(feats, ignore_index=True)


def truth_keys(gt):
    t = explode_id_lists(gt, "matched_entity_ids")
    return set(zip(t["source1_entity_id"], t["candidate_entity_id"]))


def owner_mask(s1, cand, p):
    """True where p is the record's highest probability (ties -> first)."""
    c = pd.factorize(cand)[0]
    order = np.lexsort((-p, c))
    first = np.r_[True, c[order][1:] != c[order][:-1]]
    m = np.zeros(len(p), bool)
    m[order[first]] = True
    return m


def official(gt_sub, pred):
    lists = pred.groupby("s1")["cand"].agg(",".join)
    return Evaluator().evaluate(gt_sub, pd.DataFrame({
        "source1_entity_id": gt_sub["source1_entity_id"],
        "matched_entity_ids": lists.reindex(gt_sub["source1_entity_id"]).fillna("").to_numpy()}))


def cmd_normalize(a):
    t0 = time.time()
    L = DataLoader()
    V3.mkdir(parents=True, exist_ok=True)
    writer, n = None, 0
    for s in (1, 2, 3):  # one source at a time, streamed to disk: the laptop has 16 GB
        df = L.load_source(a.split, s, columns=["entity_id", "business_name", "business_address"])
        for part in iter_normalized(df, processes=a.processes):
            t = pa.Table.from_pandas(part, preserve_index=False)
            writer = writer or pq.ParquetWriter(V3 / f"norm_{a.split}.parquet", t.schema)
            writer.write_table(t)
            n += len(part)
        del df
        logging.info("source %d done (%d so far, %.0fs)", s, n, time.time() - t0)
    writer.close()
    nn = pd.read_parquet(V3 / f"norm_{a.split}.parquet", columns=["entity_id", "nn"])
    tr = token_rarity(nn["nn"])
    tr.insert(0, "entity_id", nn["entity_id"].to_numpy())
    tr.to_parquet(V3 / f"tokdf_{a.split}.parquet", index=False)
    logging.info("normalized %d %s entities in %.0fs", n, a.split, time.time() - t0)


def cmd_train(a):
    t0 = time.time()
    gt = DataLoader().load_ground_truth()
    train_ids, val_ids = create_validation_split(gt)
    if a.exclude_s1:  # S1 treated as absent (orphan simulation): never sampled, never evaluated
        gone = set(open(a.exclude_s1, encoding="utf-8").read().split())
        train_ids, val_ids = train_ids - gone, val_ids - gone
    rng = np.random.default_rng(a.seed)
    sample = set(rng.choice(sorted(train_ids), a.n_s1, replace=False))
    pairs = stream_pairs(a.train_cands, a.K, s1_keep=sample)
    logging.info("train pairs %d (%d S1) in %.0fs", len(pairs), pairs["s1"].nunique(), time.time() - t0)
    tk = truth_keys(gt[gt["source1_entity_id"].isin(sample)])
    y = np.fromiter((k in tk for k in zip(pairs["s1"], pairs["cand"])), bool, len(pairs))
    w = np.ones(len(pairs), np.float32)
    if a.deep_neg_keep < 1:  # ranks > 100 are almost all negatives: subsample them, reweight to keep calibration
        deep = (pairs["rank"].to_numpy() > 100) & ~y
        drop = deep & (rng.random(len(pairs)) >= a.deep_neg_keep)
        w[deep] = 1 / a.deep_neg_keep
        pairs, y, w = pairs[~drop].reset_index(drop=True), y[~drop], w[~drop]
    ids = set(pairs["s1"]) | set(pairs["cand"])
    X = featurize(pairs, load_norm("train", ids), load_stats(a.stats, set(pairs["cand"])))
    logging.info("features %s pos %.4f in %.0fs", X.shape, y.mean(), time.time() - t0)
    m = HistGradientBoostingClassifier(max_iter=a.iters, learning_rate=a.lr, max_leaf_nodes=a.leaves,
                                       min_samples_leaf=200, l2_regularization=1.0, random_state=42,
                                       early_stopping=False)
    m.fit(X, y, sample_weight=w)
    V3.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": m, "features": list(X.columns), "K": a.K, "n_s1": a.n_s1, "sample": sorted(sample)},
                V3 / f"m1{a.tag}.joblib")
    logging.info("trained m1 in %.0fs", time.time() - t0)
    # quick check on a val sample (competitors restricted to the sample: optimistic for the owner rule)
    vs = set(np.random.default_rng(43).choice(sorted(val_ids), a.eval_n, replace=False))  # fixed across runs
    vp = stream_pairs(a.val_cands, a.K, s1_keep=vs)
    vids = set(vp["s1"]) | set(vp["cand"])
    vp["p"] = m.predict_proba(featurize(vp, load_norm("train", vids), load_stats(a.stats, set(vp["cand"]))))[:, 1]
    gsub = gt[gt["source1_entity_id"].isin(vs)].reset_index(drop=True)
    own = owner_mask(vp["s1"].to_numpy(), vp["cand"].to_numpy(), vp["p"].to_numpy())
    res = {}
    for t in (0.3, 0.4, 0.5, 0.6, 0.7, 0.8):
        res[t] = (official(gsub, vp[vp["p"] >= t])["macro_f05"], official(gsub, vp[(vp["p"] >= t) & own])["macro_f05"])
        logging.info("val-sample t=%.1f  pairwise %.4f  owner %.4f", t, *res[t])
    log_run({"experiment_id": "V3-M1" + a.tag, "stage": "train", "blocking": "BLK-020", "K": a.K, "n_s1": a.n_s1,
             "iters": a.iters, "features": list(X.columns), "val_sample": a.eval_n,
             "val_sample_f05": {str(k): v for k, v in res.items()}, "runtime_s": round(time.time() - t0, 1)})


def cmd_score(a):
    t0 = time.time()
    i, n = map(int, a.part.split("/"))
    n_s1 = a.n_s1
    lo, hi = (i - 1) * n_s1 // n, i * n_s1 // n
    pairs = stream_pairs(a.cands, a.K, row_range=(lo * ROWS_PER_S1, hi * ROWS_PER_S1))
    assert pairs["s1"].nunique() == hi - lo, (pairs["s1"].nunique(), hi - lo)
    logging.info("part %d/%d: %d S1, %d pairs in %.0fs", i, n, hi - lo, len(pairs), time.time() - t0)
    ids = set(pairs["s1"]) | set(pairs["cand"])
    bundles = [joblib.load(m) for m in a.model.split(",")]  # several models -> average (simple ensemble)
    X = featurize(pairs, load_norm(a.split, ids), load_stats(a.stats, set(pairs["cand"])), workers=a.threads)
    ps = [b["model"].predict_proba(X[b["features"]])[:, 1].astype("float32") for b in bundles]
    pairs["p"] = np.mean(ps, axis=0).astype("float32")
    if len(ps) > 1:
        for j, q in enumerate(ps):
            pairs[f"p_m{j}"] = q
    for c in KEEP_FEATURES:  # a few raw similarities for stage 2 (interactions with sibling support)
        pairs[c] = X[c].to_numpy()
    keep = pairs[pairs["p"] >= P_KEEP]
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    keep.to_parquet(out / f"{a.name}.part{i:02d}of{n}.parquet", index=False)
    logging.info("part %d/%d: kept %d pairs (p>=%.2f) in %.0fs", i, n, len(keep), P_KEEP, time.time() - t0)


KEEP_FEATURES = ["n_tset", "n_ratio", "a_tset", "num_cov_a", "num_cov_b", "dig_ratio", "n1_in_b", "b_alen",
                 "c_is_best", "c_margin"]
S2_FEATURES = ["p", "score", "rank", "rec_n", "rec_sum", "rec_rank", "rec_other", "p_minus_other",
               "s1_n", "s1_n05", "s1_sum", "s1_max", "s1_rank", "p_over_s1max"]


def context(df):
    """Stage-2 context over one pool universe (train+val together, or test): how each pair's p compares
    with the record's other S1 (one owner per record) and with the S1's other candidates."""
    p = df["p"].to_numpy(np.float32)
    c, s = pd.factorize(df["cand"])[0], pd.factorize(df["s1"])[0]
    order = np.lexsort((-p, c))
    cs, ps = c[order], p[order]
    start = np.r_[True, cs[1:] != cs[:-1]]
    grp = np.cumsum(start) - 1
    top1 = ps[start][grp]
    second = np.where(np.r_[cs[1:] == cs[:-1], False] & start, np.r_[ps[1:], 0], 0)  # 2nd value at group start
    top2 = np.maximum.reduceat(second, np.flatnonzero(start))[grp]
    rank_in = np.arange(len(ps)) - np.flatnonzero(start)[grp] + 1
    other = np.where(rank_in == 1, top2, top1)
    out = pd.DataFrame(index=df.index)
    inv = np.empty_like(order)
    inv[order] = np.arange(len(order))
    out["rec_other"] = other[inv]
    out["rec_rank"] = rank_in[inv].astype(np.float32)
    out["rec_n"] = np.bincount(c)[c].astype(np.float32)
    out["rec_sum"] = np.bincount(c, weights=p)[c].astype(np.float32)
    out["p_minus_other"] = p - out["rec_other"].to_numpy()
    gs = pd.Series(p).groupby(s)
    out["s1_n"] = np.bincount(s)[s].astype(np.float32)
    out["s1_n05"] = np.bincount(s, weights=(p >= 0.5))[s].astype(np.float32)
    out["s1_sum"] = np.bincount(s, weights=p)[s].astype(np.float32)
    out["s1_max"] = gs.transform("max").to_numpy(np.float32)
    out["s1_rank"] = gs.rank(ascending=False, method="first").to_numpy(np.float32)
    out["p_over_s1max"] = p / np.maximum(out["s1_max"].to_numpy(), 1e-6)
    return pd.concat([df, out], axis=1)


SIB_KEYS = ("n1", "nn", "na", "nums")
S2_FEATURES += [f"{k}_{x}" for k in SIB_KEYS for x in ("sup", "cnt", "share", "eq_s1", "s1sup")]
S2_FEATURES += ["s1_name_share", "rec_name_eq_n", "name_eq"]


def siblings(df, norm):
    """Sibling consensus: the true records of an S1 are independent noisy copies of one entity and agree with
    each other (house number, name/address key) even where the S1 itself was perturbed; distractor twins don't.
    For each key: p-mass / count of the S1's OTHER candidates sharing r's value, its share, and equality with S1."""
    s = pd.factorize(df["s1"])[0].astype(np.int64)
    p = df["p"].to_numpy(np.float64)
    s1_sum = np.bincount(s, weights=p)[s]
    out = {}
    for k in SIB_KEYS:
        val = norm[k].reindex(df["cand"]).fillna("").to_numpy()
        empty = val == ""
        kc = pd.factorize(val)[0].astype(np.int64)
        kk = pd.factorize(s * (kc.max() + 2) + kc)[0]
        sup = np.bincount(kk, weights=p)[kk] - p
        out[f"{k}_sup"] = np.where(empty, -1, sup)
        out[f"{k}_cnt"] = np.where(empty, -1, np.bincount(kk)[kk] - 1)
        out[f"{k}_share"] = np.where(empty, -1, sup / np.maximum(s1_sum - p, 1e-6))
        s1val = norm[k].reindex(df["s1"]).fillna("").to_numpy()
        eq = (val == s1val) & ~empty
        out[f"{k}_eq_s1"] = eq
        # twin contrast: p-mass of the S1's OTHER candidates carrying the S1's own value. A twin cluster has its
        # own support (k_sup) while the S1's value is backed by a different group (k_s1sup)
        out[f"{k}_s1sup"] = np.bincount(s, weights=p * eq)[s] - p * eq
    # name ambiguity: 49% of S1 share their normalized name with another S1, so the name alone can't pick the owner
    s1nn = norm["nn"].reindex(df["s1"]).fillna("").to_numpy()
    rnn = norm["nn"].reindex(df["cand"]).fillna("").to_numpy()
    per_s1 = pd.Series(s1nn).groupby(s).first()
    out["s1_name_share"] = pd.Series(s1nn).map(per_s1.value_counts()).to_numpy()
    name_eq = (s1nn == rnn) & (rnn != "")
    c = pd.factorize(df["cand"])[0]
    out["rec_name_eq_n"] = np.bincount(c, weights=name_eq)[c]
    out["name_eq"] = name_eq
    return pd.concat([df, pd.DataFrame({k: np.asarray(v, np.float32) for k, v in out.items()}, index=df.index)],
                     axis=1)


def norm_keys(split, ids):
    t = pq.read_table(V3 / f"norm_{split}.parquet", columns=["entity_id", *SIB_KEYS],
                      filters=[("entity_id", "in", list(ids))])
    return t.to_pandas().set_index("entity_id")


def read_scored(d, name, K=None):
    parts = sorted(Path(d).glob(f"{name}.part*.parquet"))
    assert parts, f"no {name} parts in {d}"
    df = pd.concat([pd.read_parquet(p) for p in parts], ignore_index=True)
    return df if K is None else df[df["rank"] <= K].reset_index(drop=True)


def decisions(df, prob, t):
    """One owner per record on `prob`, then threshold."""
    own = owner_mask(df["s1"].to_numpy(), df["cand"].to_numpy(), prob)
    return own & (prob >= t)


def efo_mask(s1, q, bias=0.0):
    """Per S1, predict the top-k candidates (by q) that maximize plug-in expected F0.5:
    E[F_k] ~ 1.25*sum_{i<=k} q_i / (0.25*sum_all q + k + bias), and E[F_0] = prod(1 - q_i) (empty = singleton).
    `bias` (tuned on val) makes the rule more conservative (> 0) or more generous (< 0)."""
    s = pd.factorize(s1)[0]
    order = np.lexsort((-q, s))
    ss, qs = s[order], q[order].astype(np.float64)
    start = np.r_[True, ss[1:] != ss[:-1]]
    grp = np.cumsum(start) - 1
    st = np.flatnonzero(start)
    csum = np.cumsum(qs)
    tp = csum - np.r_[0, csum[st[1:] - 1]][grp]                  # within-group cumulative q
    total = np.add.reduceat(qs, st)[grp]
    k = np.arange(len(qs)) - st[grp] + 1
    ef = 1.25 * tp / (0.25 * total + k + bias)
    f0 = np.exp(np.add.reduceat(np.log1p(-np.minimum(qs, 1 - 1e-6)), st))  # P(no true match among candidates)
    best_k_val = np.maximum.reduceat(ef, st)
    best_pos = pd.Series(ef).groupby(grp).idxmax().to_numpy()    # first argmax position per group
    kstar = np.where(best_k_val > f0, k[best_pos], 0)
    m = np.zeros(len(q), bool)
    m[order[k <= kstar[grp]]] = True
    return m


def owned_q(df, prob, owner):
    """The rule's score per pair; with `owner`, 0 unless the S1 is the record's top-`prob` S1 (whole universe)."""
    q = df[prob].to_numpy(np.float64)
    return np.where(owner_mask(df["s1"].to_numpy(), df["cand"].to_numpy(), q), q, 0.0) if owner else q


def select(df, r, q=None, rows=None):
    """Apply a decision rule {prob, kind: thr|efo, x: threshold|bias, owner}. Owner is decided over all of df;
    the rule itself is per S1, so it can be evaluated on a subset of S1 rows (`rows`)."""
    q = owned_q(df, r["prob"], r["owner"]) if q is None else q
    s1 = df["s1"].to_numpy()
    if rows is not None:
        q, s1 = q[rows], s1[rows]
    return q >= r["x"] if r["kind"] == "thr" else efo_mask(s1, q, r["x"]) & (q > 0)


def fast_f05(s1_codes, n_true, hit, m):
    n_pred = np.bincount(s1_codes[m], minlength=len(n_true))
    tp = np.bincount(s1_codes[m & hit], minlength=len(n_true))
    prec = np.divide(tp, n_pred, out=np.zeros(len(tp)), where=n_pred > 0)
    rec = np.divide(tp, n_true, out=np.zeros(len(tp)), where=n_true > 0)
    from entity_resolution.evaluation.evaluator import f05
    return float(np.where((n_true == 0) & (n_pred == 0), 1.0, f05(prec, rec)).mean())


def write_candidate_lists(cands_path, K, out_path):
    """candidate_pairs.tsv straight from the (S1-contiguous) candidate file: every rank <= K pair = the scored set."""
    n_rows = 0
    with open(out_path, "w", encoding="utf-8", newline="\n") as f:
        f.write("source1_entity_id\tcandidate_entity_ids\n")
        carry = None
        stream = pa.input_stream(str(cands_path), compression="gzip" if str(cands_path).endswith(".gz") else None)
        reader = pacsv.open_csv(stream, parse_options=pacsv.ParseOptions(delimiter="\t", quote_char=False),
                                convert_options=pacsv.ConvertOptions(column_types=TYPES),
                                read_options=pacsv.ReadOptions(block_size=1 << 26))
        for b in reader:
            d = b.filter(pc.less_equal(b.column(3), K)).select([0, 1]).to_pandas()
            d.columns = ["s1", "cand"]
            if carry is not None:
                d = pd.concat([carry, d], ignore_index=True)
            last = d["s1"].iloc[-1]
            carry, d = d[d["s1"] == last], d[d["s1"] != last]
            lists = d.groupby("s1", sort=False)["cand"].agg(",".join)
            f.writelines(f"{k}\t{v}\n" for k, v in lists.items())
            n_rows += len(lists)
        lists = carry.groupby("s1", sort=False)["cand"].agg(",".join)
        f.writelines(f"{k}\t{v}\n" for k, v in lists.items())
        n_rows += len(lists)
    return n_rows


def cmd_decide(a):
    t0 = time.time()
    if a.reuse:  # test only, with the stage-2 model + rule already tuned on val
        saved = joblib.load(V3 / f"m2{a.tag}.joblib")
        return apply_test(a, saved["m2"], saved["features"], saved["best"], t0)
    gt = DataLoader().load_ground_truth()
    train_ids, val_ids = create_validation_split(gt)
    m1_sample = set().union(*(joblib.load(m)["sample"] for m in a.m1.split(",")))  # never fit stage 2 on them
    tv = context(pd.concat([read_scored(a.scored, "train", a.K), read_scored(a.scored, "val", a.K)], ignore_index=True))
    tv = siblings(tv, norm_keys("train", set(tv["s1"]) | set(tv["cand"])))
    logging.info("trainval scored pairs %d in %.0fs", len(tv), time.time() - t0)
    tk = truth_keys(gt)
    tv["y"] = np.fromiter((k in tk for k in zip(tv["s1"], tv["cand"])), bool, len(tv))
    is_val = tv["s1"].isin(val_ids).to_numpy()
    fit_s1 = np.array(sorted(set(tv.loc[~is_val, "s1"]) - m1_sample))
    fit_s1 = set(np.random.default_rng(7).choice(fit_s1, min(a.s2_n, len(fit_s1)), replace=False))
    fit = tv["s1"].isin(fit_s1).to_numpy()
    m2 = HistGradientBoostingClassifier(max_iter=a.s2_iters, learning_rate=0.08, max_leaf_nodes=63, min_samples_leaf=200,
                                        random_state=42, early_stopping=False)
    feats = S2_FEATURES + [c for c in KEEP_FEATURES if c in tv.columns]
    m2.fit(tv.loc[fit, feats], tv.loc[fit, "y"])
    tv["p2"] = m2.predict_proba(tv[feats])[:, 1].astype(np.float32)
    logging.info("stage 2 fit on %d pairs (%d S1) in %.0fs", fit.sum(), len(fit_s1), time.time() - t0)

    val_gt = gt[gt["source1_entity_id"].isin(val_ids)].reset_index(drop=True)
    s1_index = pd.Index(val_gt["source1_entity_id"])
    true = explode_id_lists(val_gt, "matched_entity_ids")
    n_true = np.bincount(s1_index.get_indexer(true["source1_entity_id"]), minlength=len(s1_index))
    codes, hit = s1_index.get_indexer(tv.loc[is_val, "s1"]), tv.loc[is_val, "y"].to_numpy()
    rules = [{"prob": c, "kind": "thr", "x": float(t), "owner": o}
             for c in ("p", "p2") for t in np.round(np.arange(0.2, 0.96, 0.025), 3) for o in (True, False)]
    rules += [{"prob": c, "kind": "efo", "x": float(b), "owner": o}
              for c in ("p", "p2") for b in np.round(np.arange(-0.6, 1.61, 0.2), 2) for o in (True, False)]
    qs = {(c, o): owned_q(tv, c, o) for c in ("p", "p2") for o in (True, False)}
    res = [dict(r, f05=fast_f05(codes, n_true, hit, select(tv, r, qs[r["prob"], r["owner"]], is_val)))
           for r in rules]
    res = pd.DataFrame(res).sort_values("f05", ascending=False)
    print(res.head(15).to_string(index=False))
    best = res.iloc[0].to_dict()
    pred = tv[is_val][select(tv, best, rows=is_val)]
    off = official(val_gt, pred)
    print("official evaluator (full frozen val):", json.dumps(off))
    joblib.dump({"m2": m2, "features": feats, "best": best}, V3 / f"m2{a.tag}.joblib")
    log_run({"experiment_id": "V3" + a.tag, "stage": "decide-val", "blocking": "BLK-020", "K": a.K,
             "best": best, "official": off, "top": res.head(12).to_dict("records"),
             "runtime_s": round(time.time() - t0, 1)})
    if a.test_cands:
        apply_test(a, m2, feats, best, t0)


def check_candidate_file(path, s1_ids, matches):
    seen, problems = set(), 0
    with open(path, encoding="utf-8") as f:
        assert f.readline().rstrip("\n") == "source1_entity_id\tcandidate_entity_ids"
        for line in f:
            k, v = line.rstrip("\n").split("\t")
            ids = v.split(",") if v else []
            problems += (k not in s1_ids) + (k in seen) + (len(set(ids)) != len(ids)) + (not matches.get(k, set()) <= set(ids))
            seen.add(k)
    problems += len(s1_ids - seen)
    logging.info("candidate file check: %d rows, %d problems", len(seen), problems)
    return problems == 0


def apply_test(a, m2, feats, best, t0):
    te = context(read_scored(a.scored, "test", a.K))
    te = siblings(te, norm_keys("test", set(te["s1"]) | set(te["cand"])))
    te["p2"] = m2.predict_proba(te[feats])[:, 1].astype(np.float32)
    tp = te[select(te, best)]
    out = Path(a.out)
    s1_ids = DataLoader().load_source("test", 1, columns=["entity_id"])["entity_id"].tolist()
    match = SubmissionGenerator(out).generate(s1_ids, tp.rename(columns={"s1": "source1_entity_id",
                                                                         "cand": "candidate_entity_id"}))
    n_c = write_candidate_lists(a.test_cands, a.K, out / "candidate_pairs.tsv")
    logging.info("test: %d predicted pairs on %d S1; candidate rows %d; %.0fs", len(tp), tp["s1"].nunique(), n_c,
                 time.time() - t0)
    # the official validator holds candidate_pairs in memory (173M IDs > laptop RAM): run it on the matches with
    # --check-ids, and stream-check the candidate file (all S1 once, no duplicates, matches within candidates)
    ok = SubmissionValidator().validate(match, None, check_ids=True) and check_candidate_file(
        out / "candidate_pairs.tsv", set(s1_ids), tp.groupby("s1")["cand"].agg(set).to_dict())
    log_run({"experiment_id": "V3" + a.tag, "stage": "decide-test", "K": a.K, "best": best,
             "pred_pairs": len(tp), "validator_pass": ok})


def cmd_variants(a):
    """Test-time decision variants from one stage-2 pass: per-country logit shifts of p2 before the tuned rule.
    Test has ~23% more records per S1 than train (denser distractor twins), so val-tuned decisions are optimistic;
    the leaderboard calibrates the shift. --variants '{"base": 0, "cons1": {"US": -0.3, "India": -0.6, "France": -0.9}}'"""
    t0 = time.time()
    saved = joblib.load(a.m2)
    m2, feats, best = saved["m2"], saved["features"], saved["best"]
    te = context(read_scored(a.scored, "test", a.K))
    te = siblings(te, norm_keys("test", set(te["s1"]) | set(te["cand"])))
    te["p2"] = m2.predict_proba(te[feats])[:, 1]
    s1 = DataLoader().load_source("test", 1, columns=["entity_id", "country"])
    cty = s1.set_index("entity_id")["country"].reindex(te["s1"]).to_numpy()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    te[["s1", "cand", "rank", "p", "p2"]].assign(country=cty).to_parquet(out / "test_p2.parquet", index=False)
    z = np.log(np.clip(te["p2"].to_numpy(), 1e-7, 1 - 1e-7) / np.clip(1 - te["p2"].to_numpy(), 1e-7, 1))
    n_s1 = s1["country"].value_counts()
    for name, shift in json.loads(a.variants).items():
        d = (pd.Series(cty).map(shift).fillna(shift.get("*", 0)).to_numpy() if isinstance(shift, dict)
             else np.full(len(te), float(shift)))
        te["q"] = 1 / (1 + np.exp(-(z + d)))
        tp = te[select(te, dict(best, prob="q"))]
        SubmissionGenerator(out / name).generate(s1["entity_id"].tolist(), tp.rename(
            columns={"s1": "source1_entity_id", "cand": "candidate_entity_id"}))
        per = (pd.Series(cty[select(te, dict(best, prob="q"))]).value_counts() / n_s1).round(3).to_dict()
        logging.info("variant %s shift=%s: %d pairs; per S1 by country %s", name, shift, len(tp), per)
    logging.info("variants done in %.0fs", time.time() - t0)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("normalize")
    s.add_argument("--split", required=True, choices=["train", "test"])
    s.add_argument("--processes", type=int, default=10)
    s = sub.add_parser("train")
    s.add_argument("--train-cands", required=True)
    s.add_argument("--val-cands", required=True)
    s.add_argument("--stats", required=True)
    s.add_argument("--n-s1", type=int, default=150_000)
    s.add_argument("--eval-n", type=int, default=20_000)
    s.add_argument("--K", type=int, default=100)
    s.add_argument("--iters", type=int, default=400)
    s.add_argument("--tag", default="", help="model file suffix: output/v3/m1<tag>.joblib")
    s.add_argument("--deep-neg-keep", type=float, default=1.0, help="keep this share of rank>100 negatives")
    s.add_argument("--lr", type=float, default=0.08)
    s.add_argument("--leaves", type=int, default=127)
    s.add_argument("--seed", type=int, default=42, help="train-S1 sample seed (different seeds = ensemble diversity)")
    s.add_argument("--exclude-s1", help="file of S1 IDs treated as absent (pair it with --stats computed without them)")
    s = sub.add_parser("score")
    s.add_argument("--cands", required=True)
    s.add_argument("--split", required=True, choices=["train", "test"], help="which normalized table")
    s.add_argument("--name", required=True, help="output prefix, e.g. train / val / test")
    s.add_argument("--n-s1", type=int, required=True, help="number of S1 in the candidate file")
    s.add_argument("--part", required=True)
    s.add_argument("--K", type=int, default=100)
    s.add_argument("--model", required=True)
    s.add_argument("--stats", required=True)
    s.add_argument("--out", required=True)
    s.add_argument("--threads", type=int, default=-1)
    s = sub.add_parser("decide")
    s.add_argument("--scored", required=True, help="dir with train/val/test .partNN parquet from `score`")
    s.add_argument("--test-cands", help="test candidate TSV(.gz); omit to only tune on val")
    s.add_argument("--K", type=int, default=100)
    s.add_argument("--s2-n", type=int, default=400_000, help="train S1 (not in the m1 sample) for stage 2")
    s.add_argument("--m1", default=str(V3 / "m1.joblib"), help="the stage-1 model the scored parts came from")
    s.add_argument("--s2-iters", type=int, default=300)
    s.add_argument("--reuse", action="store_true", help="skip val tuning; apply the saved m2<tag> + rule to test")
    s.add_argument("--tag", default="")
    s.add_argument("--out", default=str(config.OUTPUT_DIR / "submissions" / "V3"))
    s = sub.add_parser("variants")
    s.add_argument("--scored", required=True)
    s.add_argument("--m2", required=True, help="m2<tag>.joblib saved by decide")
    s.add_argument("--K", type=int, default=200)
    s.add_argument("--variants", required=True, help='JSON {name: shift | {country: shift, "*": default}}')
    s.add_argument("--out", required=True)
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    {"normalize": cmd_normalize, "train": cmd_train, "score": cmd_score, "decide": cmd_decide, "variants": cmd_variants}[a.cmd](a)


if __name__ == "__main__":
    main()
