"""Matcher v3: vectorized pair features (RapidFuzz + numbers + blocking/competition context) for a GBDT.

normalize_records(df)  -> one row per entity: nn (name key), na (address key), nums, dig, n1, flags
pair_features(...)     -> float32 feature frame for aligned (S1 row, record row) index arrays
All string work is per-entity (done once); per-pair work is RapidFuzz cpdist (C++, multi-threaded) + numpy.
"""
import re
from multiprocessing import Pool

import numpy as np
import pandas as pd
from anyascii import anyascii
from rapidfuzz import fuzz, process
from rapidfuzz.distance import JaroWinkler

LEGAL = set("""private pvt pvtltd limited limite ltd llp llc inc incorporated corp corporation co company companies pc lp
plc gmbh sa sas sarl eurl sarlu the dr mr mrs ms sri shri shree smt m s formerly""".split())
ABBR = {"street": "st", "saint": "st", "str": "st", "road": "rd", "avenue": "ave", "av": "ave", "drive": "dr",
        "boulevard": "blvd", "lane": "ln", "highway": "hwy", "court": "ct", "place": "pl", "parkway": "pkwy",
        "pky": "pkwy", "circle": "cir", "square": "sq", "suite": "ste", "apartment": "apt", "building": "bldg",
        "floor": "fl", "north": "n", "south": "s", "east": "e", "west": "w", "number": "no", "near": "nr",
        "opposite": "opp", "sector": "sec", "house": "h", "hno": "h no", "door": "d"}
STATES = {  # full name -> code, applied to the normalized address string before tokenizing
    "alabama": "al", "alaska": "ak", "arizona": "az", "arkansas": "ar", "california": "ca", "colorado": "co",
    "connecticut": "ct", "delaware": "de", "florida": "fl", "georgia": "ga", "hawaii": "hi", "idaho": "id",
    "illinois": "il", "indiana": "in", "iowa": "ia", "kansas": "ks", "kentucky": "ky", "louisiana": "la",
    "maine": "me", "maryland": "md", "massachusetts": "ma", "michigan": "mi", "minnesota": "mn",
    "mississippi": "ms", "missouri": "mo", "montana": "mt", "nebraska": "ne", "nevada": "nv",
    "new hampshire": "nh", "new jersey": "nj", "new mexico": "nm", "new york": "ny", "north carolina": "nc",
    "north dakota": "nd", "ohio": "oh", "oklahoma": "ok", "oregon": "or", "pennsylvania": "pa",
    "rhode island": "ri", "south carolina": "sc", "south dakota": "sd", "tennessee": "tn", "texas": "tx",
    "utah": "ut", "vermont": "vt", "virginia": "va", "washington": "wa", "west virginia": "wv",
    "wisconsin": "wi", "wyoming": "wy", "district of columbia": "dc",
    "andhra pradesh": "ap", "arunachal pradesh": "ar", "assam": "as", "bihar": "br", "chhattisgarh": "cg",
    "goa": "ga", "gujarat": "gj", "haryana": "hr", "himachal pradesh": "hp", "jharkhand": "jh",
    "karnataka": "ka", "kerala": "kl", "madhya pradesh": "mp", "maharashtra": "mh", "manipur": "mn",
    "meghalaya": "ml", "mizoram": "mz", "nagaland": "nl", "odisha": "od", "orissa": "od", "punjab": "pb",
    "rajasthan": "rj", "sikkim": "sk", "tamil nadu": "tn", "telangana": "tg", "tripura": "tr",
    "uttar pradesh": "up", "uttarakhand": "uk", "west bengal": "wb", "delhi": "dl", "new delhi": "dl",
    "chandigarh": "ch", "puducherry": "py", "pondicherry": "py", "jammu and kashmir": "jk"}
_STATE_RE = re.compile(r"\b(" + "|".join(sorted(map(re.escape, STATES), key=len, reverse=True)) + r")\b")
_DOMAIN = re.compile(r"\.(com|in|net|org|co|biz|info|fr)\b|\bwww\.")
_NONALNUM = re.compile(r"[^a-z0-9]+")
_ALPHA = re.compile("[a-z]")
_DIGITS = re.compile(r"\d+")
_LEET = str.maketrans("013457", "oleast")


LEGAL_CLASS = {"private": "pvt", "pvt": "pvt", "pvtltd": "pvt", "limited": "ltd", "limite": "ltd", "ltd": "ltd",
               "llp": "llp", "llc": "llc", "inc": "inc", "incorporated": "inc", "corp": "corp",
               "corporation": "corp", "co": "co", "company": "co", "companies": "co", "pc": "pc", "lp": "lp",
               "plc": "plc", "public": "public", "gmbh": "gmbh", "sa": "sa", "sas": "sas", "sarl": "sarl",
               "eurl": "eurl", "sarlu": "sarl"}


def legal_classes(s: str) -> str:
    """Sorted legal-form classes in a name ('l l c' counts as llc). Distractors often swap the legal form."""
    s = _NONALNUM.sub(" ", anyascii(s).lower())
    s = s.replace("l l c", "llc").replace("l l p", "llp")
    return " ".join(sorted({LEGAL_CLASS[t] for t in s.split() if t in LEGAL_CLASS}))


def norm_name(s: str) -> str:
    s = _NONALNUM.sub(" ", _DOMAIN.sub(" ", anyascii(s).lower()))
    toks = [t.translate(_LEET) if _ALPHA.search(t) else t for t in s.split()]
    return " ".join(t for t in toks if t not in LEGAL)


def norm_addr(s: str) -> str:
    s = anyascii(s).lower().replace("<null>", " ").replace("null", " ")
    s = _STATE_RE.sub(lambda m: STATES[m.group(1)], _NONALNUM.sub(" ", s))
    return " ".join(ABBR.get(t, t) for t in s.split())


def _norm_chunk(args):
    names, addrs = args
    out = []
    for n, a in zip(names, addrs):
        nums = [x.lstrip("0") or "0" for x in _DIGITS.findall(a)]
        out.append((norm_name(n), norm_addr(a), " ".join(sorted(set(nums))), "".join(nums),
                    nums[0] if nums else "", not n.isascii(), legal_classes(n)))
    return out


def _norm_frame(args):
    ids, names, addrs = args
    out = pd.DataFrame(_norm_chunk((names, addrs)), columns=["nn", "na", "nums", "dig", "n1", "nonlatin", "lg"])
    out.insert(0, "entity_id", ids)
    return out


def iter_normalized(df: pd.DataFrame, processes: int = 0, chunk: int = 100_000):
    """Yield normalized frames chunk by chunk (bounded memory; order kept)."""
    ids = df["entity_id"].astype(str).tolist()
    names = df["business_name"].fillna("").astype(str).tolist()
    addrs = df["business_address"].fillna("").astype(str).tolist()
    jobs = ((ids[i:i + chunk], names[i:i + chunk], addrs[i:i + chunk]) for i in range(0, len(ids), chunk))
    if processes == 1:
        yield from map(_norm_frame, jobs)
        return
    with Pool(processes or None) as pool:
        yield from pool.imap(_norm_frame, jobs)


def normalize_records(df: pd.DataFrame, processes: int = 0, chunk: int = 100_000) -> pd.DataFrame:
    """df: entity_id, business_name, business_address -> entity_id + normalized columns (order kept)."""
    return pd.concat(list(iter_normalized(df, processes, chunk)), ignore_index=True)


def token_diff(an, bn, tol=65):
    """Per pair: record-name tokens absent from the S1 name (typo-tolerant), S1 tokens absent from the record,
    and the extra tokens' total length. Distractors ADD descriptive words; true copies mostly drop or garble them."""
    ex = np.zeros(len(an), np.float32); ms = np.zeros(len(an), np.float32); exl = np.zeros(len(an), np.float32)
    for i, (x, y) in enumerate(zip(an, bn)):
        a, b = set(x.split()), set(y.split())
        e, m = b - a, a - b
        if e:
            e = [t for t in e if not a or max(fuzz.ratio(t, u) for u in a) < tol]
            ex[i], exl[i] = len(e), sum(map(len, e))
        if m:
            ms[i] = sum(1 for t in m if not b or max(fuzz.ratio(t, u) for u in b) < tol)
    return ex, ms, exl


def _cp(a, b, scorer, workers):
    return process.cpdist(a, b, scorer=scorer, workers=workers, dtype=np.float32)


def pair_features(ia, ib, A: pd.DataFrame, B: pd.DataFrame, score, rank, s1_top, is_s3,
                  st_best, st_second, st_n, st_is_best, workers=-1) -> pd.DataFrame:
    """ia/ib: row positions into A (S1 table) / B (record table), aligned per pair.
    s1_top: the S1's top-1 blocking score; st_*: the record's competition stats (over ALL S1 of the split)."""
    an, bn = A["nn"].to_numpy()[ia], B["nn"].to_numpy()[ib]
    aa, ba = A["na"].to_numpy()[ia], B["na"].to_numpy()[ib]
    f = {"score": score, "rank": rank, "ratio_top": score / np.maximum(s1_top, 1e-6), "gap_top": s1_top - score,
         "src3": is_s3}
    for k, sc in (("n_ratio", fuzz.ratio), ("n_tset", fuzz.token_set_ratio), ("n_tsort", fuzz.token_sort_ratio),
                  ("n_part", fuzz.partial_ratio), ("n_jw", JaroWinkler.normalized_similarity)):
        f[k] = _cp(an, bn, sc, workers)
    ans = np.array([x.replace(" ", "") for x in an], dtype=object)
    bns = np.array([x.replace(" ", "") for x in bn], dtype=object)
    f["n_nosp"] = _cp(ans, bns, fuzz.ratio, workers)
    f["n_nosp_part"] = _cp(ans, bns, fuzz.partial_ratio, workers)
    f["n_eq"] = an == bn
    f["a_nlen"] = np.fromiter((len(x) for x in an), np.float32, len(an))
    f["b_nlen"] = np.fromiter((len(x) for x in bn), np.float32, len(bn))
    for k, sc in (("a_tset", fuzz.token_set_ratio), ("a_tsort", fuzz.token_sort_ratio),
                  ("a_part", fuzz.partial_ratio), ("a_ratio", fuzz.ratio)):
        f[k] = _cp(aa, ba, sc, workers)
    f["a_alen"] = np.fromiter((len(x) for x in aa), np.float32, len(aa))
    f["b_alen"] = np.fromiter((len(x) for x in ba), np.float32, len(ba))
    # numbers: the generator perturbs house/unit numbers, so keep graded similarities, not only equality
    na_, nb_ = A["nums"].to_numpy()[ia], B["nums"].to_numpy()[ib]
    inter = np.fromiter((len(set(x.split()) & set(y.split())) for x, y in zip(na_, nb_)), np.float32, len(na_))
    la = np.fromiter((len(x.split()) for x in na_), np.float32, len(na_))
    lb = np.fromiter((len(y.split()) for y in nb_), np.float32, len(nb_))
    f["num_inter"], f["num_la"], f["num_lb"] = inter, la, lb
    f["num_cov_a"] = np.where(la > 0, inter / np.maximum(la, 1), -1)
    f["num_cov_b"] = np.where(lb > 0, inter / np.maximum(lb, 1), -1)
    f["dig_ratio"] = _cp(A["dig"].to_numpy()[ia], B["dig"].to_numpy()[ib], fuzz.ratio, workers)
    a1, b1 = A["n1"].to_numpy()[ia], B["n1"].to_numpy()[ib]
    f["n1_ratio"] = _cp(a1, b1, fuzz.ratio, workers)
    f["n1_in_b"] = np.fromiter((x != "" and x in y.split() for x, y in zip(a1, nb_)), np.float32, len(a1))
    f["a_nonlatin"] = A["nonlatin"].to_numpy()[ia]
    f["b_nonlatin"] = B["nonlatin"].to_numpy()[ib]
    # legal form: distractor twins often swap it (LLC -> Co, Private -> Public); true copies keep or drop it
    if "lg" in A.columns:
        la_, lb_ = A["lg"].to_numpy()[ia], B["lg"].to_numpy()[ib]
        sa, sb = [set(x.split()) for x in la_], [set(y.split()) for y in lb_]
        f["lg_a_n"] = np.fromiter((len(x) for x in sa), np.float32, len(sa))
        f["lg_b_n"] = np.fromiter((len(y) for y in sb), np.float32, len(sb))
        f["lg_eq"] = la_ == lb_
        f["lg_b_extra"] = np.fromiter((len(y - x) for x, y in zip(sa, sb)), np.float32, len(sa))
        f["lg_a_miss"] = np.fromiter((len(x - y) for x, y in zip(sa, sb)), np.float32, len(sa))
        f["lg_conflict"] = np.fromiter((bool(x) and bool(y) and not (x & y) for x, y in zip(sa, sb)), np.float32,
                                       len(sa))
        ex, ms, exl = token_diff(an, bn)
        f["nm_extra_b"], f["nm_miss_a"], f["nm_extra_len"] = ex, ms, exl
    # competition for the record across all S1 of the split
    f["c_n"] = st_n
    f["c_is_best"] = st_is_best
    f["c_margin"] = st_best - score
    f["c_second_gap"] = st_best - st_second
    f["c_ratio"] = score / np.maximum(st_best, 1e-6)
    return pd.DataFrame({k: np.asarray(v, dtype=np.float32) for k, v in f.items()})


if __name__ == "__main__":  # self-check on hand-picked noise patterns from the data
    assert norm_name("Crysta1 Lending  PC") == norm_name("Crystal Lending") == "crystal lending"
    assert norm_name("Bombaypower.Com") == "bombaypower"
    assert norm_addr("66 Edgewood Street, Bridgeport, Connecticut") == norm_addr("66 EDGEWOOD ST, BRIDGEPORT, CT")
    assert norm_addr("Pune, Madhya Pradesh") == "pune mp"
    r = normalize_records(pd.DataFrame({"entity_id": ["x"], "business_name": ["ಬಾಂಬೆ ಪವರ್"],
                                        "business_address": ["No: 032/2, 2Nd Floor"]}), processes=1).iloc[0]
    assert r.nums == "2 32" and r.dig == "3222" and r.n1 == "32" and r.nonlatin, r
    print("ok")
