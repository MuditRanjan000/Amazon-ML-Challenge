"""Shared text normalization (blocking AND matching must use the same function).

Pipeline: NFKC -> casefold -> '&'->' and ' -> Latin accents stripped
-> punctuation/symbols to space -> optional legal-suffix removal -> collapse spaces.

Do NOT use regex `[^\\w\\s]` for punctuation: Indic vowel signs (Unicode Mn/Mc)
are not `\\w`, so it shreds "मार्केटिंग" into "म र क ट ग". Here punctuation is
removed by Unicode category (P*, S*) and combining marks are only stripped from
Latin letters (é -> e), never from Indic scripts.
"""
import sys
import unicodedata

import pandas as pd


def _build_translation_table() -> dict:
    table = {}
    for cp in range(sys.maxunicode + 1):
        ch = chr(cp)
        cat = unicodedata.category(ch)
        if cat[0] in "PS":
            table[cp] = " "
        elif 0xC0 <= cp <= 0x24F and cat[0] == "L":  # Latin-1 Supplement + Latin Extended-A/B
            base = "".join(c for c in unicodedata.normalize("NFKD", ch) if not unicodedata.combining(c))
            if base and base != ch:
                table[cp] = base
    return table


_TABLE = _build_translation_table()

# Legal / generic suffixes across US, India, France. Only used when strip_legal=True.
LEGAL_TOKENS = (
    "ltd limited pvt private llc llp inc incorporated corp corporation co company plc "
    "sarl sas sasu sa eurl sci snc the"
).split()
_LEGAL_RE = r"\b(?:" + "|".join(LEGAL_TOKENS) + r")\b"


def normalize_series(s: pd.Series, strip_legal: bool = False) -> pd.Series:
    """Vectorized normalization; returns an Arrow-backed string Series ('' for missing)."""
    s = s.astype("string[pyarrow]").fillna("")
    s = s.str.normalize("NFKC").str.casefold()
    s = s.str.replace("&", " and ", regex=False).str.translate(_TABLE)
    if strip_legal:
        s = s.str.replace(_LEGAL_RE, " ", regex=True)
    return s.str.replace(r"\s+", " ", regex=True).str.strip()


def normalize_text(text, strip_legal: bool = False) -> str:
    """Scalar convenience wrapper with identical semantics to normalize_series."""
    return normalize_series(pd.Series([text]), strip_legal=strip_legal).iloc[0]


def combine_fields(df: pd.DataFrame, fields, strip_legal: bool = False) -> pd.Series:
    """Normalize and space-join several text columns (e.g. name + address)."""
    parts = [normalize_series(df[f], strip_legal=strip_legal and f == "business_name") for f in fields]
    out = parts[0]
    for p in parts[1:]:
        out = (out + " " + p).str.strip()
    return out
