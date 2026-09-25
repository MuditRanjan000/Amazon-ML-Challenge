"""Blocking keys: alternative text views of a record, each indexed by its own channel.

name_key targets the misses of word TF-IDF on name+address (BLK-013 error analysis, 20k val):
~40% are cross-script Indic names ("सुपर हॉस्पिटैलिटी प्राइवेट लिमिटेड" vs "Super Hospitality
Private Limited"), ~21% typo'd names with an empty candidate address, plus names written as
domains ("kirkaerospace.com"). Hence: transliterate to ASCII (anyascii, ISC licence), drop
legal/web tokens (they are in millions of records and only slow retrieval down), drop spaces,
then char n-grams.
"""
import pandas as pd
from anyascii import anyascii

from ..data.preprocessing import normalize_series

_WEB = r"\b(?:www|com|net|org|co|in|us|biz|info|http|https)\b"
# anyascii renderings of Indic legal words (प्राइवेट -> praivet)
_TRANSLIT_LEGAL = r"\b(?:praivet|prayivet|praiveta|limitedd|limiteda|pvt|ltd|llc|inc|corp)\b"


def name_key(names: pd.Series, skeleton: bool = False) -> pd.Series:
    """ASCII, legal/web-stripped, space-free business name. skeleton=True also drops
    non-initial vowels and repeated letters (2x faster retrieval, ~0.05pp lower recall)."""
    s = pd.Series([anyascii(x) for x in names.astype("string[pyarrow]").fillna("").to_numpy(dtype=object)],
                  index=names.index, dtype="string[pyarrow]")
    s = normalize_series(s, strip_legal=True).str.replace(_WEB, " ", regex=True)
    s = s.str.replace(_TRANSLIT_LEGAL, " ", regex=True).str.replace(r"[^a-z0-9]", "", regex=True)
    if skeleton:  # RE2 (Arrow) has no lookbehind/backrefs: use Python re on object dtype
        s = s.astype(object).str.replace(r"(?<=.)[aeiouy]", "", regex=True).str.replace(r"(.)\1+", r"\1", regex=True)
    return s.astype("string[pyarrow]")
