import pandas as pd

from entity_resolution.data.preprocessing import combine_fields, normalize_series, normalize_text


def test_devanagari_is_not_shredded():
    assert normalize_text("राम मार्केटिंग प्राइवेट लिमिटेड") == "राम मार्केटिंग प्राइवेट लिमिटेड"


def test_latin_accents_punctuation_ampersand():
    assert normalize_text("Léarning Center, S.A.S.") == "learning center s a s"
    assert normalize_text('"""chrs & cie sasu"') == "chrs and cie sasu"
    assert normalize_text("B+ Retail Inc") == "b retail inc"


def test_fullwidth_and_casefold():
    assert normalize_text("ＡＢＣ　Ｃｏｒｐ") == "abc corp"
    assert normalize_text("Straße") == "strasse"


def test_missing_and_literal_na():
    assert normalize_text(None) == ""
    assert normalize_text("NA") == "na"  # a real name, not a missing value


def test_legal_suffix_strip():
    assert normalize_text("Prime Money Pvt. Ltd.", strip_legal=True) == "prime money"
    assert normalize_text("Colt Inc", strip_legal=True) == "colt"


def test_combine_fields_strips_legal_only_from_name():
    df = pd.DataFrame({"business_name": ["Acme Ltd"], "business_address": ["Ltd Road, Pune"]})
    assert combine_fields(df, ["business_name", "business_address"], strip_legal=True).iloc[0] == "acme ltd road pune"


def test_series_keeps_length_and_index():
    s = pd.Series(["A", None, "b"], index=[5, 6, 7])
    out = normalize_series(s)
    assert list(out.index) == [5, 6, 7] and list(out) == ["a", "", "b"]
