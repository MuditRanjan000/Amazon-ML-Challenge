import json

import pandas as pd
import pytest

from entity_resolution import config
from entity_resolution.data.loader import DataLoader, explode_id_lists
from entity_resolution.data.validation_split import create_validation_split, _sha256
from entity_resolution.submission.generator import SubmissionGenerator
from entity_resolution.submission.validator import SubmissionValidator


def _write(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(text.encode("utf-8"))


def test_loader_reads_raw_text_crlf_quotes_and_na(tmp_path):
    _write(tmp_path / "data/train/train_source1.tsv",
           'entity_id\tbusiness_name\tbusiness_address\tcountry\r\n'
           'S1-1\t"""chrs & cie sasu"\t\tFrance\r\n'
           'S1-2\tNA\tNone\tUS\r\n')
    df = DataLoader(tmp_path / "data", tmp_path / "cache").load_source("train", 1)
    assert list(df["business_name"]) == ['"""chrs & cie sasu"', "NA"]
    assert list(df["business_address"]) == ["", "None"]
    assert list(df["country"]) == ["France", "US"]  # no trailing \r
    # second load comes from the parquet cache and is identical
    assert DataLoader(tmp_path / "data", tmp_path / "cache").load_source("train", 1).equals(df)


def test_explode_id_lists():
    df = pd.DataFrame({"source1_entity_id": ["a", "b", "c"], "matched_entity_ids": ["x,y,x", "", None]})
    long = explode_id_lists(df)
    assert list(zip(long["source1_entity_id"], long["candidate_entity_id"])) == [("a", "x"), ("a", "y")]


def test_split_is_deterministic_and_manifest_checked(tmp_path):
    gt = pd.DataFrame({"source1_entity_id": [f"S1-{i}" for i in range(100)], "matched_entity_ids": ""})
    tr, va = create_validation_split(gt, split_dir=tmp_path / "s", manifest_path=tmp_path / "none.json")
    assert len(va) == 20 and not (tr & va)
    manifest = {"train_id_file_sha256": _sha256(tmp_path / "s/train_ids.csv"),
                "validation_id_file_sha256": "0" * 64}
    (tmp_path / "m.json").write_text(json.dumps(manifest))
    with pytest.raises(RuntimeError, match="does not match"):
        create_validation_split(split_dir=tmp_path / "s", manifest_path=tmp_path / "m.json")


@pytest.mark.skipif(not config.OFFICIAL_VALIDATOR.is_file(), reason="official validator not present")
def test_generated_files_pass_official_validator(tmp_path):
    test_dir = tmp_path / "test"
    _write(test_dir / "test_source1.tsv", "entity_id\tbusiness_name\tbusiness_address\tcountry\n"
                                           "S1-2\ta\t\tUS\nS1-1\tb\t\tFrance\nS1-3\tc\t\tIndia\n")
    for src in (2, 3):
        _write(test_dir / f"test_source{src}.tsv", "entity_id\tbusiness_name\tbusiness_address\tcountry\n"
                                                    f"S{src}-1\tx\t\tUS\nS{src}-2\ty\t\tUS\n")
    gen = SubmissionGenerator(tmp_path / "out")
    order = ["S1-2", "S1-1", "S1-3"]
    cands = pd.DataFrame({"source1_entity_id": ["S1-2", "S1-2", "S1-1", "S1-2"],
                          "candidate_entity_id": ["S3-2", "S2-1", "S2-2", "S2-1"]})
    cpath = gen.generate_candidates(order, cands)
    mpath = gen.generate(order, {"S1-2": ["S2-1", "S2-1"], "S1-9": ["S2-2"]})  # dup + unknown S1 dropped
    assert (tmp_path / "out/matching_results.tsv").read_text() == (
        "source1_entity_id\tmatched_entity_ids\nS1-2\tS2-1\nS1-1\t\nS1-3\t\n")
    assert SubmissionValidator(test_dir).validate(mpath, cpath, check_ids=True)
