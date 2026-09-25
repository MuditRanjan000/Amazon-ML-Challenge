"""Write the two official output files (vectorized, deterministic, spec-exact).

Both files: one row per test S1 entity in the order given (the test file order),
comma-separated IDs with no duplicates, empty for no match, TAB-separated, UTF-8, LF.
"""
import csv
from pathlib import Path

import pandas as pd

from .. import config


def _write_id_lists(s1_ids, pairs: pd.DataFrame, list_col: str, path: Path) -> Path:
    """pairs: long frame with source1_entity_id, candidate_entity_id (any dtype, duplicates ok)."""
    s1 = pd.Index(pd.Series(list(s1_ids), dtype="string[pyarrow]"), name="source1_entity_id")
    if s1.has_duplicates:
        raise ValueError("duplicate Source-1 IDs passed to the writer")
    long = pairs[["source1_entity_id", "candidate_entity_id"]].astype("string[pyarrow]").drop_duplicates()
    long = long[long["source1_entity_id"].isin(s1)].sort_values(["source1_entity_id", "candidate_entity_id"])
    lists = long.groupby("source1_entity_id")["candidate_entity_id"].agg(",".join)
    out = pd.DataFrame({"source1_entity_id": s1, list_col: lists.reindex(s1).fillna("").to_numpy()})
    path.parent.mkdir(parents=True, exist_ok=True)
    # QUOTE_NONE: IDs never need quoting; if one ever did, fail loudly instead of writing quotes.
    out.to_csv(path, sep="\t", index=False, lineterminator="\n", quoting=csv.QUOTE_NONE, encoding="utf-8")
    return path


def _dict_to_pairs(predicted: dict) -> pd.DataFrame:
    rows = [(s1, c) for s1, cands in predicted.items() for c in cands]
    return pd.DataFrame(rows, columns=["source1_entity_id", "candidate_entity_id"], dtype="string[pyarrow]")


class SubmissionGenerator:
    def __init__(self, output_dir=None):
        self.output_dir = Path(output_dir or config.OUTPUT_DIR)

    def generate(self, test_s1_ids, predicted_matches, filename: str = "matching_results.tsv") -> Path:
        """test_s1_ids: every test S1 ID (keep file order; a set is sorted for determinism).
        predicted_matches: {s1_id: iterable of S2/S3 IDs} or a long pairs DataFrame."""
        pairs = predicted_matches if isinstance(predicted_matches, pd.DataFrame) else _dict_to_pairs(predicted_matches)
        ids = sorted(test_s1_ids) if isinstance(test_s1_ids, (set, frozenset)) else test_s1_ids
        return _write_id_lists(ids, pairs, "matched_entity_ids", self.output_dir / filename)

    def generate_candidates(self, test_s1_ids, candidate_pairs: pd.DataFrame,
                            filename: str = "candidate_pairs.tsv") -> Path:
        """candidate_pairs: the exact long pair set the final model scored."""
        ids = sorted(test_s1_ids) if isinstance(test_s1_ids, (set, frozenset)) else test_s1_ids
        return _write_id_lists(ids, candidate_pairs, "candidate_entity_ids", self.output_dir / filename)
