"""Memory-efficient, exact loading of the challenge TSVs.

Why not plain `pd.read_csv(sep="\\t")`:
  * default quoting rewrites fields that contain `"` (e.g. `\"\"\"chrs & cie sasu\"`
    becomes `\"chrs & cie sasu`) - the official validator treats files as raw text;
  * default NA parsing turns literal names such as "NA"/"None" into NaN;
  * object-dtype strings cost ~5x the RAM of Arrow strings.

Files are parsed once with pyarrow (tab-delimited, no quoting, every column a
string, empty -> "") and cached as parquet; later loads take seconds.
"""
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.csv as pacsv
import pyarrow.parquet as pq

from .. import config

SOURCE_COLUMNS = ["entity_id", "business_name", "business_address", "country"]
GT_COLUMNS = ["source1_entity_id", "matched_entity_ids"]


def read_tsv(path: Path) -> pa.Table:
    """Parse a challenge TSV exactly as the official validator sees it."""
    return pacsv.read_csv(
        path,
        read_options=pacsv.ReadOptions(block_size=64 << 20),
        parse_options=pacsv.ParseOptions(delimiter="\t", quote_char=False, newlines_in_values=False),
        convert_options=pacsv.ConvertOptions(column_types=_all_string_types(path), strings_can_be_null=False),
    )


def _all_string_types(path: Path) -> dict:
    with open(path, encoding="utf-8") as f:
        header = f.readline().rstrip("\r\n").split("\t")
    return {name: pa.string() for name in header}


class DataLoader:
    def __init__(self, data_dir=None, cache_dir=None):
        self.data_dir = Path(data_dir or config.DATA_DIR)
        self.cache_dir = Path(cache_dir or config.CACHE_DIR)

    def _load(self, rel_path: str, columns=None) -> pd.DataFrame:
        tsv = self.data_dir / rel_path
        if not tsv.exists():
            raise FileNotFoundError(f"Data file not found: {tsv} (set ER_DATA_DIR)")
        cached = self.cache_dir / (Path(rel_path).stem + ".parquet")
        if not cached.exists() or cached.stat().st_mtime < tsv.stat().st_mtime:
            cached.parent.mkdir(parents=True, exist_ok=True)
            tmp = cached.with_suffix(".parquet.tmp")
            pq.write_table(read_tsv(tsv), tmp)
            tmp.replace(cached)  # atomic: a crash never leaves a half-written cache
        df = pq.read_table(cached, columns=columns).to_pandas(types_mapper=pd.ArrowDtype)
        # pd.ArrowDtype(string) -> pandas' native Arrow-backed "str" dtype
        return df.astype({c: "string[pyarrow]" for c in df.columns})

    def load_source(self, split: str, source: int, columns=None) -> pd.DataFrame:
        """split: 'train' | 'test'; source: 1 | 2 | 3. `columns` limits memory."""
        return self._load(f"{split}/{split}_source{source}.tsv", columns)

    def load_ground_truth(self) -> pd.DataFrame:
        return self._load("train/train_ground_truth.tsv")

    def load_candidates_pool(self, split: str, columns=None) -> pd.DataFrame:
        """S2 and S3 of a split stacked into one frame (the retrieval index)."""
        return pd.concat([self.load_source(split, 2, columns), self.load_source(split, 3, columns)],
                         ignore_index=True)


def explode_id_lists(df: pd.DataFrame, list_col: str = "matched_entity_ids") -> pd.DataFrame:
    """Wide (S1, "id1,id2,...") -> long unique (source1_entity_id, candidate_entity_id) pairs.

    Works for ground truth, matching_results and candidate_pairs frames.
    """
    ids = df[list_col].astype("string[pyarrow]").fillna("").str.split(",")
    long = pd.DataFrame({"source1_entity_id": df["source1_entity_id"].to_numpy(), "candidate_entity_id": ids})
    long = long.explode("candidate_entity_id", ignore_index=True)
    long = long[long["candidate_entity_id"].notna() & (long["candidate_entity_id"] != "")]
    return long.astype("string[pyarrow]").drop_duplicates(ignore_index=True)


def explode_ground_truth(gt: pd.DataFrame) -> pd.DataFrame:
    return explode_id_lists(gt, "matched_entity_ids")
