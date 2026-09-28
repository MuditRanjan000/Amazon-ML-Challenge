import numpy as np
import pandas as pd

from ..data.preprocessing import normalize_series
from .base import BaseBlocker


class ExactNameBlocker(BaseBlocker):
    """Candidates = S2/S3 records whose normalized name (legal suffixes removed) and country equal the S1's."""

    def __init__(self, strip_legal: bool = True):
        self.strip_legal = strip_legal
        self.index = None

    def _keyed(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df[["entity_id", "country"]].copy()
        out["norm_name"] = normalize_series(df["business_name"], strip_legal=self.strip_legal)
        return out[out["norm_name"] != ""]

    def build_index(self, s2_df: pd.DataFrame, s3_df: pd.DataFrame):
        self.index = self._keyed(pd.concat([s2_df, s3_df], ignore_index=True))
        return self

    def generate_candidates(self, s1_df: pd.DataFrame, k: int = 100) -> pd.DataFrame:
        m = self._keyed(s1_df).merge(self.index, on=["country", "norm_name"], suffixes=("_s1", "_cand"))
        m = m.sort_values(["entity_id_s1", "entity_id_cand"], ignore_index=True)  # deterministic ranks
        out = pd.DataFrame({"source1_entity_id": m["entity_id_s1"], "candidate_entity_id": m["entity_id_cand"],
                            "score": np.float32(1.0)})
        out["rank"] = (out.groupby("source1_entity_id").cumcount() + 1).astype(np.int16)
        return out[out["rank"] <= k].reset_index(drop=True)
