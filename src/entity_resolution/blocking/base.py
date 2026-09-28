from abc import ABC, abstractmethod

import pandas as pd

KEY = ["source1_entity_id", "candidate_entity_id"]


class BaseBlocker(ABC):
    @abstractmethod
    def build_index(self, s2_df: pd.DataFrame, s3_df: pd.DataFrame):
        """Build any necessary search index from S2 and S3 records."""

    @abstractmethod
    def generate_candidates(self, s1_df: pd.DataFrame, k: int = 100) -> pd.DataFrame:
        """Top-K candidates per S1 record.

        Returns a DataFrame with columns: ['source1_entity_id', 'candidate_entity_id', 'score', 'rank'].
        """


def union_candidates(channels: dict) -> pd.DataFrame:
    """Outer-join several candidate frames on (S1, candidate).

    channels: {name: frame with KEY columns + any score/rank columns}. Each channel's
    extra columns are prefixed with its name, and `n_channels` counts how many
    channels proposed the pair - both are useful matcher features.
    """
    out = None
    for name, frame in channels.items():
        f = frame.astype({c: "string[pyarrow]" for c in KEY}).drop_duplicates(KEY)
        f = f.rename(columns={c: f"{name}_{c}" for c in f.columns if c not in KEY})
        f[f"{name}_hit"] = True
        out = f if out is None else out.merge(f, on=KEY, how="outer")
    hits = [c for c in out.columns if c.endswith("_hit")]
    out[hits] = out[hits].fillna(False).astype(bool)
    out["n_channels"] = out[hits].sum(axis=1).astype("int8")
    return out.drop(columns=hits)
