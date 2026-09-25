from abc import ABC, abstractmethod
import pandas as pd
from typing import Dict

class BaseBlocker(ABC):
    @abstractmethod
    def build_index(self, s2_df: pd.DataFrame, s3_df: pd.DataFrame):
        """Build any necessary search index from S2 and S3 records."""
        pass
        
    @abstractmethod
    def generate_candidates(self, s1_df: pd.DataFrame, k: int = 100) -> pd.DataFrame:
        """
        Generate top-K candidates for each S1 record.
        Returns a DataFrame with columns: ['source1_entity_id', 'candidate_entity_id', 'score', 'rank']
        """
        pass
