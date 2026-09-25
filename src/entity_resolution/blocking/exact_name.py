import pandas as pd
import re
from .base import BaseBlocker

class ExactNameBlocker(BaseBlocker):
    def __init__(self):
        self.candidates_df = None
        
    def _normalize(self, series: pd.Series) -> pd.Series:
        s = series.fillna("").astype(str).str.lower()
        # Remove punctuation
        s = s.str.replace(r'[^\w\s]', ' ', regex=True)
        # Suffix removal
        suffixes = r'\b(ltd|limited|pvt|private|corp|corporation)\b'
        s = s.str.replace(suffixes, ' ', regex=True)
        # Whitespace normalization
        s = s.str.replace(r'\s+', ' ', regex=True).str.strip()
        return s

    def build_index(self, s2_df: pd.DataFrame, s3_df: pd.DataFrame):
        df2 = s2_df[['entity_id', 'business_name', 'country']].copy()
        df3 = s3_df[['entity_id', 'business_name', 'country']].copy()
        
        self.candidates_df = pd.concat([df2, df3], ignore_index=True)
        self.candidates_df['norm_name'] = self._normalize(self.candidates_df['business_name'])
        
        # We only want to keep valid normalized names
        self.candidates_df = self.candidates_df[self.candidates_df['norm_name'] != ""]

    def generate_candidates(self, s1_df: pd.DataFrame, k: int = 100) -> pd.DataFrame:
        s1 = s1_df[['entity_id', 'business_name', 'country']].copy()
        s1['norm_name'] = self._normalize(s1['business_name'])
        
        # Exact match join on country and normalized name
        matches = pd.merge(s1, self.candidates_df, on=['country', 'norm_name'], how='inner', suffixes=('_s1', '_cand'))
        
        # Format output
        results = matches[['entity_id_s1', 'entity_id_cand']].rename(columns={
            'entity_id_s1': 'source1_entity_id',
            'entity_id_cand': 'candidate_entity_id'
        })
        
        # Since it's exact match, all scores are 1.0. We rank arbitrarily.
        results['score'] = 1.0
        results['rank'] = results.groupby('source1_entity_id').cumcount() + 1
        
        # Filter to top K per S1
        results = results[results['rank'] <= k]
        
        return results
