import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from joblib import Parallel, delayed
from .base import BaseBlocker
import gc
import os
import joblib

class TfidfBlocker(BaseBlocker):
    def __init__(self, text_fields=['business_name'], ngram_range=(3, 5), cache_dir="output/tfidf_cache"):
        self.text_fields = text_fields
        self.ngram_range = ngram_range
        self.cache_dir = cache_dir
        self.vectorizers = {} # Not storing in memory globally
        self.matrices = {} # Not storing in memory globally
        self.candidate_ids = {} # Not storing in memory globally
        
        if not os.path.exists(self.cache_dir):
            os.makedirs(self.cache_dir)
        
    def _prepare_text(self, df: pd.DataFrame) -> pd.Series:
        # Fill NaN and convert to string
        text = pd.Series(index=df.index, data="", dtype=str)
        for field in self.text_fields:
            col_text = df[field].fillna("").astype(str).str.lower()
            text = text + " " + col_text
            
        # Clean
        text = text.str.replace(r'[^\w\s]', ' ', regex=True)
        text = text.str.replace(r'\s+', ' ', regex=True).str.strip()
        return text

    def build_index(self, s2_df: pd.DataFrame, s3_df: pd.DataFrame):
        """Build partitioned TF-IDF indexes by country."""
        df2 = s2_df[['entity_id', 'country'] + self.text_fields].copy()
        df3 = s3_df[['entity_id', 'country'] + self.text_fields].copy()
        
        candidates = pd.concat([df2, df3], ignore_index=True)
        candidates['combined_text'] = self._prepare_text(candidates)
        
        # Drop empty
        candidates = candidates[candidates['combined_text'] != ""]
        
        # Partition by country
        for country, grp in candidates.groupby('country'):
            vec = TfidfVectorizer(analyzer='char_wb', ngram_range=self.ngram_range, min_df=2)
            mat = vec.fit_transform(grp['combined_text'])
            # We do NOT save to self.vectorizers/matrices to save RAM
            # Only cache the index to disk
            joblib.dump(vec, os.path.join(self.cache_dir, f'vec_{country}.joblib'))
            joblib.dump(mat, os.path.join(self.cache_dir, f'mat_{country}.joblib'))
            joblib.dump(grp['entity_id'].values, os.path.join(self.cache_dir, f'ids_{country}.joblib'))
            
            del vec, mat
            gc.collect()
            
        del candidates
        gc.collect()

    def load_index(self, countries):
        """Not needed since we load lazily per country now."""
        pass

    def _process_chunk(self, q_mat_chunk, c_mat, c_ids, s1_ids, k):
        results = []
        # TfidfVectorizer outputs L2 normalized matrices by default.
        # c_mat is CSR, q_mat_chunk.T is CSC. CSR * CSC avoids allocating massive new indices
        # It converts q_mat_chunk.T to CSR (tiny memory because only 500 rows)
        sim = c_mat.dot(q_mat_chunk.T).T.toarray()
        for row_idx in range(sim.shape[0]):
            row_sim = sim[row_idx]
            top_indices = np.argsort(row_sim)[-k:][::-1]
            s1_id = s1_ids[row_idx]
            for rank_idx, cand_idx in enumerate(top_indices):
                score = row_sim[cand_idx]
                if score > 0.0:
                    results.append({
                        'source1_entity_id': s1_id,
                        'candidate_entity_id': c_ids[cand_idx],
                        'score': score,
                        'rank': rank_idx + 1
                    })
        return results

    def generate_candidates(self, s1_df: pd.DataFrame, k: int = 100) -> pd.DataFrame:
        """Batch inference partitioned by country."""
        s1 = s1_df[['entity_id', 'country'] + self.text_fields].copy()
        s1['combined_text'] = self._prepare_text(s1)
        
        all_results = []
        
        for country, grp in s1.groupby('country'):
            vec_path = os.path.join(self.cache_dir, f'vec_{country}.joblib')
            if not os.path.exists(vec_path):
                continue
                
            # Load only the current country into memory
            vec = joblib.load(vec_path)
            c_mat = joblib.load(os.path.join(self.cache_dir, f'mat_{country}.joblib'))
            c_ids = joblib.load(os.path.join(self.cache_dir, f'ids_{country}.joblib'))
            
            # Transform queries
            q_mat = vec.transform(grp['combined_text'])
            
            # Batch queries (Chunks of 500)
            chunk_size = 500
            chunks = []
            for i in range(0, q_mat.shape[0], chunk_size):
                end = min(i + chunk_size, q_mat.shape[0])
                chunks.append((q_mat[i:end], grp['entity_id'].iloc[i:end].values))
                
            # Safely parallelize because partitioned matrices are small
            chunk_results = Parallel(n_jobs=-1, backend='threading')(
                delayed(self._process_chunk)(q, c_mat, c_ids, ids, k) for q, ids in chunks
            )
            
            for res in chunk_results:
                all_results.extend(res)
                
            # Free memory for this country before moving to the next
            del vec, c_mat, c_ids, q_mat
            gc.collect()
                
        return pd.DataFrame(all_results)
