import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from joblib import Parallel, delayed
from .base import BaseBlocker
import gc

class TfidfBlocker(BaseBlocker):
    def __init__(self, text_fields=['business_name'], ngram_range=(3, 5), partition_by_country=False):
        self.text_fields = text_fields
        self.ngram_range = ngram_range
        self.partition_by_country = partition_by_country
        self.vectorizers = {} # country -> vectorizer or 'all' -> vectorizer
        self.matrices = {} # country -> sparse matrix
        self.candidate_ids = {} # country -> list of entity ids
        
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
        df2 = s2_df[['entity_id', 'country'] + self.text_fields].copy()
        df3 = s3_df[['entity_id', 'country'] + self.text_fields].copy()
        
        candidates = pd.concat([df2, df3], ignore_index=True)
        candidates['combined_text'] = self._prepare_text(candidates)
        
        # Drop empty
        candidates = candidates[candidates['combined_text'] != ""]
        
        if self.partition_by_country:
            for country, grp in candidates.groupby('country'):
                vec = TfidfVectorizer(analyzer='char_wb', ngram_range=self.ngram_range, min_df=2)
                mat = vec.fit_transform(grp['combined_text'])
                self.vectorizers[country] = vec
                self.matrices[country] = mat
                self.candidate_ids[country] = grp['entity_id'].values
        else:
            vec = TfidfVectorizer(analyzer='char_wb', ngram_range=self.ngram_range, min_df=2)
            mat = vec.fit_transform(candidates['combined_text'])
            self.vectorizers['all'] = vec
            self.matrices['all'] = mat
            self.candidate_ids['all'] = candidates['entity_id'].values
            
        del candidates
        gc.collect()

    def _process_chunk(self, q_mat_chunk, c_mat, c_ids, s1_ids, k):
        results = []
        sim = cosine_similarity(q_mat_chunk, c_mat)
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
        s1 = s1_df[['entity_id', 'country'] + self.text_fields].copy()
        s1['combined_text'] = self._prepare_text(s1)
        
        all_results = []
        
        if self.partition_by_country:
            for country, grp in s1.groupby('country'):
                if country not in self.vectorizers:
                    continue
                vec = self.vectorizers[country]
                c_mat = self.matrices[country]
                c_ids = self.candidate_ids[country]
                
                # Transform queries
                q_mat = vec.transform(grp['combined_text'])
                
                # Chunk to avoid OOM
                chunk_size = 100
                chunks = []
                for i in range(0, q_mat.shape[0], chunk_size):
                    end = min(i + chunk_size, q_mat.shape[0])
                    chunks.append((q_mat[i:end], grp['entity_id'].iloc[i:end].values))
                    
                chunk_results = Parallel(n_jobs=-1, backend='threading')(
                    delayed(self._process_chunk)(q, c_mat, c_ids, ids, k) for q, ids in chunks
                )
                
                for res in chunk_results:
                    all_results.extend(res)
                    
        else:
            vec = self.vectorizers['all']
            c_mat = self.matrices['all']
            c_ids = self.candidate_ids['all']
            
            q_mat = vec.transform(s1['combined_text'])
            
            chunk_size = 10
            chunks = []
            for i in range(0, q_mat.shape[0], chunk_size):
                end = min(i + chunk_size, q_mat.shape[0])
                chunks.append((q_mat[i:end], s1['entity_id'].iloc[i:end].values))
                
            chunk_results = Parallel(n_jobs=-1, backend='threading')(
                delayed(self._process_chunk)(q, c_mat, c_ids, ids, k) for q, ids in chunks
            )
            
            for res in chunk_results:
                all_results.extend(res)
                            
        return pd.DataFrame(all_results)
