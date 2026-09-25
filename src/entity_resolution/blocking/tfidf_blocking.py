import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import HashingVectorizer, TfidfTransformer
from sklearn.pipeline import make_pipeline
from joblib import Parallel, delayed
import gc
import os
import joblib
from .base import BaseBlocker

class TfidfBlocker(BaseBlocker):
    def __init__(self, text_fields=['business_name'], ngram_range=(3, 5), cache_dir='output/tfidf_cache'):
        self.text_fields = text_fields
        self.ngram_range = ngram_range
        self.cache_dir = cache_dir
        if not os.path.exists(self.cache_dir): 
            os.makedirs(self.cache_dir)
        
    def _prepare_text(self, df):
        text = pd.Series(index=df.index, data='', dtype=str)
        for field in self.text_fields:
            text = text + ' ' + df[field].fillna('').astype(str).str.lower()
        return text.str.replace(r'[^\w\s]', ' ', regex=True).str.replace(r'\s+', ' ', regex=True).str.strip()

    def build_index(self, s2_df, s3_df):
        df2 = s2_df[['entity_id', 'country'] + self.text_fields].copy()
        df3 = s3_df[['entity_id', 'country'] + self.text_fields].copy()
        cands = pd.concat([df2, df3], ignore_index=True)
        cands['combined_text'] = self._prepare_text(cands)
        cands = cands[cands['combined_text'] != '']
        
        chunk_size = 500_000
        for country, grp in cands.groupby('country'):
            vec = make_pipeline(
                HashingVectorizer(analyzer='char_wb', ngram_range=self.ngram_range, n_features=2**21, norm=None, alternate_sign=False),
                TfidfTransformer()
            )
            # Fit IDF on a sample to save memory
            sample = grp['combined_text'].sample(n=min(300_000, len(grp)), random_state=42)
            vec.fit(sample)
            joblib.dump(vec, os.path.join(self.cache_dir, f'vec_{country}.joblib'))
            
            # Process and save in chunks
            num_chunks = int(np.ceil(len(grp) / chunk_size))
            joblib.dump(num_chunks, os.path.join(self.cache_dir, f'num_chunks_{country}.joblib'))
            
            for i in range(num_chunks):
                chunk_grp = grp.iloc[i*chunk_size : (i+1)*chunk_size]
                mat = vec.transform(chunk_grp['combined_text'])
                joblib.dump(mat, os.path.join(self.cache_dir, f'mat_{country}_{i}.joblib'))
                joblib.dump(chunk_grp['entity_id'].values, os.path.join(self.cache_dir, f'ids_{country}_{i}.joblib'))
                del mat
                gc.collect()
        del cands
        gc.collect()

    def load_index(self, countries):
        pass

    def _process_chunk(self, q_mat_chunk, c_mat, c_ids, s1_ids, k):
        # Do sparse-sparse dot product to avoid materializing a massive (200, 2M) dense array
        # sim_sparse shape: (c_mat.shape[0], q_mat_chunk.shape[0]) -> (500000, 200)
        sim_sparse = c_mat.dot(q_mat_chunk.T)
        # Convert to dense (500000 x 200 = 800MB) which fits safely in RAM, then transpose
        sim = sim_sparse.toarray().T
        results = []
        for row_idx in range(sim.shape[0]):
            row_sim = sim[row_idx]
            if len(row_sim) > k:
                top_indices = np.argpartition(row_sim, -k)[-k:]
                top_indices = top_indices[np.argsort(row_sim[top_indices])[::-1]]
            else:
                top_indices = np.argsort(row_sim)[::-1]
            s1_id = s1_ids[row_idx]
            for rank_idx, cand_idx in enumerate(top_indices):
                score = row_sim[cand_idx]
                if score > 0.0:
                    results.append({'source1_entity_id': s1_id, 'candidate_entity_id': c_ids[cand_idx], 'score': score})
        return results

    def generate_candidates(self, s1_df, k=100):
        s1 = s1_df[['entity_id', 'country'] + self.text_fields].copy()
        s1['combined_text'] = self._prepare_text(s1)
        
        all_results = []
        for country, grp in s1.groupby('country'):
            vec_path = os.path.join(self.cache_dir, f'vec_{country}.joblib')
            if not os.path.exists(vec_path): continue
            
            vec = joblib.load(vec_path)
            q_mat = vec.transform(grp['combined_text'])
            num_chunks = joblib.load(os.path.join(self.cache_dir, f'num_chunks_{country}.joblib'))
            
            country_results = []
            for chunk_idx in range(num_chunks):
                c_mat = joblib.load(os.path.join(self.cache_dir, f'mat_{country}_{chunk_idx}.joblib'))
                c_ids = joblib.load(os.path.join(self.cache_dir, f'ids_{country}_{chunk_idx}.joblib'))
                
                chunk_size = max(1, 50_000_000 // c_mat.shape[0])
                chunks = []
                for i in range(0, q_mat.shape[0], chunk_size):
                    end = min(i + chunk_size, q_mat.shape[0])
                    chunks.append((q_mat[i:end], grp['entity_id'].iloc[i:end].values))
                    
                chunk_res = Parallel(n_jobs=2, backend='threading')(delayed(self._process_chunk)(q, c_mat, c_ids, ids, k) for q, ids in chunks)
                for res in chunk_res: country_results.extend(res)
                del c_mat, c_ids
                gc.collect()
            all_results.extend(country_results)
            
        df = pd.DataFrame(all_results)
        if df.empty: return df
        # Sort and keep top K across all chunks
        df = df.sort_values(['source1_entity_id', 'score'], ascending=[True, False])
        df = df.groupby('source1_entity_id').head(k).reset_index(drop=True)
        # Assign ranks
        df['rank'] = df.groupby('source1_entity_id').cumcount() + 1
        return df