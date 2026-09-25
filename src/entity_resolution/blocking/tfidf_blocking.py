"""Scalable TF-IDF character n-gram retrieval (forward or reverse blocking).

Memory model (why this does not OOM like the dense version it replaces):
  * HashingVectorizer: no n-gram vocabulary dict (that alone is GBs at 10M rows);
    stateless, so text -> vectors is embarrassingly parallel.
  * sparse_dot_topn.sp_matmul_topn keeps only the top-k products per query row
    instead of materialising a dense (queries x index) similarity matrix.
  * queries are processed in fixed-size chunks; output IDs are int32 categorical codes.
Peak RAM ~ index matrix (~8 bytes x nnz) + one query chunk + k x n_queries result rows.

Countries are an open set: queries whose country has no index partition are
searched against every partition (France exists only in test).
"""
import logging

import numpy as np
import pandas as pd
import scipy.sparse as sp
from joblib import Parallel, delayed
from sklearn.feature_extraction.text import HashingVectorizer, TfidfTransformer
from sklearn.preprocessing import normalize
from sparse_dot_topn import sp_matmul_topn

from .. import config
from ..data.preprocessing import combine_fields
from .base import BaseBlocker

log = logging.getLogger(__name__)


class TfidfBlocker(BaseBlocker):
    def __init__(self, text_fields=("business_name", "business_address"), ngram_range=(1, 1), analyzer="word",
                 n_features=2 ** 22, partition_by_country=True, strip_legal=False, max_df=1.0,
                 chunk_size=100_000, n_jobs=None):
        """max_df: drop n-grams present in more than this fraction of index records.
        Very common n-grams carry almost no IDF weight but dominate sparse-matmul cost
        (every query walks their multi-million posting lists)."""
        self.text_fields = list(text_fields)
        self.partition_by_country = partition_by_country
        self.strip_legal = strip_legal
        self.max_df = max_df
        self.chunk_size = chunk_size
        self.n_jobs = n_jobs or config.N_JOBS
        self.hasher = HashingVectorizer(analyzer=analyzer, ngram_range=tuple(ngram_range), n_features=n_features,
                                        alternate_sign=False, norm=None, dtype=np.float32)
        self.tfidf = TfidfTransformer(sublinear_tf=True, norm=None)
        self.feature_mask = None   # diagonal 0/1 matrix of kept n-grams (max_df pruning)
        self.index_ids = None      # pd.Index of indexed entity IDs (categories of the output)
        self.partitions = {}       # country -> (global row positions, index matrix transposed, CSR)

    # ------------------------------------------------------------------ vectors
    def _hash(self, text: pd.Series) -> sp.csr_matrix:
        values = text.to_numpy(dtype=object)
        step = 50_000
        if len(values) <= step or self.n_jobs == 1:
            return self.hasher.transform(values)
        parts = Parallel(n_jobs=self.n_jobs)(
            delayed(self.hasher.transform)(values[i:i + step]) for i in range(0, len(values), step))
        return sp.vstack(parts, format="csr")

    def _text(self, df: pd.DataFrame) -> pd.Series:
        return combine_fields(df, self.text_fields, strip_legal=self.strip_legal)

    def _weight(self, counts: sp.csr_matrix) -> sp.csr_matrix:
        """Counts -> TF-IDF on kept n-grams -> L2 rows (cosine over the kept vocabulary)."""
        m = self.tfidf.transform(counts)
        if self.feature_mask is not None:
            m = m @ self.feature_mask
            m.eliminate_zeros()
        return normalize(m).astype(np.float32)

    def vectors(self, df: pd.DataFrame) -> sp.csr_matrix:
        """Public: the exact vectors used for retrieval (useful as a matcher feature)."""
        return self._weight(self._hash(self._text(df)))

    # ------------------------------------------------------------------ index
    def fit(self, index_df: pd.DataFrame) -> "TfidfBlocker":
        """Index records with columns entity_id, country + text_fields."""
        text = self._text(index_df)
        keep = (text != "").to_numpy()
        counts = self._hash(text[keep])
        self.tfidf.fit(counts)
        self.feature_mask = None
        if self.max_df < 1.0:
            doc_freq = np.bincount(counts.indices, minlength=counts.shape[1])
            kept = doc_freq <= self.max_df * counts.shape[0]
            self.feature_mask = sp.diags(kept.astype(np.float32), format="csr")
            log.info("max_df=%s pruned %d of %d observed n-grams", self.max_df,
                     int(((doc_freq > 0) & ~kept).sum()), int((doc_freq > 0).sum()))
        matrix = self._weight(counts)
        del counts
        self.index_ids = pd.Index(index_df["entity_id"].to_numpy()[keep])
        countries = (index_df["country"].fillna("").to_numpy()[keep] if self.partition_by_country
                     else np.full(len(self.index_ids), ""))
        self.partitions = {}
        for country in pd.unique(countries):
            rows = np.flatnonzero(countries == country).astype(np.int32)
            self.partitions[country] = (rows, matrix[rows].T.tocsr())
        log.info("indexed %d records (%d empty skipped) in %d partition(s), nnz=%d",
                 len(self.index_ids), int((~keep).sum()), len(self.partitions), matrix.nnz)
        return self

    # ------------------------------------------------------------------ query
    def _topk(self, q: sp.csr_matrix, country, k: int):
        """Top-k against one partition (or all, for unseen countries): (q_row, global_pos, score)."""
        parts = [self.partitions[country]] if country in self.partitions else list(self.partitions.values())
        rows_out, pos_out, score_out = [], [], []
        for positions, index_t in parts:
            res = sp_matmul_topn(q, index_t, top_n=k, sort=True, n_threads=self.n_jobs).tocoo()
            rows_out.append(res.row)
            pos_out.append(positions[res.col])
            score_out.append(res.data)
        rows, pos, score = np.concatenate(rows_out), np.concatenate(pos_out), np.concatenate(score_out)
        if len(parts) > 1:  # merge per-partition top-k lists into a global top-k
            order = np.lexsort((-score, rows))
            rows, pos, score = rows[order], pos[order], score[order]
            keep = _rank_within(rows) <= k
            rows, pos, score = rows[keep], pos[keep], score[keep]
        return rows, pos, score

    def query(self, query_df: pd.DataFrame, k: int = 50) -> pd.DataFrame:
        """Top-k index records per query row.

        Returns columns query_entity_id, index_entity_id (categorical), score (float32), rank (int16),
        sorted by query then descending score. Queries with empty text get no rows.
        """
        if self.index_ids is None:
            raise RuntimeError("call fit() before query()")
        q_ids = pd.Index(query_df["entity_id"].to_numpy())
        countries = (query_df["country"].fillna("").to_numpy() if self.partition_by_country
                     else np.full(len(query_df), ""))
        q_rows, i_pos, scores = [], [], []
        for start in range(0, len(query_df), self.chunk_size):
            chunk = query_df.iloc[start:start + self.chunk_size]
            q = self.vectors(chunk)
            chunk_countries = countries[start:start + self.chunk_size]
            for country in pd.unique(chunk_countries):
                local = np.flatnonzero(chunk_countries == country)
                rows, pos, score = self._topk(q[local], country, k)
                q_rows.append(local[rows] + start)
                i_pos.append(pos)
                scores.append(score)
            log.info("queried %d / %d", min(start + self.chunk_size, len(query_df)), len(query_df))
        q_rows = np.concatenate(q_rows).astype(np.int32) if q_rows else np.empty(0, np.int32)
        i_pos = np.concatenate(i_pos).astype(np.int32) if i_pos else np.empty(0, np.int32)
        scores = np.concatenate(scores).astype(np.float32) if scores else np.empty(0, np.float32)
        order = np.lexsort((-scores, q_rows))
        q_rows, i_pos, scores = q_rows[order], i_pos[order], scores[order]
        return pd.DataFrame({
            "query_entity_id": pd.Categorical.from_codes(q_rows, categories=q_ids),
            "index_entity_id": pd.Categorical.from_codes(i_pos, categories=self.index_ids),
            "score": scores,
            "rank": _rank_within(q_rows).astype(np.int16),
        })

    # ------------------------------------------------------------------ BaseBlocker contract
    def build_index(self, s2_df: pd.DataFrame, s3_df: pd.DataFrame):
        return self.fit(pd.concat([s2_df, s3_df], ignore_index=True))

    def generate_candidates(self, s1_df: pd.DataFrame, k: int = 100) -> pd.DataFrame:
        """Forward blocking: S1 queries against the S2+S3 index."""
        return self.query(s1_df, k).rename(columns={"query_entity_id": "source1_entity_id",
                                                     "index_entity_id": "candidate_entity_id"})


def reverse_candidates(s1_df: pd.DataFrame, pool_df: pd.DataFrame, k: int = 5, **blocker_kwargs) -> pd.DataFrame:
    """Reverse blocking: index S1, query every S2/S3 record for its top-k S1 entities.

    Justified by the data: each S2/S3 ID belongs to at most one S1 entity (verified on
    train GT), so every true pair only needs to rank within its own record's top-k.
    """
    res = TfidfBlocker(**blocker_kwargs).fit(s1_df).query(pool_df, k)
    return res.rename(columns={"query_entity_id": "candidate_entity_id", "index_entity_id": "source1_entity_id",
                               "rank": "rev_rank", "score": "rev_score"})


def _rank_within(sorted_groups: np.ndarray) -> np.ndarray:
    """1-based position of each element inside its run of equal (pre-sorted) group ids."""
    if len(sorted_groups) == 0:
        return np.empty(0, np.int64)
    starts = np.r_[0, np.flatnonzero(np.diff(sorted_groups)) + 1]
    run_start = np.repeat(starts, np.diff(np.r_[starts, len(sorted_groups)]))
    return np.arange(len(sorted_groups)) - run_start + 1
