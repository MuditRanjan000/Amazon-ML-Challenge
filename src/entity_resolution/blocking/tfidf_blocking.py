"""Country-partitioned TF-IDF n-gram blocking with sparse top-k (forward or reverse).

Memory model:
  * HashingVectorizer: no n-gram vocabulary dict (GBs at 10M rows).
  * The index is stored as row shards of the transposed TF-IDF matrix, in RAM or, when
    `cache_dir` is set, on disk (a char (3,5) name+address index over the 10.3M train
    pool is ~11 GB, more than a 16 GB laptop can hold next to everything else).
  * sparse_dot_topn keeps only the top-k products per query; shard results are merged.
IDF is exact per partition (two hashing passes), with the same formula as sklearn's
TfidfVectorizer (smooth_idf, l2 rows); min_df / max_df drop n-grams by index doc freq.

Countries are an open set: a query whose country has no partition is searched against
every partition (France exists only in test).
"""
import hashlib
import json
import logging
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import scipy.sparse as sp
from joblib import Parallel, delayed
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.preprocessing import normalize
from sparse_dot_topn import sp_matmul_topn

from .. import config
from ..data.preprocessing import combine_fields
from .base import BaseBlocker

log = logging.getLogger(__name__)


class TfidfBlocker(BaseBlocker):
    def __init__(self, text_fields=("business_name", "business_address"), ngram_range=(1, 1), analyzer="word",
                 n_features=2 ** 22, partition_by_country=True, strip_legal=False, min_df=1, max_df=1.0,
                 chunk_size=100_000, shard_size=2_000_000, cache_dir=None, n_jobs=None):
        """min_df / max_df: keep n-grams whose index doc count is >= min_df and <= max_df * n_docs.
        Very common n-grams carry little IDF weight but dominate sparse-matmul cost."""
        self.config = dict(text_fields=list(text_fields), ngram_range=list(ngram_range), analyzer=analyzer,
                           n_features=n_features, partition_by_country=partition_by_country,
                           strip_legal=strip_legal, min_df=min_df, max_df=max_df)
        self.chunk_size = chunk_size
        self.shard_size = shard_size
        self.cache_dir = cache_dir
        self.n_jobs = n_jobs or config.N_JOBS
        self.hasher = HashingVectorizer(analyzer=analyzer, ngram_range=tuple(ngram_range), n_features=n_features,
                                        alternate_sign=False, norm=None, dtype=np.float32)
        self.index_ids = None  # pd.Index of indexed entity IDs (categories of the output)
        self.parts = {}        # country -> {"idf": float32[n_features], "shards": [(positions, index_T | npz path)]}
        self.nnz = 0

    # ------------------------------------------------------------------ vectors
    def _text(self, df: pd.DataFrame) -> pd.Series:
        return combine_fields(df, self.config["text_fields"], strip_legal=self.config["strip_legal"])

    def _hash(self, text: pd.Series) -> sp.csr_matrix:
        values = text.to_numpy(dtype=object)
        step = 50_000
        if len(values) <= step or self.n_jobs == 1:
            return self.hasher.transform(values)
        parts = Parallel(n_jobs=self.n_jobs)(
            delayed(self.hasher.transform)(values[i:i + step]) for i in range(0, len(values), step))
        return sp.vstack(parts, format="csr")

    @staticmethod
    def _weight(counts: sp.csr_matrix, idf: np.ndarray) -> sp.csr_matrix:
        """Counts -> TF-IDF (dropped n-grams have idf 0) -> L2 rows, so dot product = cosine."""
        m = counts.copy()
        m.data *= idf[m.indices]
        m.eliminate_zeros()
        return normalize(m).astype(np.float32)

    def vectors(self, df: pd.DataFrame, country="") -> sp.csr_matrix:
        """The exact vectors used for retrieval against `country`'s partition."""
        return self._weight(self._hash(self._text(df)), self.parts[country]["idf"])

    # ------------------------------------------------------------------ index
    def _cache_path(self, ids, countries, text):
        if not self.cache_dir:
            return None
        h = hashlib.sha1(json.dumps(self.config, sort_keys=True).encode())
        h.update(pd.util.hash_pandas_object(pd.DataFrame({"i": ids, "c": countries, "t": text}),
                                            index=False).to_numpy().tobytes())
        path = Path(self.cache_dir) / h.hexdigest()[:12]
        path.mkdir(parents=True, exist_ok=True)
        return path

    def fit(self, index_df: pd.DataFrame) -> "TfidfBlocker":
        """Index records with columns entity_id, country + text_fields."""
        text = self._text(index_df)
        keep = (text != "").to_numpy()
        text = text[keep].reset_index(drop=True)
        ids = index_df["entity_id"].to_numpy()[keep]
        countries = (index_df["country"].fillna("").to_numpy()[keep] if self.config["partition_by_country"]
                     else np.full(len(ids), "", dtype=object))
        cache = self._cache_path(ids, countries, text.to_numpy(dtype=object))
        if cache and (cache / "meta.joblib").exists():
            self.index_ids, self.parts, self.nnz = joblib.load(cache / "meta.joblib")
            log.info("loaded cached index %s (%d records, nnz=%d)", cache, len(self.index_ids), self.nnz)
            return self

        self.index_ids, self.parts, self.nnz = pd.Index(ids), {}, 0
        for j, country in enumerate(pd.unique(countries)):
            rows = np.flatnonzero(countries == country).astype(np.int32)
            shards = [rows[i:i + self.shard_size] for i in range(0, len(rows), self.shard_size)]
            doc_freq = np.zeros(self.config["n_features"], np.int64)
            for s in shards:  # pass 1: document frequencies (hashing again beats holding all counts)
                doc_freq += np.bincount(self._hash(text.iloc[s]).indices, minlength=len(doc_freq))
            n = len(rows)
            idf = (np.log((1 + n) / (1 + doc_freq)) + 1).astype(np.float32)
            idf[(doc_freq < max(self.config["min_df"], 1)) | (doc_freq > self.config["max_df"] * n)] = 0
            stored = []
            for i, s in enumerate(shards):  # pass 2: weighted, transposed shards
                index_t = self._weight(self._hash(text.iloc[s]), idf).T.tocsr()
                self.nnz += index_t.nnz
                if cache:
                    path = cache / f"p{j}_s{i}.npz"
                    sp.save_npz(path, index_t, compressed=False)
                    index_t = path
                stored.append((s, index_t))
            self.parts[country] = {"idf": idf, "shards": stored}
            log.info("partition %r: %d records, %d shard(s), %d n-grams kept", country, n, len(shards),
                     int((idf > 0).sum()))
        if cache:
            joblib.dump((self.index_ids, self.parts, self.nnz), cache / "meta.joblib")
        log.info("indexed %d records (%d empty skipped) in %d partition(s), nnz=%d",
                 len(ids), int((~keep).sum()), len(self.parts), self.nnz)
        return self

    # ------------------------------------------------------------------ query
    def _topk(self, counts: sp.csr_matrix, country, k: int):
        """Top-k against one partition (or all, for unseen countries): (q_row, global_pos, score)."""
        rows = pos = score = None
        for c in ([country] if country in self.parts else list(self.parts)):
            q = self._weight(counts, self.parts[c]["idf"])
            for positions, index_t in self.parts[c]["shards"]:
                if isinstance(index_t, Path):
                    index_t = sp.load_npz(index_t)
                res = sp_matmul_topn(q, index_t, top_n=k, n_threads=self.n_jobs).tocoo()
                r, p, s = res.row, positions[res.col], res.data
                if rows is not None:
                    r, p, s = np.concatenate([rows, r]), np.concatenate([pos, p]), np.concatenate([score, s])
                rows, pos, score = _keep_topk(r, p, s, k)
        return rows, pos, score

    def query(self, query_df: pd.DataFrame, k: int = 50) -> pd.DataFrame:
        """Top-k index records per query row.

        Returns columns query_entity_id, index_entity_id (categorical), score (float32), rank (int16),
        sorted by query then descending score. Queries with empty text get no rows.
        """
        if self.index_ids is None:
            raise RuntimeError("call fit() before query()")
        q_ids = pd.Index(query_df["entity_id"].to_numpy())
        countries = (query_df["country"].fillna("").to_numpy() if self.config["partition_by_country"]
                     else np.full(len(query_df), "", dtype=object))
        q_rows, i_pos, scores = [np.empty(0, np.int32)], [np.empty(0, np.int32)], [np.empty(0, np.float32)]
        for start in range(0, len(query_df), self.chunk_size):
            counts = self._hash(self._text(query_df.iloc[start:start + self.chunk_size]))
            chunk_countries = countries[start:start + self.chunk_size]
            for country in pd.unique(chunk_countries):
                local = np.flatnonzero(chunk_countries == country)
                rows, pos, score = self._topk(counts[local], country, k)
                q_rows.append(local[rows] + start)
                i_pos.append(pos)
                scores.append(score)
            log.info("queried %d / %d", min(start + self.chunk_size, len(query_df)), len(query_df))
        q_rows, i_pos, scores = _keep_topk(np.concatenate(q_rows).astype(np.int32),
                                           np.concatenate(i_pos).astype(np.int32),
                                           np.concatenate(scores).astype(np.float32), k)
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


def _keep_topk(rows, pos, score, k):
    """Sort by (row, -score) and keep the first k per row."""
    order = np.lexsort((-score, rows))
    rows, pos, score = rows[order], pos[order], score[order]
    keep = _rank_within(rows) <= k
    return rows[keep], pos[keep], score[keep]


def _rank_within(sorted_groups: np.ndarray) -> np.ndarray:
    """1-based position of each element inside its run of equal (pre-sorted) group ids."""
    if len(sorted_groups) == 0:
        return np.empty(0, np.int64)
    starts = np.r_[0, np.flatnonzero(np.diff(sorted_groups)) + 1]
    run_start = np.repeat(starts, np.diff(np.r_[starts, len(sorted_groups)]))
    return np.arange(len(sorted_groups)) - run_start + 1
