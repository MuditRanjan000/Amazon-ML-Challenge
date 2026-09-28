"""Configurable, fit/transform pairwise features for candidate records."""

from __future__ import annotations

import pickle
from itertools import chain
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

from ..matching.normalization import (
    build_record_representation,
    common_token_count,
    jaccard,
    jaro_winkler_similarity_normalized,
    length_ratio_normalized,
    levenshtein_distance_normalized,
    normalize_for_matching,
    normalized_levenshtein_similarity_normalized,
    to_text,
)

RECORD_PAIR_COLUMNS = (
    "source1_entity_id",
    "candidate_entity_id",
    "source1_business_name",
    "source1_business_address",
    "source1_country",
    "candidate_business_name",
    "candidate_business_address",
    "candidate_country",
)


@dataclass(frozen=True)
class FeatureConfig:
    """Versioned parameters for lexical and learned feature representations."""

    char_ngram_size: int = 3
    tfidf_ngram_min: int = 2
    tfidf_ngram_max: int = 4
    tfidf_max_features: int = 50_000


def _diagonal_cosine(left, right) -> np.ndarray:
    return np.asarray(left.multiply(right).sum(axis=1)).ravel()


class PairwiseFeatureExtractor:
    """Fits TF-IDF only on caller-provided training records, then transforms pairs."""

    def __init__(self, config: FeatureConfig | None = None):
        self.config = config or FeatureConfig()
        self.name_vectorizer: TfidfVectorizer | None = None
        self.address_vectorizer: TfidfVectorizer | None = None
        self.feature_order: list[str] = []
        self.fitted = False

    def _validate_records(self, records: pd.DataFrame) -> None:
        missing = [column for column in RECORD_PAIR_COLUMNS if column not in records.columns]
        if missing:
            raise ValueError(f"Joined records missing required columns: {missing}.")

    def _fit_vectorizer(self, corpus) -> TfidfVectorizer | None:
        """Fit from a bounded list or a one-pass disk-backed corpus iterator."""

        documents = (text for text in corpus if text)
        try:
            first = next(documents)
        except StopIteration:
            return None
        vectorizer = TfidfVectorizer(
            analyzer="char_wb",
            lowercase=False,
            ngram_range=(self.config.tfidf_ngram_min, self.config.tfidf_ngram_max),
            max_features=self.config.tfidf_max_features,
            dtype=np.float32,
        )
        vectorizer.fit(chain((first,), documents))
        return vectorizer

    def fit_from_normalized_corpora(self, names, addresses) -> "PairwiseFeatureExtractor":
        """Fit from already-normalized, de-duplicated training-only corpora.

        This keeps large preflight corpus collection disk-backed while retaining
        the same TF-IDF semantics as :meth:`fit`.
        """

        self.name_vectorizer = self._fit_vectorizer(names)
        self.address_vectorizer = self._fit_vectorizer(addresses)
        self.fitted = True
        return self

    def fit(self, training_records: pd.DataFrame) -> "PairwiseFeatureExtractor":
        """Fit learned representations using training pairs only (never validation/test)."""

        self._validate_records(training_records)
        names = pd.concat(
            [training_records["source1_business_name"], training_records["candidate_business_name"]]
        ).map(normalize_for_matching).tolist()
        addresses = pd.concat(
            [training_records["source1_business_address"], training_records["candidate_business_address"]]
        ).map(normalize_for_matching).tolist()
        # Candidate fan-out must not change document-frequency statistics.  Keep
        # first-seen order for reproducibility while fitting only unique records.
        return self.fit_from_normalized_corpora(dict.fromkeys(names), dict.fromkeys(addresses))

    def _tfidf_cosine(self, vectorizer: TfidfVectorizer | None, left: list[str], right: list[str]) -> np.ndarray:
        if vectorizer is None:
            return np.zeros(len(left), dtype=np.float32)
        unique_texts = list(dict.fromkeys([*left, *right]))
        index_by_text = {text: index for index, text in enumerate(unique_texts)}
        vectors = vectorizer.transform(unique_texts)
        left_rows = vectors[[index_by_text[text] for text in left]]
        right_rows = vectors[[index_by_text[text] for text in right]]
        return _diagonal_cosine(left_rows, right_rows)

    def transform(self, records: pd.DataFrame) -> pd.DataFrame:
        if not self.fitted:
            raise RuntimeError("Call fit(training_records) before transform(records).")
        self._validate_records(records)

        output: dict[str, list[object] | np.ndarray] = {
            "source1_entity_id": records["source1_entity_id"].astype("string").tolist(),
            "candidate_entity_id": records["candidate_entity_id"].astype("string").tolist(),
        }
        feature_rows: list[dict[str, float | int]] = []
        normalized_names_left: list[str] = []
        normalized_names_right: list[str] = []
        normalized_addresses_left: list[str] = []
        normalized_addresses_right: list[str] = []
        representations = {}

        def representation_for(name: object, address: object, country: object):
            key = (to_text(name), to_text(address), to_text(country))
            if key not in representations:
                representations[key] = build_record_representation(
                    name, address, country, self.config.char_ngram_size
                )
            return representations[key]

        for row in records.loc[:, RECORD_PAIR_COLUMNS].to_dict("records"):
            name_left, name_right = row["source1_business_name"], row["candidate_business_name"]
            address_left, address_right = row["source1_business_address"], row["candidate_business_address"]
            country_left, country_right = to_text(row["source1_country"]), to_text(row["candidate_country"])
            source_representation = representation_for(name_left, address_left, country_left)
            candidate_representation = representation_for(name_right, address_right, country_right)
            norm_name_left, norm_name_right = (
                source_representation.normalized_name,
                candidate_representation.normalized_name,
            )
            norm_address_left, norm_address_right = (
                source_representation.normalized_address,
                candidate_representation.normalized_address,
            )
            normalized_names_left.append(norm_name_left)
            normalized_names_right.append(norm_name_right)
            normalized_addresses_left.append(norm_address_left)
            normalized_addresses_right.append(norm_address_right)

            name_tokens_left, name_tokens_right = source_representation.name_tokens, candidate_representation.name_tokens
            address_tokens_left, address_tokens_right = source_representation.address_tokens, candidate_representation.address_tokens
            numeric_left, numeric_right = (
                source_representation.numeric_address_tokens,
                candidate_representation.numeric_address_tokens,
            )
            postal_left, postal_right = (
                source_representation.postal_address_tokens,
                candidate_representation.postal_address_tokens,
            )
            locality_left, locality_right = source_representation.locality_tokens, candidate_representation.locality_tokens
            source_script, candidate_script = dict(source_representation.script_flags), dict(candidate_representation.script_flags)
            name_available = int(bool(norm_name_left and norm_name_right))
            address_available = int(bool(norm_address_left and norm_address_right))
            numeric_available = int(bool(numeric_left and numeric_right))
            postal_available = int(bool(postal_left and postal_right and country_left == country_right))
            locality_available = int(bool(locality_left and locality_right))

            row_features: dict[str, float | int] = {
                "source1_name_missing": int(not norm_name_left),
                "candidate_name_missing": int(not norm_name_right),
                "source1_address_missing": int(not norm_address_left),
                "candidate_address_missing": int(not norm_address_right),
                "name_exact": int(name_available and norm_name_left == norm_name_right),
                "name_jaro_winkler": jaro_winkler_similarity_normalized(norm_name_left, norm_name_right),
                "name_levenshtein_distance": levenshtein_distance_normalized(norm_name_left, norm_name_right),
                "name_levenshtein_similarity": normalized_levenshtein_similarity_normalized(norm_name_left, norm_name_right),
                "name_token_jaccard": jaccard(name_tokens_left, name_tokens_right),
                "name_common_token_count": common_token_count(name_tokens_left, name_tokens_right),
                "name_char_ngram_similarity": jaccard(source_representation.name_ngrams, candidate_representation.name_ngrams),
                "name_length_ratio": length_ratio_normalized(norm_name_left, norm_name_right),
                "address_exact": int(address_available and norm_address_left == norm_address_right),
                "address_token_jaccard": jaccard(address_tokens_left, address_tokens_right),
                "address_common_token_count": common_token_count(address_tokens_left, address_tokens_right),
                "address_char_ngram_similarity": jaccard(source_representation.address_ngrams, candidate_representation.address_ngrams),
                "address_length_ratio": length_ratio_normalized(norm_address_left, norm_address_right),
                "numeric_overlap": jaccard(numeric_left, numeric_right),
                "numeric_disagreement": int(numeric_available and numeric_left != numeric_right),
                "numeric_both_available": numeric_available,
                "postal_evidence_available": postal_available,
                "postal_match": int(postal_available and bool(postal_left & postal_right)),
                "postal_disagreement": int(postal_available and not (postal_left & postal_right)),
                "locality_evidence_available": locality_available,
                "locality_token_jaccard": jaccard(locality_left, locality_right),
                "country_match": int(bool(country_left and country_right and country_left == country_right)),
                "country_disagreement": int(bool(country_left and country_right and country_left != country_right)),
                "candidate_is_s2": int(str(row["candidate_entity_id"]).startswith("S2-")),
                "candidate_is_s3": int(str(row["candidate_entity_id"]).startswith("S3-")),
                "both_name_address_available": int(bool(name_available and address_available)),
                "name_address_mean_lexical": (
                    jaro_winkler_similarity_normalized(norm_name_left, norm_name_right)
                    + jaccard(source_representation.address_ngrams, candidate_representation.address_ngrams)
                )
                / (1 + int(bool(norm_address_left and norm_address_right))),
                "both_exact_name_address": int(
                    name_available and address_available and norm_name_left == norm_name_right and norm_address_left == norm_address_right
                ),
            }
            for script in ("latin", "devanagari", "other"):
                row_features[f"source1_name_has_{script}"] = source_script[script]
                row_features[f"candidate_name_has_{script}"] = candidate_script[script]
                row_features[f"name_{script}_script_match"] = int(bool(source_script[script] and candidate_script[script]))
            feature_rows.append(row_features)

        feature_frame = pd.DataFrame(feature_rows)
        feature_frame["name_tfidf_cosine"] = self._tfidf_cosine(
            self.name_vectorizer, normalized_names_left, normalized_names_right
        )
        feature_frame["address_tfidf_cosine"] = self._tfidf_cosine(
            self.address_vectorizer, normalized_addresses_left, normalized_addresses_right
        )
        self.feature_order = feature_frame.columns.tolist()
        return pd.concat([pd.DataFrame(output), feature_frame], axis=1)

    def save(self, path: str | Path) -> None:
        if not self.fitted:
            raise RuntimeError("Only a fitted feature extractor can be persisted.")
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("wb") as handle:
            pickle.dump(self, handle)
        target.with_suffix(target.suffix + ".json").write_text(
            pd.Series({"config": asdict(self.config), "feature_order": self.feature_order}).to_json(indent=2),
            encoding="utf-8",
        )

    @classmethod
    def load(cls, path: str | Path) -> "PairwiseFeatureExtractor":
        with Path(path).open("rb") as handle:
            extractor = pickle.load(handle)
        if not isinstance(extractor, cls) or not extractor.fitted:
            raise ValueError("Artifact does not contain a fitted PairwiseFeatureExtractor.")
        return extractor
