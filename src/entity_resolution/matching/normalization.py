"""Matching-specific text representations that preserve Unicode information."""

from __future__ import annotations

import math
import re
import unicodedata
from collections import Counter
from dataclasses import dataclass

import pandas as pd

try:
    from rapidfuzz.distance import JaroWinkler as _RapidFuzzJaroWinkler
    from rapidfuzz.distance import Levenshtein as _RapidFuzzLevenshtein
except ImportError:  # pragma: no cover - retained for constrained environments.
    _RapidFuzzJaroWinkler = None
    _RapidFuzzLevenshtein = None

_POSTAL_PATTERNS = {
    "US": re.compile(r"\b\d{5}(?:-\d{4})?\b"),
    "India": re.compile(r"\b\d{6}\b"),
    "France": re.compile(r"\b\d{5}\b"),
}


def to_text(value: object) -> str:
    """Return a safe string while treating pandas nulls as missing."""

    if value is None or value is pd.NA:
        return ""
    if isinstance(value, float) and math.isnan(value):
        return ""
    text = str(value).strip()
    return "" if text.casefold() in {"", "nan", "null", "none"} else text


def normalize_for_matching(value: object) -> str:
    """Casefold and tokenize text without transliterating or discarding scripts/digits."""

    text = unicodedata.normalize("NFKC", to_text(value)).casefold().replace("&", " and ")
    characters = [
        character if character.isalnum() or unicodedata.category(character).startswith("M") else " "
        for character in text
    ]
    return " ".join("".join(characters).split())


def tokens(value: object) -> tuple[str, ...]:
    return tuple(normalize_for_matching(value).split())


def token_set(value: object) -> set[str]:
    return set(tokens(value))


def jaccard(left: set[str], right: set[str]) -> float:
    if not left or not right:
        return 0.0
    return len(left & right) / len(left | right)


def common_token_count(left: set[str], right: set[str]) -> int:
    return len(left & right) if left and right else 0


def character_ngrams(value: object, n: int = 3) -> set[str]:
    return character_ngrams_normalized(normalize_for_matching(value), n)


def character_ngrams_normalized(text: str, n: int = 3) -> set[str]:
    text = text.replace(" ", "")
    if not text:
        return set()
    if len(text) < n:
        return {text}
    return {text[index : index + n] for index in range(len(text) - n + 1)}


def character_ngram_similarity(left: object, right: object, n: int = 3) -> float:
    return character_ngram_similarity_normalized(normalize_for_matching(left), normalize_for_matching(right), n)


def character_ngram_similarity_normalized(left: str, right: str, n: int = 3) -> float:
    return jaccard(character_ngrams_normalized(left, n), character_ngrams_normalized(right, n))


def levenshtein_distance(left: object, right: object) -> int:
    """Memory-bounded Levenshtein distance over normalized strings."""

    return levenshtein_distance_normalized(normalize_for_matching(left), normalize_for_matching(right))


def levenshtein_distance_normalized(first: str, second: str) -> int:
    if _RapidFuzzLevenshtein is not None:
        return int(_RapidFuzzLevenshtein.distance(first, second))
    return _python_levenshtein_distance_normalized(first, second)


def _python_levenshtein_distance_normalized(first: str, second: str) -> int:
    if len(first) < len(second):
        first, second = second, first
    if not second:
        return len(first)
    previous = list(range(len(second) + 1))
    for index, first_char in enumerate(first, start=1):
        current = [index]
        for second_index, second_char in enumerate(second, start=1):
            current.append(
                min(
                    current[-1] + 1,
                    previous[second_index] + 1,
                    previous[second_index - 1] + (first_char != second_char),
                )
            )
        previous = current
    return previous[-1]


def normalized_levenshtein_similarity(left: object, right: object) -> float:
    first, second = normalize_for_matching(left), normalize_for_matching(right)
    return normalized_levenshtein_similarity_normalized(first, second)


def normalized_levenshtein_similarity_normalized(first: str, second: str) -> float:
    if not first or not second:
        return 0.0
    return 1.0 - levenshtein_distance_normalized(first, second) / max(len(first), len(second))


def jaro_winkler_similarity(left: object, right: object) -> float:
    """Dependency-free Jaro-Winkler similarity for bounded pairwise features."""

    first, second = normalize_for_matching(left), normalize_for_matching(right)
    return jaro_winkler_similarity_normalized(first, second)


def jaro_winkler_similarity_normalized(first: str, second: str) -> float:
    if _RapidFuzzJaroWinkler is not None:
        return float(_RapidFuzzJaroWinkler.normalized_similarity(first, second))
    return _python_jaro_winkler_similarity_normalized(first, second)


def _python_jaro_winkler_similarity_normalized(first: str, second: str) -> float:
    if not first or not second:
        return 0.0
    if first == second:
        return 1.0
    match_window = max(len(first), len(second)) // 2 - 1
    first_matches = [False] * len(first)
    second_matches = [False] * len(second)
    matches = 0
    for index, character in enumerate(first):
        start, end = max(0, index - match_window), min(index + match_window + 1, len(second))
        for other_index in range(start, end):
            if not second_matches[other_index] and character == second[other_index]:
                first_matches[index] = second_matches[other_index] = True
                matches += 1
                break
    if not matches:
        return 0.0
    first_sequence = [first[index] for index, matched in enumerate(first_matches) if matched]
    second_sequence = [second[index] for index, matched in enumerate(second_matches) if matched]
    transpositions = sum(a != b for a, b in zip(first_sequence, second_sequence)) / 2
    jaro = (matches / len(first) + matches / len(second) + (matches - transpositions) / matches) / 3
    prefix = 0
    for a, b in zip(first, second):
        if a != b or prefix == 4:
            break
        prefix += 1
    return jaro + prefix * 0.1 * (1 - jaro)


def length_ratio(left: object, right: object) -> float:
    first, second = normalize_for_matching(left), normalize_for_matching(right)
    return length_ratio_normalized(first, second)


def length_ratio_normalized(first: str, second: str) -> float:
    if not first or not second:
        return 0.0
    return min(len(first), len(second)) / max(len(first), len(second))


def numeric_tokens(value: object) -> set[str]:
    return set(re.findall(r"\d+", normalize_for_matching(value)))


def numeric_tokens_normalized(text: str) -> set[str]:
    return set(re.findall(r"\d+", text))


def postal_tokens(value: object, country: object) -> set[str]:
    pattern = _POSTAL_PATTERNS.get(to_text(country))
    if not pattern:
        return set()
    text = unicodedata.normalize("NFKC", to_text(value))
    return {match.casefold() for match in pattern.findall(text)}


def extract_locality_tokens(value: object, country: object) -> set[str]:
    """Extract a locality only from a conservative comma-plus-short-region pattern.

    This intentionally omits uncertain cases (including many French addresses),
    allowing the corresponding feature to represent unavailable evidence as zero.
    """

    parts = [normalize_for_matching(part) for part in to_text(value).split(",")]
    parts = [part for part in parts if part]
    region = parts[-1].split() if len(parts) >= 3 else []
    if len(parts) < 3 or len(region) != 1 or len(region[0]) not in {2, 3}:
        return set()
    return set(parts[-2].split())


def script_indicators(value: object) -> dict[str, int]:
    """Return coarse script flags without transliteration or language claims."""

    return script_indicators_normalized(normalize_for_matching(value))


def script_indicators_normalized(text: str) -> dict[str, int]:
    observed = Counter()
    for character in text:
        if not character.isalpha():
            continue
        name = unicodedata.name(character, "")
        if "DEVANAGARI" in name:
            observed["devanagari"] += 1
        elif "LATIN" in name:
            observed["latin"] += 1
        else:
            observed["other"] += 1
    return {key: int(observed[key] > 0) for key in ("latin", "devanagari", "other")}


@dataclass(frozen=True)
class RecordRepresentation:
    """Reusable record-level lexical state for a single bounded feature batch."""

    normalized_name: str
    normalized_address: str
    name_tokens: frozenset[str]
    address_tokens: frozenset[str]
    name_ngrams: frozenset[str]
    address_ngrams: frozenset[str]
    numeric_address_tokens: frozenset[str]
    postal_address_tokens: frozenset[str]
    locality_tokens: frozenset[str]
    script_flags: tuple[tuple[str, int], ...]


def build_record_representation(name: object, address: object, country: object, n: int = 3) -> RecordRepresentation:
    """Build all reusable representations once; no record-level cache persists past a batch."""

    normalized_name, normalized_address = normalize_for_matching(name), normalize_for_matching(address)
    return RecordRepresentation(
        normalized_name=normalized_name,
        normalized_address=normalized_address,
        name_tokens=frozenset(normalized_name.split()),
        address_tokens=frozenset(normalized_address.split()),
        name_ngrams=frozenset(character_ngrams_normalized(normalized_name, n)),
        address_ngrams=frozenset(character_ngrams_normalized(normalized_address, n)),
        numeric_address_tokens=frozenset(numeric_tokens_normalized(normalized_address)),
        postal_address_tokens=frozenset(postal_tokens(address, country)),
        locality_tokens=frozenset(extract_locality_tokens(address, country)),
        script_flags=tuple(sorted(script_indicators_normalized(normalized_name).items())),
    )
