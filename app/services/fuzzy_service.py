"""Fuzzy name-matching service.

Primary metric: token edit-distance ratio (thefuzz), thresholded at 90% as required.
Secondary metric: Jaro-Winkler (prefix-weighted, well suited to names) as a
corroborating signal. It is reported alongside the primary score and used only to flag
agreement, never to override the spec-mandated primary decision (DECISION_LOG D-2).
"""

from __future__ import annotations

from typing import Protocol

from Levenshtein import jaro_winkler
from thefuzz import fuzz

# Percent (0-100). Requirement mandates a 90% similarity threshold.
SIMILARITY_THRESHOLD = 90


class NameMatcher(Protocol):
    """Interface for name matchers."""

    def match(
        self, extracted_names: list[str], query_names: list[dict]
    ) -> list[dict]:
        ...


class FuzzyNameMatcher:
    """Edit-distance matcher with a Jaro-Winkler corroboration signal."""

    def __init__(self, threshold: int = SIMILARITY_THRESHOLD) -> None:
        self._threshold = threshold

    def match(
        self, extracted_names: list[str], query_names: list[dict]
    ) -> list[dict]:
        matches: list[dict] = []
        for query in query_names:
            # strip() keeps matching clean when a caller passes an empty last name
            # (e.g. a full name in one field), avoiding a trailing-space penalty.
            query_full = f"{query['first_name']} {query['last_name']}".strip()

            best_match = None
            best_score = 0
            for extracted in extracted_names:
                # Full-string ratio (not partial_ratio, which inflates short
                # substrings so "Jo Sm" wrongly matches "John Smith").
                score = fuzz.ratio(query_full, extracted)
                if score > best_score:
                    best_score = score
                    best_match = extracted

            if best_match is not None and best_score >= self._threshold:
                jw = jaro_winkler(query_full, best_match)
                matches.append(
                    {
                        "extracted_name": best_match,
                        "matched_name": query_full,
                        "score": best_score / 100.0,
                        "jaro_winkler": round(jw, 4),
                        "corroborated": jw >= self._threshold / 100.0,
                    }
                )
        return matches


_default_matcher = FuzzyNameMatcher()
fuzzy_match_names = _default_matcher.match
