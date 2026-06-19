"""Fuzzy name-matching service.

Scores each requested ``first last`` pair against the extracted names and keeps
the best match when it clears the configured similarity threshold (90% per the
spec). Uses ``fuzz.ratio`` (full-string Levenshtein ratio) rather than
``partial_ratio`` so short substrings (e.g. ``Jo Sm``) do not spuriously match
longer names.
"""

import logging

from thefuzz import fuzz

from app.config import settings

logger = logging.getLogger(__name__)

# Module-level constant (read by callers/tests); sourced from settings.
SIMILARITY_THRESHOLD = settings.fuzzy_threshold


class FuzzyMatcher:
    """Best-match fuzzy matcher with a configurable similarity threshold."""

    def match(self, extracted_names: list, query_names: list) -> list:
        matches = []
        for query in query_names:
            query_full = f"{query['first_name']} {query['last_name']}"

            best_match = None
            best_score = 0
            for extracted in extracted_names:
                score = fuzz.ratio(query_full, extracted)
                if score > best_score:
                    best_score = score
                    best_match = extracted

            if best_score >= SIMILARITY_THRESHOLD:
                matches.append(
                    {
                        "extracted_name": best_match,
                        "matched_name": query_full,
                        "score": best_score / 100.0,
                    }
                )
        return matches


_matcher = FuzzyMatcher()
fuzzy_match_names = _matcher.match
