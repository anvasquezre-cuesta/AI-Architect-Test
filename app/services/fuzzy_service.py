"""Fuzzy matching between extracted names and requested name pairs.

Bug fixes vs. original implementation:
- Threshold raised from 70 to the required 90 (configurable via env).
- ``fuzz.ratio`` (full-string Levenshtein similarity) replaces
  ``fuzz.partial_ratio``, which inflated scores for partial substrings
  ('Jo Sm' matched 'John Smith' at 100%).
- Comparison is case-insensitive.
- Comparison is accent-insensitive: 'Maria Gonzalez' matches
  'María González' (needed for the Spanish/multilingual profiles, and for
  OCR output that often drops or garbles diacritics).
"""

import unicodedata

from thefuzz import fuzz

from app.core.config import get_settings
from app.core.logging import get_logger
from app.services.interfaces import FuzzyMatcher

logger = get_logger(__name__)

# Required threshold: 90% similarity (configurable via SIMILARITY_THRESHOLD env).
SIMILARITY_THRESHOLD = get_settings().similarity_threshold


class FuzzyNameMatcher(FuzzyMatcher):
    """Levenshtein-ratio-based name matcher with a configurable threshold."""

    def __init__(self, threshold: int | None = None) -> None:
        self._threshold = threshold

    @staticmethod
    def _strip_accents(text: str) -> str:
        """Remove diacritics ('María' -> 'Maria')."""
        decomposed = unicodedata.normalize("NFD", text)
        return "".join(ch for ch in decomposed if not unicodedata.combining(ch))

    @classmethod
    def _normalize(cls, name: str) -> str:
        return cls._strip_accents(name.strip().lower())

    @property
    def threshold(self) -> int:
        return self._threshold if self._threshold is not None else SIMILARITY_THRESHOLD

    def fuzzy_match_names(
        self,
        extracted_names: list[str],
        query_names: list[dict],
    ) -> list[dict]:
        """Perform fuzzy matching between extracted and query names."""
        matches: list[dict] = []

        for query in query_names:
            first = str(query.get("first_name", "")).strip()
            last = str(query.get("last_name", "")).strip()
            query_full = f"{first} {last}".strip()
            if not query_full:
                continue

            best_match: str | None = None
            best_score = 0

            query_norm = self._normalize(query_full)
            for extracted in extracted_names:
                score = fuzz.ratio(query_norm, self._normalize(extracted))
                if score > best_score:
                    best_score = score
                    best_match = extracted

            if best_match is not None and best_score >= self.threshold:
                matches.append(
                    {
                        "extracted_name": best_match,
                        "matched_name": query_full,
                        "score": best_score / 100.0,
                    }
                )

        logger.debug(
            "Fuzzy matching: %d/%d query name(s) matched at threshold %d",
            len(matches), len(query_names), self.threshold,
        )
        return matches


# Default instance + module-level callable (stable public API / test seam).
_default_service = FuzzyNameMatcher()
fuzzy_match_names = _default_service.fuzzy_match_names
