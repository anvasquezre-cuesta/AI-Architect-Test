"""Tests for fuzzy matching service.

These tests verify fuzzy name matching behavior.
Some tests will FAIL due to bugs in the current implementation.
"""

import pytest


class TestFuzzyMatchNames:
    """Tests for fuzzy_match_names function."""

    def test_threshold_is_90_percent(self):
        """FAILS: Bug — threshold is set to 70 instead of required 90.
        Names with 75% similarity should NOT match.
        """
        from app.services.fuzzy_service import SIMILARITY_THRESHOLD

        assert SIMILARITY_THRESHOLD == 90, (
            f"Threshold should be 90, got {SIMILARITY_THRESHOLD}"
        )

    def test_exact_match_scores_100(self):
        """Exact name matches should score 1.0 (100%)."""
        from app.services.fuzzy_service import fuzzy_match_names

        extracted = ["John Smith", "Maria Garcia"]
        query = [{"first_name": "John", "last_name": "Smith"}]

        matches = fuzzy_match_names(extracted, query)

        assert len(matches) == 1
        assert matches[0]["score"] == 1.0
        assert matches[0]["extracted_name"] == "John Smith"

    def test_no_match_below_threshold(self):
        """FAILS: Bug — partial_ratio gives inflated scores.
        'Jo' partial-matches 'John Smith' with high score.
        """
        from app.services.fuzzy_service import fuzzy_match_names

        extracted = ["John Smith"]
        query = [{"first_name": "Jo", "last_name": "Sm"}]

        matches = fuzzy_match_names(extracted, query)

        assert len(matches) == 0, (
            "Partial names should not match — 'Jo Sm' is not similar enough to 'John Smith'"
        )

    def test_close_match_above_threshold(self):
        """Names with minor typos should still match at 90% threshold."""
        from app.services.fuzzy_service import FuzzyService

        # Initialize with explicit threshold to ensure test stability
        service = FuzzyService(threshold=90)
        extracted = ["John Smith"]
        query = [{"first_name": "John", "last_name": "Smth"}]  # minor typo

        matches = service.match_names(extracted, query)

        assert len(matches) == 1
        assert matches[0]["score"] >= 0.9

    def test_completely_different_names_no_match(self):
        """Completely different names should not match."""
        from app.services.fuzzy_service import fuzzy_match_names

        extracted = ["John Smith"]
        query = [{"first_name": "Xavier", "last_name": "Rodriguez"}]

        matches = fuzzy_match_names(extracted, query)

        assert len(matches) == 0
