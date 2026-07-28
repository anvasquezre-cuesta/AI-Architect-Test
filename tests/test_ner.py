"""Tests for NER service.

These tests verify named entity extraction.
Some tests will FAIL due to bugs in the current implementation.
"""

import pytest
from unittest.mock import patch, MagicMock


class TestExtractNames:
    """Tests for extract_names function."""

    def test_extracts_only_person_entities(self):
        """FAILS: Bug — function also extracts ORG entities.
        'Acme Corp' should NOT appear in results.
        """
        mock_doc = MagicMock()
        mock_ents = [
            MagicMock(text="John Smith", label_="PERSON"),
            MagicMock(text="Acme Corp", label_="ORG"),
            MagicMock(text="Maria Garcia", label_="PERSON"),
        ]
        mock_doc.ents = mock_ents

        mock_model = MagicMock(side_effect=lambda x: mock_doc)
        mock_spacy_model = MagicMock()
        mock_spacy_model.get_instance.return_value = mock_model

        with patch("app.services.ner_service.SpacyModel", mock_spacy_model):
            from app.services.ner_service import NerService

            text = "John Smith works at Acme Corp with Maria Garcia"
            ner_service = NerService(text)
            result = ner_service.extract_names(include_orgs=False)

        assert "John Smith" in result
        assert "Maria Garcia" in result
        assert "Acme Corp" not in result, (
            "ORG entities should not be returned as person names"
        )

    def test_returns_empty_list_for_no_names(self):
        """Should return empty list when no person entities found."""
        mock_doc = MagicMock()
        mock_doc.ents = []
        mock_model = MagicMock(side_effect=lambda x: mock_doc)
        mock_spacy_model = MagicMock()
        mock_spacy_model.get_instance.return_value = mock_model

        with patch("app.services.ner_service.SpacyModel", mock_spacy_model):
            from app.services.ner_service import NerService

            text = "This text has no names in it."
            ner_service = NerService(text)
            result = ner_service.extract_names()

        assert result == []

    def test_handles_multiple_entity_types(self):
        """Should only return PERSON entities, ignoring GPE, DATE, etc."""
        mock_doc = MagicMock()
        mock_ents = [
            MagicMock(text="Sarah Williams", label_="PERSON"),
            MagicMock(text="New York", label_="GPE"),
            MagicMock(text="January 2024", label_="DATE"),
            MagicMock(text="MIT", label_="ORG"),
        ]
        mock_doc.ents = mock_ents
        mock_model = MagicMock(side_effect=lambda x: mock_doc)
        mock_spacy_model = MagicMock()
        mock_spacy_model.get_instance.return_value = mock_model

        with patch("app.services.ner_service.SpacyModel", mock_spacy_model):
            from app.services.ner_service import NerService

            text = "Sarah Williams visited New York in January 2024."
            ner_service = NerService(text)
            result = ner_service.extract_names(include_orgs=False)

        assert result == ["Sarah Williams"]


class TestExtractNamesWithPositions:
    """Tests for extract_names_with_positions function."""

    def test_returns_positions_for_person_only(self):
        """FAILS: Bug — includes ORG entities in results."""
        mock_doc = MagicMock()
        mock_ents = [
            MagicMock(
                text="John Smith",
                label_="PERSON",
                start_char=0,
                end_char=10,
            ),
            MagicMock(
                text="Google",
                label_="ORG",
                start_char=25,
                end_char=31,
            ),
        ]
        mock_doc.ents = mock_ents
        mock_model = MagicMock(side_effect=lambda x: mock_doc)
        mock_spacy_model = MagicMock()
        mock_spacy_model.get_instance.return_value = mock_model

        with patch("app.services.ner_service.SpacyModel", mock_spacy_model):
            from app.services.ner_service import NerService

            text = "John Smith works at Google."
            ner_service = NerService(text)
            result = ner_service.extract_names_with_positions(include_orgs=False)

        assert len(result) == 1, "Should only return PERSON entities"
        assert result[0]["name"] == "John Smith"
        assert result[0]["label"] == "PERSON"
