"""Tests for API endpoints.

These tests verify the FastAPI endpoints.
Some tests will FAIL due to bugs in the current implementation.
"""

import json
import pytest
from unittest.mock import patch, MagicMock


class TestExtractEndpoint:
    """Tests for POST /api/extract endpoint."""

    def test_rejects_non_pdf_files(self, app_client):
        """FAILS: Bug — no file type validation.
        Uploading a .txt file should return 400, not process it.
        """
        response = app_client.post(
            "/api/extract",
            files={"pdf_file": ("test.txt", b"not a pdf", "text/plain")},
            data={"names": json.dumps([{"first_name": "John", "last_name": "Smith"}])},
        )

        assert response.status_code == 400, (
            "Should reject non-PDF files with 400 status"
        )

    def test_response_includes_fuzzy_matches(self, app_client):
        """FAILS: Bug — ExtractionResponse schema missing fuzzy_matches field.
        Fuzzy match results are computed but silently dropped.
        """
        mock_name_boxes = [
            {"name": "John Smith", "page": 0, "x": 100, "y": 50, "width": 90, "height": 12}
        ]
        mock_matches = [
            {"extracted_name": "John Smith", "matched_name": "John Smith", "score": 1.0}
        ]

        with (
            patch(
                "app.api.extract.OCRService.extract_text_from_pdf", return_value="John Smith"
            ),
            patch(
                "app.api.extract.OCRService.find_name_bounding_boxes",
                return_value=mock_name_boxes,
            ),
            patch(
                "app.api.extract.FuzzyService.fuzzy_match_names", return_value=mock_matches
            ),
        ):
            response = app_client.post(
                "/api/extract",
                files={
                    "pdf_file": ("test.pdf", b"%PDF-1.4 fake", "application/pdf")
                },
                data={
                    "names": json.dumps(
                        [{"first_name": "John", "last_name": "Smith"}]
                    )
                },
            )

        data = response.json()
        assert "fuzzy_matches" in data, (
            "Response should include fuzzy_matches field"
        )
        assert len(data["fuzzy_matches"]) == 1

    def test_response_includes_page_number(self, app_client):
        """FAILS: Bug — BoundingBox schema missing page_number field."""
        mock_name_boxes = [
            {"name": "John Smith", "page": 0, "x": 100, "y": 50, "width": 90, "height": 12}
        ]

        with (
            patch(
                "app.api.extract.OCRService.extract_text_from_pdf", return_value="John Smith"
            ),
            patch(
                "app.api.extract.OCRService.find_name_bounding_boxes",
                return_value=mock_name_boxes,
            ),
            patch("app.api.extract.FuzzyService.fuzzy_match_names", return_value=[]),
        ):
            response = app_client.post(
                "/api/extract",
                files={
                    "pdf_file": ("test.pdf", b"%PDF-1.4 fake", "application/pdf")
                },
                data={"names": json.dumps([])},
            )

        data = response.json()
        for name_entry in data["extracted_names"]:
            assert "page_number" in name_entry["bounding_box"], (
                "Bounding box should include page_number"
            )


class TestRAGEndpoints:
    """Tests for RAG-related endpoints."""

    def test_ask_returns_sources(self, app_client):
        """FAILS: Bug — RAGResponse schema missing sources field."""
        mock_result = {
            "answer": "Test answer",
            "sources": ["chunk 1", "chunk 2"],
        }

        with patch("app.api.rag.generate_answer", return_value=mock_result):
            response = app_client.post(
                "/api/ask",
                json={"question": "What is this about?"},
            )

        data = response.json()
        assert "sources" in data, (
            "RAG response should include source chunks for transparency"
        )

    def test_health_endpoint_exists(self, app_client):
        """FAILS: No health check endpoint implemented."""
        response = app_client.get("/health")
        assert response.status_code == 200
