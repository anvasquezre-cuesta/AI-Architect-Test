"""Edge-case tests: bad input, error handling, empty input, and unicode.

These complement the bug-fix tests by covering failure modes a user can actually hit.
"""

import io
import json
from unittest.mock import patch


class TestExtractEndpointEdges:
    def test_ingest_rejects_non_pdf(self, app_client):
        """A non-PDF upload to /ingest must be rejected with 400."""
        res = app_client.post(
            "/api/ingest",
            files={"pdf_file": ("note.txt", b"hello", "text/plain")},
        )
        assert res.status_code == 400

    def test_extract_rejects_invalid_names_json(self, app_client):
        """A malformed `names` payload must 400 before any processing."""
        res = app_client.post(
            "/api/extract",
            files={"pdf_file": ("doc.pdf", b"%PDF-1.4 x", "application/pdf")},
            data={"names": "not-json"},
        )
        assert res.status_code == 400

    def test_extract_processing_error_returns_clean_500(self, app_client):
        """An OCR/processing failure returns a clean 500, not a stack trace."""
        with patch(
            "app.api.extract.extract_text_from_pdf", side_effect=RuntimeError("boom")
        ):
            res = app_client.post(
                "/api/extract",
                files={"pdf_file": ("doc.pdf", b"%PDF-1.4 x", "application/pdf")},
                data={"names": json.dumps([])},
            )
        assert res.status_code == 500


class TestServiceEdges:
    def test_chunk_text_empty_returns_empty(self):
        from app.services.rag_service import chunk_text

        assert chunk_text("") == []

    def test_generate_answer_without_llm_returns_notice(self):
        """With no API key, /ask should return retrieved sources plus a clear notice."""
        from app.services.rag_service import RAGService

        rag = RAGService(
            embed_query=lambda q: [0.1] * 384,
            retriever=lambda vec, top_k=3: [{"text": "context chunk", "score": 0.9}],
            api_key="",
        )
        result = rag.generate_answer("anything?")
        assert "No LLM configured" in result["answer"]
        assert result["sources"] == ["context chunk"]

    def test_fuzzy_handles_unicode_names(self):
        """Accented names must be handled and score 100 on an exact match."""
        from app.services.fuzzy_service import fuzzy_match_names

        matches = fuzzy_match_names(
            ["José García"], [{"first_name": "José", "last_name": "García"}]
        )
        assert len(matches) == 1
        assert matches[0]["score"] == 1.0

    def test_ner_empty_text_returns_empty(self):
        from app.services.ner_service import extract_names

        assert extract_names("") == []
