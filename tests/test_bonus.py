"""Additional tests written as part of the submission.

These cover behaviour the original suite did not exercise: configuration,
the Protocol/implementation contract, the loss-less chunker invariant, and
input-validation edge cases on the API.
"""

import pytest


class TestConfiguration:
    """Settings come from defaults and can be overridden via the environment."""

    def test_defaults(self):
        from app.config import Settings

        s = Settings()
        assert s.fuzzy_threshold == 90
        assert s.vector_size == 384
        assert s.qdrant_port == 6333
        assert s.ocr_dpi == 150

    def test_env_override(self, monkeypatch):
        monkeypatch.setenv("FUZZY_THRESHOLD", "85")
        monkeypatch.setenv("QDRANT_HOST", "qdrant.internal")
        from app.config import Settings

        s = Settings()
        assert s.fuzzy_threshold == 85
        assert s.qdrant_host == "qdrant.internal"


class TestProtocolContracts:
    """Concrete services structurally satisfy their declared interfaces."""

    def test_implementations_match_protocols(self):
        from app.services.interfaces import (
            EmbeddingModel,
            NERModel,
            OCREngine,
            VectorStore,
        )
        from app.services.embedding_service import SentenceTransformerEmbedder
        from app.services.ner_service import SpacyNERService
        from app.services.ocr_service import TesseractOCRService
        from app.services.vector_service import QdrantVectorStore

        assert isinstance(TesseractOCRService(), OCREngine)
        assert isinstance(SpacyNERService(), NERModel)
        assert isinstance(SentenceTransformerEmbedder(), EmbeddingModel)
        assert isinstance(QdrantVectorStore(), VectorStore)


class TestChunkerInvariant:
    """The chunker must be loss-less and respect word boundaries."""

    @pytest.mark.parametrize("size", [5, 10, 23, 100])
    def test_lossless_and_word_safe(self, size):
        from app.services.rag_service import chunk_text

        text = "The quick brown fox jumps over the lazy dog near the river bank."
        chunks = chunk_text(text, chunk_size=size)

        assert "".join(chunks) == text  # no characters lost or added
        for chunk in chunks:
            # within budget, unless a single token is itself larger than size
            assert len(chunk) <= size or len(chunk.split()) == 1

    def test_empty_text_yields_no_chunks(self):
        from app.services.rag_service import chunk_text

        assert chunk_text("") == []


class TestApiValidation:
    """Input-validation edge cases on the endpoints."""

    def test_extract_rejects_malformed_names_json(self, app_client):
        response = app_client.post(
            "/api/extract",
            files={"pdf_file": ("t.pdf", b"%PDF-1.4 x", "application/pdf")},
            data={"names": "this is not json"},
        )
        assert response.status_code == 400

    def test_ingest_rejects_non_pdf(self, app_client):
        response = app_client.post(
            "/api/ingest",
            files={"pdf_file": ("t.txt", b"hello world", "text/plain")},
        )
        assert response.status_code == 400
