"""Tests for vector database service.

These tests verify Qdrant integration.
Some tests will FAIL due to bugs in the current implementation.
"""

import pytest
from unittest.mock import patch, MagicMock, call
from qdrant_client.models import Distance


class TestInitCollection:
    """Tests for init_collection function."""

    def test_creates_collection_with_cosine_distance(self, mock_qdrant_client):
        """Verify collection uses correct distance metric."""
        from app.services.vector_service import init_collection

        init_collection()

        mock_qdrant_client.create_collection.assert_called_once()
        call_kwargs = mock_qdrant_client.create_collection.call_args
        vector_config = call_kwargs.kwargs.get(
            "vectors_config", call_kwargs[1].get("vectors_config")
        )
        assert vector_config.distance == Distance.COSINE

    def test_skips_creation_if_exists(self, mock_qdrant_client):
        """Should not recreate collection if it already exists."""
        existing = MagicMock()
        existing.name = "pdf_documents"
        mock_qdrant_client.get_collections.return_value = MagicMock(
            collections=[existing]
        )

        from app.services.vector_service import init_collection

        init_collection()

        mock_qdrant_client.create_collection.assert_not_called()


class TestStoreDocumentChunks:
    """Tests for store_document_chunks function."""

    def test_generates_unique_ids(self, mock_qdrant_client):
        """FAILS: Bug — IDs start from 0 every time.
        Storing a second document overwrites the first document's chunks.
        """
        from app.services.vector_service import store_document_chunks

        mock_embeddings = [[0.1] * 384, [0.2] * 384]

        with patch(
            "app.services.vector_service.get_embeddings",
            return_value=mock_embeddings,
        ):
            # Store first document
            store_document_chunks(["chunk 1", "chunk 2"])
            first_call_points = mock_qdrant_client.upsert.call_args_list[0]

            # Store second document
            store_document_chunks(["chunk A", "chunk B"])
            second_call_points = mock_qdrant_client.upsert.call_args_list[1]

        # Extract point IDs from both calls
        first_ids = [
            p.id for p in first_call_points.kwargs.get("points", first_call_points[1].get("points", []))
        ]
        second_ids = [
            p.id for p in second_call_points.kwargs.get("points", second_call_points[1].get("points", []))
        ]

        # IDs should NOT overlap
        assert set(first_ids).isdisjoint(set(second_ids)), (
            f"Point IDs overlap: {first_ids} vs {second_ids}. "
            "Second document will overwrite first."
        )


class TestSearchSimilar:
    """Tests for search_similar function."""

    def test_returns_text_and_score(self, mock_qdrant_client):
        """Should return matching chunks with text and similarity score."""
        mock_hit = MagicMock()
        mock_hit.payload = {"text": "relevant chunk"}
        mock_hit.score = 0.85
        
        mock_result = MagicMock()
        mock_result.points = [mock_hit]
        mock_qdrant_client.query_points.return_value = mock_result

        from app.services.vector_service import search_similar

        results = search_similar([0.1] * 384, top_k=3)

        assert len(results) == 1
        assert results[0]["text"] == "relevant chunk"
        assert results[0]["score"] == 0.85
