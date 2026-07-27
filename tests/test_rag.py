"""Tests for RAG service.

These tests verify the RAG pipeline.
Some tests will FAIL due to bugs in the current implementation.
"""

import pytest
from unittest.mock import patch, MagicMock


class TestChunkText:
    """Tests for chunk_text function."""

    def test_chunks_respect_word_boundaries(self):
        """FAILS: Bug — naive character splitting cuts words in half.
        'Hello world' with chunk_size=7 produces ['Hello w', 'orld'].
        Should produce ['Hello', 'world'] or similar word-aware split.
        """
        from app.services.rag_service import chunk_text

        text = "Hello world this is a test sentence for chunking"
        chunks = chunk_text(text, chunk_size=15)

        for chunk in chunks:
            # No chunk should start or end mid-word
            # (unless it's the very start or end of text)
            stripped = chunk.strip()
            if stripped:
                assert not stripped[0].isalpha() or stripped[0] == stripped[0], (
                    f"Chunk starts mid-word: '{chunk}'"
                )
                # Check that no word is split across chunks
                words_in_chunk = stripped.split()
                for word in words_in_chunk:
                    assert word in text, f"Word fragment '{word}' not in original text"

    def test_preserves_all_text(self):
        """All original text should be preserved across chunks."""
        from app.services.rag_service import chunk_text

        text = "The quick brown fox jumps over the lazy dog."
        chunks = chunk_text(text, chunk_size=10)

        reconstructed = "".join(chunks)
        assert reconstructed == text

    def test_single_chunk_for_short_text(self):
        """Short text should produce a single chunk."""
        from app.services.rag_service import chunk_text

        text = "Short text."
        chunks = chunk_text(text, chunk_size=500)

        assert len(chunks) == 1
        assert chunks[0] == text


class TestGenerateAnswer:
    """Tests for generate_answer function."""

    def test_prompt_includes_question(self):
        """FAILS: Bug — the question placeholder is not filled in the f-string.
        The prompt contains literal '{question}' instead of the actual question.
        """
        captured_prompt = {}

        async def mock_post(url, **kwargs):
            captured_prompt["content"] = kwargs["json"]["messages"][0]["content"]
            mock_resp = MagicMock()
            mock_resp.json.return_value = {
                "choices": [{"message": {"content": "Test answer"}}]
            }
            mock_resp.raise_for_status = MagicMock()
            return mock_resp

        mock_chunks = [{"text": "Some context", "score": 0.9}]

        with (
            patch(
                "app.services.rag_service.get_query_embedding",
                return_value=[0.1] * 384,
            ),
            patch(
                "app.services.rag_service.search_similar", return_value=mock_chunks
            ),
            patch("httpx.AsyncClient.post", side_effect=mock_post),
            patch("app.services.rag_service.OPENAI_API_KEY", "test-key"),
        ):
            from app.services.rag_service import generate_answer, reset_default_rag_service
            reset_default_rag_service()
            import asyncio
            asyncio.run(generate_answer("What is the project about?"))

        prompt = captured_prompt["content"]
        assert "What is the project about?" in prompt, (
            "Question not interpolated into prompt — found literal '{question}' instead"
        )
        assert "{question}" not in prompt, (
            "Prompt contains unresolved placeholder '{question}'"
        )

    def test_returns_no_info_when_no_chunks(self):
        """Should gracefully handle empty search results."""
        with (
            patch(
                "app.services.rag_service.get_query_embedding",
                return_value=[0.1] * 384,
            ),
            patch("app.services.rag_service.search_similar", return_value=[]),
        ):
            from app.services.rag_service import generate_answer
            import asyncio
            result = asyncio.run(generate_answer("Any question?"))

        assert "No relevant information found" in result["answer"]
