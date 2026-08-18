"""Embedding service (Sentence-Transformers / MiniLM by default).

The model is injected from the composition root and loaded once (lazily). Callers
patch the re-exported names in the consuming modules (vector_service.get_embeddings,
rag_service.get_query_embedding), so those stay module-level aliases.
"""

from __future__ import annotations

from typing import Protocol

from app.dependencies import get_embedding_model


class EmbeddingService(Protocol):
    """Interface for embedding backends so the model can be swapped."""

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        ...

    def embed_query(self, query: str) -> list[float]:
        ...


class SentenceTransformerEmbeddingService:
    """Sentence-Transformers embedder. The model is injected, not loaded here."""

    def __init__(self, model) -> None:
        self._model = model

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        return self._model.encode(texts).tolist()

    def embed_query(self, query: str) -> list[float]:
        return self._model.encode([query])[0].tolist()


# Aliases resolve the cached model lazily at call time (no load at import).
get_embeddings = lambda texts: SentenceTransformerEmbeddingService(  # noqa: E731
    get_embedding_model()
).embed_texts(texts)
get_query_embedding = lambda query: SentenceTransformerEmbeddingService(  # noqa: E731
    get_embedding_model()
).embed_query(query)
