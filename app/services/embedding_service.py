"""Text-embedding service.

Wraps a sentence-transformers model. The model is created lazily and cached in
the composition root (``create_embedder``) instead of being re-instantiated on
every call as before — loading the model per request was both a correctness
risk and a severe performance bug.
"""

import logging

from app.dependencies import create_embedder

logger = logging.getLogger(__name__)


class SentenceTransformerEmbedder:
    """Concrete :class:`EmbeddingModel` backed by sentence-transformers."""

    def embed_documents(self, texts: list) -> list:
        return create_embedder().encode(texts).tolist()

    def embed_query(self, text: str) -> list:
        return create_embedder().encode([text])[0].tolist()


_embedder = SentenceTransformerEmbedder()
get_embeddings = _embedder.embed_documents
get_query_embedding = _embedder.embed_query
