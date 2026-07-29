"""Text embedding service backed by sentence-transformers.

Improvements vs. original implementation:
- The model was reloaded on EVERY call (very expensive). It is now
  lazily loaded once through the composition root and reused.
- The model can be injected, enabling tests and model swaps.
"""

from app.core.logging import get_logger
from app.core.providers import get_sentence_transformer
from app.services.interfaces import EmbeddingService

logger = get_logger(__name__)


class SentenceTransformerEmbeddingService(EmbeddingService):
    """Concrete embedding implementation using sentence-transformers."""

    def __init__(self, model=None, model_name: str | None = None) -> None:
        self._model = model
        self._model_name = model_name

    @property
    def model(self):
        """Return the injected model, or lazily resolve one by name.

        With no name given, the shared default model from configuration is
        used; profiles pass their own name (cached per name in providers).
        """
        if self._model is None:
            self._model = get_sentence_transformer(self._model_name)
        return self._model

    def get_embeddings(self, texts: list[str]) -> list[list[float]]:
        """Convert a batch of texts to vector embeddings."""
        embeddings = self.model.encode(list(texts), show_progress_bar=False)
        return embeddings.tolist()

    def get_query_embedding(self, query: str) -> list[float]:
        """Convert a single query to a vector embedding."""
        embedding = self.model.encode([query], show_progress_bar=False)
        return embedding[0].tolist()


# Default instance + module-level callables (stable public API / test seam).
_default_service = SentenceTransformerEmbeddingService()
get_embeddings = _default_service.get_embeddings
get_query_embedding = _default_service.get_query_embedding
