"""Composition root: lazy, cached construction of heavy dependencies.

Models and clients are instantiated HERE (not inside services) so that:
- services stay decoupled and receive dependencies via injection,
- expensive objects are created once and reused,
- alternative implementations can be swapped in tests or deployments.
"""

from functools import lru_cache
from typing import TYPE_CHECKING, Any

from app.core.config import get_settings
from app.core.logging import get_logger

if TYPE_CHECKING:  # pragma: no cover - typing only
    from qdrant_client import QdrantClient
    from sentence_transformers import SentenceTransformer

logger = get_logger(__name__)


@lru_cache
def get_spacy_pipeline(model_name: str | None = None) -> Any:
    """Load (once) and return the spaCy NLP pipeline."""
    import spacy

    name = model_name or get_settings().spacy_model
    logger.info("Loading spaCy model '%s'", name)
    return spacy.load(name)


@lru_cache
def get_sentence_transformer(model_name: str | None = None) -> "SentenceTransformer":
    """Load (once) and return the sentence-transformers embedding model."""
    from sentence_transformers import SentenceTransformer

    name = model_name or get_settings().embedding_model_name
    logger.info("Loading embedding model '%s'", name)
    return SentenceTransformer(name)


@lru_cache
def get_qdrant_client() -> "QdrantClient":
    """Create (once) and return the Qdrant client."""
    from qdrant_client import QdrantClient

    settings = get_settings()
    logger.info(
        "Connecting to Qdrant at %s:%d", settings.qdrant_host, settings.qdrant_port
    )
    return QdrantClient(
        host=settings.qdrant_host,
        port=settings.qdrant_port,
        timeout=settings.qdrant_timeout_seconds,
    )
