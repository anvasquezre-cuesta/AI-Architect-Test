"""Composition root for heavy ML dependencies.

This module deliberately lives *outside* ``app/services`` so the service
modules never construct their own models or clients. Instead they depend on
these cached factory functions (dependency inversion): the concrete spaCy /
sentence-transformers / Qdrant objects are created here, once, and injected.

Each factory is ``lru_cache``-wrapped so the (expensive) model load or client
construction happens at most once per process and can be swapped in tests.
"""

import logging
from functools import lru_cache

from app.config import settings

logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def create_ner_model():
    """Load and cache the spaCy NER pipeline."""
    import spacy

    logger.info("Loading spaCy model '%s'", settings.spacy_model)
    return spacy.load(settings.spacy_model)


@lru_cache(maxsize=1)
def create_embedder():
    """Load and cache the sentence-transformers embedding model (lazily)."""
    from sentence_transformers import SentenceTransformer

    logger.info("Loading embedding model '%s'", settings.embedding_model)
    return SentenceTransformer(settings.embedding_model)


@lru_cache(maxsize=1)
def create_vector_client():
    """Construct and cache the Qdrant client (connection is lazy)."""
    from qdrant_client import QdrantClient

    logger.info(
        "Configuring Qdrant client for %s:%s",
        settings.qdrant_host,
        settings.qdrant_port,
    )
    return QdrantClient(host=settings.qdrant_host, port=settings.qdrant_port)
