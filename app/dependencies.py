"""Composition root: construct and cache the heavy, injectable dependencies.

Deliberately kept OUTSIDE ``app/services/``. The architecture tests audit only the
services package and forbid constructing models/clients (``spacy.load``,
``QdrantClient``, ``SentenceTransformer``) inside service modules or functions.
Centralizing construction here keeps services dependency-free and swappable, and
``lru_cache`` guarantees each heavy object is built once per process (load-once),
lazily on first use.
"""

from __future__ import annotations

from functools import lru_cache

from app import config


@lru_cache(maxsize=1)
def get_spacy_nlp():
    """Load the spaCy pipeline once."""
    import spacy

    return spacy.load(config.SPACY_MODEL)


@lru_cache(maxsize=1)
def get_embedding_model():
    """Load the Sentence-Transformers embedding model once."""
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(config.EMBEDDING_MODEL_NAME)


@lru_cache(maxsize=1)
def get_qdrant_client():
    """Construct the Qdrant client once (connection is lazy)."""
    from qdrant_client import QdrantClient

    return QdrantClient(host=config.QDRANT_HOST, port=config.QDRANT_PORT)
