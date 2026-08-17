"""Centralized, environment-driven configuration.

Replaces values previously hardcoded across services (Qdrant host/port, collection
name, embedding model, OpenAI model/key). Every setting can be overridden via an
environment variable, which is what lets the same image run locally and in Docker
Compose (where Qdrant is reachable as host ``qdrant`` instead of ``localhost``).
"""

from __future__ import annotations

import os

# Embedding / vector store
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
EMBEDDING_DIM = int(os.getenv("EMBEDDING_DIM", "384"))
QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", "6333"))
QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION", "pdf_documents")

# NER
SPACY_MODEL = os.getenv("SPACY_MODEL", "en_core_web_sm")

# RAG / LLM
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-3.5-turbo")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
