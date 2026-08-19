"""Vector store service (Qdrant).

Wraps Qdrant behind a ``VectorStore`` interface so the backend is swappable. The
client and embedder are injected; construction of the real client happens in the
composition root.
"""

from __future__ import annotations

import uuid
from typing import Protocol

from qdrant_client.models import Distance, PointStruct, VectorParams

from app import config
from app.dependencies import get_qdrant_client
from app.services.embedding_service import get_embeddings

COLLECTION_NAME = config.QDRANT_COLLECTION
VECTOR_SIZE = config.EMBEDDING_DIM


class VectorStore(Protocol):
    """Interface for a vector store (Qdrant, Chroma, ...)."""

    def init_collection(self) -> None:
        ...

    def store_document_chunks(self, chunks: list[str], metadata=None) -> None:
        ...

    def search_similar(self, query_embedding: list[float], top_k: int = 5) -> list[dict]:
        ...


class QdrantVectorStore:
    """Qdrant-backed vector store. Client and embedder are injected."""

    def __init__(self, client, embedder) -> None:
        self._client = client
        self._embed = embedder

    def init_collection(self) -> None:
        existing = [c.name for c in self._client.get_collections().collections]
        if COLLECTION_NAME not in existing:
            self._client.create_collection(
                collection_name=COLLECTION_NAME,
                vectors_config=VectorParams(
                    size=VECTOR_SIZE, distance=Distance.COSINE
                ),
            )

    def store_document_chunks(self, chunks: list[str], metadata=None) -> None:
        embeddings = self._embed(chunks)
        points = []
        for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
            payload = {"text": chunk}
            if metadata and i < len(metadata):
                payload.update(metadata[i])
            # UUID id, not a per-call index, so a second document does not
            # overwrite the first (previous bug used id=i starting at 0).
            points.append(
                PointStruct(id=str(uuid.uuid4()), vector=embedding, payload=payload)
            )
        self._client.upsert(collection_name=COLLECTION_NAME, points=points)

    def search_similar(
        self, query_embedding: list[float], top_k: int = 5
    ) -> list[dict]:
        results = self._client.search(
            collection_name=COLLECTION_NAME,
            query_vector=query_embedding,
            limit=top_k,
            score_threshold=config.RETRIEVAL_SCORE_THRESHOLD,
        )
        return [
            {"text": hit.payload["text"], "score": hit.score} for hit in results
        ]


# Module-level patchable client + aliases that inject the current module globals
# (`client`, `get_embeddings`) at call time so tests' patches take effect.
client = get_qdrant_client()
init_collection = lambda: QdrantVectorStore(  # noqa: E731
    client, get_embeddings
).init_collection()
store_document_chunks = lambda chunks, metadata=None: QdrantVectorStore(  # noqa: E731
    client, get_embeddings
).store_document_chunks(chunks, metadata)
search_similar = lambda query_embedding, top_k=5: QdrantVectorStore(  # noqa: E731
    client, get_embeddings
).search_similar(query_embedding, top_k)
