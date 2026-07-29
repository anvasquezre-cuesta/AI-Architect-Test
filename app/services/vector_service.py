"""Vector store service backed by Qdrant.

Bug fixes vs. original implementation:
- Point IDs are now UUIDs: the original used enumerate() indexes starting
  at 0, so every new document overwrote the previous document's chunks.
- The client is no longer created at import time; it is lazily resolved
  through the composition root or injected via the constructor.
- Connection parameters and collection settings come from configuration.
"""

import uuid

from qdrant_client.models import Distance, PointStruct, VectorParams

from app.core.config import get_settings
from app.core.exceptions import VectorStoreError
from app.core.logging import get_logger
from app.core.providers import get_qdrant_client
from app.services.embedding_service import get_embeddings
from app.services.interfaces import VectorStore

logger = get_logger(__name__)

# Module-level client handle: lazily populated on first use.
# Kept as a module attribute so tests can patch `app.services.vector_service.client`.
client = None

COLLECTION_NAME = get_settings().qdrant_collection
VECTOR_SIZE = get_settings().embedding_vector_size


class QdrantVectorStore(VectorStore):
    """Concrete vector store implementation using Qdrant."""

    def __init__(
        self,
        qdrant_client=None,
        embedding_service=None,
        collection_name: str | None = None,
        vector_size: int | None = None,
        score_threshold: float | None = None,
    ) -> None:
        settings = get_settings()
        self._client = qdrant_client
        self._embedder = embedding_service
        self.collection_name = collection_name or settings.qdrant_collection
        self.vector_size = vector_size or settings.embedding_vector_size
        self.score_threshold = (
            score_threshold
            if score_threshold is not None
            else settings.qdrant_score_threshold
        )

    @property
    def qdrant(self):
        """Return the injected client, or lazily resolve the shared default."""
        if self._client is not None:
            return self._client
        global client
        if client is None:
            client = get_qdrant_client()
        return client

    def _embed(self, chunks: list[str]) -> list[list[float]]:
        if self._embedder is not None:
            return self._embedder.get_embeddings(chunks)
        # Module-level lookup at call time keeps this seam patchable in tests.
        return get_embeddings(chunks)

    def init_collection(self) -> None:
        """Create the vector collection if it doesn't exist."""
        try:
            collections = self.qdrant.get_collections().collections
            existing = [c.name for c in collections]

            if self.collection_name not in existing:
                logger.info("Creating Qdrant collection '%s'", self.collection_name)
                self.qdrant.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=VectorParams(
                        size=self.vector_size,
                        distance=Distance.COSINE,
                    ),
                )
        except VectorStoreError:
            raise
        except Exception as exc:
            raise VectorStoreError(f"Failed to initialize collection: {exc}") from exc

    def store_document_chunks(
        self, chunks: list[str], metadata: list[dict] | None = None
    ) -> None:
        """Store text chunks with their embeddings in the vector database."""
        if not chunks:
            return

        embeddings = self._embed(chunks)

        points = []
        for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
            payload = {"text": chunk}
            if metadata and i < len(metadata):
                payload.update(metadata[i])

            points.append(
                PointStruct(
                    # UUIDs guarantee globally-unique IDs across documents
                    # (sequential ints overwrote previously stored chunks).
                    id=str(uuid.uuid4()),
                    vector=embedding,
                    payload=payload,
                )
            )

        try:
            self.qdrant.upsert(collection_name=self.collection_name, points=points)
        except Exception as exc:
            raise VectorStoreError(f"Failed to store chunks: {exc}") from exc
        logger.info("Stored %d chunk(s) in '%s'", len(points), self.collection_name)

    def search_similar(
        self, query_embedding: list[float], top_k: int = 5
    ) -> list[dict]:
        """Search for similar text chunks."""
        try:
            results = self.qdrant.search(
                collection_name=self.collection_name,
                query_vector=query_embedding,
                limit=top_k,
                score_threshold=self.score_threshold,
            )
        except Exception as exc:
            raise VectorStoreError(f"Similarity search failed: {exc}") from exc

        return [
            {
                "text": (hit.payload or {}).get("text", ""),
                "score": hit.score,
            }
            for hit in results
        ]


# Default instance + module-level callables (stable public API / test seam).
_default_store = QdrantVectorStore()
init_collection = _default_store.init_collection
store_document_chunks = _default_store.store_document_chunks
search_similar = _default_store.search_similar
