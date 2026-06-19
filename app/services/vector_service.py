"""Vector-store service (Qdrant).

Stores document-chunk embeddings and runs similarity search. The Qdrant client
and collection settings are injected/configured rather than hard-coded, and
point IDs are globally unique (UUIDs) so ingesting a second document no longer
overwrites the first (the previous ``id=i`` reset to 0 on every call).
"""

import logging
import uuid

from qdrant_client.models import Distance, PointStruct, VectorParams

from app.config import settings
from app.dependencies import create_vector_client
from app.services.embedding_service import get_embeddings

logger = logging.getLogger(__name__)

# Injected Qdrant client (constructed once in the composition root).
client = create_vector_client()
COLLECTION_NAME = settings.qdrant_collection
VECTOR_SIZE = settings.vector_size


class QdrantVectorStore:
    """Concrete :class:`VectorStore` backed by Qdrant."""

    def init_collection(self) -> None:
        existing = [c.name for c in client.get_collections().collections]
        if COLLECTION_NAME not in existing:
            logger.info("Creating Qdrant collection '%s'", COLLECTION_NAME)
            client.create_collection(
                collection_name=COLLECTION_NAME,
                vectors_config=VectorParams(
                    size=VECTOR_SIZE, distance=Distance.COSINE
                ),
            )

    def store(self, chunks: list, metadata: list = None) -> None:
        embeddings = get_embeddings(chunks)
        points = []
        for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
            payload = {"text": chunk}
            if metadata and i < len(metadata):
                payload.update(metadata[i])
            points.append(
                PointStruct(
                    id=str(uuid.uuid4()),  # unique across documents/ingests
                    vector=embedding,
                    payload=payload,
                )
            )
        client.upsert(collection_name=COLLECTION_NAME, points=points)

    def search(self, query_embedding: list, top_k: int = None) -> list:
        limit = top_k if top_k is not None else settings.search_top_k
        results = client.search(
            collection_name=COLLECTION_NAME,
            query_vector=query_embedding,
            limit=limit,
            score_threshold=settings.search_score_threshold,
        )
        return [{"text": hit.payload["text"], "score": hit.score} for hit in results]


_store = QdrantVectorStore()
init_collection = _store.init_collection
store_document_chunks = _store.store
search_similar = _store.search
