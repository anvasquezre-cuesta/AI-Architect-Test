import os
import uuid
from typing import Protocol, List, Optional, Dict, Any
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from loguru import logger
from dotenv import load_dotenv

load_dotenv()

from app.services.embedding_service import get_embeddings

class VectorStore(Protocol):
    """Protocol defining the interface for a vector store."""
    def init_collection(self) -> None:
        ...

    def store_document_chunks(self, chunks: List[str], metadata: Optional[List[Dict[str, Any]]] = None) -> None:
        ...

    def search_similar(self, query_embedding: List[float], top_k: int = 5) -> List[Dict[str, Any]]:
        ...


class QdrantVectorStoreService:
    """
    Service for interacting with Qdrant vector database.
    Implements production best practices including configuration via environment variables,
    proper error handling, and logging.
    """
    def __init__(
        self,
        host: Optional[str] = None,
        port: Optional[int] = None,
        client: Optional[QdrantClient] = None
    ):
        """
        Initialize the Qdrant client.

        Args:
            host: Qdrant host. Defaults to QDRANT_HOST env var or 'localhost'.
            port: Qdrant port. Defaults to QDRANT_PORT env var or 6333.
            client: Optional pre-configured QdrantClient instance.
        """
        if client:
            self.client = client
        else:
            host = host or os.getenv("QDRANT_HOST", "localhost")
            port = port or int(os.getenv("QDRANT_PORT", "6333"))
            api_key = os.getenv("QDRANT_API_KEY")
            self.client = QdrantClient(host=host, port=port, api_key=api_key, check_compatibility=False)
        
        self.collection_name = os.getenv("QDRANT_COLLECTION", "pdf_documents")
        self.vector_size = int(os.getenv("QDRANT_VECTOR_SIZE", "384"))

    def init_collection(self) -> None:
        """Create the vector collection if it doesn't exist."""
        logger.debug(f"Ensuring collection exists: {self.collection_name}")
        
        try:
            collections = self.client.get_collections().collections
            existing = [c.name for c in collections]

            if self.collection_name not in existing:
                logger.info(f"Creating collection: {self.collection_name}")
                self.client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=VectorParams(
                        size=self.vector_size,
                        distance=Distance.COSINE,
                    ),
                )
        except Exception as e:
            logger.exception(f"Failed to initialize collection {self.collection_name}: {e}")
            raise

    def store_document_chunks(
        self, 
        chunks: List[str], 
        metadata: Optional[List[Dict[str, Any]]] = None
    ) -> None:
        """
        Store text chunks with their embeddings in the vector database.

        Args:
            chunks: List of text strings to store.
            metadata: Optional list of dictionaries containing metadata for each chunk.
        """
        if not chunks:
            return

        logger.info(f"Storing {len(chunks)} chunks in collection {self.collection_name}")
        
        try:
            embeddings = get_embeddings(chunks)

            points = []
            for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
                payload = {"text": chunk}
                if metadata and i < len(metadata):
                    payload.update(metadata[i])

                points.append(
                    PointStruct(
                        id=str(uuid.uuid4()),
                        vector=embedding,
                        payload=payload,
                    )
                )

            self.client.upsert(collection_name=self.collection_name, points=points)
            logger.debug(f"Successfully upserted {len(points)} points")

        except Exception as e:
            logger.exception(f"Error storing document chunks: {e}")
            raise

    def search_similar(self, query_embedding: List[float], top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Search for similar text chunks.

        Args:
            query_embedding: The vector to search for.
            top_k: Number of results to return.

        Returns:
            List of dictionaries containing 'text' and 'score'.
        """
        logger.debug(f"Searching for similar vectors in {self.collection_name}, top_k={top_k}")

        try:
            # Use query_points method which is available in newer versions of qdrant-client
            results = self.client.query_points(
                collection_name=self.collection_name,
                query=query_embedding,
                limit=top_k,
                score_threshold=0.5,
            )

            return [
                {
                    "text": hit.payload.get("text", ""),
                    "score": hit.score,
                    "metadata": {k: v for k, v in hit.payload.items() if k != "text"}
                }
                for hit in results.points
            ]

        except Exception as e:
            logger.exception(f"Error searching for similar vectors: {e}")
            return []



# Singleton instance management
_default_store: Optional[QdrantVectorStoreService] = None

def get_default_store() -> QdrantVectorStoreService:
    """Get or create the default singleton instance of the vector store."""
    global _default_store
    if _default_store is None:
        _default_store = QdrantVectorStoreService()
    return _default_store

def reset_default_store() -> None:
    """Reset the default store (mainly for testing)."""
    global _default_store
    _default_store = None

def init_collection() -> None:
    """Initialize the default collection."""
    get_default_store().init_collection()

def store_document_chunks(chunks: List[str], metadata: Optional[List[Dict[str, Any]]] = None) -> None:
    """Store chunks in the default collection."""
    get_default_store().store_document_chunks(chunks, metadata)

def search_similar(query_embedding: List[float], top_k: int = 5) -> List[Dict[str, Any]]:
    """Search for similar vectors in the default collection."""
    return get_default_store().search_similar(query_embedding, top_k)
