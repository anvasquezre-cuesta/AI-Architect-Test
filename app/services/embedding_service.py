from typing import Protocol, List, Optional, Union
import torch
from sentence_transformers import SentenceTransformer
from loguru import logger
import os
from dotenv import load_dotenv

load_dotenv()

class EmbeddingService(Protocol):
    def get_embeddings(self, texts: List[str]) -> List[List[float]]:
        ...

    def get_query_embedding(self, query: str) -> List[float]:
        ...

class SentenceTransformerEmbedding:
    """
    Service for generating vector embeddings using SentenceTransformers.
    Includes production best practices like error handling, device management,
    and input validation.
    """
    def __init__(
        self, 
        model_name: Optional[str] = None, 
        device: Optional[str] = None,
        batch_size: Optional[int] = None
    ):
        self.model_name = model_name or os.getenv("EMBEDDING_MODEL_NAME", "all-MiniLM-L6-v2")
        self.batch_size = batch_size or int(os.getenv("EMBEDDING_BATCH_SIZE", "32"))
        self._model = None
        
        # Auto-detect device if not provided
        if device is None:
            device = os.getenv("EMBEDDING_DEVICE")
        
        if device is None:
            if torch.cuda.is_available():
                self.device = "cuda"
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                self.device = "mps"
            else:
                self.device = "cpu"
        else:
            self.device = device
            
        logger.info(f"SentenceTransformerEmbedding initialized with model={model_name}, device={self.device}")

    @property
    def model(self) -> SentenceTransformer:
        """Lazy-load the model with error handling."""
        if self._model is None:
            try:
                logger.info(f"Loading SentenceTransformer model: {self.model_name} on {self.device}")
                self._model = SentenceTransformer(self.model_name, device=self.device)
            except Exception as e:
                logger.error(f"Failed to load model {self.model_name}: {e}")
                raise RuntimeError(f"Could not initialize embedding model: {e}")
        return self._model

    def get_embeddings(self, texts: List[str]) -> List[List[float]]:
        """
        Convert a list of texts to vector embeddings.
        
        Args:
            texts: List of strings to embed.
            
        Returns:
            List of embedding vectors (lists of floats).
        """
        if not texts:
            return []
            
        if not isinstance(texts, list) or not all(isinstance(t, str) for t in texts):
            raise ValueError("Input 'texts' must be a list of strings.")

        try:
            logger.debug(f"Generating embeddings for {len(texts)} texts using {self.model_name}")
            embeddings = self.model.encode(
                texts, 
                batch_size=self.batch_size,
                show_progress_bar=False,
                convert_to_numpy=True
            )
            return embeddings.tolist()
        except Exception as e:
            logger.error(f"Error generating embeddings: {e}")
            raise

    def get_query_embedding(self, query: str) -> List[float]:
        """
        Convert a single query string to a vector embedding.
        
        Args:
            query: The query string to embed.
            
        Returns:
            An embedding vector (list of floats).
        """
        if not query or not isinstance(query, str):
            raise ValueError("Input 'query' must be a non-empty string.")

        try:
            # Re-use get_embeddings for consistency
            result = self.get_embeddings([query])
            return result[0]
        except Exception as e:
            logger.error(f"Error generating query embedding: {e}")
            raise

# Singleton instance management
_default_embedding_service: Optional[SentenceTransformerEmbedding] = None

def get_default_embedding_service() -> SentenceTransformerEmbedding:
    """Get or create the default singleton instance of the embedding service."""
    global _default_embedding_service
    if _default_embedding_service is None:
        _default_embedding_service = SentenceTransformerEmbedding()
    return _default_embedding_service

def get_embeddings(texts: List[str]) -> List[List[float]]:
    """Convenience wrapper for get_embeddings."""
    return get_default_embedding_service().get_embeddings(texts)

def get_query_embedding(query: str) -> List[float]:
    """Convenience wrapper for get_query_embedding."""
    return get_default_embedding_service().get_query_embedding(query)
