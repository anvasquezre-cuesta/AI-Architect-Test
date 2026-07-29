"""Abstract interfaces (ports) for all services.

Defining these contracts decouples the pipeline from concrete engines:
the OCR engine, NER model, embedding model, vector database, and LLM can
each be swapped (or mocked in tests) without touching consuming code.
"""

from abc import ABC, abstractmethod


class OCRService(ABC):
    """Extracts text and word-level bounding boxes from PDF documents."""

    @abstractmethod
    def extract_text_from_pdf(self, pdf_path: str) -> str:
        """Return the full text of the document (all pages)."""

    @abstractmethod
    def get_word_bounding_boxes(self, pdf_path: str) -> list[dict]:
        """Return one entry per recognized word with PDF-space coordinates."""


class NERService(ABC):
    """Identifies person names in free text."""

    @abstractmethod
    def extract_names(self, text: str) -> list[str]:
        """Return the person names found in ``text``."""

    @abstractmethod
    def extract_names_with_positions(self, text: str) -> list[dict]:
        """Return person names with character offsets and labels."""


class BoundingBoxService(ABC):
    """Maps extracted names to their locations in the document."""

    @abstractmethod
    def find_name_bounding_boxes(self, pdf_path: str, text: str) -> list[dict]:
        """Return merged bounding boxes for every occurrence of each name."""


class FuzzyMatcher(ABC):
    """Fuzzy-matches extracted names against user-provided name pairs."""

    @abstractmethod
    def fuzzy_match_names(
        self, extracted_names: list[str], query_names: list[dict]
    ) -> list[dict]:
        """Return matches whose similarity meets the configured threshold."""


class EmbeddingService(ABC):
    """Converts text into dense vector representations."""

    @abstractmethod
    def get_embeddings(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of texts."""

    @abstractmethod
    def get_query_embedding(self, query: str) -> list[float]:
        """Embed a single query string."""


class VectorStore(ABC):
    """Stores and searches embedded document chunks."""

    @abstractmethod
    def init_collection(self) -> None:
        """Create the collection if it does not exist."""

    @abstractmethod
    def store_document_chunks(
        self, chunks: list[str], metadata: list[dict] | None = None
    ) -> None:
        """Persist chunks with their embeddings."""

    @abstractmethod
    def search_similar(
        self, query_embedding: list[float], top_k: int = 5
    ) -> list[dict]:
        """Return the most similar stored chunks."""


class AnswerGenerator(ABC):
    """Generates answers from a question using retrieval-augmented context."""

    @abstractmethod
    def chunk_text(self, text: str, chunk_size: int | None = None) -> list[str]:
        """Split text into word-aware chunks."""

    @abstractmethod
    def generate_answer(self, question: str) -> dict:
        """Answer the question grounded on retrieved chunks."""
