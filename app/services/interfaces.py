"""Service interfaces (structural typing via ``typing.Protocol``).

Defining these abstractions lets the rest of the codebase depend on
capabilities rather than concrete implementations, so an OCR engine, NER
model, embedding model, vector store or LLM client can be swapped (Tesseract →
cloud OCR, Qdrant → Chroma, OpenAI → a local model) without touching callers.
"""

from typing import List, Protocol, runtime_checkable


@runtime_checkable
class OCREngine(Protocol):
    """Extracts text and word-level bounding boxes from a PDF."""

    def extract_text(self, pdf_path: str) -> str: ...

    def word_boxes(self, pdf_path: str) -> List[dict]: ...


@runtime_checkable
class NERModel(Protocol):
    """Extracts person names from free text."""

    def names(self, text: str) -> List[str]: ...

    def names_with_positions(self, text: str) -> List[dict]: ...


@runtime_checkable
class EmbeddingModel(Protocol):
    """Turns text into dense vector representations."""

    def embed_documents(self, texts: List[str]) -> List[List[float]]: ...

    def embed_query(self, text: str) -> List[float]: ...


@runtime_checkable
class VectorStore(Protocol):
    """Persists and searches document-chunk embeddings."""

    def init_collection(self) -> None: ...

    def store(self, chunks: List[str], metadata: List[dict] = None) -> None: ...

    def search(self, query_embedding: List[float], top_k: int = None) -> List[dict]: ...


@runtime_checkable
class LLMClient(Protocol):
    """Generates a completion for a prompt."""

    def complete(self, prompt: str) -> str: ...
