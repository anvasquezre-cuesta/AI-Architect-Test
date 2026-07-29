"""Domain-specific exceptions for clearer error handling at the API layer."""


class AppError(Exception):
    """Base class for application errors."""


class InvalidFileError(AppError):
    """Uploaded file is not a valid PDF (wrong type, empty, or corrupt)."""


class InvalidRequestError(AppError):
    """Request payload is malformed (e.g., bad names JSON)."""


class OCRError(AppError):
    """OCR pipeline failed to process the document."""


class VectorStoreError(AppError):
    """Vector database operation failed."""


class LLMError(AppError):
    """LLM completion call failed after retries."""
