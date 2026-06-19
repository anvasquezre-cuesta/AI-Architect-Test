"""Centralised application configuration.

Every value that used to be hard-coded across the services (Qdrant host/port,
model names, the fuzzy-match threshold, OCR DPI, chunk size, OpenAI settings)
lives here and is overridable via environment variables or a local ``.env``
file. This keeps the services free of environment-specific constants and makes
the application 12-factor friendly.
"""

import logging

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Strongly-typed settings sourced from the environment / ``.env``."""

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # --- Vector database (Qdrant) ---
    qdrant_host: str = "localhost"
    qdrant_port: int = 6333
    qdrant_collection: str = "pdf_documents"
    vector_size: int = 384
    search_top_k: int = 3
    search_score_threshold: float = 0.5

    # --- Models ---
    spacy_model: str = "en_core_web_sm"
    embedding_model: str = "all-MiniLM-L6-v2"

    # --- OCR ---
    ocr_dpi: int = 150

    # --- Fuzzy matching ---
    fuzzy_threshold: int = 90

    # --- RAG chunking ---
    chunk_size: int = 500

    # --- LLM (OpenAI-compatible) ---
    openai_api_key: str = ""
    openai_base_url: str = "https://api.openai.com/v1"
    openai_model: str = "gpt-3.5-turbo"
    llm_timeout: float = 30.0

    # --- Observability ---
    log_level: str = "INFO"


settings = Settings()


def configure_logging() -> None:
    """Configure structured root logging once, at application startup."""
    logging.basicConfig(
        level=settings.log_level.upper(),
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    )
