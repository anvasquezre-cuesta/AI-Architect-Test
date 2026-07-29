"""Centralized application configuration.

All tunables live here and can be overridden via environment variables
(or a local ``.env`` file), removing hardcoded values from the services.
"""

from functools import lru_cache

from pydantic import BaseModel
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings, overridable via environment variables."""

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # API
    app_name: str = "PDF Name Extractor & RAG API"
    log_level: str = "INFO"
    max_upload_bytes: int = 20 * 1024 * 1024  # 20 MB

    # OCR
    ocr_dpi: int = 150
    pdf_point_dpi: float = 72.0  # PDF coordinate space is 72 points/inch
    ocr_min_confidence: float = 0.0
    ocr_lang: str = "eng"  # Tesseract language(s), e.g. "eng", "spa", "eng+spa"

    # NER
    spacy_model: str = "en_core_web_sm"

    # Model profile selected when a request doesn't specify one.
    default_profile: str = "english_fast"

    # Fuzzy matching
    similarity_threshold: int = 90

    # Embeddings
    embedding_model_name: str = "all-MiniLM-L6-v2"
    embedding_vector_size: int = 384

    # Vector store (Qdrant)
    qdrant_host: str = "localhost"
    qdrant_port: int = 6333
    qdrant_collection: str = "pdf_documents"
    qdrant_score_threshold: float = 0.5
    qdrant_timeout_seconds: float = 10.0

    # RAG / LLM
    openai_api_key: str | None = None
    openai_base_url: str = "https://api.openai.com/v1"
    llm_model: str = "gpt-4o-mini"
    llm_timeout_seconds: float = 30.0
    llm_max_retries: int = 3
    rag_top_k: int = 3
    rag_chunk_size: int = 500


@lru_cache
def get_settings() -> Settings:
    """Return a cached Settings instance (one read of the environment)."""
    return Settings()


class ModelProfile(BaseModel):
    """A named bundle of model choices exposed to API clients.

    Clients pick a friendly profile key (like choosing "fast" vs "thinking"
    tiers on an LLM API) instead of internal model names — the mapping to
    concrete models stays a server-side implementation detail.
    """

    key: str
    description: str
    ocr_lang: str
    spacy_model: str
    embedding_model_name: str
    embedding_vector_size: int
    # Each profile stores vectors in a collection tied to its EMBEDDING
    # MODEL (not the profile itself): embeddings from different models
    # live in different vector spaces and must never be mixed, while
    # profiles sharing a model can safely share a collection.
    collection_suffix: str
    # NER engine: "spacy" (one statistical model), "spacy_ensemble"
    # (union of spacy_model + extra_spacy_models — different models have
    # different blind spots), or "llm" (zero-shot via OpenAI-compatible
    # API — requires OPENAI_API_KEY).
    ner_engine: str = "spacy"
    extra_spacy_models: list[str] = []


MODEL_PROFILES: dict[str, ModelProfile] = {
    "english_fast": ModelProfile(
        key="english_fast",
        description=(
            "Default. Fastest pipeline, optimized for English documents "
            "(Tesseract eng, spaCy en_core_web_sm, MiniLM embeddings)."
        ),
        ocr_lang="eng",
        spacy_model="en_core_web_sm",
        embedding_model_name="all-MiniLM-L6-v2",
        embedding_vector_size=384,
        collection_suffix="",
    ),
    "spanish_precision": ModelProfile(
        key="spanish_precision",
        description=(
            "Best free/offline quality for Spanish documents: Tesseract "
            "spa+eng and an ENSEMBLE of the LARGE Spanish NER model + the "
            "multilingual model (each covers the other's blind spots), "
            "multilingual embeddings. Slightly slower than a single model."
        ),
        ocr_lang="spa+eng",
        # Decision trail (see TASK.md): es sm and md both mislabel
        # 'La señora Lucía Fernández' as MISC; lg tags her PER correctly.
        spacy_model="es_core_news_lg",
        ner_engine="spacy_ensemble",
        extra_spacy_models=["xx_ent_wiki_sm"],
        embedding_model_name="paraphrase-multilingual-MiniLM-L12-v2",
        embedding_vector_size=384,
        collection_suffix="_es",
    ),
    "multilingual": ModelProfile(
        key="multilingual",
        description=(
            "Mixed/unknown languages (Tesseract eng+spa, spaCy multilingual "
            "NER, multilingual embeddings). Lower NER precision than the "
            "language-specific profiles."
        ),
        ocr_lang="eng+spa",
        spacy_model="xx_ent_wiki_sm",
        embedding_model_name="paraphrase-multilingual-MiniLM-L12-v2",
        embedding_vector_size=384,
        collection_suffix="_multi",
    ),
    "llm_precision": ModelProfile(
        key="llm_precision",
        description=(
            "Highest NER accuracy, any language: zero-shot LLM extraction "
            "(handles honorifics and noisy OCR that statistical models "
            "mislabel). Requires OPENAI_API_KEY or a local OpenAI-compatible "
            "server; slower (~1-3 s/doc) and paid per token."
        ),
        ocr_lang="eng+spa",
        spacy_model="xx_ent_wiki_sm",  # unused (ner_engine="llm"); kept valid
        ner_engine="llm",
        embedding_model_name="paraphrase-multilingual-MiniLM-L12-v2",
        embedding_vector_size=384,
        # Same embedding model as "multilingual" -> same vector space ->
        # intentionally SHARES that collection.
        collection_suffix="_multi",
    ),
}
