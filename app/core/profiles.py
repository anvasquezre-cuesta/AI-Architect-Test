"""Per-profile service bundles.

API clients select a friendly profile key ("english_fast",
"spanish_precision", "multilingual") — similar to picking a "fast" vs
"thinking" tier on an LLM API — and this module maps it to a concrete,
cached bundle of services. Internal model names never leak into the API.

Notes:
- Bundles are cheap to build: every underlying model loads lazily on
  first use, so listing or validating profiles never downloads anything.
- Each profile gets its own Qdrant collection (suffix per profile):
  embeddings from different models live in incompatible vector spaces.
- The default profile is served by the existing module-level services so
  behavior (and the test seams that patch them) is unchanged.
"""

from dataclasses import dataclass
from functools import lru_cache

from app.core.config import MODEL_PROFILES, ModelProfile, get_settings


class UnknownProfileError(ValueError):
    """Raised when a request asks for a profile that is not registered."""

    def __init__(self, key: str) -> None:
        valid = ", ".join(sorted(MODEL_PROFILES))
        super().__init__(f"Unknown profile '{key}'. Valid profiles: {valid}.")


def resolve_profile(key: str | None) -> ModelProfile:
    """Return the profile for `key`, or the default profile when None."""
    if key is None or key == "":
        key = get_settings().default_profile
    try:
        return MODEL_PROFILES[key]
    except KeyError as exc:
        raise UnknownProfileError(key) from exc


def list_profiles() -> list[dict]:
    """Public, client-facing view of the available profiles."""
    default = get_settings().default_profile
    return [
        {
            "key": p.key,
            "description": p.description,
            "default": p.key == default,
        }
        for p in MODEL_PROFILES.values()
    ]


@dataclass(frozen=True)
class ServiceBundle:
    """The services that make up one profile's pipeline."""

    profile: ModelProfile
    ocr: "object"
    ner: "object"
    bbox: "object"
    embedder: "object"
    store: "object"
    rag: "object"


@lru_cache
def get_service_bundle(profile_key: str) -> ServiceBundle:
    """Build (once per profile) the service bundle for a profile key.

    Imports are local to avoid import cycles (services import config).
    """
    from app.services.bbox_service import NameBoundingBoxService
    from app.services.embedding_service import SentenceTransformerEmbeddingService
    from app.services.llm_ner_service import LLMNERService
    from app.services.ner_service import EnsembleNERService, SpacyNERService
    from app.services.ocr_service import TesseractOCRService
    from app.services.rag_service import RAGService
    from app.services.vector_service import QdrantVectorStore

    profile = resolve_profile(profile_key)
    settings = get_settings()

    ocr = TesseractOCRService(lang=profile.ocr_lang)
    if profile.ner_engine == "llm":
        ner = LLMNERService()
    elif profile.ner_engine == "spacy_ensemble":
        ner = EnsembleNERService(
            [
                SpacyNERService(model_name=name)
                for name in [profile.spacy_model, *profile.extra_spacy_models]
            ]
        )
    else:
        ner = SpacyNERService(model_name=profile.spacy_model)
    bbox = NameBoundingBoxService(ner_service=ner, ocr_service=ocr)
    embedder = SentenceTransformerEmbeddingService(
        model_name=profile.embedding_model_name
    )
    store = QdrantVectorStore(
        embedding_service=embedder,
        collection_name=settings.qdrant_collection + profile.collection_suffix,
        vector_size=profile.embedding_vector_size,
    )
    rag = RAGService(embedding_service=embedder, vector_store=store)

    return ServiceBundle(
        profile=profile,
        ocr=ocr,
        ner=ner,
        bbox=bbox,
        embedder=embedder,
        store=store,
        rag=rag,
    )
