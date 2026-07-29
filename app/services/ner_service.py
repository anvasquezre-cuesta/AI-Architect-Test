"""NER service backed by spaCy.

Bug fixes vs. original implementation:
- Only PERSON entities are returned (ORG/GPE/DATE were polluting results).
- The spaCy pipeline is no longer loaded at import time; it is lazily
  resolved through the composition root (``app.core.providers``) or
  injected via the constructor.
"""

from app.core.logging import get_logger
from app.core.providers import get_spacy_pipeline
from app.services.interfaces import NERService

logger = get_logger(__name__)

# Module-level pipeline handle: lazily populated on first use.
# Kept as a module attribute so tests can patch `app.services.ner_service.nlp`.
nlp = None

PERSON_LABEL = "PERSON"
# spaCy's English models tag people as "PERSON"; Spanish and multilingual
# models (es_core_news_sm, xx_ent_wiki_sm) use "PER". Accept both.
PERSON_LABELS = frozenset({"PERSON", "PER"})


class SpacyNERService(NERService):
    """Concrete NER implementation using a spaCy pipeline."""

    def __init__(self, pipeline=None, model_name: str | None = None) -> None:
        self._pipeline = pipeline
        self._model_name = model_name

    @property
    def pipeline(self):
        """Return the injected pipeline, or lazily resolve one by name."""
        if self._pipeline is not None:
            return self._pipeline
        if self._model_name is not None:
            # Profile-specific pipeline: cached per model name in providers,
            # without touching the shared default (test seam below).
            self._pipeline = get_spacy_pipeline(self._model_name)
            return self._pipeline
        global nlp
        if nlp is None:
            nlp = get_spacy_pipeline()
        return nlp

    def extract_names(self, text: str) -> list[str]:
        """Extract person names from text (PERSON entities only)."""
        doc = self.pipeline(text)
        names = [ent.text for ent in doc.ents if ent.label_ in PERSON_LABELS]
        logger.debug("NER found %d person name(s)", len(names))
        return names

    def extract_names_with_positions(self, text: str) -> list[dict]:
        """Extract person names with their character positions in text."""
        doc = self.pipeline(text)
        return [
            {
                "name": ent.text,
                "start_char": ent.start_char,
                "end_char": ent.end_char,
                "label": ent.label_,
            }
            for ent in doc.ents
            if ent.label_ in PERSON_LABELS
        ]


class EnsembleNERService(NERService):
    """Union of multiple NER engines.

    Statistical models have different blind spots (es_core_news_sm tags
    'La señora Lucía Fernández' as MISC; xx_ent_wiki_sm finds her but adds
    other noise). Running several engines and taking the union improves
    recall; the extra false positives are tolerable because the downstream
    fuzzy-matching threshold (90%) filters what actually gets matched.
    """

    def __init__(self, services: list[NERService]) -> None:
        if not services:
            raise ValueError("EnsembleNERService requires at least one engine")
        self._services = list(services)

    def extract_names(self, text: str) -> list[str]:
        """Union of every engine's names (order-preserving, deduped)."""
        seen: set[str] = set()
        names: list[str] = []
        for service in self._services:
            for name in service.extract_names(text):
                key = name.strip().lower()
                if key and key not in seen:
                    seen.add(key)
                    names.append(name)
        return names

    def extract_names_with_positions(self, text: str) -> list[dict]:
        """Union of every engine's positioned names (deduped by name+offset)."""
        seen: set[tuple] = set()
        results: list[dict] = []
        for service in self._services:
            for item in service.extract_names_with_positions(text):
                key = (item["name"].strip().lower(), item["start_char"])
                if key not in seen:
                    seen.add(key)
                    results.append(item)
        return results


# Default instance + module-level callables (stable public API / test seam).
_default_service = SpacyNERService()
extract_names = _default_service.extract_names
extract_names_with_positions = _default_service.extract_names_with_positions
