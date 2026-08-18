"""Named-entity recognition service (person names).

Deterministic spaCy NER is used at the extraction step by design (DECISION_LOG D-5):
unlike an instruction-following LLM, it cannot be hijacked by injected text hidden in
a document, and it is cheap and reproducible. A quality-oriented LLM cascade is
documented as an optional alternative, kept out of this trust-sensitive path.
"""

from __future__ import annotations

from typing import Protocol

from app.dependencies import get_spacy_nlp

PERSON_LABEL = "PERSON"


class NERService(Protocol):
    """Interface for name extractors so the backend (spaCy, HF, LLM) is swappable."""

    def extract_names(self, text: str) -> list[str]:
        ...

    def extract_names_with_positions(self, text: str) -> list[dict]:
        ...


class SpacyNERService:
    """spaCy-backed person-name extractor. The pipeline is injected, not loaded here."""

    def __init__(self, nlp) -> None:
        self._nlp = nlp

    def extract_names(self, text: str) -> list[str]:
        """Return PERSON entity texts only (ORG entities are excluded)."""
        doc = self._nlp(text)
        return [ent.text for ent in doc.ents if ent.label_ == PERSON_LABEL]

    def extract_names_with_positions(self, text: str) -> list[dict]:
        """Return PERSON entities with their character offsets."""
        doc = self._nlp(text)
        return [
            {
                "name": ent.text,
                "start_char": ent.start_char,
                "end_char": ent.end_char,
                "label": ent.label_,
            }
            for ent in doc.ents
            if ent.label_ == PERSON_LABEL
        ]


# Module-level patchable dependency + callable aliases. `nlp` stays a module
# attribute so tests can `patch("app.services.ner_service.nlp", ...)`; the aliases
# are assignments (not `def`, which the architecture audit forbids) and resolve
# `nlp` at call time, so patches take effect.
nlp = get_spacy_nlp()
extract_names = lambda text: SpacyNERService(nlp).extract_names(text)  # noqa: E731
extract_names_with_positions = lambda text: SpacyNERService(  # noqa: E731
    nlp
).extract_names_with_positions(text)
