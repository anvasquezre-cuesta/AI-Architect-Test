"""Named-entity recognition service.

Wraps the spaCy pipeline (injected from the composition root) and returns only
``PERSON`` entities. Module-level ``nlp`` plus the function aliases keep the
existing import/patch surface; the logic itself lives in
:class:`SpacyNERService` (see ``app.services.interfaces.NERModel``).
"""

import logging

from app.dependencies import create_ner_model

logger = logging.getLogger(__name__)

# Injected spaCy pipeline (constructed once in the composition root).
nlp = create_ner_model()

# Only people are names — organisations, places, dates, etc. are excluded.
PERSON_LABELS = ("PERSON",)


class SpacyNERService:
    """Concrete :class:`NERModel` that extracts person names with spaCy."""

    def names(self, text: str) -> list:
        doc = nlp(text)
        return [ent.text for ent in doc.ents if ent.label_ in PERSON_LABELS]

    def names_with_positions(self, text: str) -> list:
        doc = nlp(text)
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


_service = SpacyNERService()
extract_names = _service.names
extract_names_with_positions = _service.names_with_positions
