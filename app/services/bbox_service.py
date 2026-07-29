"""Maps NER-extracted names to word bounding boxes from OCR.

Bug fixes vs. original implementation:
- Case-insensitive, punctuation-insensitive word comparison (OCR may
  return 'JOHN' or 'Smith,' while NER returns 'John Smith').
- All occurrences of a name are returned (the original always matched
  only the first occurrence of each word).
- Words are matched as a *consecutive sequence* on the same page, so the
  parts of one name can no longer be merged across unrelated locations.
"""

import re

from app.core.logging import get_logger
from app.services.interfaces import BoundingBoxService
from app.services.ner_service import extract_names
from app.services.ocr_service import get_word_bounding_boxes

logger = get_logger(__name__)

_NORMALIZE_RE = re.compile(r"[^\w]", re.UNICODE)


class NameBoundingBoxService(BoundingBoxService):
    """Finds bounding boxes for every occurrence of each extracted name."""

    def __init__(self, ner_service=None, ocr_service=None) -> None:
        self._ner = ner_service
        self._ocr = ocr_service

    @staticmethod
    def _normalize(word: str) -> str:
        """Lowercase and strip punctuation for robust comparisons."""
        return _NORMALIZE_RE.sub("", word).lower()

    @staticmethod
    def _merge_boxes(name: str, boxes: list[dict]) -> dict:
        """Merge word boxes into a single enclosing bounding box."""
        min_x = min(b["x"] for b in boxes)
        min_y = min(b["y"] for b in boxes)
        max_x = max(b["x"] + b["width"] for b in boxes)
        max_y = max(b["y"] + b["height"] for b in boxes)
        return {
            "name": name,
            "page": boxes[0]["page"],
            "x": min_x,
            "y": min_y,
            "width": max_x - min_x,
            "height": max_y - min_y,
        }

    def find_name_bounding_boxes(self, pdf_path: str, text: str) -> list[dict]:
        """Match extracted names to their bounding boxes in the PDF."""
        names = (
            self._ner.extract_names(text) if self._ner else extract_names(text)
        )
        word_boxes = (
            self._ocr.get_word_bounding_boxes(pdf_path)
            if self._ocr
            else get_word_bounding_boxes(pdf_path)
        )

        normalized_words = [self._normalize(wb["word"]) for wb in word_boxes]
        name_boxes: list[dict] = []
        seen_names: set[str] = set()

        for name in names:
            key = self._normalize(name)
            if key in seen_names:  # avoid duplicating repeated NER mentions
                continue
            seen_names.add(key)

            parts = [p for p in (self._normalize(p) for p in name.split()) if p]
            if not parts:
                continue

            occurrences = self._find_occurrences(parts, word_boxes, normalized_words)
            for boxes in occurrences:
                name_boxes.append(self._merge_boxes(name, boxes))

        logger.debug("Matched %d bounding box(es) for %d name(s)",
                     len(name_boxes), len(names))
        return name_boxes

    @staticmethod
    def _find_occurrences(
        parts: list[str],
        word_boxes: list[dict],
        normalized_words: list[str],
    ) -> list[list[dict]]:
        """Find every consecutive same-page occurrence of ``parts``."""
        occurrences: list[list[dict]] = []
        n = len(parts)
        for i in range(len(word_boxes) - n + 1):
            window = word_boxes[i : i + n]
            if any(wb["page"] != window[0]["page"] for wb in window):
                continue
            if normalized_words[i : i + n] == parts:
                occurrences.append(window)
        return occurrences


# Default instance + module-level callable (stable public API / test seam).
_default_service = NameBoundingBoxService()
find_name_bounding_boxes = _default_service.find_name_bounding_boxes
