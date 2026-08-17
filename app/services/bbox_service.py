"""Bounding-box service.

Maps each extracted person name to its location on the page by matching the name's
word parts against the OCR word boxes and merging them into one box per name.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from app.services.ner_service import extract_names
from app.services.ocr_service import get_word_bounding_boxes


@runtime_checkable
class BoundingBoxService(Protocol):
    """Interface for locating names on a page."""

    def find_name_bounding_boxes(self, pdf_path: str, text: str) -> list[dict]:
        ...


class NameBoundingBoxService:
    """Merge OCR word boxes into one box per name.

    The name extractor and word-box provider are injected so this class has no
    hard dependency on spaCy or Tesseract.
    """

    def __init__(self, name_extractor, word_box_provider) -> None:
        self._extract_names = name_extractor
        self._word_boxes = word_box_provider

    def find_name_bounding_boxes(self, pdf_path: str, text: str) -> list[dict]:
        names = self._extract_names(text)
        word_boxes = self._word_boxes(pdf_path)

        name_boxes: list[dict] = []
        used: set[int] = set()  # consumed boxes, so a repeated name maps to the next occurrence
        for name in names:
            matched = []
            for part in name.split():
                for idx, box in enumerate(word_boxes):
                    if idx in used:
                        continue
                    # Case-insensitive: OCR may yield "JOHN" while NER yields "John".
                    if box["word"].lower() == part.lower():
                        matched.append(box)
                        used.add(idx)
                        break
            if not matched:
                continue
            min_x = min(b["x"] for b in matched)
            min_y = min(b["y"] for b in matched)
            max_x = max(b["x"] + b["width"] for b in matched)
            max_y = max(b["y"] + b["height"] for b in matched)
            name_boxes.append(
                {
                    "name": name,
                    "page": matched[0]["page"],
                    "x": min_x,
                    "y": min_y,
                    "width": max_x - min_x,
                    "height": max_y - min_y,
                }
            )
        return name_boxes


# Module-level aliases resolve the (patchable) `extract_names` /
# `get_word_bounding_boxes` module globals at call time and inject them.
find_name_bounding_boxes = lambda pdf_path, text: NameBoundingBoxService(  # noqa: E731
    extract_names, get_word_bounding_boxes
).find_name_bounding_boxes(pdf_path, text)
