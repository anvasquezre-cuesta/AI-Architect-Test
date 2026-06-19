"""Bounding-box service.

Joins NER person names to the OCR word boxes that spell them out and merges the
per-word boxes into a single span per name. Matching is case-insensitive
because OCR frequently returns a different case than the NER text (e.g. ``JOHN``
vs ``John``).
"""

import logging

from app.services.ner_service import extract_names
from app.services.ocr_service import get_word_bounding_boxes

logger = logging.getLogger(__name__)


class BBoxMatcher:
    """Locates each extracted name's bounding box within the OCR word boxes."""

    def find(self, pdf_path: str, text: str) -> list:
        names = extract_names(text)
        word_boxes = get_word_bounding_boxes(pdf_path)

        name_boxes = []
        for name in names:
            matched = []
            for part in name.split():
                for word_box in word_boxes:
                    # Case-insensitive: OCR casing need not match the NER text.
                    if word_box["word"].lower() == part.lower():
                        matched.append(word_box)
                        break

            if matched:
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


_matcher = BBoxMatcher()
find_name_bounding_boxes = _matcher.find
