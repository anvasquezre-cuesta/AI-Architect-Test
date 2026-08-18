"""Bounding-box service.

Maps each extracted person name to its location on the page. A name only counts as
found when its word parts appear together: same page, same line, left to right, and
adjacent. That co-location rule (plus punctuation-tolerant matching, since OCR yields
"Thompson,") is what stops an unrelated "Margaret" at the top of a page from merging
with a "Thompson" at the bottom into a box that spans the whole page. Every emitted box
is validated: its words must reconstruct the name and it must stay about one line tall.
"""

from __future__ import annotations

import logging
import re
from typing import Protocol

from app.services.ner_service import extract_names
from app.services.ocr_service import get_word_bounding_boxes

logger = logging.getLogger(__name__)

SAME_LINE_TOLERANCE = 0.7   # max vertical offset for "same line", in line heights
MAX_GAP_FACTOR = 3.0        # max horizontal gap between adjacent words, in line heights
MAX_HEIGHT_FACTOR = 2.0     # a real name box is about one line tall


class BoundingBoxService(Protocol):
    """Interface for locating names on a page."""

    def find_name_bounding_boxes(self, pdf_path: str, text: str) -> list[dict]:
        ...


class NameBoundingBoxService:
    """Locate co-located word runs that spell each name and merge them into one box.

    The name extractor and word-box provider are injected so this class has no hard
    dependency on spaCy or Tesseract.
    """

    def __init__(self, name_extractor, word_box_provider) -> None:
        self._extract_names = name_extractor
        self._word_boxes = word_box_provider

    @staticmethod
    def _norm(word: str) -> str:
        """Lowercase and strip surrounding punctuation (OCR often yields 'Thompson,')."""
        return re.sub(r"^[^\w]+|[^\w]+$", "", word).lower()

    def find_name_bounding_boxes(self, pdf_path: str, text: str) -> list[dict]:
        names = self._extract_names(text)
        word_boxes = self._word_boxes(pdf_path)

        name_boxes: list[dict] = []
        used: set[int] = set()
        for name in names:
            parts = [p for p in name.split() if p]
            if not parts:
                continue
            occurrence = self._find_occurrence(word_boxes, parts, used)
            if occurrence is None:
                # No co-located run of words spells this name; skip rather than
                # emit an implausible box. This is the tenacity path: if a first
                # candidate is not co-located, _find_occurrence keeps looking.
                logger.debug("no co-located match for name", extra={"name": name})
                continue
            boxes = [word_boxes[i] for i in occurrence]
            merged = self._merge(name, boxes)
            if not self._is_valid(merged, boxes, parts):
                logger.warning("rejected implausible bounding box", extra={"name": name})
                continue
            used.update(occurrence)
            name_boxes.append(merged)
        return name_boxes

    def _find_occurrence(
        self, word_boxes: list[dict], parts: list[str], used: set[int]
    ) -> list[int] | None:
        """Return indices of the first unused, co-located run spelling `parts`."""
        n = len(word_boxes)
        for i in range(n):
            if i in used or self._norm(word_boxes[i]["word"]) != self._norm(parts[0]):
                continue
            matched = [i]
            prev = word_boxes[i]
            complete = True
            for part in parts[1:]:
                nxt = self._next_part(word_boxes, part, prev, used, matched)
                if nxt is None:
                    complete = False
                    break
                matched.append(nxt)
                prev = word_boxes[nxt]
            if complete:
                return matched
        return None

    def _next_part(
        self, word_boxes: list[dict], part: str, prev: dict, used: set[int], matched: list[int]
    ) -> int | None:
        """Index of the next word equal to `part`, co-located just right of `prev`."""
        for j, b in enumerate(word_boxes):
            if j in used or j in matched or self._norm(b["word"]) != self._norm(part):
                continue
            if b["page"] != prev["page"]:
                continue
            line_height = max(prev["height"], b["height"])
            if abs(b["y"] - prev["y"]) > line_height * SAME_LINE_TOLERANCE:
                continue  # different line
            gap = b["x"] - (prev["x"] + prev["width"])
            if b["x"] < prev["x"] or gap > line_height * MAX_GAP_FACTOR:
                continue  # to the left, or too far away to be the next word
            return j
        return None

    @staticmethod
    def _merge(name: str, boxes: list[dict]) -> dict:
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

    def _is_valid(self, merged: dict, boxes: list[dict], parts: list[str]) -> bool:
        """Check the coordinates actually correspond to the extracted name.

        The matched words must reconstruct the name in order, and the merged box must
        stay about one line tall (a page-spanning box means the parts were unrelated).
        """
        if [self._norm(b["word"]) for b in boxes] != [self._norm(p) for p in parts]:
            return False
        max_height = max(b["height"] for b in boxes)
        return merged["height"] <= max_height * MAX_HEIGHT_FACTOR


# Module-level alias resolves the (patchable) `extract_names` /
# `get_word_bounding_boxes` module globals at call time and injects them.
find_name_bounding_boxes = lambda pdf_path, text: NameBoundingBoxService(  # noqa: E731
    extract_names, get_word_bounding_boxes
).find_name_bounding_boxes(pdf_path, text)
