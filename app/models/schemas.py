"""Pydantic request/response models.

Required fields that the API previously computed but dropped are included here:
``BoundingBox.page_number``, ``ExtractionResponse.fuzzy_matches`` and
``RAGResponse.sources``. The fuzzy-match corroboration signal is exposed as optional
fields so the primary contract stays stable.
"""

from __future__ import annotations

from pydantic import BaseModel


class BoundingBox(BaseModel):
    x: float
    y: float
    width: float
    height: float
    page_number: int


class ExtractedName(BaseModel):
    name: str
    bounding_box: BoundingBox


class FuzzyMatch(BaseModel):
    extracted_name: str
    matched_name: str
    score: float
    jaro_winkler: float | None = None
    corroborated: bool | None = None


class ExtractionResponse(BaseModel):
    extracted_names: list[ExtractedName]
    fuzzy_matches: list[FuzzyMatch]


class NamePair(BaseModel):
    first_name: str
    last_name: str


class RAGRequest(BaseModel):
    question: str


class RAGResponse(BaseModel):
    answer: str
    sources: list[str] = []
