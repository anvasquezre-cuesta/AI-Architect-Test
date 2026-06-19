"""Pydantic request/response schemas for the API."""

from typing import List

from pydantic import BaseModel, Field


class BoundingBox(BaseModel):
    x: float
    y: float
    width: float
    height: float
    page_number: int = 0


class ExtractedName(BaseModel):
    name: str
    bounding_box: BoundingBox


class FuzzyMatch(BaseModel):
    extracted_name: str
    matched_name: str
    score: float


class ExtractionResponse(BaseModel):
    extracted_names: List[ExtractedName]
    # Previously computed but dropped because it was missing from the schema.
    fuzzy_matches: List[FuzzyMatch] = Field(default_factory=list)


class NamePair(BaseModel):
    first_name: str
    last_name: str


class RAGRequest(BaseModel):
    question: str


class RAGResponse(BaseModel):
    answer: str
    # Source chunks used to ground the answer (transparency / citations).
    sources: List[str] = Field(default_factory=list)
