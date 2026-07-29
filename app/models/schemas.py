"""Pydantic schemas for API requests and responses."""

from pydantic import BaseModel, Field


class BoundingBox(BaseModel):
    """Bounding box in PDF coordinate space (points, 72 DPI)."""

    page_number: int = Field(ge=0, description="0-based page index")
    x: float
    y: float
    width: float
    height: float


class ExtractedName(BaseModel):
    name: str
    bounding_box: BoundingBox


class FuzzyMatch(BaseModel):
    extracted_name: str
    matched_name: str
    score: float = Field(ge=0.0, le=1.0)


class ExtractionResponse(BaseModel):
    extracted_names: list[ExtractedName]
    fuzzy_matches: list[FuzzyMatch] = []


class NamePair(BaseModel):
    first_name: str = Field(min_length=1)
    last_name: str = Field(min_length=1)


class IngestResponse(BaseModel):
    status: str
    chunks_stored: int


class RAGRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)
    profile: str | None = Field(
        default=None,
        description=(
            "Model profile: english_fast (default), spanish_precision, "
            "or multilingual. See GET /api/profiles."
        ),
    )


class ProfileInfo(BaseModel):
    """Client-facing description of a selectable model profile."""

    key: str
    description: str
    default: bool


class RAGResponse(BaseModel):
    answer: str
    sources: list[str] = []


class HealthResponse(BaseModel):
    status: str
    version: str
