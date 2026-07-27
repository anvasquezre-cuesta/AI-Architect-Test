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


class ExtractionResponse(BaseModel):
    extracted_names: list[ExtractedName]
    fuzzy_matches: list[FuzzyMatch]
    
    class Config:
        extra = "allow"


class NamePair(BaseModel):
    first_name: str
    last_name: str


class RAGRequest(BaseModel):
    question: str


class RAGResponse(BaseModel):
    answer: str
    sources: list[str]
    
    class Config:
        extra = "allow"
