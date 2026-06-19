"""FastAPI application entry point."""

from fastapi import FastAPI

from app.api.extract import router as extract_router
from app.api.rag import router as rag_router
from app.config import configure_logging

configure_logging()

app = FastAPI(title="PDF Name Extractor & RAG API")

app.include_router(extract_router, prefix="/api", tags=["extraction"])
app.include_router(rag_router, prefix="/api", tags=["rag"])


@app.get("/health", tags=["health"])
def health():
    """Liveness probe."""
    return {"status": "ok"}
