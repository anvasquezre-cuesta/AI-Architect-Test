"""FastAPI application entrypoint."""

from __future__ import annotations

import logging

from fastapi import FastAPI
from fastapi.responses import RedirectResponse

from app.api.extract import router as extract_router
from app.api.rag import router as rag_router

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)

app = FastAPI(title="PDF Name Extractor & RAG API")

app.include_router(extract_router, prefix="/api", tags=["extraction"])
app.include_router(rag_router, prefix="/api", tags=["rag"])


@app.get("/", include_in_schema=False)
def root():
    """Redirect the bare host to the interactive API docs."""
    return RedirectResponse(url="/docs")


@app.get("/health", tags=["health"])
def health():
    """Liveness probe."""
    return {"status": "ok"}
