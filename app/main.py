"""FastAPI application entrypoint."""

from __future__ import annotations

import logging
import os
import time
import uuid

from fastapi import FastAPI, Request
from fastapi.responses import FileResponse

from app.api.extract import router as extract_router
from app.api.rag import router as rag_router
from app.logging_config import configure_logging

_STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")

configure_logging()
logger = logging.getLogger("app.access")

app = FastAPI(title="PDF Name Extractor & RAG API")

app.include_router(extract_router, prefix="/api", tags=["extraction"])
app.include_router(rag_router, prefix="/api", tags=["rag"])


@app.middleware("http")
async def access_log(request: Request, call_next):
    """Assign a request id and log one structured line per request with timing."""
    request_id = uuid.uuid4().hex[:12]
    request.state.request_id = request_id
    start = time.perf_counter()
    try:
        response = await call_next(request)
    except Exception:
        elapsed_ms = round((time.perf_counter() - start) * 1000, 1)
        logger.exception(
            "request failed",
            extra={
                "request_id": request_id,
                "method": request.method,
                "path": request.url.path,
                "elapsed_ms": elapsed_ms,
            },
        )
        raise
    elapsed_ms = round((time.perf_counter() - start) * 1000, 1)
    logger.info(
        "request",
        extra={
            "request_id": request_id,
            "method": request.method,
            "path": request.url.path,
            "status": response.status_code,
            "elapsed_ms": elapsed_ms,
        },
    )
    response.headers["X-Request-ID"] = request_id
    return response


@app.get("/", include_in_schema=False)
def root():
    """Serve the minimal single-page UI (calls the /api endpoints via fetch)."""
    return FileResponse(os.path.join(_STATIC_DIR, "index.html"))


@app.get("/health", tags=["health"])
def health():
    """Liveness probe."""
    return {"status": "ok"}
