"""FastAPI application entrypoint."""

from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from app.api.extract import router as extract_router
from app.api.rag import router as rag_router
from app.core.config import get_settings
from app.core.exceptions import AppError
from app.core.logging import configure_logging, get_logger

APP_VERSION = "1.0.0"

configure_logging()
logger = get_logger(__name__)

settings = get_settings()

app = FastAPI(title=settings.app_name, version=APP_VERSION)

app.include_router(extract_router, prefix="/api", tags=["extraction"])
app.include_router(rag_router, prefix="/api", tags=["rag"])

# Lightweight demo frontend: PDF viewer with bounding-box overlay.
# Self-contained (vendored pdf.js) so it works fully offline.
_STATIC_DIR = Path(__file__).parent / "static"
app.mount("/static", StaticFiles(directory=_STATIC_DIR), name="static")


@app.get("/", include_in_schema=False)
def viewer():
    """Serve the demo PDF viewer frontend."""
    return FileResponse(_STATIC_DIR / "viewer.html")


@app.get("/health")
def health():
    """Liveness probe for container orchestration and monitoring."""
    return {"status": "ok", "version": APP_VERSION}


@app.exception_handler(AppError)
async def app_error_handler(request: Request, exc: AppError):
    """Last-resort handler so domain errors never leak stack traces."""
    logger.error("Unhandled application error on %s: %s", request.url.path, exc)
    return JSONResponse(status_code=500, content={"detail": str(exc)})
