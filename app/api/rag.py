"""RAG endpoints: PDF ingestion into the vector store and question answering."""

import os
import tempfile

from fastapi import APIRouter, File, Form, HTTPException, UploadFile

from app.core.config import get_settings
from app.core.exceptions import AppError, LLMError, VectorStoreError
from app.core.logging import get_logger
from app.core.profiles import (
    UnknownProfileError,
    get_service_bundle,
    list_profiles,
    resolve_profile,
)
from app.models.schemas import IngestResponse, ProfileInfo, RAGRequest, RAGResponse
from app.services.ocr_service import extract_text_from_pdf
from app.services.rag_service import chunk_text, generate_answer
from app.services.vector_service import init_collection, store_document_chunks

logger = get_logger(__name__)

router = APIRouter()

PDF_MAGIC = b"%PDF"


def _resolve_bundle_or_400(profile: str | None):
    """Resolve a profile key to (profile, bundle-or-None); 400 on bad keys.

    Returns bundle=None for the default profile so those requests keep
    flowing through the stable module-level services.
    """
    try:
        selected = resolve_profile(profile)
    except UnknownProfileError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    if selected.key == get_settings().default_profile:
        return selected, None
    return selected, get_service_bundle(selected.key)


@router.get("/profiles", response_model=list[ProfileInfo])
def get_profiles():
    """List the model profiles clients can select on the other endpoints."""
    return list_profiles()


@router.post("/ingest", response_model=IngestResponse)
def ingest_pdf(
    pdf_file: UploadFile = File(...),
    profile: str | None = Form(
        None,
        description=(
            "Model profile: english_fast (default), spanish_precision, "
            "or multilingual. See GET /api/profiles."
        ),
    ),
):
    """Ingest a PDF document into the vector database."""
    content = pdf_file.file.read()

    filename = (pdf_file.filename or "").lower()
    content_type = (pdf_file.content_type or "").lower()
    if (
        not content
        or (not filename.endswith(".pdf") and content_type != "application/pdf")
        or not content.startswith(PDF_MAGIC)
    ):
        raise HTTPException(status_code=400, detail="A valid PDF file is required.")

    _, bundle = _resolve_bundle_or_400(profile)

    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
    try:
        tmp.write(content)
        tmp.close()

        source = pdf_file.filename or "unknown.pdf"
        if bundle is None:
            init_collection()
            text = extract_text_from_pdf(tmp.name)
            chunks = [c for c in chunk_text(text) if c.strip()]
            store_document_chunks(
                chunks, metadata=[{"source": source} for _ in chunks]
            )
        else:
            bundle.store.init_collection()
            text = bundle.ocr.extract_text_from_pdf(tmp.name)
            chunks = [c for c in bundle.rag.chunk_text(text) if c.strip()]
            bundle.store.store_document_chunks(
                chunks, metadata=[{"source": source} for _ in chunks]
            )

        logger.info("Ingested '%s' as %d chunk(s)", source, len(chunks))
        return {"status": "success", "chunks_stored": len(chunks)}
    except HTTPException:
        raise
    except VectorStoreError as exc:
        logger.exception("Vector store unavailable during ingestion")
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Unexpected error ingesting PDF")
        raise HTTPException(status_code=500, detail="Failed to ingest the PDF.") from exc
    finally:
        tmp.close()
        os.unlink(tmp.name)


@router.post("/ask", response_model=RAGResponse)
def ask_question(request: RAGRequest):
    """Answer a question using RAG."""
    _, bundle = _resolve_bundle_or_400(request.profile)
    try:
        if bundle is None:
            return generate_answer(request.question)
        return bundle.rag.generate_answer(request.question)
    except VectorStoreError as exc:
        logger.exception("Vector store unavailable during search")
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except LLMError as exc:
        logger.exception("LLM call failed")
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    except AppError as exc:
        logger.exception("RAG pipeline failed")
        raise HTTPException(status_code=500, detail=str(exc)) from exc
