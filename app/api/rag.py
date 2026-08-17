"""RAG ingest / ask endpoints."""

from __future__ import annotations

import logging
import os
import tempfile

from fastapi import APIRouter, File, HTTPException, UploadFile

from app.models.schemas import RAGRequest, RAGResponse
from app.services.ocr_service import extract_text_from_pdf
from app.services.rag_service import chunk_text, generate_answer
from app.services.vector_service import init_collection, store_document_chunks

logger = logging.getLogger(__name__)

router = APIRouter()


def _is_pdf(upload: UploadFile) -> bool:
    filename = (upload.filename or "").lower()
    return upload.content_type == "application/pdf" or filename.endswith(".pdf")


@router.post("/ingest")
def ingest_pdf(pdf_file: UploadFile = File(...)):
    """Ingest a PDF document into the vector database."""
    if not _is_pdf(pdf_file):
        raise HTTPException(status_code=400, detail="Only PDF files are accepted.")

    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
    try:
        tmp.write(pdf_file.file.read())
        tmp.close()

        init_collection()
        text = extract_text_from_pdf(tmp.name)
        chunks = chunk_text(text)
        store_document_chunks(chunks)

        logger.info(
            "ingest complete",
            extra={"pdf": pdf_file.filename, "chunks_stored": len(chunks)},
        )
        return {"status": "success", "chunks_stored": len(chunks)}
    except Exception:
        logger.exception("ingest failed", extra={"pdf": pdf_file.filename})
        raise HTTPException(status_code=500, detail="Failed to ingest the PDF.")
    finally:
        os.unlink(tmp.name)


@router.post("/ask", response_model=RAGResponse)
def ask_question(request: RAGRequest):
    """Answer a question using the RAG pipeline."""
    result = generate_answer(request.question)
    logger.info(
        "ask complete",
        extra={
            "question_len": len(request.question),
            "n_sources": len(result.get("sources", [])),
        },
    )
    return result
