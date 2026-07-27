import tempfile
import os

from fastapi import APIRouter, File, UploadFile, HTTPException, status
import anyio

from app.models.schemas import RAGRequest, RAGResponse
from app.services.ocr_service import OCRService
from app.services.rag_service import OpenAIRAGService
from app.services.vector_service import QdrantVectorStoreService

from loguru import logger

router = APIRouter()

vector_store = QdrantVectorStoreService()
rag_service = OpenAIRAGService(vector_store=vector_store)

async def generate_answer(question: str):
    """Convenience wrapper for rag_service.generate_answer."""
    return await rag_service.generate_answer(question)


@router.post("/ingest", status_code=status.HTTP_201_CREATED)
async def ingest_pdf(pdf_file: UploadFile = File(...)):
    """Ingest a PDF document into the vector database."""

    logger.info(f"ACCESSING API Operation ingest_pdf: file={pdf_file.filename}")

    if not pdf_file.filename.endswith(".pdf"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Only PDF files are supported"
        )

    tmp_path = ""
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            content = await pdf_file.read()
            tmp.write(content)
            tmp_path = tmp.name

        logger.info(f"Getting PDF from tmp file={tmp_path}")

        # Offload heavy/blocking tasks to a thread pool to avoid blocking the event loop
        await anyio.to_thread.run_sync(vector_store.init_collection)
        logger.info("Initialized collection in Vector_Store")

        ocr_service = OCRService() # Fixed: OCRService() doesn't take path in __init__
        text = await anyio.to_thread.run_sync(ocr_service.extract_text_from_pdf, tmp_path)
        logger.info("Extracted text with OCRService")

        if not text.strip():
             logger.warning("No text extracted from PDF")
             return {"status": "success", "chunks_stored": 0, "message": "No text found in document"}

        chunks = await anyio.to_thread.run_sync(rag_service.chunk_text, text)
        logger.info(f"Chunked text into {len(chunks)} parts")

        await anyio.to_thread.run_sync(vector_store.store_document_chunks, chunks)
        logger.info("Stored document chunks using Vector_Store")

        return {"status": "success", "chunks_stored": len(chunks)}

    except Exception as e:
        logger.error(f"Error in ingest_pdf: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to ingest PDF: {str(e)}"
        )

    finally:
        if tmp_path and os.path.exists(tmp_path):
            logger.info(f"UNLINKING ingest_pdf:: file={tmp_path}")
            os.unlink(tmp_path)


@router.get("/health")
def health_check():
    return {"status": "ok"}


@router.post("/ask", response_model=RAGResponse)
async def ask_question(request: RAGRequest):
    """Answer a question using RAG."""

    logger.info(f"ACCESSING API Operation ask_question: question={request.question}")

    try:
        # Generate answer using the now-async rag_service
        result = await rag_service.generate_answer(request.question)

        logger.info("Successfully generated answer")
        return result

    except Exception as e:
        logger.error(f"Error in ask_question: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Error generating answer from RAG service"
        )



# Start the server (Development only)
if __name__ == "__main__":
    import uvicorn
    from fastapi import FastAPI
    app = FastAPI()
    app.include_router(router)
    logger.info("API for RAG Application launched (Dev mode)")
    uvicorn.run(app, host="0.0.0.0", port=8004)
