import os
import tempfile
import json
import ast

from typing import Any, List
from loguru import logger

from fastapi import APIRouter, File, Form, UploadFile, HTTPException, FastAPI, Depends

import uvicorn

from app.models.schemas import ExtractionResponse, NamePair
from app.services.ocr_service import OCRService
from app.services.fuzzy_service import FuzzyService

router = APIRouter()

# Singleton services for dependency injection
_ocr_service = None
_fuzzy_service = None

def get_ocr_service() -> OCRService:
    global _ocr_service
    if _ocr_service is None:
        _ocr_service = OCRService()
    return _ocr_service

def get_fuzzy_service() -> FuzzyService:
    global _fuzzy_service
    if _fuzzy_service is None:
        _fuzzy_service = FuzzyService()
    return _fuzzy_service

@router.get("/health")
def health_check():
    return {"status": "ok"}


@router.post("/extract", response_model=ExtractionResponse)
async def extract_names_from_pdf(
    pdf_file: UploadFile = File(...),
    names: str = Form(...),
    ocr_service: OCRService = Depends(get_ocr_service),
    fuzzy_service: FuzzyService = Depends(get_fuzzy_service),
) -> Any:

    logger.info(f"ACCESSING API Operation extract_names_from_pdf: file={pdf_file.filename} names={names}")

    if not pdf_file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported")

    """Extract names from PDF and perform fuzzy matching."""
    import anyio
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp_path = tmp.name
        content = await pdf_file.read()
        tmp.write(content)

    logger.info(f"Extracting names from PDF plus Fuzzy Matching from tmp file={tmp_path}")

    try:
        text = await anyio.to_thread.run_sync(ocr_service.extract_text_from_pdf, tmp_path)
        logger.debug(f"Extracted Text: {text[:100]}...")

        name_boxes = await anyio.to_thread.run_sync(ocr_service.find_name_bounding_boxes, tmp_path, text)
        logger.debug(f"Found {len(name_boxes)} bounding boxes")

        try:
            # First try standard JSON parsing
            query_names_raw = json.loads(names)
        except json.JSONDecodeError:
            try:
                # If JSON fails, try literal_eval for single quotes (python style)
                query_names_raw = ast.literal_eval(names)
            except (ValueError, SyntaxError):
                logger.warning(f"Invalid format for names: {names}")
                raise HTTPException(status_code=400, detail="Invalid format for names. Must be a JSON list of name objects.")

        # Validate with Pydantic
        try:
            if not isinstance(query_names_raw, list):
                raise ValueError("names must be a list")
            query_names = [NamePair(**item).model_dump() for item in query_names_raw]
        except Exception as e:
            logger.warning(f"Invalid name data structure: {str(e)}")
            raise HTTPException(status_code=400, detail=f"Invalid name data structure: {str(e)}")

        extracted_name_strings = [nb["name"] for nb in name_boxes]
        matches = await anyio.to_thread.run_sync(fuzzy_service.match_names, extracted_name_strings, query_names)

        extract_response = {
            "extracted_names": [
                {
                    "name": nb["name"],
                    "bounding_box": {
                        "x": nb["x"],
                        "y": nb["y"],
                        "width": nb["width"],
                        "height": nb["height"],
                        "page_number": nb["page"],
                    },
                }
                for nb in name_boxes
            ],
            "fuzzy_matches": matches,
        }

        logger.info(f"Successfully processed extraction for {pdf_file.filename}")
        return extract_response

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in extract_names_from_pdf: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error during PDF processing")

    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)



# Start the server
if __name__ == "__main__":
    def main ():
        app = FastAPI()
        app.include_router(router)
        logger.info(f"\n\nAPI for EXTRACT Application launched.\n\n")
        uvicorn.run(app, host="0.0.0.0", port=8003)

    main ()
