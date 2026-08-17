"""PDF name-extraction endpoint."""

from __future__ import annotations

import json
import logging
import os
import tempfile

from fastapi import APIRouter, File, Form, HTTPException, UploadFile

from app.models.schemas import ExtractionResponse
from app.services.bbox_service import find_name_bounding_boxes
from app.services.fuzzy_service import fuzzy_match_names
from app.services.ocr_service import extract_text_from_pdf

logger = logging.getLogger(__name__)

router = APIRouter()


def _is_pdf(upload: UploadFile) -> bool:
    filename = (upload.filename or "").lower()
    return upload.content_type == "application/pdf" or filename.endswith(".pdf")


@router.post("/extract", response_model=ExtractionResponse)
def extract_names_from_pdf(
    pdf_file: UploadFile = File(...),
    names: str = Form(...),
):
    """Extract person names + bounding boxes from a PDF and fuzzy-match them."""
    if not _is_pdf(pdf_file):
        raise HTTPException(status_code=400, detail="Only PDF files are accepted.")

    try:
        query_names = json.loads(names)
    except json.JSONDecodeError:
        raise HTTPException(
            status_code=400, detail="`names` must be a valid JSON array."
        )

    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
    try:
        tmp.write(pdf_file.file.read())
        tmp.close()

        text = extract_text_from_pdf(tmp.name)
        name_boxes = find_name_bounding_boxes(tmp.name, text)
        extracted_name_strings = [nb["name"] for nb in name_boxes]
        matches = fuzzy_match_names(extracted_name_strings, query_names)

        logger.info(
            "extract complete",
            extra={
                "pdf": pdf_file.filename,
                "n_query_names": len(query_names),
                "n_extracted": len(name_boxes),
                "n_matches": len(matches),
            },
        )
        return {
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
    except HTTPException:
        raise
    except Exception:
        logger.exception("extract failed", extra={"pdf": pdf_file.filename})
        raise HTTPException(status_code=500, detail="Failed to process the PDF.")
    finally:
        os.unlink(tmp.name)
