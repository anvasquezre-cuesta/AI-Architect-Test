"""PDF name-extraction and region-crop endpoints."""

from __future__ import annotations

import json
import logging

from fastapi import APIRouter, File, Form, HTTPException, Response, UploadFile

from app.api._helpers import require_pdf, saved_pdf
from app.models.schemas import ExtractionResponse
from app.services.bbox_service import find_name_bounding_boxes
from app.services.fuzzy_service import fuzzy_match_names
from app.services.ocr_service import extract_text_from_pdf
from app.services.pdf_render_service import render_region

logger = logging.getLogger(__name__)

router = APIRouter()


@router.post("/extract", response_model=ExtractionResponse)
def extract_names_from_pdf(
    pdf_file: UploadFile = File(...),
    names: str = Form(...),
):
    """Extract person names + bounding boxes from a PDF and fuzzy-match them."""
    require_pdf(pdf_file)
    try:
        query_names = json.loads(names)
    except json.JSONDecodeError:
        raise HTTPException(
            status_code=400, detail="`names` must be a valid JSON array."
        )

    try:
        with saved_pdf(pdf_file) as path:
            text = extract_text_from_pdf(path)
            name_boxes = find_name_bounding_boxes(path, text)
            matches = fuzzy_match_names(
                [nb["name"] for nb in name_boxes], query_names
            )

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


@router.post("/crop")
def crop_bounding_box(
    pdf_file: UploadFile = File(...),
    page: int = Form(...),
    x: float = Form(...),
    y: float = Form(...),
    width: float = Form(...),
    height: float = Form(...),
):
    """Return a PNG crop of the PDF region for one bounding box (box outlined)."""
    require_pdf(pdf_file)
    try:
        with saved_pdf(pdf_file) as path:
            png = render_region(path, page, x, y, width, height)
    except ValueError:
        raise HTTPException(status_code=400, detail="Page out of range.")
    except Exception:
        logger.exception("crop failed", extra={"pdf": pdf_file.filename})
        raise HTTPException(status_code=500, detail="Failed to render the crop.")

    return Response(content=png, media_type="image/png")
