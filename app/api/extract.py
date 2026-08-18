"""PDF name-extraction endpoint."""

from __future__ import annotations

import json
import logging
import os
import tempfile

import fitz
from fastapi import APIRouter, File, Form, HTTPException, Response, UploadFile

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


@router.post("/crop")
def crop_bounding_box(
    pdf_file: UploadFile = File(...),
    page: int = Form(...),
    x: float = Form(...),
    y: float = Form(...),
    width: float = Form(...),
    height: float = Form(...),
):
    """Return a PNG crop of the PDF region for one bounding box.

    Renders that page area with a little surrounding context and the box outlined,
    so the user can visually verify what was extracted at those coordinates.
    """
    if not _is_pdf(pdf_file):
        raise HTTPException(status_code=400, detail="Only PDF files are accepted.")

    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
    try:
        tmp.write(pdf_file.file.read())
        tmp.close()

        doc = fitz.open(tmp.name)
        try:
            if page < 0 or page >= len(doc):
                raise HTTPException(status_code=400, detail="Page out of range.")
            pdf_page = doc[page]
            box = fitz.Rect(x, y, x + width, y + height)
            pdf_page.draw_rect(box, color=(0.22, 0.74, 0.98), width=1.5)
            pad = 24  # context around the box, in PDF points
            clip = fitz.Rect(
                max(0, x - pad),
                max(0, y - pad),
                min(pdf_page.rect.width, x + width + pad),
                min(pdf_page.rect.height, y + height + pad),
            )
            png = pdf_page.get_pixmap(clip=clip, dpi=150).tobytes("png")
        finally:
            doc.close()

        return Response(content=png, media_type="image/png")
    except HTTPException:
        raise
    except Exception:
        logger.exception("crop failed", extra={"pdf": pdf_file.filename})
        raise HTTPException(status_code=500, detail="Failed to render the crop.")
    finally:
        os.unlink(tmp.name)
