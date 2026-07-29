"""Extraction endpoint: OCR -> NER -> bounding boxes -> fuzzy matching."""

import json
import os
import tempfile

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from pydantic import ValidationError

from app.core.config import get_settings
from app.core.exceptions import AppError
from app.core.logging import get_logger
from app.core.profiles import UnknownProfileError, get_service_bundle, resolve_profile
from app.models.schemas import ExtractionResponse, NamePair
from app.services.bbox_service import find_name_bounding_boxes
from app.services.fuzzy_service import fuzzy_match_names
from app.services.ocr_service import extract_text_from_pdf

logger = get_logger(__name__)

router = APIRouter()

PDF_MAGIC = b"%PDF"


def resolve_profile_or_400(profile: str | None):
    """Resolve a client-provided profile key, mapping bad keys to 400."""
    try:
        return resolve_profile(profile)
    except UnknownProfileError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


def _validate_pdf_upload(pdf_file: UploadFile, content: bytes) -> None:
    """Reject uploads that are not valid PDF files (400)."""
    settings = get_settings()

    filename = (pdf_file.filename or "").lower()
    content_type = (pdf_file.content_type or "").lower()

    if not filename.endswith(".pdf") and content_type != "application/pdf":
        raise HTTPException(
            status_code=400,
            detail="Only PDF files are accepted (expected .pdf / application/pdf).",
        )
    if not content:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")
    if not content.startswith(PDF_MAGIC):
        raise HTTPException(
            status_code=400,
            detail="Uploaded file is not a valid PDF (bad file signature).",
        )
    if len(content) > settings.max_upload_bytes:
        raise HTTPException(
            status_code=413,
            detail=f"File too large (max {settings.max_upload_bytes} bytes).",
        )


def _parse_query_names(raw_names: str) -> list[dict]:
    """Parse and validate the `names` form field (JSON list of name pairs)."""
    try:
        parsed = json.loads(raw_names)
    except json.JSONDecodeError as exc:
        raise HTTPException(
            status_code=400, detail=f"'names' is not valid JSON: {exc}"
        ) from exc

    if not isinstance(parsed, list):
        raise HTTPException(
            status_code=400, detail="'names' must be a JSON list of name pairs."
        )

    try:
        return [NamePair.model_validate(item).model_dump() for item in parsed]
    except ValidationError as exc:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid name pair in 'names': {exc.errors()}",
        ) from exc


@router.post("/extract", response_model=ExtractionResponse)
def extract_names_from_pdf(
    pdf_file: UploadFile = File(...),
    names: str = Form(...),
    profile: str | None = Form(
        None,
        description=(
            "Model profile: english_fast (default), spanish_precision, "
            "or multilingual. See GET /api/profiles."
        ),
    ),
):
    """Extract person names from a PDF and fuzzy-match them against `names`."""
    content = pdf_file.file.read()
    _validate_pdf_upload(pdf_file, content)
    query_names = _parse_query_names(names)
    selected = resolve_profile_or_400(profile)
    # The default profile keeps using the module-level services (stable
    # public API); other profiles use their dedicated service bundle.
    is_default = selected.key == get_settings().default_profile
    bundle = None if is_default else get_service_bundle(selected.key)

    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
    try:
        tmp.write(content)
        tmp.close()

        if bundle is None:
            text = extract_text_from_pdf(tmp.name)
            name_boxes = find_name_bounding_boxes(tmp.name, text)
        else:
            text = bundle.ocr.extract_text_from_pdf(tmp.name)
            name_boxes = bundle.bbox.find_name_bounding_boxes(tmp.name, text)

        extracted_name_strings = [nb["name"] for nb in name_boxes]
        matches = fuzzy_match_names(extracted_name_strings, query_names)

        return {
            "extracted_names": [
                {
                    "name": nb["name"],
                    "bounding_box": {
                        "page_number": nb["page"],
                        "x": nb["x"],
                        "y": nb["y"],
                        "width": nb["width"],
                        "height": nb["height"],
                    },
                }
                for nb in name_boxes
            ],
            "fuzzy_matches": matches,
        }
    except HTTPException:
        raise
    except AppError as exc:
        logger.exception("Extraction pipeline failed")
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Unexpected error processing PDF")
        raise HTTPException(
            status_code=500, detail="Failed to process the PDF document."
        ) from exc
    finally:
        tmp.close()
        os.unlink(tmp.name)
