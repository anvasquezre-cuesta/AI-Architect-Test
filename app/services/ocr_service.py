"""OCR service.

Extracts text and per-word bounding boxes from scanned (image-only) PDFs by
rasterizing each page with PyMuPDF and running Tesseract via pytesseract.

Design notes:
- ``OCRService`` is a Protocol so the engine (Tesseract today, EasyOCR/cloud OCR
  tomorrow) can be swapped without touching callers.
- ``TesseractOCRService`` holds no heavy model, so a module-level default instance
  is cheap; the ``extract_text_from_pdf`` / ``get_word_bounding_boxes`` module
  aliases below preserve the original functional call sites and patch targets.
"""

from __future__ import annotations

import io
from typing import Protocol

import fitz
import pytesseract
from PIL import Image

# Rasterization DPI for OCR, and the PDF coordinate-space DPI we map boxes back to.
OCR_DPI = 150
PDF_DPI = 72


class OCRService(Protocol):
    """Interface for OCR engines."""

    def extract_text(self, pdf_path: str) -> str:
        ...

    def get_word_bounding_boxes(self, pdf_path: str) -> list[dict]:
        ...


class TesseractOCRService:
    """Tesseract-backed OCR over PyMuPDF-rasterized pages."""

    def __init__(self, dpi: int = OCR_DPI) -> None:
        self._dpi = dpi

    def extract_text(self, pdf_path: str) -> str:
        """Return the concatenated OCR text of every page.

        Fixes: iterate from page 0 (was ``range(1, len(doc))``, which dropped the
        first page) and always close the document.
        """
        doc = fitz.open(pdf_path)
        try:
            parts = []
            for page_num in range(len(doc)):
                page = doc[page_num]
                # Rasterize at OCR DPI (not the ~72 DPI default) so Tesseract has
                # enough resolution. At the default, glyphs like R get misread as P.
                pix = page.get_pixmap(dpi=self._dpi)
                img = Image.open(io.BytesIO(pix.tobytes("png")))
                parts.append(pytesseract.image_to_string(img))
            return "".join(part + "\n" for part in parts)
        finally:
            doc.close()

    def get_word_bounding_boxes(self, pdf_path: str) -> list[dict]:
        """Return one box per recognized word, in PDF coordinate space.

        Fixes: rescale OCR image-space coordinates (``self._dpi``) back to PDF
        points (``PDF_DPI``) instead of leaving them in image pixels; close the doc.
        """
        doc = fitz.open(pdf_path)
        try:
            scale = PDF_DPI / self._dpi
            results: list[dict] = []
            for page_num in range(len(doc)):
                page = doc[page_num]
                pix = page.get_pixmap(dpi=self._dpi)
                img = Image.open(io.BytesIO(pix.tobytes("png")))
                ocr_data = pytesseract.image_to_data(
                    img, output_type=pytesseract.Output.DICT
                )
                for i in range(len(ocr_data["text"])):
                    word = ocr_data["text"][i].strip()
                    if not word:
                        continue
                    results.append(
                        {
                            "word": word,
                            "page": page_num,
                            "x": ocr_data["left"][i] * scale,
                            "y": ocr_data["top"][i] * scale,
                            "width": ocr_data["width"][i] * scale,
                            "height": ocr_data["height"][i] * scale,
                        }
                    )
            return results
        finally:
            doc.close()


# Default instance + functional aliases: preserve the original import sites
# (`from app.services.ocr_service import extract_text_from_pdf`) and patch targets
# (`patch("app.services.ocr_service.fitz.open")`) without a module-level `def`.
_default_service = TesseractOCRService()
extract_text_from_pdf = _default_service.extract_text
get_word_bounding_boxes = _default_service.get_word_bounding_boxes
