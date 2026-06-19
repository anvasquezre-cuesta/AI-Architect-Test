"""OCR service.

Rasterises PDF pages with PyMuPDF and runs Tesseract to recover text and
word-level bounding boxes. The public module-level functions are thin aliases
over :class:`TesseractOCRService` so callers and tests keep a stable surface
while the engine itself is encapsulated and swappable (see
``app.services.interfaces.OCREngine``).
"""

import io
import logging

import fitz
import pytesseract
from PIL import Image

from app.config import settings

logger = logging.getLogger(__name__)


class TesseractOCRService:
    """Concrete :class:`OCREngine` backed by PyMuPDF + Tesseract."""

    def __init__(self, dpi: int = None) -> None:
        self._dpi = dpi if dpi is not None else settings.ocr_dpi

    def extract_text(self, pdf_path: str) -> str:
        """Return the OCR'd text of every page (page 0 included)."""
        doc = fitz.open(pdf_path)
        try:
            full_text = ""
            for page_num in range(len(doc)):  # was range(1, len(doc)) — skipped page 0
                page = doc[page_num]
                # Render at the configured DPI (default 150); the previous
                # default 72-DPI raster produced poor, error-prone OCR.
                pix = page.get_pixmap(dpi=self._dpi)
                img = Image.open(io.BytesIO(pix.tobytes("png")))
                full_text += pytesseract.image_to_string(img) + "\n"
            return full_text
        finally:
            doc.close()  # the document was previously leaked

    def word_boxes(self, pdf_path: str) -> list:
        """Return every OCR'd word with its bounding box in **PDF point space**."""
        doc = fitz.open(pdf_path)
        try:
            results = []
            # Pixmap is rendered at ``ocr_dpi``; convert coordinates back to the
            # PDF's native 72-DPI point space so boxes line up with the document.
            scale = 72.0 / self._dpi
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


_service = TesseractOCRService()
extract_text_from_pdf = _service.extract_text
get_word_bounding_boxes = _service.word_boxes
