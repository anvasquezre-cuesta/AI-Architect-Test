"""OCR service backed by Tesseract + PyMuPDF.

Bug fixes vs. original implementation:
- Off-by-one: iteration now starts at page 0 (first page was skipped).
- Documents are always closed (try/finally) to avoid leaking file handles.
- Word bounding boxes are converted from OCR image space (rendered DPI)
  back to PDF coordinate space (72 points/inch).
"""

import io

import fitz
import pytesseract
from PIL import Image

from app.core.config import get_settings
from app.core.logging import get_logger
from app.services.interfaces import OCRService

logger = get_logger(__name__)


class TesseractOCRService(OCRService):
    """Concrete OCR implementation using Tesseract via pytesseract."""

    def __init__(
        self,
        dpi: int | None = None,
        pdf_point_dpi: float | None = None,
        min_confidence: float | None = None,
        lang: str | None = None,
    ) -> None:
        settings = get_settings()
        self.dpi = dpi or settings.ocr_dpi
        self.pdf_point_dpi = pdf_point_dpi or settings.pdf_point_dpi
        self.min_confidence = (
            min_confidence if min_confidence is not None else settings.ocr_min_confidence
        )
        # Tesseract language(s), e.g. "eng", "spa", "eng+spa". Requires the
        # matching tesseract-ocr-<lang> packs to be installed.
        self.lang = lang or settings.ocr_lang

    def _render_page(self, page) -> Image.Image:
        """Render a PDF page to a PIL image at the configured DPI."""
        pix = page.get_pixmap(dpi=self.dpi)
        return Image.open(io.BytesIO(pix.tobytes("png")))

    def extract_text_from_pdf(self, pdf_path: str) -> str:
        """Extract text from every page of a (scanned) PDF using OCR."""
        doc = fitz.open(pdf_path)
        try:
            page_texts: list[str] = []
            for page_num in range(len(doc)):  # start at 0: include first page
                img = self._render_page(doc[page_num])
                page_texts.append(pytesseract.image_to_string(img, lang=self.lang))
            logger.debug("OCR extracted %d page(s) from %s", len(page_texts), pdf_path)
            return "\n".join(page_texts)
        finally:
            doc.close()

    def get_word_bounding_boxes(self, pdf_path: str) -> list[dict]:
        """Get PDF-space bounding boxes for all recognized words."""
        scale = self.pdf_point_dpi / self.dpi  # image px -> PDF points
        doc = fitz.open(pdf_path)
        try:
            results: list[dict] = []
            for page_num in range(len(doc)):
                img = self._render_page(doc[page_num])
                ocr_data = pytesseract.image_to_data(
                    img, lang=self.lang, output_type=pytesseract.Output.DICT
                )
                confidences = ocr_data.get("conf")
                for i in range(len(ocr_data["text"])):
                    word = ocr_data["text"][i].strip()
                    if not word:
                        continue
                    if not self._confidence_ok(confidences, i):
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

    def _confidence_ok(self, confidences, index: int) -> bool:
        """Filter out words below the configured OCR confidence."""
        if not confidences or index >= len(confidences):
            return True
        try:
            conf = float(confidences[index])
        except (TypeError, ValueError):
            return True
        # Tesseract uses -1 for structural (non-word) entries; keep those
        # out of the threshold comparison.
        return conf < 0 or conf >= self.min_confidence


# Default instance + module-level callables kept as the stable public API
# (also what unit tests patch/import).
_default_service = TesseractOCRService()
extract_text_from_pdf = _default_service.extract_text_from_pdf
get_word_bounding_boxes = _default_service.get_word_bounding_boxes
