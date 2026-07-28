"""Tests for OCR service.

These tests verify text extraction from PDF documents.
Some tests will FAIL due to bugs in the current implementation.
"""

import pytest
from unittest.mock import patch, MagicMock
from PIL import Image
import io


class TestExtractTextFromPDF:
    """Tests for extract_text_from_pdf function."""

    def test_extracts_text_from_all_pages(self):
        """FAILS: Bug — first page is skipped due to off-by-one error.
        The function starts iteration at page 1 instead of page 0.
        """
        mock_doc = MagicMock()
        mock_doc.__len__ = MagicMock(return_value=3)

        mock_pages = []
        for i in range(3):
            page = MagicMock()
            pix = MagicMock()
            # Create a minimal valid PNG
            img = Image.new("RGB", (100, 100))
            buf = io.BytesIO()
            img.save(buf, format="PNG")
            pix.tobytes.return_value = buf.getvalue()
            page.get_pixmap.return_value = pix
            mock_pages.append(page)

        mock_doc.__getitem__ = MagicMock(side_effect=lambda i: mock_pages[i])

        with (
            patch("app.services.ocr_service.pymupdf.open", return_value=mock_doc),
            patch("app.services.ocr_service.easyocr.Reader") as mock_reader,
        ):
            mock_instance = mock_reader.return_value
            mock_instance.readtext.side_effect = [
                [[None, f"Page {i} text", 0.95]] for i in range(3)
            ]
            from app.services.ocr_service import OCRService

            ocr_service = OCRService()
            result = ocr_service.extract_text_from_pdf("test.pdf")

        # Should contain text from ALL 3 pages including page 0
        assert "Page 0 text" in result, "First page text is missing — off-by-one bug"
        assert "Page 1 text" in result
        assert "Page 2 text" in result

    def test_returns_empty_string_for_empty_pdf(self):
        """Should handle empty PDFs gracefully."""
        mock_doc = MagicMock()
        mock_doc.__len__ = MagicMock(return_value=0)

        with (
            patch("app.services.ocr_service.pymupdf.open", return_value=mock_doc),
            patch("app.services.ocr_service.easyocr.Reader"),
        ):
            from app.services.ocr_service import OCRService

            ocr_service = OCRService()
            result = ocr_service.extract_text_from_pdf("empty.pdf")

        assert result == ""

    def test_document_is_properly_closed(self):
        """FAILS: Bug — document is never closed after processing.
        Should use context manager or explicit close().
        """
        mock_doc = MagicMock()
        mock_doc.__len__ = MagicMock(return_value=0)

        with (
            patch("app.services.ocr_service.pymupdf.open", return_value=mock_doc),
            patch("app.services.ocr_service.easyocr.Reader"),
        ):
            from app.services.ocr_service import OCRService

            ocr_service = OCRService()
            ocr_service.extract_text_from_pdf("test.pdf")

        mock_doc.close.assert_called_once()


class TestGetWordBoundingBoxes:
    """Tests for get_word_bounding_boxes function."""

    def test_coordinates_in_pdf_space(self):
        """FAILS: Bug — coordinates are in OCR image space (150 DPI),
        not converted back to PDF coordinate space (72 DPI).
        """
        mock_doc = MagicMock()
        mock_doc.__len__ = MagicMock(return_value=1)

        page = MagicMock()
        # PDF page is 612x792 points (8.5x11 inches at 72 DPI)
        page.rect = MagicMock()
        page.rect.width = 612
        page.rect.height = 792

        pix = MagicMock()
        img = Image.new("RGB", (1275, 1650))  # 8.5x11 at 150 DPI
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        pix.tobytes.return_value = buf.getvalue()
        page.get_pixmap.return_value = pix

        mock_doc.__getitem__ = MagicMock(return_value=page)

        # EasyOCR format: (bbox, text, confidence)
        # bbox: [[x1, y1], [x2, y2], [x3, y3], [x4, y4]]
        ocr_data = [
            [[[150, 300], [350, 300], [350, 330], [150, 330]], "Hello", 0.95]
        ]

        with (
            patch("app.services.ocr_service.pymupdf.open", return_value=mock_doc),
            patch("app.services.ocr_service.easyocr.Reader") as mock_easyocr,
        ):
            mock_instance = mock_easyocr.return_value
            mock_instance.readtext.return_value = ocr_data
            from app.services.ocr_service import OCRService

            ocr_service = OCRService()
            results = ocr_service.get_word_bounding_boxes("test.pdf")

        assert len(results) == 1
        # At 150 DPI, 150px = 1 inch = 72 PDF points
        # Coordinates should be converted: x * (72/150)
        assert results[0]["x"] == pytest.approx(72.0, abs=1.0), (
            "X coordinate should be in PDF space (72 DPI), not OCR space (150 DPI)"
        )
