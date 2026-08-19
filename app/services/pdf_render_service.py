"""PDF region rendering.

Renders a full-width strip around a bounding box, with the box outlined, to PNG bytes.
This is the implementation the ``/crop`` endpoint used to hold inline; keeping it here
makes that controller thin (validate, delegate, respond) and the rendering reusable.
"""

from __future__ import annotations

from typing import Protocol

import fitz

HIGHLIGHT_COLOR = (0.22, 0.74, 0.98)
VERTICAL_PAD = 18  # context above and below the line, in PDF points


class PdfRegionRenderer(Protocol):
    """Interface for rendering a highlighted region of a PDF page."""

    def render_region(
        self, pdf_path: str, page: int, x: float, y: float, width: float, height: float
    ) -> bytes:
        ...


class FitzPdfRegionRenderer:
    """PyMuPDF-backed renderer. Raises ``ValueError`` for an out-of-range page."""

    def render_region(
        self, pdf_path: str, page: int, x: float, y: float, width: float, height: float
    ) -> bytes:
        doc = fitz.open(pdf_path)
        try:
            if page < 0 or page >= len(doc):
                raise ValueError("page out of range")
            pdf_page = doc[page]
            pdf_page.draw_rect(
                fitz.Rect(x, y, x + width, y + height),
                color=HIGHLIGHT_COLOR,
                width=1.5,
            )
            # Full page width so no word is cut off, focused vertically on the line.
            clip = fitz.Rect(
                0,
                max(0, y - VERTICAL_PAD),
                pdf_page.rect.width,
                min(pdf_page.rect.height, y + height + VERTICAL_PAD),
            )
            return pdf_page.get_pixmap(clip=clip, dpi=150).tobytes("png")
        finally:
            doc.close()


_default_renderer = FitzPdfRegionRenderer()
render_region = _default_renderer.render_region
