"""Shared helpers for the API layer.

Small HTTP-adjacent utilities used by more than one endpoint, so the controllers do
not repeat the PDF check or the temp-file dance. Kept in the api package (not services)
because they are about the HTTP boundary, not domain logic.
"""

from __future__ import annotations

import contextlib
import os
import tempfile

from fastapi import HTTPException, UploadFile


def require_pdf(upload: UploadFile) -> None:
    """Raise 400 unless the upload looks like a PDF (by content type or suffix)."""
    filename = (upload.filename or "").lower()
    if not (upload.content_type == "application/pdf" or filename.endswith(".pdf")):
        raise HTTPException(status_code=400, detail="Only PDF files are accepted.")


@contextlib.contextmanager
def saved_pdf(upload: UploadFile):
    """Persist an uploaded PDF to a temp file for the duration of the block, then remove it."""
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
    try:
        tmp.write(upload.file.read())
        tmp.close()
        yield tmp.name
    finally:
        os.unlink(tmp.name)
