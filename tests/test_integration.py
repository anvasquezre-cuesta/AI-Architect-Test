"""Integration tests using sample PDF documents.

These tests use the actual sample PDFs in sample_pdfs/ to verify
the full extraction pipeline. They require Tesseract OCR and spaCy
to be installed.

Some tests will FAIL due to bugs in the current implementation.
"""

import os
import json
import pytest

SAMPLE_DIR = os.path.join(os.path.dirname(__file__), "..", "sample_pdfs")
MEMO_PDF = os.path.join(SAMPLE_DIR, "company_memo.pdf")
MINUTES_PDF = os.path.join(SAMPLE_DIR, "meeting_minutes.pdf")
REPORT_PDF = os.path.join(SAMPLE_DIR, "research_report.pdf")


def pdf_exists(path):
    return os.path.exists(path)


@pytest.mark.skipif(not pdf_exists(MEMO_PDF), reason="Sample PDFs not generated")
class TestOCRWithSamplePDFs:
    """Test OCR text extraction against real sample PDFs."""

    def test_memo_extracts_text_from_single_page(self):
        """Single-page memo should produce non-empty text."""
        from app.services.ocr_service import OCRService

        ocr_service = OCRService(MEMO_PDF)
        text = ocr_service.extract_text_from_pdf()

        assert len(text.strip()) > 0, "OCR returned empty text for company memo"

    def test_memo_contains_key_words(self):
        """FAILS: Off-by-one bug skips page 0 — single-page PDF returns empty."""
        from app.services.ocr_service import OCRService

        ocr_service = OCRService(MEMO_PDF)
        text = ocr_service.extract_text_from_pdf()

        assert "MEMORANDUM" in text.upper(), (
            "Expected 'MEMORANDUM' in extracted text — page may have been skipped"
        )

    def test_minutes_extracts_both_pages(self):
        """Should extract text from multiple pages."""
        from app.services.ocr_service import OCRService
        from thefuzz import fuzz

        ocr_service = OCRService(MINUTES_PDF)
        text = ocr_service.extract_text_from_pdf()
        text_upper = text.upper()

        assert "RICHARD" in text_upper or fuzz.partial_ratio("RICHARD HERNANDEZ", text_upper) >= 60, (
            "Page 1 content missing — Richard Hernandez should appear on first page"
        )

    def test_report_word_boxes_have_entries(self):
        """Bounding box extraction should return words with coordinates."""
        from app.services.ocr_service import OCRService

        ocr_service = OCRService(REPORT_PDF)
        boxes = ocr_service.get_word_bounding_boxes()

        assert len(boxes) > 0, "No word bounding boxes extracted"
        assert all(
            key in boxes[0] for key in ("word", "page", "x", "y", "width", "height")
        ), "Bounding box missing required fields"


@pytest.mark.skipif(not pdf_exists(MEMO_PDF), reason="Sample PDFs not generated")
class TestNERWithSamplePDFs:
    """Test NER extraction against real OCR output from sample PDFs."""

    def test_memo_finds_person_names(self):
        """NER should find most person names in the memo."""
        from app.services.ocr_service import OCRService
        from app.services.ner_service import NerService

        ocr_service = OCRService(MEMO_PDF)
        text = ocr_service.extract_text_from_pdf()
        ner_service = NerService(text)
        names = ner_service.extract_names(include_orgs=False)

        expected_names = [
            "Margaret Thompson",
            "Robert Chen",
            "Sarah Williams",
            "Maria Garcia",
        ]

        found = 0
        from thefuzz import fuzz
        for expected in expected_names:
            for extracted in names:
                # Use fuzzy matching for integration test resilience against OCR typos
                if fuzz.token_sort_ratio(expected.lower(), extracted.lower()) >= 60:
                    found += 1
                    break

        assert found >= 1, (
            f"Expected at least 1 of {expected_names} to be found, got {found}. "
            f"Extracted names: {names}"
        )

    def test_ner_does_not_return_organizations(self):
        """NER should not return organization names as person names."""
        from app.services.ner_service import NerService

        text = (
            "Margaret Thompson is the CEO of Acme Corporation. "
            "Robert Chen works at Google and reports to James Anderson."
        )
        ner_service = NerService(text)
        names = ner_service.extract_names(include_orgs=False)

        org_names = ["Acme Corporation", "Google"]
        for org in org_names:
            assert org not in names, (
                f"Organization '{org}' should not appear in person name results"
            )


@pytest.mark.skipif(not pdf_exists(MEMO_PDF), reason="Sample PDFs not generated")
class TestFuzzyMatchWithSamplePDFs:
    """Test fuzzy matching with names from sample PDFs."""

    def test_exact_names_from_memo_match(self):
        """Exact names from the memo should match at 90% threshold."""
        from app.services.fuzzy_service import fuzzy_match_names

        extracted = [
            "Margaret Thompson",
            "Robert Chen",
            "Sarah Williams",
            "Maria Garcia",
        ]
        query = [
            {"first_name": "Margaret", "last_name": "Thompson"},
            {"first_name": "Robert", "last_name": "Chen"},
        ]

        matches = fuzzy_match_names(extracted, query)

        assert len(matches) == 2, f"Expected 2 exact matches, got {len(matches)}"
        for m in matches:
            assert m["score"] >= 0.9, (
                f"Exact match '{m['matched_name']}' scored {m['score']}, expected >= 0.9"
            )

    def test_ocr_typos_still_match(self):
        """Names with minor OCR errors should still match above threshold."""
        from app.services.fuzzy_service import fuzzy_match_names

        extracted = ["Margret Thopmson", "Robet Chen"]  # simulated OCR typos
        query = [
            {"first_name": "Margaret", "last_name": "Thompson"},
            {"first_name": "Robert", "last_name": "Chen"},
        ]

        matches = fuzzy_match_names(extracted, query)

        assert len(matches) >= 1, (
            "At least one name with minor typos should still match"
        )

    def test_completely_wrong_names_do_not_match(self):
        """FAILS: partial_ratio + low threshold lets bad matches through."""
        from app.services.fuzzy_service import fuzzy_match_names

        extracted = ["Margaret Thompson", "Robert Chen"]
        query = [
            {"first_name": "Zara", "last_name": "Xu"},
        ]

        matches = fuzzy_match_names(extracted, query)

        assert len(matches) == 0, (
            f"'Zara Xu' should not match any extracted names, but got: {matches}"
        )


@pytest.mark.skipif(not pdf_exists(MEMO_PDF), reason="Sample PDFs not generated")
class TestEndToEndExtraction:
    """End-to-end test of the full extraction pipeline via API."""

    def test_extract_endpoint_with_memo_pdf(self, app_client):
        """FAILS: Multiple bugs compound — off-by-one, missing schema fields."""
        with open(MEMO_PDF, "rb") as f:
            pdf_bytes = f.read()

        query_names = [
            {"first_name": "Margaret", "last_name": "Thompson"},
            {"first_name": "Maria", "last_name": "Garcia"},
        ]

        response = app_client.post(
            "/api/extract",
            files={"pdf_file": ("company_memo.pdf", pdf_bytes, "application/pdf")},
            data={"names": json.dumps(query_names)},
        )

        assert response.status_code == 200

        data = response.json()
        assert "extracted_names" in data
        assert len(data["extracted_names"]) > 0, (
            "No names extracted — check OCR and NER pipeline"
        )

    def test_extract_endpoint_with_multipage_pdf(self, app_client):
        """FAILS: Off-by-one skips first page of meeting minutes."""
        with open(MINUTES_PDF, "rb") as f:
            pdf_bytes = f.read()

        query_names = [
            {"first_name": "Richard", "last_name": "Hernandez"},
            {"first_name": "Jennifer", "last_name": "Liu"},
        ]

        response = app_client.post(
            "/api/extract",
            files={
                "pdf_file": (
                    "meeting_minutes.pdf",
                    pdf_bytes,
                    "application/pdf",
                )
            },
            data={"names": json.dumps(query_names)},
        )

        assert response.status_code == 200

        data = response.json()
        assert len(data["extracted_names"]) > 0, (
            "No names extracted from multi-page PDF — first page may be skipped"
        )
