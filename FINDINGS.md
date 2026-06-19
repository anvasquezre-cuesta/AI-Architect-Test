# Findings Report

Documents every bug fixed and improvement made. See [DECISIONS.md](DECISIONS.md)
for the rationale behind each change and [DESIGN.md](DESIGN.md) for the system
design.

## Bugs Found & Fixed

| # | File | Bug Description | How You Fixed It |
|---|------|-----------------|------------------|
| 1 | `app/services/ocr_service.py` | `extract_text_from_pdf` iterated `range(1, len(doc))`, skipping page 0 (single-page PDFs returned nothing). | Iterate `range(len(doc))`. |
| 2 | `app/services/ocr_service.py` | The `fitz` document was opened but never closed (handle leak). | Close it in a `try/finally`. |
| 3 | `app/services/ocr_service.py` | Word-box coordinates were returned in 150-DPI image space, not PDF space. | Scale every coordinate by `72/dpi` to PDF point space. |
| 4 | `app/services/ocr_service.py` | Text extraction rendered at the default 72 DPI, producing OCR errors (e.g. "Richard"→"Pichard"). | Rasterise at the configured DPI (default 150). |
| 5 | `app/services/ner_service.py` | `extract_names`/`extract_names_with_positions` returned `ORG` entities as names. | Filter to `PERSON` only. |
| 6 | `app/services/bbox_service.py` | OCR words matched name parts with case-sensitive `==`, so `JOHN` ≠ `John`. | Compare lower-cased on both sides. |
| 7 | `app/services/fuzzy_service.py` | `SIMILARITY_THRESHOLD` was 70, not the required 90. | Source threshold from config (default 90). |
| 8 | `app/services/fuzzy_service.py` | `fuzz.partial_ratio` inflated scores, letting substrings match. | Use `fuzz.ratio` (full-string). |
| 9 | `app/services/vector_service.py` | `store_document_chunks` used `id=i` (restarts at 0), so new documents overwrote old ones. | Use `uuid.uuid4()` point ids. |
| 10 | `app/services/rag_service.py` | Prompt f-string escaped the placeholder (`{{question}}`), so the question was never inserted. | Interpolate the real `question`. |
| 11 | `app/services/rag_service.py` | `chunk_text` sliced on raw character offsets, cutting words in half. | Word-aware, loss-less chunker. |
| 12 | `app/models/schemas.py` / `app/api/extract.py` | `ExtractionResponse` had no `fuzzy_matches`; matches were computed then dropped. | Add `fuzzy_matches` field and populate it. |
| 13 | `app/models/schemas.py` / `app/api/extract.py` | `BoundingBox` had no `page_number`. | Add `page_number` and populate from OCR page. |
| 14 | `app/models/schemas.py` / `app/api/rag.py` | `RAGResponse` had no `sources`. | Add `sources` field and return retrieved chunks. |
| 15 | `app/api/extract.py` | No file-type validation — any upload was processed. | Reject non-PDF with `400` (magic bytes / content-type) + `400` on bad `names` JSON. |
| 16 | `app/main.py` | No `/health` endpoint. | Add `GET /health`. |

## Architecture & Design Improvements

| # | What You Changed | Why |
|---|------------------|-----|
| 1 | Refactored all 7 services from loose functions into classes implementing `Protocol` interfaces (`OCREngine`, `NERModel`, `EmbeddingModel`, `VectorStore`, `LLMClient`). | Encapsulation + swappable implementations (SOLID: SRP, DIP). |
| 2 | Added `app/dependencies.py` composition root with `lru_cache` factories that own all model/client construction, injected into services. | Dependency inversion; models built once, in one place; testable; lazy embedder load. |
| 3 | Added `app/config.py` (pydantic-settings) and removed every hard-coded constant (host/port, model names, DPI, threshold, chunk size, OpenAI settings). | Twelve-factor configuration; same image runs in every environment. |
| 4 | Preserved module-level function aliases over the new classes. | Backwards-compatible public surface; no churn for callers/tests. |

## Engineering Best Practices Added

| # | Practice | Where / How You Implemented It |
|---|----------|-------------------------------|
| 1 | Structured logging | `configure_logging()` in `app/config.py`, called at startup; per-module loggers. |
| 2 | Externalised configuration | `Settings` + `.env.example`; no secrets or hosts in code. |
| 3 | Input validation | PDF magic-byte/content-type checks and `names` JSON validation → `400`. |
| 4 | Resilience | LLM call has a timeout + `raise_for_status`; fail-fast with a clear message when `OPENAI_API_KEY` is unset. |
| 5 | Resource safety | Temp files and PDF documents always released in `finally`. |
| 6 | Containerisation | `Dockerfile` (Tesseract + baked models) and `docker-compose.yml` (app + Qdrant). |
| 7 | Health checks | `GET /health` for liveness/orchestration. |
| 8 | Extra test coverage | `tests/test_bonus.py` (config, Protocol conformance, chunker invariant, validation). |

## Test status

`pytest` → **59 passed** (49 original incl. architecture + integration, 10 bonus).
End-to-end verified: `/health`, `/api/extract` (real OCR/NER/bbox/fuzzy on the
sample PDFs), `/api/ingest` (real embeddings + Qdrant), and `/api/ask`
(graceful no-context path; clean failure when no LLM key is configured).
