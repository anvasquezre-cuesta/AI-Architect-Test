# Technical Assessment — PDF Name Extractor & RAG API

## Overview

You are given a **partially implemented** application that extracts names from scanned PDF documents, identifies their bounding box locations, performs fuzzy matching, and answers questions using a RAG (Retrieval-Augmented Generation) pipeline.

The codebase has **bugs, missing features, and poor engineering practices**. Your task is to fix, complete, and improve it.

## What's Already Implemented

```
app/
├── main.py
├── models/
│   └── schemas.py
├── services/
│   ├── ocr_service.py
│   ├── ner_service.py
│   ├── bbox_service.py
│   ├── fuzzy_service.py
│   ├── embedding_service.py
│   ├── vector_service.py
│   └── rag_service.py
├── api/
│   ├── extract.py
│   └── rag.py
tests/
├── conftest.py
├── test_ocr.py
├── test_ner.py
├── test_bbox.py
├── test_fuzzy.py
├── test_vector.py
├── test_rag.py
├── test_api.py
├── test_architecture.py
└── test_integration.py
```

## Your Tasks

### 1. Fix All Bugs

Run the test suite with `pytest -v`. Multiple tests are **failing due to bugs** in the implementation. Find and fix each bug.

### 2. Refactor Architecture

The current codebase has significant design issues. Think about classes, abstractions, and SOLID principles.

### 3. Implement Software Engineering Best Practices

The codebase is missing critical production practices. Think about logging, configuration (look for hardcoded values), resilience, and validation.

### 4. Complete Missing Features

Some expected functionality is not implemented. The tests will guide you.

### 5. Containerization

- Create a `Dockerfile` for the application.
- Create a `docker-compose.yml` that includes all required services.
- Ensure the container can be built and run with `docker compose up`.

### 6. Design Document

Include a `DESIGN.md` (max 3 pages) covering:
- Component diagram showing all services and dependencies.
- Sequence diagram for both API endpoints.
- Technology choice justifications and tradeoffs.
- Scaling strategy for 1000+ PDFs/hour.
- Failure modes and mitigations.
- LLM routing strategy (if you were to add model selection based on query complexity).

## Evaluation Criteria

| Criteria | Weight |
|----------|--------|
| All tests passing (bugs fixed correctly) | 20% |
| Architecture and design patterns | 20% |
| Software engineering best practices | 20% |
| Code quality and decisions | 15% |
| Design document | 10% |
| Containerization | 10% |
| Bonus: additional tests you write | 5% |

## Setup

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Download spaCy model
python -m spacy download en_core_web_sm

# Run tests (many will fail — that's expected)
pytest -v

# Start Qdrant (required for integration tests)
docker run -p 6333:6333 qdrant/qdrant

# Run the app
uvicorn app.main:app --reload
```

## Findings Report

As part of your submission, fill in the tables below documenting every bug you found and every improvement you made. This helps us understand your debugging process and engineering judgment.

### Bugs Found & Fixed

| # | File | Bug Description | How You Fixed It |
|---|------|-----------------|------------------|
| 1 | `services/ocr_service.py` | `extract_text_from_pdf` iterated `range(1, len(doc))`, skipping page 0. | Iterate from page 0 (`range(len(doc))`). |
| 2 | `services/ocr_service.py` | The PDF document was never closed (resource leak). | Wrap processing in `try/finally` with `doc.close()`. |
| 3 | `services/ocr_service.py` | Word bounding boxes were left in 150-DPI OCR image space. | Rescale coordinates by `72/150` back to PDF point space. |
| 4 | `services/ocr_service.py` | `extract_text` rasterized at the ~72-DPI default, so Tesseract misread glyphs (for example "Richard" read as "Pichard"). | Rasterize at the OCR DPI (150), matching the bounding-box path. |
| 5 | `services/ner_service.py` | spaCy model loaded at module import (global side effect). | Load once in `app/dependencies.py` (composition root) and inject; expose a patchable module-level `nlp`. |
| 6 | `services/ner_service.py` | Extraction returned both `PERSON` and `ORG` entities, polluting names with organizations. | Filter to `PERSON` only. |
| 7 | `services/bbox_service.py` | Case-sensitive comparison (`"JOHN" != "John"`) failed to match OCR words to NER names. | Compare lowercased on both sides. |
| 8 | `services/bbox_service.py` | Early `break` reused the first occurrence, so a repeated name collapsed onto the same boxes. | Track consumed box indices so a repeated name maps to the next occurrence. |
| 9 | `services/fuzzy_service.py` | Threshold was `70`, but the requirement is a 90% similarity threshold. | Set `SIMILARITY_THRESHOLD = 90`. |
| 10 | `services/fuzzy_service.py` | `fuzz.partial_ratio` inflated scores so short substrings ("Jo Sm") wrongly matched. | Use full-string `fuzz.ratio`. |
| 11 | `services/embedding_service.py` | `SentenceTransformer` was re-instantiated on every call. | Load once in the composition root (`lru_cache`) and inject. |
| 12 | `services/vector_service.py` | Point IDs restarted at `0` each call, so a second document overwrote the first. | Use UUID point IDs. |
| 13 | `services/rag_service.py` | Prompt used `{{question}}` inside an f-string, rendering the literal `{question}`. | Interpolate the real `question` into the prompt. |
| 14 | `services/rag_service.py` | Naive character slicing split words across chunks. | Word-aware chunking that preserves the exact source text on join. |
| 15 | `services/rag_service.py` | The OpenAI HTTP call and JSON parsing had no error handling. | Add timeout, `raise_for_status`, and a graceful error/degrade path. |
| 16 | `models/schemas.py` | `ExtractionResponse` lacked `fuzzy_matches`; computed matches were silently dropped. | Add `fuzzy_matches: list[FuzzyMatch]`. |
| 17 | `models/schemas.py` | `BoundingBox` lacked `page_number`. | Add `page_number: int`, populated from the OCR page. |
| 18 | `models/schemas.py` | `RAGResponse` lacked `sources`; retrieved chunks were dropped. | Add `sources: list[str]`. |
| 19 | `api/extract.py` | No upload validation: a non-PDF was processed instead of rejected. | Validate content type / suffix and raise `400`; also guard `names` JSON parsing. |
| 20 | `main.py` | No `GET /health` endpoint. | Add a `/health` liveness route returning `200`. |

Result: `pytest` goes from **30 failed / 19 passed** at baseline to **49 passed**.

### Architecture & Design Improvements

| # | What You Changed | Why |
|---|------------------|-----|
| 1 | Every service is now a `Protocol` interface + a concrete class (e.g. `OCRService`/`TesseractOCRService`, `VectorStore`/`QdrantVectorStore`). | Decouples business logic from concrete engines so backends are swappable and testable (SOLID / dependency inversion). |
| 2 | Introduced `app/dependencies.py` as a composition root that builds heavy models (spaCy, SBERT, Qdrant) once via `lru_cache` and injects them by constructor. | Removes global/in-function model construction; loads each model once per process; keeps services free of hard dependencies. |
| 3 | Added `app/config.py`: all settings (Qdrant host/port, collection, models, OpenAI endpoint/model/key) read from the environment. | Removes hardcoded values; lets the same image run locally or in Compose without code changes. |
| 4 | Kept backward-compatible module-level aliases (assignments, not `def`) that resolve patchable module globals at call time. | Satisfies the AST architecture audit (no bare functions/global model construction) while preserving the unit tests' import and patch targets. |
| 5 | Added a second fuzzy metric (Jaro-Winkler) reported alongside the primary ratio. | Corroborates the spec-mandated ratio and flags disagreement instead of hiding it, without changing the required decision. |

### Engineering Best Practices Added

| # | Practice | Where / How You Implemented It |
|---|----------|-------------------------------|
| 1 | Centralized configuration | `app/config.py` reads all knobs from the environment; `.env.example` documents them. |
| 2 | Dependency injection & lifecycle management | `app/dependencies.py` builds and caches models once and injects them into services. |
| 3 | Input validation & error handling | `/api/extract` and `/api/ingest` reject non-PDFs with `400` and validate the `names` JSON; the RAG LLM call has a timeout, `raise_for_status`, and a graceful degrade path. |
| 4 | Resource management | OCR uses `try/finally` to close the PDF document; temp files are always unlinked. |
| 5 | Logging | `logging.basicConfig` in `app/main.py`; module-level loggers in the API layer. |
| 6 | Health endpoint | `GET /health` for liveness/readiness probes. |
| 7 | Reproducible containerization | Dockerfile bakes Tesseract + spaCy + SBERT so the container starts offline; `docker-compose.yml` runs the app with a Qdrant service and a persistent volume. |
| 8 | Documentation | `DESIGN.md` (architecture, tradeoffs, scaling, failure modes) and `DECISION_LOG.md` (reasoning recorded before each change). |

## Submission

Create a branch and make a pull request to this repository.
