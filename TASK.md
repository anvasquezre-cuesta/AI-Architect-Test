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
| 1 | `app/services/ocr_service.py` | Off-by-one: `range(1, len(doc))` skipped page 0, so single-page PDFs returned empty text | Iterate `range(len(doc))` starting at page 0 |
| 2 | `app/services/ocr_service.py` | PDF documents were never closed (leaked file handles) | `try/finally: doc.close()` in both OCR methods |
| 3 | `app/services/ocr_service.py` | Word bounding boxes returned in OCR image space (150 DPI pixels) instead of PDF space (72 points/inch) | Multiply all coordinates by `72 / dpi` |
| 4 | `app/services/ner_service.py` | `extract_names` / `extract_names_with_positions` also returned `ORG` entities as person names | Filter entities to `PERSON` only |
| 5 | `app/services/bbox_service.py` | Case-sensitive word comparison — OCR `'JOHN'` never matched NER `'John'` | Normalize (lowercase + strip punctuation) before comparing |
| 6 | `app/services/bbox_service.py` | Early `break` always matched only the first occurrence of each word; parts of a name could merge from unrelated locations/pages | Match name parts as a *consecutive same-page sequence* and return every occurrence |
| 7 | `app/services/fuzzy_service.py` | `SIMILARITY_THRESHOLD = 70` instead of the required 90 | Set to 90 (env-configurable) |
| 8 | `app/services/fuzzy_service.py` | `fuzz.partial_ratio` inflated scores — `'Jo Sm'` matched `'John Smith'` ~100% | Use full-string `fuzz.ratio`, case-insensitive |
| 9 | `app/services/vector_service.py` | Point IDs from `enumerate()` restarted at 0, so each new document overwrote the previous one | UUIDv4 point IDs |
| 10 | `app/services/rag_service.py` | Prompt contained escaped `{{question}}` — the literal string `'{question}'` was sent to the LLM | Proper interpolation via `str.format(context=…, question=…)` |
| 11 | `app/services/rag_service.py` | `chunk_text` cut words in half (naive character slicing) | Word-aware chunker that keeps whitespace attached so `''.join(chunks) == text` |
| 12 | `app/services/embedding_service.py` | `SentenceTransformer` model reloaded on **every** call (very slow, memory churn) | Model loaded once via cached provider, injectable |
| 13 | `app/api/extract.py` | No file validation — any file type was processed | 400 on wrong extension/content-type/magic bytes/empty; 413 on oversize |
| 14 | `app/api/extract.py` | `fuzzy_matches` computed but silently dropped (missing from `ExtractionResponse`) | Added `fuzzy_matches: list[FuzzyMatch]` to the schema |
| 15 | `app/models/schemas.py` | `BoundingBox` had no page number, so multi-page results were ambiguous | Added `page_number` field |
| 16 | `app/models/schemas.py` | `RAGResponse` dropped the `sources` returned by the service | Added `sources: list[str]` |
| 17 | `app/main.py` | No health check endpoint | Added `GET /health` (used by Docker healthcheck) |
| 18 | `app/api/extract.py` | Unvalidated `json.loads(names)` — malformed JSON crashed with 500 | Parse + validate with Pydantic `NamePair`, return 400 with details |
| 19 | `app/services/ner_service.py` | NER filtered only the `PERSON` label; Spanish/multilingual spaCy models tag people as `PER`, so they returned zero names (found live-testing the `spanish_precision` profile) | Accept both labels (`PERSON`, `PER`) |
| 20 | `app/services/fuzzy_service.py` | Accented names never matched their unaccented query ("Maria Gonzalez" vs "María González" scored ~86%, below the 90 threshold) | Accent-insensitive normalization (Unicode NFD, strip combining marks) before comparing |

### Architecture & Design Improvements

| # | What You Changed | Why |
|---|------------------|-----|
| 1 | All services refactored into classes implementing ABC interfaces (`app/services/interfaces.py`: `OCRService`, `NERService`, `BoundingBoxService`, `FuzzyMatcher`, `EmbeddingService`, `VectorStore`, `AnswerGenerator`) | SOLID: depend on abstractions — OCR engine, NER model, embedding model, vector DB, and LLM are all swappable without touching consumers; enables mocking in tests |
| 2 | Composition root (`app/core/providers.py`) with lazy, cached construction of spaCy pipeline, SentenceTransformer, and Qdrant client | Removes import-time side effects and hardcoded instantiation from services; heavy objects created once; dependency injection via constructors |
| 3 | Centralized configuration (`app/core/config.py`, pydantic-settings) | Eliminates hardcoded values (host, ports, DPI, thresholds, model names, timeouts) — everything overridable via environment / `.env` |
| 4 | Module-level function aliases kept as the public API (`extract_text_from_pdf = _default_service.extract_text_from_pdf`) | Preserves a stable, simple facade (and the seam the test-suite patches) while the implementation lives in classes |
| 5 | Layered error model (`app/core/exceptions.py`) mapped to HTTP codes at the API boundary (400/413/422/502/503) | Clear failure semantics; no stack traces leaked to clients |
| 6 | RAG chunking, retrieval and LLM call isolated in `RAGService` with injectable embedder/store | Single responsibility; LLM routing or reranking can be added without touching endpoints |
| 7 | Model profiles (`app/core/profiles.py` + `GET /api/profiles`): clients select `english_fast` / `spanish_precision` / `multilingual` / `llm_precision` instead of internal model names; each profile bundles OCR language + NER engine + embedding model + a Qdrant collection per embedding space, built lazily and cached | Multi-language support without leaking implementation details into the API (like "fast"/"thinking" tiers on LLM APIs); collections are per embedding model because vectors from different models are not comparable |
| 8 | `LLMNERService` (zero-shot LLM NER behind the same `NERService` ABC, used by the `llm_precision` profile) | Statistical models mislabel systematically (es sm+md tag "La señora Lucía Fernández" as MISC — found live-testing); the LLM handles honorifics/mixed languages/noisy OCR. Fails fast with a clear 502 when no API key is configured — no silent fallback |
| 9 | `EnsembleNERService`: `spanish_precision` unions the LARGE Spanish model (`es_core_news_lg`) + `xx` instead of `es_core_news_sm` alone | Decision trail: es sm ❌ and md ❌ mislabel "La señora Lucía Fernández" as MISC; **lg ✅** tags her PER — a pure config upgrade, cheapest option first. Union with xx covers remaining blind spots; 90% fuzzy threshold filters ensemble noise. A Spanish transformer NER was evaluated and deferred (config fix sufficed). Engine upgraded without any API change |

### Engineering Best Practices Added

| # | Practice | Where / How You Implemented It |
|---|----------|-------------------------------|
| 1 | Structured logging | `app/core/logging.py`; namespaced loggers in every service and endpoint, configurable `LOG_LEVEL` |
| 2 | Configuration via environment | `app/core/config.py` (pydantic-settings): Qdrant host/port, DPI, thresholds, model names, timeouts, upload size limit |
| 3 | Resilience: timeouts + retries with exponential backoff | `app/core/retry.py`; LLM calls have a timeout and 3 retries; Qdrant timeout configured on the client; graceful extractive fallback when no LLM key is set |
| 4 | Input validation | PDF magic-bytes/content-type/size checks, Pydantic validation of `names` and `question`, `Field` constraints on schemas |
| 5 | Proper resource management | Temp files removed in `finally`; PDF documents always closed |
| 6 | Error handling with meaningful HTTP codes | 400 bad input, 413 too large, 422 schema, 502 LLM failure, 503 vector store down |
| 7 | Containerization hardening | Non-root user, models baked at build time, layer-cached deps, Docker `HEALTHCHECK`, pinned Qdrant version, persistent volume |
| 8 | Additional tests (bonus) | `tests/test_additional.py` — 21 tests: validation, chunker edge cases, retry behavior, bbox occurrences/pages, config overrides, health |

## Submission

Create a branch and make a pull request to this repository.
