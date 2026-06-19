# Changelog

All notable changes are documented here. Format based on
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased] — german-sanchez-test

### Fixed
- **OCR:** first page was skipped (`range(1, len(doc))`); PDF document was never
  closed; word-box coordinates were in image space; text extraction ran at 72
  DPI, causing OCR errors.
- **NER:** organisations (`ORG`) were returned as person names.
- **Bounding boxes:** OCR words were matched to names case-sensitively, so
  different casing never matched.
- **Fuzzy matching:** threshold was 70 (spec: 90) and `partial_ratio` inflated
  scores, allowing bad matches.
- **Vector store:** point ids restarted at 0 each call, so a second document
  overwrote the first.
- **RAG:** the prompt never interpolated the question (escaped `{question}`);
  the chunker split words mid-token.
- **API:** `ExtractionResponse` dropped `fuzzy_matches`, `BoundingBox` lacked
  `page_number`, `RAGResponse` lacked `sources`.

### Added
- `app/config.py` (pydantic-settings) — centralised, env-overridable configuration.
- `app/dependencies.py` — DI composition root with cached model/client factories.
- `app/services/interfaces.py` — `Protocol` interfaces for every service.
- `GET /health` liveness endpoint.
- PDF upload validation (magic bytes / content-type) and `names` JSON validation.
- `Dockerfile`, `docker-compose.yml`, `.dockerignore` (app + Qdrant; models baked in).
- `tests/test_bonus.py` — configuration, Protocol conformance, chunker invariant,
  API validation.
- `DESIGN.md`, `DECISIONS.md`, `FINDINGS.md`, project `README.md`, `.env.example`.

### Changed
- All services refactored from loose module functions into classes implementing
  their Protocols, with dependencies injected from the composition root. The
  module-level function aliases are preserved for backwards compatibility.
- Structured logging configured at application startup.
- LLM client is now OpenAI-compatible and fully configuration-driven, with a
  request timeout, `raise_for_status`, and a clear failure when no key is set.

### Test status
- **59 passing** (49 original suite incl. architecture + integration, 10 bonus).
