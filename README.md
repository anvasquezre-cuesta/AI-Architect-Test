# PDF Name Extractor & RAG API

A FastAPI service that extracts person names and their bounding boxes from
scanned PDFs, fuzzy-matches them against a supplied list, and answers questions
over ingested PDFs using a Retrieval-Augmented Generation (RAG) pipeline.

> This is German Sanchez's submission for the AI Architect technical assessment.
> The original brief is in [TASK.md](TASK.md). What changed and why:
> [FINDINGS.md](FINDINGS.md) · [DECISIONS.md](DECISIONS.md) ·
> [DESIGN.md](DESIGN.md) · [CHANGELOG.md](CHANGELOG.md).

## Architecture at a glance

```
Client ─▶ FastAPI ─▶ Extraction pipeline: OCR → NER → bounding box → fuzzy match
                 └─▶ RAG pipeline:        chunk → embed → Qdrant → LLM answer

Config (pydantic-settings) + DI composition root (app/dependencies.py)
inject Tesseract, spaCy, MiniLM, Qdrant and an OpenAI-compatible LLM behind
Protocol interfaces, so every component is swappable.
```

See [DESIGN.md](DESIGN.md) for the full component and sequence diagrams.

## Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/api/extract` | Upload a PDF + `names` JSON; returns extracted names, bounding boxes and fuzzy matches (≥ 90%). |
| `POST` | `/api/ingest`  | Upload a PDF; OCRs, chunks, embeds and stores it in Qdrant. |
| `POST` | `/api/ask`     | Ask a question; returns a RAG answer with its source chunks. |
| `GET`  | `/health`      | Liveness probe. |

## Run with Docker (recommended)

```bash
export OPENAI_API_KEY=sk-...        # required only for /api/ask generation
docker compose up --build
```

The app is on http://localhost:8000 (docs at `/docs`); Qdrant on `:6333`.
The image bakes in the spaCy and MiniLM models so it starts offline.

## Run locally

```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
python -m spacy download en_core_web_sm
# Tesseract is a system binary:
#   macOS: brew install tesseract   |   Debian/Ubuntu: apt-get install tesseract-ocr

# Qdrant (needed for /api/ingest and /api/ask):
docker run -p 6333:6333 qdrant/qdrant:v1.11.0

export OPENAI_API_KEY=sk-...        # for /api/ask
uvicorn app.main:app --reload
```

Configuration is via environment variables or a `.env` file — see
[.env.example](.env.example) and `app/config.py`.

## Example

```bash
# Extract + fuzzy match
curl -X POST http://localhost:8000/api/extract \
  -F "pdf_file=@sample_pdfs/company_memo.pdf;type=application/pdf" \
  -F 'names=[{"first_name":"Margaret","last_name":"Thompson"}]'

# Ingest, then ask
curl -X POST http://localhost:8000/api/ingest \
  -F "pdf_file=@sample_pdfs/research_report.pdf;type=application/pdf"
curl -X POST http://localhost:8000/api/ask \
  -H "Content-Type: application/json" -d '{"question":"What is the report about?"}'
```

## Tests

```bash
pytest -v
```

**59 passing** — the full original suite (unit, API, architecture, integration)
plus `tests/test_bonus.py`. Integration tests use the bundled `sample_pdfs/` and
require Tesseract; they self-skip if the PDFs are absent.

## Project layout

```
app/
  config.py            # pydantic-settings configuration
  dependencies.py      # DI composition root (cached model/client factories)
  main.py              # FastAPI app + /health + logging
  models/schemas.py    # Pydantic request/response models
  api/                 # extract.py, rag.py (routers)
  services/
    interfaces.py      # Protocol interfaces
    ocr_service.py     # TesseractOCRService
    ner_service.py     # SpacyNERService
    bbox_service.py    # BBoxMatcher
    fuzzy_service.py   # FuzzyMatcher
    embedding_service.py  # SentenceTransformerEmbedder
    vector_service.py     # QdrantVectorStore
    rag_service.py        # RAGService
tests/                 # full suite incl. test_bonus.py
sample_pdfs/           # bundled scanned-style PDFs for integration tests
Dockerfile · docker-compose.yml
```
