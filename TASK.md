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

| # | File                              | Bug Description                                              | How You Fixed It |
|---|-----------------------------------|--------------------------------------------------------------|------------------|
| 1 | app/services/embedding_service.py | ModuleNotFoundError: No module named 'sentence_transformers' |                  |
| 2 |                                   |                                                              |                  |
| 3 |                                   |                                                              |                  |
| 4 |                                   |                                                              |                  |
| 5 |                                   |                                                              |                  |
| 6 |                                   |                                                              |                  |
| 7 |                                   |                                                              |                  |
| 8 |                                   |                                                              |                  |

(add more rows as needed)

### Architecture & Design Improvements

| # | What You Changed | Why |
|---|------------------|-----|
| 1 |                  |     |
| 2 |                  |     |
| 3 |                  |     |

(add more rows as needed)

### Engineering Best Practices Added

| # | Practice | Where / How You Implemented It |
|---|----------|-------------------------------|
| 1 |          |                               |
| 2 |          |                               |
| 3 |          |                               |

(add more rows as needed)

## Submission

Create a branch and make a pull request to this repository.


## EXTRACT API TESTING

python3 -m uvicorn app.api.extract:router --host 0.0.0.0 --port 8000 & sleep 5 && curl -X POST http://localhost:8003/extract -F "pdf_file=@/Users/gama/repo-AIarchitect/sample_pdfs/company_memo.pdf" -F 'names=[{"first_name": "Sarah", "last_name": "Williams"}]' && kill $!

python3 -m uvicorn app.main:app --host 127.0.0.1 --port 8000

Note: In the code, app/api/extract.py has a __main__ block that runs uvicorn.run(router, ...) on port 8003, but it only includes the extraction router. Running app.main:app is preferred as it includes all routes.

curl -X POST http://127.0.0.1:8000/api/extract \
  -F "pdf_file=@sample_pdfs/company_memo.pdf" \
  -F "names=[{'first_name': 'Margaret', 'last_name': 'Thompson'}]"


## RAG API TESTING

curl -X GET http://localhost:8000/health

curl -X POST "http://localhost:8000/api/ingest" \
       -F "pdf_file=@sample_pdfs/meeting_minutes.pdf"

curl -X POST "http://localhost:8000/api/ask" \
    -H "Content-Type: application/json" \
    -d '{"question": "What were the main topics discussed in the meeting?"}'


## FINDINGS, IMPROVEMENTS, PRACTICES

See file Task-Findings-Practices.md



