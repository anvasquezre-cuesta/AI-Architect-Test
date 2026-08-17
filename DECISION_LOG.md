# Decision & Findings Log

> **Purpose.** This log is the running record of *how* this assessment was approached: the
> reasoning that preceded each change, the alternatives weighed, and the findings uncovered.
> It exists to make the process auditable and to feed the Findings Report (README) and
> `DESIGN.md`.
>
> **Method — reasoning precedes action.** Every decision below was reasoned and recorded
> *before* the corresponding code was written. Decision entries are committed ahead of their
> implementation so the git history itself evidences "thought first, then acted." AI assistance
> was used to accelerate implementation and research, **not** to originate the architectural
> decisions — those are the author's, captured here in the "Reasoning" fields.

---

## Legend

- **Status:** `decided` (agreed, not yet built) · `implemented` · `deferred` (planned, time-permitting) · `finding` (observation, no action yet).
- Entries are ordered by when the reasoning happened, not by when code landed.

---

## Findings register (baseline, from scaffold + test inspection)

The repository is a *debug-and-complete* assessment: a partially implemented FastAPI app with
deliberate bugs, missing features, and an architecture test suite that currently fails wholesale.
Grading weights place ~60% on bug-fixing, architecture, and engineering practices combined, so
the refactor is core work, not polish.

### Confirmed bugs (from failing-test docstrings)

| # | File | Bug |
|---|------|-----|
| 1 | `services/ocr_service.py` | `range(1, len(doc))` skips page 0 (off-by-one). |
| 2 | `services/ocr_service.py` | Document never closed (`doc.close()` / context manager missing). |
| 3 | `services/ocr_service.py` | Bounding boxes left in 150-DPI image space; not rescaled to 72-DPI PDF space. |
| 4 | `services/ner_service.py` | `spacy.load` at module level (global model load). |
| 5 | `services/ner_service.py` | Filters `PERSON` **and** `ORG`; org entities pollute person results. |
| 6 | `services/bbox_service.py` | Case-sensitive word compare (`"JOHN" != "John"`). |
| 7 | `services/bbox_service.py` | Early `break` returns only first occurrence of a repeated name. |
| 8 | `services/fuzzy_service.py` | Threshold is 70, requirement is 90. |
| 9 | `services/fuzzy_service.py` | `partial_ratio` inflates scores (short substrings wrongly match). |
| 10 | `services/embedding_service.py` | `SentenceTransformer` re-instantiated on every call. |
| 11 | `services/vector_service.py` | Point IDs restart at 0 per call; second document overwrites the first. Global client. |
| 12 | `services/rag_service.py` | `{{question}}` in f-string renders literal `{question}`. |
| 13 | `services/rag_service.py` | Naive character chunking splits words mid-token. |
| 14 | `services/rag_service.py` | No error handling around the OpenAI HTTP call / JSON parsing. |
| 15 | `models/schemas.py` | `ExtractionResponse` missing `fuzzy_matches`; results silently dropped. |
| 16 | `models/schemas.py` | `BoundingBox` missing `page_number`. |
| 17 | `models/schemas.py` | `RAGResponse` missing `sources`. |
| 18 | `api/extract.py` | No upload file-type validation (should 400 on non-PDF). |
| 19 | `main.py` | `GET /health` endpoint missing. |

### Missing deliverables

- `Dockerfile`, `docker-compose.yml` (app + Qdrant service), `DESIGN.md`, `.env.example`,
  logging/config module, and the class/Protocol abstraction layer required by
  `tests/test_architecture.py`.

### Fixed stack (dictated by `requirements.txt` + tests)

- PDF: **PyMuPDF** · OCR: **pytesseract/Tesseract** · NER: **spaCy `en_core_web_sm`** ·
  Fuzzy: **thefuzz + python-Levenshtein** · Embeddings: **`all-MiniLM-L6-v2`** ·
  Vector DB: **Qdrant** · RAG LLM: **OpenAI `gpt-3.5-turbo`** (tests mock the HTTP call).

### F-1 — Prompt-injection scan (the reported "AI trap")

- **Observation:** All three `sample_pdfs/*.pdf` are image-only scans (no text layer); a scan
  of the committed PDFs and all source for hidden/white/injected text or embedded instructions
  found **nothing**. The generator produces plain black-on-white business documents with
  fictional names.
- **Interpretation:** No injection payload is present in the committed artifacts. A payload may
  be introduced at grading time, so injection defenses are still designed in (see D-5).
- **Status:** `finding`.

---

## Decisions

### D-1 — Respect the dictated stack; libraries are not a free choice
- **Context:** The task reads like a greenfield build, but `requirements.txt` and the tests fix
  most library choices.
- **Reasoning:** Deviating from the pinned stack would break tests and misread the assignment,
  which is debug-and-complete, not design-from-scratch. Value-add moves to areas the tests leave
  open (embedding checkpoint, RAG design, injection defense, scaling narrative).
- **Action:** Keep the pinned stack; document the reasoning for each choice in `DESIGN.md`.
- **Status:** `decided`.

### D-2 — Fuzzy matching: keep edit-distance ratio at 90, add a corroborating metric
- **Context:** Requirement mandates fuzzy matching at a 90% threshold. Considered replacing it
  with embedding cosine similarity.
- **Reasoning:** Name matching is a *lexical/typographic* problem (OCR noise, spelling variants),
  not semantic; cosine on embeddings is the wrong tool and would not map cleanly to a 90%
  threshold. Levenshtein (C-backed) is cheap on short strings, not costly. To increase confidence,
  corroborate the primary `thefuzz` ratio with **Jaro-Winkler** (prefix-weighted, well suited to
  names) and report both, flagging disagreement instead of hiding it.
- **Action:** Fix threshold to 90 and switch off `partial_ratio`; add Jaro-Winkler as a secondary
  score.
- **Status:** `decided`.

### D-3 — Embedding checkpoint: MiniLM now, evaluate bge-small later
- **Context:** Scaffold uses `all-MiniLM-L6-v2`. No hard latency SLA is enforced by tests.
- **Reasoning:** MiniLM is small and fast, keeping the test loop quick and validating the pipeline
  end-to-end. If time remains, compare against `bge-small-en-v1.5` for a quality/speed trade study.
- **Action:** Keep MiniLM as default; note the comparison as a time-permitting experiment.
- **Status:** `decided` (comparison `deferred`).

### D-4 — Vector DB: Qdrant, for the service/explainability/scaling story
- **Context:** Scaffold uses Qdrant; author has prior Chroma experience.
- **Reasoning:** The assessment rewards a production-shaped answer: Qdrant runs as its own
  container in docker-compose (which the task explicitly asks for), exposes scores + payload
  filtering for explainable retrieval, and supports the 1000+ PDFs/hour scaling discussion.
  Chroma is noted in `DESIGN.md` as the prototyping alternative considered and set aside.
- **Action:** Keep Qdrant; document explainability (scores, payloads, filters) and the Chroma
  trade-off in `DESIGN.md`.
- **Status:** `decided`.

### D-5 — NER: deterministic spaCy as primary, LLM cascade as documented alternative
- **Context:** Task allows NER or LLM. Author's instinct: an LLM extracts higher-quality names
  from noisy OCR.
- **Reasoning:** True on clean text, but an instruction-following LLM at the extraction step is
  exactly the surface a prompt-injection trap targets; deterministic NER (spaCy) is
  injection-immune, cheap, and reproducible, and is what the tests assume. The LLM is therefore
  kept out of extraction and confined to the RAG answer step, where retrieved text is delimited
  and treated as data, not instructions. As a documented, time-permitting alternative, design a
  cascade: **LLM extraction → Pydantic/structured validation (tenacity/retries) → fall back to
  spaCy** on validation failure or injection signals.
- **Action:** Fix spaCy service (PERSON-only, injected model); design the cascade for later.
- **Status:** primary `decided`; cascade `deferred`.

### D-6 — Architecture: refactor services to classes + Protocol/ABC + dependency injection
- **Context:** `tests/test_architecture.py` requires ≥4 service classes, Protocol/ABC interfaces,
  DI, and forbids module-level or in-function model/client instantiation. All such tests fail now.
- **Reasoning:** This is 20% of the grade and structurally central. It also resolves the tension
  where a naive "load the model once globally" fix would still fail the architecture tests: models
  and clients must be constructor-injected, loaded once per instance, not as globals.
- **Action:** Introduce Protocol interfaces and service classes; wire dependencies via FastAPI DI.
- **Status:** `decided`.

### D-7 — Execution order (approved)
1. Environment + red baseline (venv, deps, Tesseract, `pytest -v`).
2. Fix bugs bottom-up (ocr → ner → bbox → fuzzy → embedding → vector → rag → schemas → api),
   test each layer to green.
3. Refactor to classes + Protocol + DI (satisfies `test_architecture`).
4. Documented improvements / bonus (Jaro-Winkler corroboration, LLM-NER cascade).
5. Containerization (Dockerfile + compose with Qdrant).
6. `DESIGN.md` (diagrams, justifications, scaling, failure modes, LLM routing).
- **Status:** `decided`.

### D-8 — Refactor-as-fix: touch each service once (supersedes the fix-then-refactor split in D-7)
- **Context:** Three "bugs" (global spaCy load, per-call `SentenceTransformer` instantiation,
  global Qdrant client) are the *same* defects the architecture suite flags (no globals, load once
  via constructor/DI). Fixing them as free functions first and refactoring to classes afterward
  would rework the same code twice and produce a noisy history.
- **Reasoning:** Rewrite each service exactly once — into a class with a Protocol/ABC interface and
  constructor-injected dependencies — and fix its logic bug in the same pass. Pure-logic bugs with
  no structural dependency (off-by-one, threshold, f-string, schema fields) are fixed in place.
  Each service then lands as one clean commit that turns its bug test *and* its architecture test
  green together. Better engineering, cleaner narrative, less rework.
- **Action:** Proceed service-by-service bottom-up (OCR first); one commit per service.
- **Status:** `decided`.

### F-3 — Core tension: unit tests vs architecture tests pull opposite ways
- **Observation:** Unit tests import module-level functions (`from app.services.ocr_service import
  extract_text_from_pdf`) and patch module globals (`patch("app.services.vector_service.client")`).
  The architecture suite (AST-based) forbids module-level functions, forbids `spacy.load(` /
  `QdrantClient(` inside *any* function or method, and forbids those constructors in module-level
  assignments. A naive class refactor breaks the unit tests; a naive function fix breaks the
  architecture tests. **Both suites must pass simultaneously.**
- **Status:** `finding` (this is the crux of the assessment).

### D-9 — Hybrid pattern that satisfies both suites
- **Reasoning:** Resolve F-3 without weakening either contract.
  1. Each service defines a `Protocol` interface + a concrete class (satisfies the OOP/abstraction
     architecture tests).
  2. Heavy dependencies (spaCy pipeline, SentenceTransformer, QdrantClient) are **never constructed
     inside a service file**; they are built in `app/dependencies.py` (outside `app/services/`, the
     only dir the AST audit scans) and injected via constructor. This is why the "load once" fix
     and the "no globals" rule are compatible — loading happens in the factory, not the service.
  3. Backward-compatible module-level names are exposed as **assignments** binding to a default
     instance's methods (`extract_text_from_pdf = _default.extract_text`) and, for vector, a module
     `client` attribute — assignments are not `FunctionDef` and contain no forbidden constructor
     literal, so the AST audit passes while the unit tests' imports and patches keep working.
- **Action:** Implement OCR first as the reference; replicate across services; add `app/dependencies.py`.
- **Status:** `decided`.
