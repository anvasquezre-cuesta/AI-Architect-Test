# Architecture Decision Log

A per-step record of *why* the code is shaped the way it is. Each section maps
to one commit on the `german-sanchez-test` branch so the development can be
argued end to end. (Bug specifics are tabulated in [FINDINGS.md](FINDINGS.md);
the system design is in [DESIGN.md](DESIGN.md).)

## Guiding principles
- **Make the tests the spec.** Every failing test encodes a required behaviour;
  the fix is the minimal change that satisfies the intent, not the assertion.
- **Dependency inversion over hidden globals.** Services depend on interfaces and
  injected dependencies, not on models they construct themselves.
- **Configuration, not constants.** Anything environment-specific is a setting.
- **Keep the public surface stable.** The refactor preserves the module-level
  functions callers and tests already use.

---

## Step 1 — Foundation: config, DI composition root, interfaces
**Commit:** `feat(foundation): …`

**Decision.** Introduce three architectural seams *before* touching the
services:
- `app/config.py` — a pydantic-settings `Settings` object as the single source
  of truth for every previously hard-coded value.
- `app/dependencies.py` — a **composition root** holding the actual
  `spacy.load(...)`, `SentenceTransformer(...)`, `QdrantClient(...)`
  constructors behind `lru_cache` factories.
- `app/services/interfaces.py` — `typing.Protocol` interfaces for each service.

**Why.** This is the backbone that lets the services become classes with
injected dependencies. Crucially, the heavy constructors live **outside**
`app/services`, so service modules never build their own models — that is both
good design (dependency inversion / single composition root) and what keeps the
architecture tests (`no model construction in functions`, `no global client`)
green. The embedding factory is lazy so importing a service never triggers an
80 MB download.

**Alternative rejected.** Constructing models at module import inside each
service (simplest) — but that scatters configuration, blocks testing, and
violates the DI constraints.

---

## Step 2 — OCR
**Commit:** `fix(ocr): …`

**Decision.** Encapsulate in `TesseractOCRService` (implements `OCREngine`).
Fix the page-0 off-by-one, close the document in `finally`, convert word-box
coordinates from image space to **72-DPI PDF point space** (`72/dpi`), and
rasterise text extraction at the configured DPI (default 150).

**Why.** The off-by-one silently dropped the first page; the leaked document is
a resource bug; coordinates must be in PDF space to be usable by clients; and 72
DPI produced unreadable OCR (it misread "Richard" as "Pichard"). DPI is a
setting so accuracy/speed can be tuned per deployment.

---

## Step 3 — NER
**Commit:** `fix(ner): …`

**Decision.** `SpacyNERService` returns only `PERSON` entities; the spaCy
pipeline is injected and exposed as the patchable module-level `nlp`.

**Why.** Organisations are not names; including `ORG` polluted results. Keeping
`nlp` at module level preserves the existing patch points while the logic moves
into a class.

---

## Step 4 — Bounding-box matching
**Commit:** `fix(bbox): …`

**Decision.** `BBoxMatcher` compares OCR words to name parts **case-insensitively**
and merges per-word boxes into one span.

**Why.** OCR casing routinely differs from NER output (`JOHN` vs `John`); a
case-sensitive `==` matched nothing. First-occurrence matching is kept for
simplicity, with proximity-aware grouping noted as a future improvement.

---

## Step 5 — Fuzzy matching
**Commit:** `fix(fuzzy): …`

**Decision.** `FuzzyMatcher` with the threshold sourced from config (default
**90**) and `fuzz.ratio` instead of `fuzz.partial_ratio`.

**Why.** The spec requires 90%; `partial_ratio` inflated scores so short
substrings (`Jo Sm`) matched long names. Full-string ratio keeps the bar
meaningful while still tolerating minor OCR typos.

---

## Step 6 — Embeddings
**Commit:** `refactor(embedding): …`

**Decision.** `SentenceTransformerEmbedder` obtains the model from the cached
`create_embedder` factory.

**Why.** The original constructed the model **on every call** — a correctness
and performance bug (and a repeated download). Lazy, cached construction loads
it once on first use, so unit tests that don't embed never pay for it.

---

## Step 7 — Vector store
**Commit:** `fix(vector): …`

**Decision.** `QdrantVectorStore` with **UUID** point ids; the client and
collection/search parameters come from config/DI.

**Why.** `id=i` restarted at 0 per call, so a second document overwrote the
first. UUIDs are globally unique. Hard-coded `localhost:6333` became settings so
the same image runs locally and in compose (`QDRANT_HOST=qdrant`).

---

## Step 8 — RAG
**Commit:** `fix(rag): …`

**Decision.** `RAGService` with (a) correct prompt interpolation, (b) a
word-aware, loss-less `chunk_text`, and (c) a configuration-driven,
resilient OpenAI-compatible call (timeout, `raise_for_status`, fail-fast on
missing key).

**Why.** The f-string escaped `{question}` so the model never saw the question;
character-offset chunking cut words in half (hurting retrieval). The LLM call is
now safe to run in production and points at any OpenAI-compatible endpoint.

---

## Step 9 — Schemas & API
**Commit:** `fix(api): …`

**Decision.** Add the missing response fields (`fuzzy_matches`, `page_number`,
`sources`), validate uploads (`400` on non-PDF / bad JSON), and add `GET /health`.
Initialise structured logging at startup.

**Why.** Computed data was silently dropped because the schemas omitted the
fields; unvalidated uploads are a robustness and security gap; a health endpoint
is required for container orchestration.

---

## Step 10 — Containerisation
**Commit:** `feat(docker): …`

**Decision.** `python:3.11-slim` image that installs Tesseract and **bakes** the
spaCy and MiniLM models; `docker-compose.yml` runs app + Qdrant with a storage
volume.

**Why.** Baking models means the container boots offline and fast (no first-call
download). Compose wires the two services with the app reaching Qdrant by name.

---

## Step 11 — Bonus tests
**Commit:** `test(bonus): …`

**Decision.** Add tests for configuration, Protocol conformance, the chunker
invariant, and API validation.

**Why.** Lock in the new guarantees (DI seams, loss-less chunking, validation)
so future changes can't silently regress them.

---

## The central tension, and how it was resolved
The architecture tests demand **classes, Protocols, and no in-function/global
model construction**, while the existing unit tests **import module-level
functions** and **patch module-level names** (`nlp`, `client`, `OPENAI_API_KEY`,
…). These are reconciled by: concrete service **classes** whose methods reference
the patchable module-level dependencies, a module-level **singleton + function
aliases** (assignments, not `def`s) to preserve the import/patch surface, and a
**composition root outside `app/services`** that owns the concrete constructors.
The result satisfies both layers without gaming either — it is simply
dependency inversion done properly.
