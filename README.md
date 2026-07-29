# ML-TechTest

> **Start here: read [TASK.md](TASK.md) for full instructions.** The repo contains a partially implemented application with bugs and missing features. Your job is to fix, refactor, and complete it.

## Task Description:
The task is to build a system that can extract names and last names from a scanned
PDF document, identify their bounding box locations, and provide an API endpoint to perform fuzzy
matching between the extracted names and the names provided in the request.

Create a Branch and make a pull request to this repository.

## Requirements:

1. Use an open-source OCR library to extract text from the scanned PDF document.
2. Implement a named entity recognition (NER) or a large language model (LLM) approach to
identify and extract names and last names from the extracted text.
3. Parse the extracted text and OCR results to find the bounding box coordinates of the identified
names and last names.
4. Create a RESTful API using FastAPI or Flask that accepts a scanned PDF file and a set of
name-last name pairs as input.
5. Perform fuzzy matching (with a similarity threshold of 90%) between the extracted names and
the provided names in the API request.
6. Return the extracted names, their bounding box coordinates, and the fuzzy matching results as a
JSON response.
7. Implement a vector database (e.g., Chroma, Qdrant) to store the extracted text from the PDF
documents.
8. Use an open-source embedding model (e.g., Sentence-BERT, Universal Sentence Encoder) to
convert the extracted text into vector representations and store them in the vector database.
9. Create another RESTful API endpoint that accepts a question as input and returns an answer
based on the RAG strategy.
10. Create a Docker container image for the entire application, including the text extraction, named
entity recognition, bounding box parsing, vector database, and RAG components. Ensure that
the container can be built and run using Docker commands or a Docker Compose file. Include
instructions for running the containerized application and accessing the API endpoints.
11. Include a DESIGN.md document (max 3 pages) that covers:
    a. **Component diagram** of your solution showing all services, databases, and external dependencies and how they connect (ASCII, Mermaid, or image).
    b. **Sequence diagram** showing the request flow for both API endpoints: (1) PDF upload with name extraction and fuzzy matching, and (2) RAG question-answering (ASCII, Mermaid, or image).
    c. Justify your choice of OCR library, NER approach, embedding model, and vector database.
    d. Explain tradeoffs you considered (accuracy vs speed, cost vs quality).
    e. Describe how you would scale this system to handle 1000+ PDFs per hour.
    f. Identify potential failure modes and how you would mitigate them.
    g. If you were to add LLM routing (selecting different models based on query complexity),
       how would you implement it?

## Test Steps:

1. **Set up the Development Environment:**
   - Create a new Python virtual environment and install the required libraries (e.g., PyMuPDF, Tesseract OCR, spaCy, Hugging Face Transformers, FastAPI or Flask).
   - Obtain a sample scanned PDF document containing names and last names for testing purposes.

2. **Implement Text Extraction from PDF:**
   - Use PyMuPDF or a similar library to extract text from the scanned PDF document.
   - Optionally, you can preprocess the extracted text (e.g., remove newlines, clean up formatting) for better NER or LLM performance.

3. **Implement Named Entity Recognition (NER) or LLM Approach:**
   - Use spaCy's pre-trained NER model or a custom NER model trained on a relevant dataset to identify names and last names in the extracted text.
   - Alternatively, use a pre-trained NER (e.g., BERT, RoBERTa) from the Hugging Face Transformers library for the named entity recognition task. It is easier to use a Generative LLM for zero-shot NER and return the response as a usable JSON.

4. **Parse Bounding Box Coordinates:**
   - Integrate the OCR library (e.g., Tesseract OCR) to obtain the bounding box coordinates of the recognized words in the PDF document.
   - Match the extracted names and last names with their corresponding bounding box coordinates from the OCR results.

5. **Implement the RESTful API:**
   - Use FastAPI or Flask to create a RESTful API endpoint that accepts a scanned PDF file and a set of name-last name pairs as input.
   - Perform the text extraction, named entity recognition, and bounding box parsing steps on the uploaded PDF file.
   - Implement fuzzy matching (e.g., using the python-fuzzy library) between the extracted names and the provided names in the API request, with a similarity threshold of 90%.
   - Return the extracted names, their bounding box coordinates, and the fuzzy matching results as a JSON response.

6. **Set up a Vector Database:**
   - Choose and install an open-source vector database solution (e.g., Chroma, Qdrant, Milvus) compatible with your development environment. If you are using a docker image of an open-source database, make sure to include it in the docker compose file.
   - Create a new database or collection within the vector database to store the extracted text from the PDF documents.

7. **Implement Text Embedding:**
   - Use an open-source embedding model like Sentence-BERT or Universal Sentence Encoder to convert the extracted text into vector representations.
   - Integrate the embedding model with the vector database to store the text embeddings along with their corresponding text snippets.

8. **Implement Retrieval-Augmented Generation (RAG):**
   - Create a new RESTful API endpoint that accepts a question as input.
   - Implement the RAG strategy to retrieve relevant text snippets from the vector database based on the similarity between the question embedding and the stored text embeddings.
   - Use a pre-trained language model (e.g., GPT-3.5, Gemini) to generate an answer based on the retrieved text snippets and the input question.
   - Return the generated answer as the response to the API request.

9. **Containerization:**
   - Create a new Dockerfile in the project directory with all the required dependencies to run the API server.
   - If required, generate a docker-compose file.

10. **Documentation and Submission:**
    - Document the code, and any external libraries or resources used.
    - Provide instructions for setting up and running the system, as well as any additional dependencies or requirements.

---

# Solution — How to Run

## With Docker (recommended)

```bash
# Optional: export OPENAI_API_KEY=sk-... to enable LLM-generated answers.
# Without it, /api/ask degrades gracefully to extractive answers.
docker compose up --build
```

This starts the API on http://localhost:8000 (Swagger UI at `/docs`) and Qdrant on port 6333 with a persistent volume. Models (spaCy + MiniLM) are baked into the image at build time.

### Fully local / open-source LLM (no external API)

The LLM client speaks the OpenAI-compatible protocol, so any local server (Ollama, vLLM, LM Studio) works. An optional Ollama service is included behind a compose profile:

```bash
docker compose --profile local-llm up --build
docker compose exec ollama ollama pull llama3.2   # once
```

Then run the API with:

```bash
OPENAI_BASE_URL=http://ollama:11434/v1 LLM_MODEL=llama3.2 OPENAI_API_KEY=ollama docker compose --profile local-llm up
```

With no LLM configured at all, `/api/ask` still works and returns an extractive answer built from the retrieved context.

## Locally

```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
python -m spacy download en_core_web_sm
# system dependency: tesseract-ocr must be installed (apt install tesseract-ocr)

# Only needed for the spanish_precision / multilingual profiles:
python -m spacy download es_core_news_sm
python -m spacy download xx_ent_wiki_sm
# apt install tesseract-ocr-spa

docker run -p 6333:6333 qdrant/qdrant   # vector DB
uvicorn app.main:app --reload
```

## Trying it out — two built-in UIs

Once the stack is up, there are two ways to exercise the API without writing any code:

| UI | URL | Best for |
|---|---|---|
| **Demo viewer** | [http://localhost:8000/](http://localhost:8000/) | Visual end-to-end test: upload a PDF, pick a model profile, and see the extracted names drawn as bounding boxes over the rendered document (green = fuzzy-matched, orange = extracted only), plus the match table with scores. |
| **Swagger UI** (OpenAPI spec) | [http://localhost:8000/docs](http://localhost:8000/docs) | Interactive documentation auto-generated from the OpenAPI schema (`/openapi.json`). Covers **all** endpoints — including `/api/ingest` and `/api/ask` (RAG), which the viewer doesn't expose — with request/response schemas and "Try it out" forms. |

Typical demo flow: open the viewer, upload `sample_pdfs/company_memo.pdf` with names `Margaret Thompson`, then switch to `sample_pdfs/memo_espanol.pdf` with the `spanish_precision` profile; use Swagger to ingest `research_report.pdf` and ask it a question.

## Using the API

```bash
# Health
curl http://localhost:8000/health

# Extract names + bounding boxes + fuzzy matching
curl -X POST http://localhost:8000/api/extract \
  -F "pdf_file=@sample_pdfs/company_memo.pdf" \
  -F 'names=[{"first_name":"Margaret","last_name":"Thompson"},{"first_name":"Maria","last_name":"Garcia"}]'

# Ingest a PDF into the vector store
curl -X POST http://localhost:8000/api/ingest \
  -F "pdf_file=@sample_pdfs/research_report.pdf"

# Ask a question (RAG)
curl -X POST http://localhost:8000/api/ask \
  -H "Content-Type: application/json" \
  -d '{"question":"What is the research report about?"}'
```

## Model profiles (multi-language support)

Instead of exposing internal model names, the API offers friendly **profiles** — like choosing a "fast" vs "thinking" tier on an LLM API. Each profile maps server-side to an OCR language, a spaCy NER model, an embedding model, and its own Qdrant collection (embeddings from different models live in incompatible vector spaces and are never mixed).

| Profile | Use for | Internally |
|---|---|---|
| `english_fast` (default) | English documents, fastest | Tesseract `eng` · spaCy `en_core_web_sm` · MiniLM |
| `spanish_precision` | Spanish documents, best free/offline quality | Tesseract `spa+eng` · **NER ensemble** (spaCy `es_lg` ∪ `xx`) · multilingual MiniLM |
| `multilingual` | Mixed/unknown languages | Tesseract `eng+spa` · spaCy `xx_ent_wiki_sm` · multilingual MiniLM |
| `llm_precision` | Highest NER accuracy, any language | Tesseract `eng+spa` · **zero-shot LLM NER** · multilingual MiniLM — requires `OPENAI_API_KEY` (or local Ollama); slower, per-token cost |

```bash
# Discover available profiles
curl http://localhost:8000/api/profiles

# Optional `profile` field on every endpoint (omit it → english_fast)
curl -X POST http://localhost:8000/api/extract \
  -F "pdf_file=@memo_espanol.pdf" \
  -F 'names=[{"first_name":"Maria","last_name":"Gonzalez"}]' \
  -F 'profile=spanish_precision'

curl -X POST http://localhost:8000/api/ask \
  -H "Content-Type: application/json" \
  -d '{"question":"¿De qué trata el informe?","profile":"spanish_precision"}'
```

An unknown profile returns `400` with the list of valid keys. Fuzzy matching is accent-insensitive (`Maria Gonzalez` matches `María González`) and works across all profiles. Note: baking the multilingual models into the Docker image adds ~500 MB; remove the corresponding lines in the `Dockerfile` if you only need English.

Why `spanish_precision` uses the LARGE Spanish model in an ensemble — decision trail: `es_core_news_sm` and `es_core_news_md` both tag `'La señora Lucía Fernández'` as `MISC` and silently drop her; `es_core_news_lg` tags her `PER` correctly (a pure configuration upgrade — no code change), and the union with the multilingual model covers the remaining blind spots. The 90% fuzzy threshold downstream filters ensemble noise out of the matches. A Spanish-specific transformer NER (e.g. `PlanTL-GOB-ES/roberta-...-capitel-ner`, F1 ~90) was evaluated and deliberately deferred: the config-level fix resolves the known failure at zero extra latency, and the transformer remains the documented next step if recall regresses. A profile's name is a promise — the engine behind it can be upgraded without clients noticing, which is the point of the profile abstraction.

Why `llm_precision` exists: even the ensemble inherits statistical noise (job titles tagged as people). Zero-shot LLM extraction returns clean, correct names in any language. Zero-shot LLM extraction handles honorifics, mixed languages, and noisy OCR correctly. Without an API key the profile fails fast with a `502` and a clear message (no silent fallback). It shares the multilingual embedding collection, since the embedding model — and therefore the vector space — is the same.

### Getting and configuring the OpenAI API key

The key enables two features at once: LLM-generated answers on `/api/ask` (instead of the extractive fallback) and the `llm_precision` NER profile. Note that API access is billed separately from ChatGPT (prepaid credits; `gpt-4o-mini` costs fractions of a cent per request).

1. Go to [platform.openai.com](https://platform.openai.com) and sign in (this is the API console — not chatgpt.com).
2. Add prepaid credit under **Settings → Billing** (minimum $5).
3. Create a key under **API keys → Create new secret key** and copy it immediately — it is shown only once.
4. Create a `.env` file next to `docker-compose.yml` (git-ignored, never commit the key):

   ```
   OPENAI_API_KEY=sk-your-key-here
   ```

5. `docker compose up` — compose reads `.env` automatically and passes the key to the API container.

The key is server-side configuration only (12-Factor style): it is never accepted from clients per-request and never reaches the frontend. To use a fully local LLM instead (no key, no cost), see the Ollama profile above.

## Tests

```bash
pytest -v        # 87 tests, no external services required (Qdrant is mocked)
```

Configuration is environment-driven (see `app/core/config.py`): `QDRANT_HOST`, `QDRANT_PORT`, `OCR_DPI`, `SIMILARITY_THRESHOLD`, `EMBEDDING_MODEL_NAME`, `LLM_MODEL`, `OPENAI_API_KEY`, `OPENAI_BASE_URL`, etc.

Architecture, tradeoffs, scaling and failure analysis: see [DESIGN.md](DESIGN.md). Bug-by-bug findings report: see the tables in [TASK.md](TASK.md).
