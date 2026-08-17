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

## Running the application

### With Docker Compose (recommended)

Builds the API image (Tesseract, spaCy, and the SBERT model are baked in) and starts
it together with a Qdrant vector database.

```bash
# Optional: copy env template and set an LLM key for the RAG /ask endpoint.
cp .env.example .env

docker compose up --build
```

The API is then available at `http://localhost:8000` and Qdrant at
`http://localhost:6333`. To run without an LLM, leave `OPENAI_API_KEY` empty:
extraction and ingestion work; only `/api/ask` needs it. Point `OPENAI_BASE_URL`
at any OpenAI-compatible server (vLLM, Ollama, and similar) to use a self-hosted open model.

### Locally (without Docker)

```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
python -m spacy download en_core_web_sm
docker run -p 6333:6333 qdrant/qdrant          # vector DB for the RAG endpoints
uvicorn app.main:app --reload
```

### API endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/health` | Liveness probe. |
| POST | `/api/extract` | Multipart: `pdf_file` (PDF) + `names` (JSON array of `{first_name, last_name}`). Returns extracted names, bounding boxes (with `page_number`), and fuzzy-match results. |
| POST | `/api/ingest` | Multipart: `pdf_file` (PDF). OCRs, chunks, embeds, and stores the document in Qdrant. |
| POST | `/api/ask` | JSON: `{ "question": "..." }`. Returns a RAG-generated `answer` and its `sources`. |

Interactive docs (Swagger UI) are served at `http://localhost:8000/docs`.

### Observability (optional)

The app emits structured JSON logs to stdout (one object per line, with a per-request
id and timings). To persist and query them, bring up the opt-in observability profile,
which adds Grafana Loki (log store), Promtail (collector), and Grafana (viewer):

```bash
docker compose --profile observability up -d
```

Grafana is then at `http://localhost:3000` (anonymous access, Loki datasource
pre-provisioned). Query the app logs with a LogQL expression like
`{container=~".*app.*"}`. The default `docker compose up` stays lean (app + Qdrant only).

Example:

```bash
curl -F "pdf_file=@sample_pdfs/company_memo.pdf" \
     -F 'names=[{"first_name":"John","last_name":"Smith"}]' \
     http://localhost:8000/api/extract
```
