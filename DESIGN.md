# Project Design Document: AI Architect PDF Extraction & RAG System

## 1. Services and Dependencies

The system is a FastAPI-based application designed for high-performance PDF text extraction, entity matching, and Retrieval-Augmented Generation (RAG).

### Core Services
- **OCR Service**: Handles PDF processing and text extraction. Uses **PyMuPDF** for digital text extraction and **EasyOCR** for scanned documents.
- **NER Service**: Uses **spaCy** (`en_core_web_sm`) to identify entities (PERSON, ORG) within extracted text.
- **Fuzzy Matching Service**: Uses **thefuzz** (Levenshtein distance) to match extracted entities against user-provided query names.
- **Embedding Service**: Leverages **sentence-transformers** (`all-MiniLM-L6-v2`) to generate vector representations of text chunks.
- **Vector Store Service**: Manages interactions with **Qdrant**, storing and searching document embeddings.
- **RAG Service**: Orchestrates context retrieval and answer generation using **OpenAI GPT-4o**.

### Key Dependencies
- **FastAPI/Uvicorn**: Web framework and ASGI server.
- **Loguru**: Structured logging for observability.
- **httpx**: Asynchronous HTTP client for LLM API calls.
- **Pydantic**: Data validation and settings management.

---

## 2. Operation Sequence

### API Endpoint: `/extract` (Extraction & Fuzzy Matching)
1. **Input**: PDF file and a JSON list of name objects (`first_name`, `last_name`).
2. **Preprocessing**: Save PDF to a temporary file.
3. **Text Extraction**:
   - `OCRService` uses PyMuPDF to attempt direct text extraction.
   - If needed, `EasyOCR` processes pages as images to extract text from scanned areas.
4. **Entity Recognition**: `NerService` (spaCy) scans the text for `PERSON` and `ORG` tags.
5. **Bounding Box Resolution**: `OCRService` maps recognized entities back to their physical coordinates (x, y, width, height) on the PDF pages.
6. **Fuzzy Matching**: `FuzzyService` compares extracted names against query names using `partial_ratio` scoring.
7. **Cleanup**: Temporary files are deleted.
8. **Output**: List of extracted names with coordinates and identified fuzzy matches.

### API Endpoint: `/ingest` (Vector Ingestion)
1. **Input**: PDF file.
2. **Text Extraction**: Same OCR pipeline as above to get raw text.
3. **Chunking**: `RAGService` splits text into overlapping chunks (default 500 chars).
4. **Embedding**: `EmbeddingService` converts chunks into 384-dimensional vectors.
5. **Storage**: `QdrantVectorStoreService` upserts vectors and raw text payload into the `pdf_documents` collection.

### API Endpoint: `/ask` (RAG Query)
1. **Input**: User question string.
2. **Query Embedding**: Convert question into a vector.
3. **Retrieval**: Search Qdrant for the top 3 most similar text chunks.
4. **Prompt Construction**: Inject retrieved context and the user question into a specialized RAG prompt.
5. **Generation**: Call OpenAI GPT-4o to generate a natural language answer based *only* on the provided context.
6. **Output**: The generated answer and the source text chunks used.

---

## 3. Technology Choice Justifications and Tradeoffs

### OCR: EasyOCR vs. Tesseract/PaddleOCR
- **Choice**: EasyOCR.
- **Reasoning**: The project environment (Mac Big Sur 11.7.11) imposed significant restrictions on Python versions and binary library dependencies. Tesseract and PaddleOCR proved difficult to build/install due to specific system library requirements. EasyOCR provided a robust, pure-python compatible alternative with excellent accuracy for the target PDF types.
- **Tradeoff**: EasyOCR can be slower than PaddleOCR on CPU; this is mitigated by offloading to worker threads.

### Vector Store: Qdrant vs. Chroma/Pinecone
- **Choice**: Qdrant.
- **Reasoning**: Qdrant offers a superior balance of performance and local development ease (via Docker) compared to Chroma (which can have SQLite locking issues in concurrent environments) or Pinecone (which requires an external cloud subscription). Qdrant's `query_points` API is highly efficient for the required scale.

### LLM: OpenAI GPT-4o
- **Choice**: GPT-4o.
- **Reasoning**: Selected for its superior reasoning capabilities, large context window, and fast inference times compared to local models or older GPT-3.5 versions.

### Utilities
- **PyMuPDF**: Chosen for its speed and ability to handle complex PDF layouts and metadata better than `PyPDF2`.
- **Loguru**: Replaced standard `logging` for better traceback formatting and easier rotation/filtering.
- **Sentence-Transformers**: Used locally for embeddings to reduce latency and costs associated with external API calls for every chunk.

---

## 4. Scaling Strategy (1000+ PDFs/hour)

To handle a sustained load of 1000+ PDFs per hour, the following scaling architecture is recommended:

1. **Horizontal Scaling**: Deploy the FastAPI app as a containerized service (Docker/K8s) with a load balancer. Since extraction is CPU-bound, increase replica count based on CPU utilization.
2. **Task Queue (Celery/RabbitMQ)**: Move the `/extract` and `/ingest` logic into background tasks. The API should return a `task_id` immediately, allowing the system to process heavy OCR jobs asynchronously without blocking HTTP workers.
3. **Dedicated Embedding Workers**: Centralize embedding generation. A dedicated service with GPU acceleration (even on MPS for Mac or CUDA for Linux) can handle batch embedding of chunks much faster than individual API workers.
4. **Qdrant Clustering**: Run Qdrant in distributed mode with multiple shards to ensure search latency remains low as the document count grows into the millions.
5. **S3 Storage**: Instead of local temp files, stream PDFs directly from an S3-compatible bucket to avoid disk I/O bottlenecks.

---

## 5. Failure Modes and Mitigations

| Failure Mode | Mitigation |
| :--- | :--- |
| **OCR Timeout/Crash** | Wrapped in `anyio.to_thread` to prevent event loop blocking. Implemented `try-except-finally` blocks to ensure temp file cleanup. |
| **OpenAI API Rate Limits** | Implement exponential backoff retries in `rag_service.py` using libraries like `tenacity`. |
| **Vector Store Connection Loss** | Health checks on startup and lazy-reconnection logic in `QdrantVectorStoreService`. |
| **Memory Exhaustion (Large PDFs)** | Implement page-by-page processing and streaming extraction instead of loading the entire document into memory. |
| **Invalid JSON in Names** | Robust parsing in `extract.py` using both `json.loads` and `ast.literal_eval` to handle malformed inputs from different clients. |

---

## 6. LLM Routing Strategy

If multiple models are available (e.g., GPT-4o, GPT-4o-mini, and a local Llama-3), a **Router Service** can be implemented to optimize cost and performance based on query complexity:

1. **Complexity Classifier**: A lightweight classifier (based on query length, keyword density, or a fast "mini" LLM call) determines if the query is "Simple" (e.g., "What is the date?") or "Complex" (e.g., "Summarize the legal implications of section 4").
2. **Routing Logic**:
   - **Simple**: Route to **GPT-4o-mini** (Low cost, high speed).
   - **Complex**: Route to **GPT-4o** (High reasoning capability).
   - **Privacy-Sensitive**: Route to a **local Llama-3 instance** if specific metadata flags are present.
3. **Fallback**: If a high-tier model fails or hits a rate limit, the router automatically downgrades to the next available model to ensure system availability.
