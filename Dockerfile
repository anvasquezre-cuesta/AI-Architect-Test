# ---------------------------------------------------------------------------
# PDF Name Extractor & RAG API
# ---------------------------------------------------------------------------
FROM python:3.11-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/opt/hf-cache

# System dependencies: Tesseract for OCR (+ Spanish language pack for the
# spanish_precision / multilingual profiles), curl for the healthcheck.
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        tesseract-ocr tesseract-ocr-spa curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /srv/app

# Install CPU-only PyTorch FIRST: the default Linux wheel bundles CUDA
# libraries this CPU-only container never uses. This single line shrinks
# the image by ~3-4 GB (torch download: ~2.5 GB -> ~200 MB).
RUN pip install torch --index-url https://download.pytorch.org/whl/cpu

# Install Python dependencies (torch is already satisfied, so
# sentence-transformers won't pull the CUDA build).
COPY requirements.txt .
RUN pip install -r requirements.txt

# Pre-download models at build time so containers start fast and can run
# without outbound internet access. One spaCy + one embedding model per
# profile (english_fast / spanish_precision / multilingual); drop the
# lines you don't need to slim the image (~500 MB for the multilingual
# embedding model).
RUN python -m spacy download en_core_web_sm \
    && python -m spacy download es_core_news_lg \
    && python -m spacy download xx_ent_wiki_sm \
    && python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')" \
    && python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')"

COPY app ./app

# Run as an unprivileged user.
RUN useradd --create-home appuser \
    && chown -R appuser:appuser /srv/app /opt/hf-cache
USER appuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
    CMD curl -fsS http://localhost:8000/health || exit 1

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
