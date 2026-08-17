# syntax=docker/dockerfile:1

FROM python:3.12-slim

# System dependency: Tesseract OCR engine (pytesseract is only a wrapper).
RUN apt-get update \
    && apt-get install -y --no-install-recommends tesseract-ocr \
    && rm -rf /var/lib/apt/lists/*

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/models/hf

WORKDIR /app

# Install Python dependencies first for better layer caching.
COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

# Bake models into the image so the container starts offline with no runtime
# download: spaCy NER pipeline and the SBERT embedding checkpoint.
RUN python -m spacy download en_core_web_sm \
    && python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"

COPY app ./app

EXPOSE 8000

# Qdrant is reached by service name inside Compose; overridable via env.
ENV QDRANT_HOST=qdrant \
    QDRANT_PORT=6333

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
