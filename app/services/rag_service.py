"""RAG service.

Chunks text, retrieves the most similar chunks from the vector store and asks
an OpenAI-compatible chat model to answer grounded in that context.

Two correctness fixes vs. the original:
* ``chunk_text`` is word-aware and loss-less (``"".join(chunks) == text``); the
  old version sliced on raw character offsets and cut words in half.
* the prompt now interpolates the real ``question`` — the old f-string escaped
  it as the literal ``{question}``.

The LLM endpoint, model and key are configuration-driven, the HTTP call has a
timeout and raises on error, and a missing key fails fast with a clear message.
"""

import logging
import re

import httpx

from app.config import settings
from app.services.embedding_service import get_query_embedding
from app.services.vector_service import search_similar

logger = logging.getLogger(__name__)

# Module-level (patchable) credential, sourced from settings.
OPENAI_API_KEY = settings.openai_api_key


class RAGService:
    """Retrieval-augmented generation pipeline."""

    def chunk_text(self, text: str, chunk_size: int = None) -> list:
        """Split text into <= ``chunk_size`` chunks without breaking words.

        Whitespace is preserved so the chunks concatenate back to the original
        text exactly.
        """
        size = chunk_size if chunk_size is not None else settings.chunk_size
        if not text:
            return []

        tokens = re.findall(r"\S+\s*", text)  # each word plus its trailing space
        chunks = []
        current = ""
        for token in tokens:
            if current and len(current) + len(token) > size:
                chunks.append(current)
                current = token
            else:
                current += token
        if current:
            chunks.append(current)
        return chunks

    def generate_answer(self, question: str) -> dict:
        query_embedding = get_query_embedding(question)
        relevant_chunks = search_similar(query_embedding, top_k=3)

        if not relevant_chunks:
            return {"answer": "No relevant information found.", "sources": []}

        context = "\n\n".join(chunk["text"] for chunk in relevant_chunks)
        prompt = (
            "Based on the following context, answer the question.\n"
            'If the answer is not in the context, say "I don\'t have enough '
            'information."\n\n'
            f"Context:\n{context}\n\n"
            f"Question:\n{question}\n\n"
            "Answer:"
        )

        if not OPENAI_API_KEY:
            logger.error("OPENAI_API_KEY is not configured")
            raise RuntimeError(
                "LLM is not configured: set OPENAI_API_KEY to use /api/ask"
            )

        response = httpx.post(
            f"{settings.openai_base_url}/chat/completions",
            headers={
                "Authorization": f"Bearer {OPENAI_API_KEY}",
                "Content-Type": "application/json",
            },
            json={
                "model": settings.openai_model,
                "messages": [{"role": "user", "content": prompt}],
            },
            timeout=settings.llm_timeout,
        )
        response.raise_for_status()

        result = response.json()
        answer = result["choices"][0]["message"]["content"]
        return {
            "answer": answer,
            "sources": [chunk["text"] for chunk in relevant_chunks],
        }


_rag = RAGService()
chunk_text = _rag.chunk_text
generate_answer = _rag.generate_answer
