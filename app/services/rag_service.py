"""RAG (Retrieval-Augmented Generation) service.

Bug fixes vs. original implementation:
- The question placeholder was escaped (``{{question}}``) so the literal
  string '{question}' was sent to the LLM. It is now interpolated.
- Chunking is word-aware: the naive character split cut words in half,
  hurting both embeddings and retrieval quality.
- The LLM call now has a timeout, retries with backoff, response
  validation, and a graceful degradation path when no API key is set.
"""

import re

import httpx

from app.core.config import get_settings
from app.core.exceptions import LLMError
from app.core.logging import get_logger
from app.core.retry import retry_call
from app.services.embedding_service import get_query_embedding
from app.services.interfaces import AnswerGenerator
from app.services.vector_service import search_similar

logger = get_logger(__name__)

# Module attribute (patched in tests); sourced from configuration.
OPENAI_API_KEY = get_settings().openai_api_key

_TOKEN_RE = re.compile(r"\S+\s*")

_PROMPT_TEMPLATE = """Based on the following context, answer the question.
If the answer is not in the context, say "I don't have enough information."

Context:
{context}

Question:
{question}

Answer:"""


class RAGService(AnswerGenerator):
    """Retrieval-augmented answer generation over the vector store."""

    def __init__(self, embedding_service=None, vector_store=None) -> None:
        self._embedder = embedding_service
        self._store = vector_store
        self._settings = get_settings()

    # ------------------------------------------------------------------ #
    # Chunking
    # ------------------------------------------------------------------ #
    def chunk_text(self, text: str, chunk_size: int | None = None) -> list[str]:
        """Split text into chunks WITHOUT breaking words.

        Whitespace is kept attached to the preceding token so that
        ``"".join(chunks) == text`` (no information is lost).
        """
        size = chunk_size or self._settings.rag_chunk_size

        matches = list(_TOKEN_RE.finditer(text))
        if not matches:
            return [text] if text else []

        tokens = [m.group(0) for m in matches]
        # Preserve any leading whitespace before the first token.
        leading = text[: matches[0].start()]
        if leading:
            tokens[0] = leading + tokens[0]

        chunks: list[str] = []
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

    # ------------------------------------------------------------------ #
    # Answer generation
    # ------------------------------------------------------------------ #
    def _retrieve(self, question: str) -> list[dict]:
        if self._embedder is not None:
            query_embedding = self._embedder.get_query_embedding(question)
        else:
            query_embedding = get_query_embedding(question)

        if self._store is not None:
            return self._store.search_similar(
                query_embedding, top_k=self._settings.rag_top_k
            )
        return search_similar(query_embedding, top_k=self._settings.rag_top_k)

    def _call_llm(self, prompt: str, api_key: str) -> str:
        settings = self._settings

        def _do_request() -> str:
            response = httpx.post(
                f"{settings.openai_base_url}/chat/completions",
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": settings.llm_model,
                    "messages": [{"role": "user", "content": prompt}],
                },
                timeout=settings.llm_timeout_seconds,
            )
            response.raise_for_status()
            payload = response.json()
            try:
                return payload["choices"][0]["message"]["content"]
            except (KeyError, IndexError, TypeError) as exc:
                raise LLMError(f"Unexpected LLM response shape: {payload}") from exc

        try:
            return retry_call(
                _do_request,
                attempts=settings.llm_max_retries,
                retry_on=(httpx.HTTPError,),
                label="LLM completion",
            )
        except httpx.HTTPError as exc:
            raise LLMError(f"LLM request failed after retries: {exc}") from exc

    def generate_answer(self, question: str) -> dict:
        """Generate an answer using the RAG strategy."""
        relevant_chunks = self._retrieve(question)

        if not relevant_chunks:
            return {"answer": "No relevant information found.", "sources": []}

        context = "\n\n".join(chunk["text"] for chunk in relevant_chunks)
        sources = [chunk["text"] for chunk in relevant_chunks]

        # NOTE: real interpolation — the original sent a literal '{question}'.
        prompt = _PROMPT_TEMPLATE.format(context=context, question=question)

        # Module-level lookup at call time keeps this patchable in tests.
        api_key = OPENAI_API_KEY
        if not api_key:
            logger.warning(
                "No LLM API key configured — returning extractive fallback answer"
            )
            return {
                "answer": (
                    "LLM is not configured (missing OPENAI_API_KEY). "
                    "Most relevant retrieved context:\n\n" + context
                ),
                "sources": sources,
            }

        answer = self._call_llm(prompt, api_key)
        return {"answer": answer, "sources": sources}


# Default instance + module-level callables (stable public API / test seam).
_default_service = RAGService()
chunk_text = _default_service.chunk_text
generate_answer = _default_service.generate_answer
