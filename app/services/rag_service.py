"""RAG service.

Word-aware chunking (preserves the exact source text), retrieval over the vector
store, and answer generation via an OpenAI-compatible chat endpoint. The retrieved
context is inserted as *data* into a fixed prompt; the model is instructed to answer
only from context, which also contains any injected text to the answer step rather
than letting it steer extraction (DECISION_LOG D-5).
"""

from __future__ import annotations

import logging
import re
from typing import Protocol, runtime_checkable

import httpx

from app import config
from app.services.embedding_service import get_query_embedding
from app.services.vector_service import search_similar

logger = logging.getLogger(__name__)

OPENAI_API_KEY = config.OPENAI_API_KEY

_NO_CONTEXT_MESSAGE = "No relevant information found."


@runtime_checkable
class AnswerGenerator(Protocol):
    """Interface for RAG answer generation."""

    def chunk_text(self, text: str, chunk_size: int = 500) -> list[str]:
        ...

    def generate_answer(self, question: str) -> dict:
        ...


class RAGService:
    """Retrieval-augmented answer generation over an injected retriever/embedder."""

    def __init__(
        self,
        embed_query=None,
        retriever=None,
        api_key: str | None = None,
        model: str | None = None,
        base_url: str | None = None,
    ) -> None:
        self._embed_query = embed_query
        self._retriever = retriever
        self._api_key = api_key if api_key is not None else config.OPENAI_API_KEY
        self._model = model or config.OPENAI_MODEL
        self._base_url = base_url or config.OPENAI_BASE_URL

    def chunk_text(self, text: str, chunk_size: int = 500) -> list[str]:
        """Split text into <=chunk_size-ish chunks without cutting words.

        Tiles the original into word and whitespace tokens so that
        ``"".join(chunks) == text`` exactly, while never splitting a word.
        """
        if not text:
            return []
        tokens = re.findall(r"\S+|\s+", text)
        chunks: list[str] = []
        current = ""
        current_word_len = 0
        for token in tokens:
            if token.isspace():
                current += token
                continue
            if current_word_len > 0 and current_word_len + len(token) > chunk_size:
                chunks.append(current)
                current = token
                current_word_len = len(token)
            else:
                current += token
                current_word_len += len(token)
        if current:
            chunks.append(current)
        return chunks

    def generate_answer(self, question: str) -> dict:
        query_embedding = self._embed_query(question)
        relevant_chunks = self._retriever(query_embedding, top_k=3)

        if not relevant_chunks:
            return {"answer": _NO_CONTEXT_MESSAGE, "sources": []}

        sources = [chunk["text"] for chunk in relevant_chunks]

        # No LLM configured: return the retrieved context instead of attempting an
        # unauthenticated call (which would fail with an opaque header error).
        if not self._api_key:
            return {
                "answer": (
                    "No LLM configured. Set OPENAI_API_KEY, or point "
                    "OPENAI_BASE_URL at a local OpenAI-compatible server. "
                    "Returning the retrieved source passages instead."
                ),
                "sources": sources,
            }

        context = "\n\n".join(chunk["text"] for chunk in relevant_chunks)
        # No explicit "say you don't know" clause: with small models it over-triggers and
        # refuses even when the answer is present. Naming a role/title as valid fixes
        # "who is X?" questions, and the model still abstains on its own when the context
        # is unrelated (verified against qwen2.5:3b).
        prompt = (
            "Answer the question using only the context below. Be specific and "
            "complete: when the context provides details such as a role, title, "
            "date, number, or name, include them in the answer.\n\n"
            f"Context:\n{context}\n\n"
            f"Question: {question}\n\n"
            "Answer:"
        )

        try:
            response = httpx.post(
                f"{self._base_url}/chat/completions",
                headers={
                    "Authorization": f"Bearer {self._api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": self._model,
                    "messages": [{"role": "user", "content": prompt}],
                    # Low but non-zero: focused and grounded, not fully rigid.
                    "temperature": 0.2,
                },
                timeout=30.0,
            )
            response.raise_for_status()
            data = response.json()
            answer = data["choices"][0]["message"]["content"]
        except (httpx.HTTPError, KeyError, IndexError, ValueError) as exc:
            logger.warning(
                "LLM call failed", extra={"model": self._model, "error": str(exc)}
            )
            answer = f"Error generating answer: {exc}"

        return {"answer": answer, "sources": sources}


# Aliases inject current module globals at call time so patches on
# get_query_embedding / search_similar / OPENAI_API_KEY / httpx.post take effect.
chunk_text = lambda text, chunk_size=500: RAGService().chunk_text(  # noqa: E731
    text, chunk_size
)
generate_answer = lambda question: RAGService(  # noqa: E731
    get_query_embedding, search_similar, OPENAI_API_KEY
).generate_answer(question)
