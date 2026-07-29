"""Zero-shot NER via an OpenAI-compatible LLM.

Implements the same ``NERService`` interface as the spaCy implementation,
so it can be swapped in through a model profile without touching any
consumer (bounding boxes, extraction endpoint, tests).

Why an LLM for NER: statistical models have systematic blind spots — the
es/xx spaCy models mislabel e.g. 'La señora Lucía Fernández' (MISC) or
tag job titles as people. A zero-shot LLM handles honorifics, mixed
languages, and noisy OCR text much better, at the cost of latency
(~1-3 s/doc), per-token pricing, and non-determinism.

Requires OPENAI_API_KEY (or an OpenAI-compatible OPENAI_BASE_URL, e.g.
a local Ollama). Without a key, extraction fails fast with a clear
LLMError — no silent fallback, so callers know which engine produced
their results.
"""

import json
import re

import httpx

from app.core.config import get_settings
from app.core.exceptions import LLMError
from app.core.logging import get_logger
from app.core.retry import retry_call
from app.services.interfaces import NERService

logger = get_logger(__name__)

_PROMPT = """Extract every PERSON name from the text below.

Rules:
- Return ONLY a JSON array of strings, e.g. ["María González", "John Smith"].
- Include first and last names exactly as they appear in the text.
- Do NOT include honorifics or titles ("Dr.", "La señora", "CEO").
- Do NOT include organizations, places, job titles, or section headers.
- If there are no person names, return [].

Text:
{text}"""


class LLMNERService(NERService):
    """Concrete NER implementation using zero-shot LLM extraction."""

    def __init__(
        self,
        model: str | None = None,
        api_key: str | None = None,
        base_url: str | None = None,
    ) -> None:
        settings = get_settings()
        self._model = model or settings.llm_model
        self._api_key = api_key if api_key is not None else settings.openai_api_key
        self._base_url = base_url or settings.openai_base_url

    # ------------------------------------------------------------------ #
    # LLM call + response parsing
    # ------------------------------------------------------------------ #
    def _call_llm(self, prompt: str) -> str:
        settings = get_settings()
        if not self._api_key:
            raise LLMError(
                "LLM-based NER requires OPENAI_API_KEY (or an OpenAI-compatible "
                "OPENAI_BASE_URL such as a local Ollama). Configure it or use a "
                "spaCy-based profile."
            )

        def _do_request() -> str:
            response = httpx.post(
                f"{self._base_url}/chat/completions",
                headers={
                    "Authorization": f"Bearer {self._api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": self._model,
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": 0,
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
                label="LLM NER",
            )
        except httpx.HTTPError as exc:
            raise LLMError(f"LLM NER request failed after retries: {exc}") from exc

    @staticmethod
    def _parse_names(raw: str) -> list[str]:
        """Parse the model's reply into a list of names (tolerant of fences)."""
        text = raw.strip()
        # Strip Markdown code fences some models insist on adding.
        text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.MULTILINE)
        # Keep only the outermost JSON array if the model added prose.
        start, end = text.find("["), text.rfind("]")
        if start == -1 or end == -1 or end <= start:
            raise LLMError(f"LLM NER reply is not a JSON array: {raw!r}")
        try:
            parsed = json.loads(text[start : end + 1])
        except json.JSONDecodeError as exc:
            raise LLMError(f"LLM NER reply is not valid JSON: {raw!r}") from exc
        if not isinstance(parsed, list):
            raise LLMError(f"LLM NER reply is not a list: {raw!r}")

        names: list[str] = []
        for item in parsed:
            name = str(item).strip()
            if name and name not in names:  # dedupe, keep order
                names.append(name)
        return names

    # ------------------------------------------------------------------ #
    # NERService interface
    # ------------------------------------------------------------------ #
    def extract_names(self, text: str) -> list[str]:
        """Extract person names via a zero-shot LLM prompt."""
        if not text.strip():
            return []
        reply = self._call_llm(_PROMPT.format(text=text))
        names = self._parse_names(reply)
        logger.debug("LLM NER found %d person name(s)", len(names))
        return names

    def extract_names_with_positions(self, text: str) -> list[dict]:
        """Extract names and locate their character offsets in the text."""
        results: list[dict] = []
        for name in self.extract_names(text):
            for match in re.finditer(re.escape(name), text):
                results.append(
                    {
                        "name": name,
                        "start_char": match.start(),
                        "end_char": match.end(),
                        "label": "PERSON",
                    }
                )
        return results
