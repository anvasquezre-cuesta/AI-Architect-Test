"""Additional tests written as part of the submission (bonus).

Covers request validation, chunking edge cases, fuzzy normalization,
bbox occurrence handling, configuration defaults, and RAG resilience.
"""

import json
from unittest.mock import MagicMock, patch

import httpx
import pytest


class TestExtractEndpointValidation:
    """Input validation on POST /api/extract."""

    def test_rejects_malformed_names_json(self, app_client):
        response = app_client.post(
            "/api/extract",
            files={"pdf_file": ("doc.pdf", b"%PDF-1.4 fake", "application/pdf")},
            data={"names": "not-json"},
        )
        assert response.status_code == 400

    def test_rejects_names_not_a_list(self, app_client):
        response = app_client.post(
            "/api/extract",
            files={"pdf_file": ("doc.pdf", b"%PDF-1.4 fake", "application/pdf")},
            data={"names": json.dumps({"first_name": "John", "last_name": "Smith"})},
        )
        assert response.status_code == 400

    def test_rejects_name_pair_missing_fields(self, app_client):
        response = app_client.post(
            "/api/extract",
            files={"pdf_file": ("doc.pdf", b"%PDF-1.4 fake", "application/pdf")},
            data={"names": json.dumps([{"first_name": "John"}])},
        )
        assert response.status_code == 400

    def test_rejects_empty_file(self, app_client):
        response = app_client.post(
            "/api/extract",
            files={"pdf_file": ("doc.pdf", b"", "application/pdf")},
            data={"names": json.dumps([])},
        )
        assert response.status_code == 400

    def test_rejects_file_with_pdf_extension_but_wrong_content(self, app_client):
        """A .pdf filename with non-PDF bytes must still be rejected."""
        response = app_client.post(
            "/api/extract",
            files={"pdf_file": ("doc.pdf", b"GIF89a...", "application/pdf")},
            data={"names": json.dumps([])},
        )
        assert response.status_code == 400


class TestChunkTextEdgeCases:
    """Edge cases for the word-aware chunker."""

    def test_empty_text_returns_no_chunks(self):
        from app.services.rag_service import chunk_text

        assert chunk_text("", chunk_size=10) == []

    def test_single_long_word_is_not_split(self):
        from app.services.rag_service import chunk_text

        word = "supercalifragilistic"
        chunks = chunk_text(word, chunk_size=5)

        assert chunks == [word], "A single word longer than chunk_size must stay whole"

    def test_leading_whitespace_is_preserved(self):
        from app.services.rag_service import chunk_text

        text = "   hello world"
        chunks = chunk_text(text, chunk_size=8)

        assert "".join(chunks) == text

    def test_join_always_reconstructs_original(self):
        from app.services.rag_service import chunk_text

        text = "Lorem ipsum dolor sit amet, consectetur adipiscing elit. " * 5
        for size in (7, 20, 100, 10_000):
            assert "".join(chunk_text(text, chunk_size=size)) == text


class TestFuzzyMatcherNormalization:
    """Case handling and scoring in the fuzzy matcher."""

    def test_matching_is_case_insensitive(self):
        from app.services.fuzzy_service import fuzzy_match_names

        matches = fuzzy_match_names(
            ["JOHN SMITH"], [{"first_name": "john", "last_name": "smith"}]
        )

        assert len(matches) == 1
        assert matches[0]["score"] == 1.0

    def test_empty_query_names_returns_no_matches(self):
        from app.services.fuzzy_service import fuzzy_match_names

        assert fuzzy_match_names(["John Smith"], []) == []

    def test_empty_extracted_names_returns_no_matches(self):
        from app.services.fuzzy_service import fuzzy_match_names

        assert (
            fuzzy_match_names([], [{"first_name": "John", "last_name": "Smith"}]) == []
        )


class TestBBoxOccurrences:
    """Occurrence handling in the bounding box service."""

    def test_returns_every_occurrence_of_a_name(self):
        mock_names = ["John Smith"]
        mock_word_boxes = [
            {"word": "John", "page": 0, "x": 100, "y": 50, "width": 40, "height": 12},
            {"word": "Smith", "page": 0, "x": 145, "y": 50, "width": 50, "height": 12},
            {"word": "John", "page": 1, "x": 200, "y": 100, "width": 40, "height": 12},
            {"word": "Smith", "page": 1, "x": 245, "y": 100, "width": 50, "height": 12},
        ]

        with (
            patch("app.services.bbox_service.extract_names", return_value=mock_names),
            patch(
                "app.services.bbox_service.get_word_bounding_boxes",
                return_value=mock_word_boxes,
            ),
        ):
            from app.services.bbox_service import find_name_bounding_boxes

            result = find_name_bounding_boxes("test.pdf", "dummy")

        assert len(result) == 2, "Both occurrences should get a bounding box"
        assert {r["page"] for r in result} == {0, 1}

    def test_ignores_punctuation_attached_by_ocr(self):
        mock_names = ["Maria Garcia"]
        mock_word_boxes = [
            {"word": "Maria", "page": 0, "x": 10, "y": 10, "width": 40, "height": 12},
            {"word": "Garcia,", "page": 0, "x": 55, "y": 10, "width": 50, "height": 12},
        ]

        with (
            patch("app.services.bbox_service.extract_names", return_value=mock_names),
            patch(
                "app.services.bbox_service.get_word_bounding_boxes",
                return_value=mock_word_boxes,
            ),
        ):
            from app.services.bbox_service import find_name_bounding_boxes

            result = find_name_bounding_boxes("test.pdf", "dummy")

        assert len(result) == 1

    def test_does_not_merge_words_from_different_pages(self):
        """Name parts split across a page boundary must not merge into one box."""
        mock_names = ["John Smith"]
        mock_word_boxes = [
            {"word": "John", "page": 0, "x": 100, "y": 700, "width": 40, "height": 12},
            {"word": "Smith", "page": 1, "x": 100, "y": 50, "width": 50, "height": 12},
        ]

        with (
            patch("app.services.bbox_service.extract_names", return_value=mock_names),
            patch(
                "app.services.bbox_service.get_word_bounding_boxes",
                return_value=mock_word_boxes,
            ),
        ):
            from app.services.bbox_service import find_name_bounding_boxes

            result = find_name_bounding_boxes("test.pdf", "dummy")

        assert result == []


class TestConfiguration:
    """Configuration defaults and overrides."""

    def test_defaults_match_requirements(self):
        from app.core.config import Settings

        settings = Settings(_env_file=None)
        assert settings.similarity_threshold == 90
        assert settings.embedding_vector_size == 384
        assert settings.qdrant_collection == "pdf_documents"

    def test_env_override(self, monkeypatch):
        from app.core.config import Settings

        monkeypatch.setenv("QDRANT_HOST", "qdrant.internal")
        monkeypatch.setenv("OCR_DPI", "300")
        settings = Settings(_env_file=None)

        assert settings.qdrant_host == "qdrant.internal"
        assert settings.ocr_dpi == 300


class TestRAGResilience:
    """Failure handling in the RAG pipeline."""

    def test_llm_http_error_is_wrapped_and_returned_as_502(self, app_client):
        from app.core.exceptions import LLMError

        with patch("app.api.rag.generate_answer", side_effect=LLMError("boom")):
            response = app_client.post("/api/ask", json={"question": "Anything?"})

        assert response.status_code == 502

    def test_llm_retries_on_transient_http_error(self):
        """The LLM call should retry transient HTTP failures before giving up."""
        call_count = {"n": 0}

        def flaky_post(url, **kwargs):
            call_count["n"] += 1
            if call_count["n"] < 2:
                raise httpx.ConnectError("connection refused")
            resp = MagicMock()
            resp.json.return_value = {
                "choices": [{"message": {"content": "recovered"}}]
            }
            return resp

        mock_chunks = [{"text": "ctx", "score": 0.9}]

        with (
            patch(
                "app.services.rag_service.get_query_embedding",
                return_value=[0.1] * 384,
            ),
            patch("app.services.rag_service.search_similar", return_value=mock_chunks),
            patch("app.services.rag_service.httpx.post", side_effect=flaky_post),
            patch("app.services.rag_service.OPENAI_API_KEY", "test-key"),
            patch("app.core.retry.time.sleep"),  # no real backoff in tests
        ):
            from app.services.rag_service import generate_answer

            result = generate_answer("q?")

        assert result["answer"] == "recovered"
        assert call_count["n"] == 2

    def test_ask_endpoint_validates_empty_question(self, app_client):
        response = app_client.post("/api/ask", json={"question": ""})
        assert response.status_code == 422


class TestHealthEndpoint:
    def test_health_reports_version(self, app_client):
        response = app_client.get("/health")
        assert response.status_code == 200
        body = response.json()
        assert body["status"] == "ok"
        assert "version" in body


class TestModelProfiles:
    """Friendly model profiles (english_fast / spanish_precision / ...)."""

    def test_profiles_endpoint_lists_profiles(self, app_client):
        response = app_client.get("/api/profiles")
        assert response.status_code == 200
        body = response.json()
        keys = {p["key"] for p in body}
        assert {"english_fast", "spanish_precision", "multilingual"} <= keys
        defaults = [p for p in body if p["default"]]
        assert len(defaults) == 1 and defaults[0]["key"] == "english_fast"

    def test_ask_rejects_unknown_profile(self, app_client):
        response = app_client.post(
            "/api/ask", json={"question": "q?", "profile": "klingon_turbo"}
        )
        assert response.status_code == 400
        assert "Valid profiles" in response.json()["detail"]

    def test_extract_rejects_unknown_profile(self, app_client):
        import json as _json

        response = app_client.post(
            "/api/extract",
            files={"pdf_file": ("doc.pdf", b"%PDF-1.4 fake", "application/pdf")},
            data={"names": _json.dumps([]), "profile": "nope"},
        )
        assert response.status_code == 400

    def test_resolve_profile_defaults_and_maps_models(self):
        from app.core.profiles import resolve_profile

        default = resolve_profile(None)
        assert default.key == "english_fast"
        assert default.ocr_lang == "eng"

        spanish = resolve_profile("spanish_precision")
        assert spanish.spacy_model == "es_core_news_lg"
        assert "multilingual" in spanish.embedding_model_name
        assert "spa" in spanish.ocr_lang

    def test_bundle_isolates_collections_without_loading_models(self):
        """Building a bundle must be lazy (no model downloads) and give
        each profile its own Qdrant collection."""
        from app.core.config import get_settings
        from app.core.profiles import get_service_bundle

        bundle = get_service_bundle("spanish_precision")
        base = get_settings().qdrant_collection
        assert bundle.store.collection_name == base + "_es"
        assert bundle.ocr.lang == "spa+eng"

    def test_ocr_service_lang_is_configurable(self):
        from app.services.ocr_service import TesseractOCRService

        assert TesseractOCRService().lang == "eng"
        assert TesseractOCRService(lang="spa").lang == "spa"

    def test_spanish_ner_label_is_accepted(self):
        """es_core_news_sm tags people as 'PER' (not 'PERSON') — bug found
        during live testing of the spanish_precision profile."""
        from unittest.mock import MagicMock

        from app.services.ner_service import SpacyNERService

        doc = MagicMock()
        doc.ents = [
            MagicMock(text="María González", label_="PER"),
            MagicMock(text="Acme Corp", label_="ORG"),
        ]
        pipeline = MagicMock(return_value=doc)
        names = SpacyNERService(pipeline=pipeline).extract_names("texto")
        assert names == ["María González"]

    def test_fuzzy_matching_is_accent_insensitive(self):
        """'Maria Gonzalez' must match 'María González' — accents pushed
        the score below the 90 threshold before normalization."""
        from app.services.fuzzy_service import FuzzyNameMatcher

        matches = FuzzyNameMatcher().fuzzy_match_names(
            ["María González"],
            [{"first_name": "Maria", "last_name": "Gonzalez"}],
        )
        assert len(matches) == 1
        assert matches[0]["score"] == 1.0


class TestLLMNERService:
    """Zero-shot LLM NER (llm_precision profile)."""

    def _service(self):
        from app.services.llm_ner_service import LLMNERService

        return LLMNERService(api_key="test-key", model="test-model")

    def test_parses_clean_json_array(self):
        response = MagicMock()
        response.json.return_value = {
            "choices": [{"message": {"content": '["María González", "John Smith"]'}}]
        }
        with patch("app.services.llm_ner_service.httpx.post", return_value=response):
            names = self._service().extract_names("some document text")
        assert names == ["María González", "John Smith"]

    def test_parses_code_fenced_reply_and_dedupes(self):
        content = '```json\n["Ana Ruiz", "Ana Ruiz"]\n```'
        response = MagicMock()
        response.json.return_value = {"choices": [{"message": {"content": content}}]}
        with patch("app.services.llm_ner_service.httpx.post", return_value=response):
            names = self._service().extract_names("text")
        assert names == ["Ana Ruiz"]

    def test_missing_api_key_raises_clear_llm_error(self):
        from app.core.exceptions import LLMError
        from app.services.llm_ner_service import LLMNERService

        service = LLMNERService(api_key="", model="m")
        with pytest.raises(LLMError, match="OPENAI_API_KEY"):
            service.extract_names("some text")

    def test_empty_text_short_circuits_without_llm_call(self):
        with patch("app.services.llm_ner_service.httpx.post") as post:
            assert self._service().extract_names("   ") == []
        post.assert_not_called()

    def test_positions_locates_every_occurrence(self):
        response = MagicMock()
        response.json.return_value = {
            "choices": [{"message": {"content": '["Ana Ruiz"]'}}]
        }
        text = "Ana Ruiz saluda. Luego Ana Ruiz firma."
        with patch("app.services.llm_ner_service.httpx.post", return_value=response):
            positions = self._service().extract_names_with_positions(text)
        assert len(positions) == 2
        assert all(text[p["start_char"]:p["end_char"]] == "Ana Ruiz" for p in positions)

    def test_llm_precision_profile_uses_llm_ner(self):
        from app.core.profiles import get_service_bundle
        from app.services.llm_ner_service import LLMNERService

        bundle = get_service_bundle("llm_precision")
        assert isinstance(bundle.ner, LLMNERService)
        # Same embedding model as "multilingual" -> shares its collection.
        from app.core.profiles import get_service_bundle as gsb

        assert bundle.store.collection_name == gsb("multilingual").store.collection_name


class TestEnsembleNER:
    """spanish_precision now unions the es + xx spaCy models."""

    def _fake(self, names):
        svc = MagicMock()
        svc.extract_names.return_value = names
        svc.extract_names_with_positions.return_value = [
            {"name": n, "start_char": i, "end_char": i + len(n), "label": "PER"}
            for i, n in enumerate(names)
        ]
        return svc

    def test_union_dedupes_case_insensitively_preserving_order(self):
        from app.services.ner_service import EnsembleNERService

        ensemble = EnsembleNERService(
            [self._fake(["María González", "Ana Ruiz"]),
             self._fake(["maría gonzález", "Lucía Fernández"])]
        )
        assert ensemble.extract_names("txt") == [
            "María González", "Ana Ruiz", "Lucía Fernández"
        ]

    def test_requires_at_least_one_engine(self):
        from app.services.ner_service import EnsembleNERService

        with pytest.raises(ValueError):
            EnsembleNERService([])

    def test_spanish_precision_profile_uses_ensemble(self):
        from app.core.profiles import get_service_bundle
        from app.services.ner_service import EnsembleNERService

        get_service_bundle.cache_clear()
        bundle = get_service_bundle("spanish_precision")
        assert isinstance(bundle.ner, EnsembleNERService)
