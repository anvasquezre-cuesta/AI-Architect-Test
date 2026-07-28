import pytest
from unittest.mock import MagicMock, patch

from fastapi.testclient import TestClient


@pytest.fixture
def app_client():
    """Create a test client for the FastAPI app."""
    from app.main import app

    return TestClient(app)


@pytest.fixture
def sample_text():
    """Sample text containing names for NER testing."""
    return (
        "John Smith is the lead engineer at Acme Corp. "
        "He works closely with Maria Garcia and Robert Johnson. "
        "The project was reviewed by Dr. Sarah Williams from MIT."
    )


@pytest.fixture
def sample_names_query():
    """Sample name pairs for fuzzy matching."""
    return [
        {"first_name": "John", "last_name": "Smith"},
        {"first_name": "Maria", "last_name": "Garcia"},
        {"first_name": "James", "last_name": "Wilson"},
    ]


@pytest.fixture
def sample_word_boxes():
    """Sample OCR word bounding boxes."""
    return [
        {"word": "John", "page": 0, "x": 100, "y": 50, "width": 40, "height": 12},
        {"word": "Smith", "page": 0, "x": 145, "y": 50, "width": 50, "height": 12},
        {"word": "is", "page": 0, "x": 200, "y": 50, "width": 15, "height": 12},
        {"word": "the", "page": 0, "x": 220, "y": 50, "width": 25, "height": 12},
        {"word": "Maria", "page": 0, "x": 100, "y": 80, "width": 45, "height": 12},
        {"word": "Garcia", "page": 0, "x": 150, "y": 80, "width": 50, "height": 12},
    ]


@pytest.fixture
def mock_qdrant_client():
    """Mock Qdrant client for testing without a running instance."""
    from app.services.vector_service import reset_default_store
    reset_default_store()
    with patch("app.services.vector_service.QdrantClient") as mock_constructor:
        mock_client = MagicMock()
        mock_constructor.return_value = mock_client
        mock_client.get_collections.return_value = MagicMock(collections=[])
        yield mock_client
    reset_default_store()
