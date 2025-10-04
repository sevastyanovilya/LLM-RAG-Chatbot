"""Smoke tests for FastAPI endpoints."""
import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock
from app.main import app


@pytest.fixture
def client():
    """Create test client."""
    return TestClient(app)


def test_health_check(client):
    """Test health endpoint."""
    response = client.get("/health")
    assert response.status_code == 200
    assert "status" in response.json()
    assert response.json()["status"] == "healthy"


def test_root_endpoint(client):
    """Test root endpoint."""
    response = client.get("/")
    assert response.status_code == 200
    assert "message" in response.json()
    assert "endpoints" in response.json()


@patch("app.main.pipeline")
def test_chat_endpoint(mock_pipeline, client):
    """Test chat endpoint with mocked pipeline."""
    # Mock the answer method
    mock_pipeline.answer.return_value = {
        "answer": "This is a test answer.",
        "sources": [
            {
                "content": "Test source content",
                "source": "test.md",
                "similarity": 0.95
            }
        ],
        "latency_seconds": 1.23
    }

    response = client.post(
        "/chat",
        json={"query": "What is this project?"}
    )

    assert response.status_code == 200
    data = response.json()
    assert "answer" in data
    assert "sources" in data
    assert "latency_seconds" in data
    assert len(data["sources"]) > 0


def test_chat_empty_query(client):
    """Test chat endpoint with empty query."""
    response = client.post(
        "/chat",
        json={"query": ""}
    )

    assert response.status_code == 400


@patch("app.main.pipeline")
def test_ingest_endpoint(mock_pipeline, client):
    """Test ingest endpoint with mocked pipeline."""
    # Mock the ingest_documents method
    mock_pipeline.ingest_documents.return_value = {
        "status": "success",
        "documents_processed": 10,
        "time_seconds": 5.67
    }

    response = client.post("/ingest")

    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert data["documents_processed"] > 0
