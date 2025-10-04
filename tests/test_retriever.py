"""Tests for FAISS retriever."""
import pytest
from pathlib import Path
import tempfile
from app.retriever import FAISSRetriever
from app.utils.chunking import Document


@pytest.fixture
def sample_documents():
    """Create sample documents for testing."""
    return [
        Document(
            content="Python is a programming language used for web development.",
            metadata={"source": "python.md", "chunk_id": 0}
        ),
        Document(
            content="FastAPI is a modern web framework for building APIs with Python.",
            metadata={"source": "fastapi.md", "chunk_id": 0}
        ),
        Document(
            content="Machine learning uses algorithms to learn from data.",
            metadata={"source": "ml.md", "chunk_id": 0}
        ),
    ]


@pytest.fixture
def retriever():
    """Create retriever instance."""
    return FAISSRetriever(embedding_model="sentence-transformers/all-MiniLM-L6-v2")


def test_build_index(retriever, sample_documents):
    """Test building FAISS index."""
    retriever.build_index(sample_documents)

    assert retriever.index is not None
    assert retriever.index.ntotal == len(sample_documents)
    assert len(retriever.documents) == len(sample_documents)


def test_search(retriever, sample_documents):
    """Test searching the index."""
    retriever.build_index(sample_documents)

    results = retriever.search("What is FastAPI?", top_k=2)

    assert len(results) == 2
    assert all(isinstance(doc, Document) for doc, _ in results)
    assert all(isinstance(score, float) for _, score in results)

    # First result should be about FastAPI
    top_doc, top_score = results[0]
    assert "FastAPI" in top_doc.content


def test_save_and_load(retriever, sample_documents):
    """Test saving and loading index."""
    with tempfile.TemporaryDirectory() as tmpdir:
        index_path = Path(tmpdir) / "test_index"

        # Build and save
        retriever.build_index(sample_documents)
        retriever.save(index_path)

        # Load in new retriever
        new_retriever = FAISSRetriever(embedding_model="sentence-transformers/all-MiniLM-L6-v2")
        new_retriever.load(index_path)

        # Verify
        assert new_retriever.index.ntotal == len(sample_documents)
        assert len(new_retriever.documents) == len(sample_documents)

        # Test search still works
        results = new_retriever.search("Python programming", top_k=1)
        assert len(results) == 1


def test_search_empty_query(retriever, sample_documents):
    """Test searching with empty query."""
    retriever.build_index(sample_documents)
    results = retriever.search("", top_k=3)

    # Should still return results
    assert len(results) > 0
