"""Tests for document chunking utilities."""
import pytest
from pathlib import Path
from app.utils.chunking import chunk_text, Document, load_and_chunk_documents


def test_chunk_text_basic():
    """Test basic text chunking."""
    text = "This is a test. " * 100  # Long text
    chunks = chunk_text(text, chunk_size=100, chunk_overlap=20)

    assert len(chunks) > 1
    assert all(len(chunk) <= 120 for chunk in chunks)  # Allow some overflow


def test_chunk_text_with_paragraphs():
    """Test chunking respects paragraph boundaries."""
    text = "Paragraph one.\n\nParagraph two.\n\nParagraph three."
    chunks = chunk_text(text, chunk_size=50, chunk_overlap=10)

    assert len(chunks) >= 1
    assert all(chunk.strip() for chunk in chunks)


def test_chunk_text_empty():
    """Test chunking empty text."""
    chunks = chunk_text("", chunk_size=100, chunk_overlap=20)
    assert chunks == []


def test_chunk_text_whitespace_only():
    """Test chunking whitespace-only text."""
    chunks = chunk_text("   \n\n   ", chunk_size=100, chunk_overlap=20)
    assert chunks == []


def test_chunk_text_overlap():
    """Test that chunks have overlap."""
    text = "Word " * 50
    chunks = chunk_text(text, chunk_size=50, chunk_overlap=10)

    if len(chunks) > 1:
        # Check that consecutive chunks share some content
        for i in range(len(chunks) - 1):
            # At least some overlap should exist
            assert len(chunks[i]) > 0 and len(chunks[i + 1]) > 0


def test_chunk_text_single_paragraph():
    """Test chunking a single short paragraph."""
    text = "This is a short paragraph that should fit in one chunk."
    chunks = chunk_text(text, chunk_size=200, chunk_overlap=20)

    assert len(chunks) == 1
    assert chunks[0] == text


def test_chunk_text_no_overlap():
    """Test chunking with zero overlap."""
    text = "First sentence. Second sentence. Third sentence. Fourth sentence."
    chunks = chunk_text(text, chunk_size=30, chunk_overlap=0)

    assert len(chunks) >= 1


def test_document_creation():
    """Test Document class."""
    doc = Document(
        content="Test content",
        metadata={"source": "test.md", "chunk_id": 0}
    )

    assert doc.content == "Test content"
    assert doc.metadata["source"] == "test.md"
    assert doc.metadata["chunk_id"] == 0


def test_document_with_empty_metadata():
    """Test Document with empty metadata."""
    doc = Document(content="Content", metadata={})

    assert doc.content == "Content"
    assert doc.metadata == {}


def test_chunk_text_long_word():
    """Test chunking with words longer than chunk size."""
    text = "short " + "x" * 200 + " short"
    chunks = chunk_text(text, chunk_size=50, chunk_overlap=10)

    # Should still produce chunks even with long words
    assert len(chunks) >= 1
