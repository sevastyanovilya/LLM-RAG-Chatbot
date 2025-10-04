"""Document loading and chunking utilities."""
import re
from pathlib import Path
from typing import List, Dict, Any
import logging

logger = logging.getLogger(__name__)


class Document:
    """Simple document representation."""

    def __init__(self, content: str, metadata: Dict[str, Any]):
        self.content = content
        self.metadata = metadata


def load_markdown(file_path: Path) -> str:
    """Load text from Markdown file."""
    return file_path.read_text(encoding="utf-8")


def load_pdf(file_path: Path) -> str:
    """Extract text from PDF file."""
    try:
        from pypdf import PdfReader

        reader = PdfReader(file_path)
        text = ""
        for page in reader.pages:
            text += page.extract_text() + "\n"
        return text
    except Exception as e:
        logger.error(f"PDF extraction failed for {file_path}: {e}")
        return ""


def load_document(file_path: Path) -> str:
    """Load document based on file extension."""
    suffix = file_path.suffix.lower()
    if suffix == ".md":
        return load_markdown(file_path)
    elif suffix == ".pdf":
        return load_pdf(file_path)
    elif suffix == ".txt":
        return file_path.read_text(encoding="utf-8")
    else:
        logger.warning(f"Unsupported file type: {suffix}")
        return ""


def chunk_text(text: str, chunk_size: int = 500, chunk_overlap: int = 50) -> List[str]:
    """Split text into overlapping chunks."""
    if not text.strip():
        return []

    # Split on double newlines (paragraphs) first
    paragraphs = re.split(r"\n\n+", text)
    chunks = []
    current_chunk = ""

    for para in paragraphs:
        para = para.strip()
        if not para:
            continue

        # If adding this paragraph exceeds chunk_size, save current chunk
        if len(current_chunk) + len(para) > chunk_size and current_chunk:
            chunks.append(current_chunk)
            # Start new chunk with overlap
            overlap_text = current_chunk[-chunk_overlap:] if len(current_chunk) > chunk_overlap else current_chunk
            current_chunk = overlap_text + " " + para
        else:
            current_chunk = current_chunk + " " + para if current_chunk else para

    if current_chunk:
        chunks.append(current_chunk)

    return [chunk.strip() for chunk in chunks if chunk.strip()]


def load_and_chunk_documents(kb_path: Path, chunk_size: int = 500, chunk_overlap: int = 50) -> List[Document]:
    """Load all documents from knowledge base and chunk them."""
    documents = []

    for file_path in kb_path.rglob("*"):
        if file_path.is_file() and file_path.suffix.lower() in [".md", ".pdf", ".txt"]:
            logger.info(f"Processing {file_path}")
            text = load_document(file_path)

            if text:
                chunks = chunk_text(text, chunk_size, chunk_overlap)
                for i, chunk in enumerate(chunks):
                    doc = Document(
                        content=chunk,
                        metadata={
                            "source": str(file_path),
                            "chunk_id": i,
                            "total_chunks": len(chunks),
                        },
                    )
                    documents.append(doc)

    logger.info(f"Loaded {len(documents)} chunks from {kb_path}")
    return documents