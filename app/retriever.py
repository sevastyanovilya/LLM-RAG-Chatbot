"""FAISS-based vector retrieval."""
import pickle
from pathlib import Path
from typing import List, Tuple
import logging
import numpy as np

logger = logging.getLogger(__name__)


class FAISSRetriever:
    """FAISS vector store wrapper."""

    def __init__(self, embedding_model: str):
        from sentence_transformers import SentenceTransformer

        self.embedder = SentenceTransformer(embedding_model)
        self.index = None
        self.documents = []
        self.dimension = self.embedder.get_sentence_embedding_dimension()

    def build_index(self, documents: List[any]) -> None:
        """Build FAISS index from documents."""
        import faiss

        logger.info(f"Building index from {len(documents)} documents")

        # Extract text and store documents
        texts = [doc.content for doc in documents]
        self.documents = documents

        # Generate embeddings
        embeddings = self.embedder.encode(texts, show_progress_bar=True, convert_to_numpy=True)

        # Create FAISS index
        self.index = faiss.IndexFlatL2(self.dimension)
        self.index.add(embeddings.astype("float32"))

        logger.info(f"Index built with {self.index.ntotal} vectors")

    def save(self, index_path: Path) -> None:
        """Save index and documents to disk."""
        import faiss

        if self.index is None:
            raise ValueError("No index to save")

        index_path.mkdir(parents=True, exist_ok=True)

        # Save FAISS index
        faiss.write_index(self.index, str(index_path / "index.faiss"))

        # Save documents
        with open(index_path / "documents.pkl", "wb") as f:
            pickle.dump(self.documents, f)

        logger.info(f"Index saved to {index_path}")

    def load(self, index_path: Path) -> None:
        """Load index and documents from disk."""
        import faiss

        # Load FAISS index
        index_file = index_path / "index.faiss"
        if not index_file.exists():
            raise FileNotFoundError(f"Index not found at {index_file}")

        self.index = faiss.read_index(str(index_file))

        # Load documents
        with open(index_path / "documents.pkl", "rb") as f:
            self.documents = pickle.load(f)

        logger.info(f"Index loaded from {index_path} with {len(self.documents)} documents")

    def search(self, query: str, top_k: int = 3) -> List[Tuple[any, float]]:
        """Search for most similar documents."""
        if self.index is None:
            raise ValueError("Index not built or loaded")

        # Encode query
        query_embedding = self.embedder.encode([query], convert_to_numpy=True)

        # Search
        distances, indices = self.index.search(query_embedding.astype("float32"), top_k)

        # Return documents with similarity scores (convert L2 distance to similarity)
        results = []
        for idx, dist in zip(indices[0], distances[0]):
            if idx < len(self.documents):
                # Convert L2 distance to similarity score (inverse)
                similarity = 1.0 / (1.0 + dist)
                results.append((self.documents[idx], similarity))

        return results