"""Evaluation metrics for RAG pipeline."""
from typing import List
import numpy as np
from sentence_transformers import SentenceTransformer


class EvalMetrics:
    """Semantic similarity metrics."""

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.embedder = SentenceTransformer(model_name)

    def cosine_similarity(self, text1: str, text2: str) -> float:
        """Compute cosine similarity between two texts."""
        embeddings = self.embedder.encode([text1, text2], convert_to_numpy=True)
        emb1, emb2 = embeddings[0], embeddings[1]

        # Cosine similarity
        similarity = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))
        return float(similarity)

    def semantic_match(self, generated: str, reference: str, threshold: float = 0.7) -> bool:
        """Check if generated text semantically matches reference."""
        similarity = self.cosine_similarity(generated, reference)
        return similarity >= threshold

    def batch_cosine_similarity(self, texts1: List[str], texts2: List[str]) -> List[float]:
        """Compute cosine similarity for batches of texts."""
        if len(texts1) != len(texts2):
            raise ValueError("Input lists must have same length")

        similarities = []
        for t1, t2 in zip(texts1, texts2):
            similarities.append(self.cosine_similarity(t1, t2))

        return similarities
