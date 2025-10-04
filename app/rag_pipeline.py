"""RAG pipeline orchestration."""
import time
import logging
from typing import Dict, Any
from pathlib import Path

from app.config import settings
from app.retriever import FAISSRetriever
from app.llm import get_llm
from app.utils.chunking import load_and_chunk_documents

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RAGPipeline:
    """Orchestrates retrieval and generation."""

    def __init__(self):
        self.retriever = FAISSRetriever(embedding_model=settings.embedding_model)
        self.llm = None
        self._init_llm()

    def _init_llm(self):
        """Initialize LLM based on configuration."""
        try:
            self.llm = get_llm(
                mode=settings.llm_mode,
                local_path=settings.local_model_path if settings.llm_mode == "local" else None,
                api_key=settings.openai_api_key if settings.llm_mode == "hosted" else None,
            )
        except Exception as e:
            logger.error(f"LLM initialization failed: {e}")
            raise

    def ingest_documents(self) -> Dict[str, Any]:
        """Load, chunk, and index documents."""
        start_time = time.time()

        # Load and chunk
        documents = load_and_chunk_documents(
            settings.kb_path,
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
        )

        if not documents:
            return {"status": "error", "message": "No documents found"}

        # Build index
        self.retriever.build_index(documents)

        # Save index
        self.retriever.save(settings.index_path)

        elapsed = time.time() - start_time
        return {
            "status": "success",
            "documents_processed": len(documents),
            "time_seconds": round(elapsed, 2),
        }

    def load_index(self) -> bool:
        """Load existing index."""
        try:
            self.retriever.load(settings.index_path)
            return True
        except FileNotFoundError:
            logger.warning("Index not found. Run ingestion first.")
            return False

    def answer(self, query: str) -> Dict[str, Any]:
        """Answer query using RAG."""
        start_time = time.time()

        # Retrieve relevant documents
        retrieve_start = time.time()
        results = self.retriever.search(query, top_k=settings.top_k)
        retrieve_time = time.time() - retrieve_start

        if not results:
            return {
                "answer": "I couldn't find relevant information to answer your question.",
                "sources": [],
                "latency_seconds": round(time.time() - start_time, 2),
            }

        # Build context from retrieved documents
        context = "\n\n".join([f"[{i+1}] {doc.content}" for i, (doc, _) in enumerate(results)])

        # Build prompt
        prompt = f"""Answer the question based on the following context. Be concise and cite sources using [1], [2], etc.

Context:
{context}

Question: {query}

Answer:"""

        # Generate answer
        generate_start = time.time()
        answer = self.llm.generate(prompt, max_tokens=256)
        generate_time = time.time() - generate_start

        # Extract sources
        sources = [
            {
                "content": doc.content[:200] + "...",
                "source": doc.metadata.get("source", "unknown"),
                "similarity": round(score, 3),
            }
            for doc, score in results
        ]

        total_time = time.time() - start_time

        logger.info(f"Query answered in {total_time:.2f}s (retrieve: {retrieve_time:.2f}s, generate: {generate_time:.2f}s)")

        return {
            "answer": answer,
            "sources": sources,
            "latency_seconds": round(total_time, 2),
            "metrics": {
                "retrieve_time": round(retrieve_time, 2),
                "generate_time": round(generate_time, 2),
            },
        }


# CLI for ingestion
if __name__ == "__main__":
    import sys

    pipeline = RAGPipeline()

    if "--ingest" in sys.argv:
        result = pipeline.ingest_documents()
        print(f"Ingestion complete: {result}")
    else:
        print("Usage: python -m app.rag_pipeline --ingest")