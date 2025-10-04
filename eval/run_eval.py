"""Evaluation harness for RAG pipeline."""
import json
import sys
import logging
from pathlib import Path
from typing import List, Dict, Any

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.rag_pipeline import RAGPipeline
from app.utils.eval_metrics import EvalMetrics

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_qa_pairs(file_path: str = "eval/qa_pairs.jsonl") -> List[Dict[str, str]]:
    """Load question-answer pairs from JSONL file."""
    qa_pairs = []
    with open(file_path, "r") as f:
        for line in f:
            qa_pairs.append(json.loads(line))
    return qa_pairs


def evaluate_retrieval(pipeline: RAGPipeline, qa_pairs: List[Dict[str, str]], top_k: int = 3) -> float:
    """Evaluate retrieval accuracy."""
    correct = 0

    for qa in qa_pairs:
        question = qa["question"]
        reference = qa["reference"]

        # Retrieve documents
        results = pipeline.retriever.search(question, top_k=top_k)

        # Check if any retrieved document contains key terms from reference
        # Simple heuristic: check for word overlap
        reference_words = set(reference.lower().split())
        reference_words = {w for w in reference_words if len(w) > 3}  # Filter short words

        for doc, _ in results:
            doc_words = set(doc.content.lower().split())
            # If at least 30% of reference words appear in doc, consider it relevant
            overlap = len(reference_words & doc_words)
            if overlap >= len(reference_words) * 0.3:
                correct += 1
                break

    accuracy = correct / len(qa_pairs) if qa_pairs else 0.0
    return accuracy


def evaluate_generation(pipeline: RAGPipeline, qa_pairs: List[Dict[str, str]], metrics: EvalMetrics) -> Dict[str, float]:
    """Evaluate generation quality using semantic similarity."""
    similarities = []
    latencies = []

    for qa in qa_pairs:
        question = qa["question"]
        reference = qa["reference"]

        # Generate answer
        result = pipeline.answer(question)
        generated = result["answer"]
        latencies.append(result["latency_seconds"])

        # Compute semantic similarity
        similarity = metrics.cosine_similarity(generated, reference)
        similarities.append(similarity)

    return {
        "avg_similarity": sum(similarities) / len(similarities) if similarities else 0.0,
        "median_latency": sorted(latencies)[len(latencies) // 2] if latencies else 0.0,
        "pass_rate": sum(1 for s in similarities if s >= 0.6) / len(similarities) if similarities else 0.0,
    }


def main():
    """Run evaluation harness."""
    logger.info("=" * 60)
    logger.info("RAG Pipeline Evaluation")
    logger.info("=" * 60)

    # Load QA pairs
    qa_pairs = load_qa_pairs()
    logger.info(f"Loaded {len(qa_pairs)} QA pairs")

    # Initialize pipeline
    logger.info("Initializing pipeline...")
    pipeline = RAGPipeline()

    # Load index
    if not pipeline.load_index():
        logger.error("Index not found. Run 'make ingest' first.")
        sys.exit(1)

    # Initialize metrics
    metrics = EvalMetrics()

    # Evaluate retrieval
    logger.info("\n--- Retrieval Evaluation ---")
    retrieval_accuracy = evaluate_retrieval(pipeline, qa_pairs, top_k=3)
    logger.info(f"Retrieval Accuracy: {retrieval_accuracy:.2%}")

    # Evaluate generation
    logger.info("\n--- Generation Evaluation ---")
    gen_metrics = evaluate_generation(pipeline, qa_pairs, metrics)
    logger.info(f"Average Semantic Similarity: {gen_metrics['avg_similarity']:.3f}")
    logger.info(f"Pass Rate (≥0.6 similarity): {gen_metrics['pass_rate']:.2%}")
    logger.info(f"Median Latency: {gen_metrics['median_latency']:.2f}s")

    # Overall results
    logger.info("\n" + "=" * 60)
    logger.info("SUMMARY")
    logger.info("=" * 60)
    logger.info(f"Retrieval Accuracy: {retrieval_accuracy:.2%} (target: ≥60%)")
    logger.info(f"Pass Rate: {gen_metrics['pass_rate']:.2%} (target: ≥60%)")
    logger.info(f"Median Latency: {gen_metrics['median_latency']:.2f}s (target: ≤8s)")

    # Check if targets met
    targets_met = (
        retrieval_accuracy >= 0.60
        and gen_metrics['pass_rate'] >= 0.60
        and gen_metrics['median_latency'] <= 8.0
    )

    if targets_met:
        logger.info("\n✓ All targets met!")
        sys.exit(0)
    else:
        logger.warning("\n✗ Some targets not met. Review results above.")
        sys.exit(1)


if __name__ == "__main__":
    main()
