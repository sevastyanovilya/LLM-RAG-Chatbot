"""FastAPI application with RAG endpoints."""
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any
import logging

from app.rag_pipeline import RAGPipeline
from app.config import settings

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="LLM RAG Chatbot",
    description="Portfolio-ready RAG chatbot with local LLM support",
    version="0.1.0",
)

# Initialize pipeline
pipeline = RAGPipeline()


@app.on_event("startup")
async def startup_event():
    """Load index on startup."""
    logger.info("Loading vector index...")
    if not pipeline.load_index():
        logger.warning("No index found. Use /ingest endpoint to build one.")


class ChatRequest(BaseModel):
    """Chat request model."""

    query: str


class ChatResponse(BaseModel):
    """Chat response model."""

    answer: str
    sources: List[Dict[str, Any]]
    latency_seconds: float


class IngestResponse(BaseModel):
    """Ingest response model."""

    status: str
    documents_processed: int
    time_seconds: float


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "llm_mode": settings.llm_mode}


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """Answer user query using RAG."""
    if not request.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty")

    try:
        result = pipeline.answer(request.query)
        return ChatResponse(**result)
    except Exception as e:
        logger.error(f"Chat error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/ingest", response_model=IngestResponse)
async def ingest():
    """Ingest documents and build index."""
    try:
        result = pipeline.ingest_documents()
        if result["status"] == "error":
            raise HTTPException(status_code=500, detail=result["message"])
        return IngestResponse(**result)
    except Exception as e:
        logger.error(f"Ingest error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/")
async def root():
    """Root endpoint with API info."""
    return {
        "message": "LLM RAG Chatbot API",
        "endpoints": {
            "health": "/health",
            "chat": "POST /chat",
            "ingest": "POST /ingest",
            "docs": "/docs",
        },
    }