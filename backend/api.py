"""
FastAPI application — single endpoint for the React UI.

Endpoints:
  GET  /               – health/status
  GET  /api/health     – health check
  POST /api/query      – main RAG query
  POST /api/query/no-retrieval  – Part E: pure-LLM baseline
"""

import sys
from pathlib import Path

# Allow importing raghana as a local package when run from backend/
sys.path.insert(0, str(Path(__file__).parent))

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Literal

from raghana.pipeline import run as pipeline_run
from raghana.generate import generate_no_retrieval

app = FastAPI(
    title="RAGhana API",
    version="1.0.0",
    description="Ghana public-sector RAG assistant",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000", "*"],
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)


# ──────────────────────────────────────────────────────────
# Request / Response schemas
# ──────────────────────────────────────────────────────────

class QueryRequest(BaseModel):
    q: str = Field(..., min_length=1, description="User query")
    k: int = Field(5, ge=1, le=20, description="Number of chunks to retrieve")
    mode: Literal["hybrid", "vector", "bm25"] = "hybrid"
    prompt_version: Literal["v2", "v1", "v3"] = "v2"


class ChunkInfo(BaseModel):
    id: str
    score: float
    text: str
    source: str
    metadata: dict


class QueryResponse(BaseModel):
    query: str
    answer: str
    retrieved_chunks: list[ChunkInfo]
    final_prompt: str
    stage_timings: dict
    multistep_trace: dict | None = None


class NoRetrievalResponse(BaseModel):
    answer: str
    mode: str = "no_retrieval"


# ──────────────────────────────────────────────────────────
# Routes
# ──────────────────────────────────────────────────────────

@app.get("/")
def root():
    return {
        "status": "RAGhana API is running",
        "docs": "/docs",
        "query_endpoint": "POST /api/query",
    }


@app.get("/api/health")
def health():
    return {"status": "ok"}


@app.post("/api/query", response_model=QueryResponse)
def query(req: QueryRequest):
    """Run the full RAG pipeline and return answer + all intermediate data."""
    try:
        result = pipeline_run(
            query=req.q.strip(),
            k=req.k,
            mode=req.mode,
            prompt_version=req.prompt_version,
        )
        return QueryResponse(**result.to_dict())
    except FileNotFoundError:
        raise HTTPException(
            status_code=503,
            detail=(
                "Index not found. "
                "Run 'python build_index.py' from the backend/ directory first."
            ),
        )
    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/query/no-retrieval", response_model=NoRetrievalResponse)
def query_no_retrieval(req: QueryRequest):
    """
    Part E: send the same query to the LLM without any retrieved context.
    Used to measure hallucination rate of pure-LLM vs RAG.
    """
    try:
        answer = generate_no_retrieval(req.q.strip())
        return NoRetrievalResponse(answer=answer)
    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=str(e))
