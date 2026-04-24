"""
End-to-end RAG pipeline orchestrator.

Stage order:
  1. Retrieve (hybrid BM25 + vector)
  2. Multi-step router (numeric queries only)
  3. Prompt construction + context packing
  4. LLM generation (Groq)

Each stage is timed and logged via logger.py.
The result object carries all intermediate data so the API can return
retrieved chunks, the final prompt, and stage timings to the UI.
"""

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, Optional

from raghana.vectorstore import VectorStore
from raghana.bm25 import BM25
from raghana.retrieve import retrieve
from raghana.prompt import build_prompt, build_prompt_v1, build_prompt_v3
from raghana.generate import generate
from raghana.multistep import is_numeric_query, compute_numeric, inject_computed_result
from raghana.logger import log_stage

_ARTIFACTS = Path(__file__).parent.parent / "artifacts"

# Module-level singletons — loaded once, reused for all requests
_vector_store: Optional[VectorStore] = None
_bm25: Optional[BM25] = None


def _load_index() -> tuple[VectorStore, BM25]:
    global _vector_store, _bm25
    if _vector_store is None:
        _vector_store = VectorStore.load(_ARTIFACTS / "embeddings")
    if _bm25 is None:
        _bm25 = BM25.load(_ARTIFACTS / "bm25.pkl")
    return _vector_store, _bm25


# ──────────────────────────────────────────────────────────
# Result dataclass
# ──────────────────────────────────────────────────────────

@dataclass
class RAGResult:
    query: str
    answer: str
    retrieved_chunks: list[dict] = field(default_factory=list)
    final_prompt: str = ""
    stage_timings: dict[str, float] = field(default_factory=dict)
    multistep_trace: Optional[dict] = None

    def to_dict(self) -> dict:
        return {
            "query": self.query,
            "answer": self.answer,
            "retrieved_chunks": [
                {
                    "id": c.get("chunk_id", ""),
                    "score": round(c.get("score", 0.0), 4),
                    "text": c.get("text", ""),
                    "source": c.get("source", ""),
                    "metadata": c.get("metadata", {}),
                }
                for c in self.retrieved_chunks
            ],
            "final_prompt": self.final_prompt,
            "stage_timings": {k: round(v, 1) for k, v in self.stage_timings.items()},
            "multistep_trace": self.multistep_trace,
        }


# ──────────────────────────────────────────────────────────
# Main entry point
# ──────────────────────────────────────────────────────────

def run(
    query: str,
    k: int = 5,
    mode: Literal["hybrid", "vector", "bm25"] = "hybrid",
    prompt_version: Literal["v2", "v1", "v3"] = "v2",
) -> RAGResult:
    """
    Execute the full RAG pipeline for one query.

    Parameters
    ----------
    query          : raw user query string
    k              : number of chunks to retrieve
    mode           : retrieval mode ("hybrid" | "vector" | "bm25")
    prompt_version : prompt template ("v2" | "v1" | "v3")
    """
    timings: dict[str, float] = {}
    result = RAGResult(query=query, answer="")

    # ── Stage 1: Retrieve ──────────────────────────────────────
    t0 = time.perf_counter()
    vs, bm = _load_index()
    chunks = retrieve(query, vs, bm, k=k, mode=mode)
    timings["retrieve"] = (time.perf_counter() - t0) * 1000

    top_score = chunks[0].get("score", 0.0) if chunks else 0.0
    log_stage(
        "retrieve",
        timings["retrieve"],
        query,
        f"{len(chunks)} chunks | top_score={top_score:.3f} | mode={mode}",
    )
    result.retrieved_chunks = chunks

    # ── Stage 2: Multi-step (numeric queries only) ─────────────
    multistep_trace: Optional[dict] = None
    if is_numeric_query(query):
        t1 = time.perf_counter()
        numeric = compute_numeric(query, chunks)
        timings["multistep"] = (time.perf_counter() - t1) * 1000
        if numeric:
            multistep_trace = numeric
            log_stage(
                "multistep",
                timings["multistep"],
                query,
                numeric["computed_result"],
            )
    result.multistep_trace = multistep_trace

    # ── Stage 3: Prompt construction ──────────────────────────
    t2 = time.perf_counter()
    if prompt_version == "v1":
        system_msg, user_msg = build_prompt_v1(query, chunks)
    elif prompt_version == "v3":
        system_msg, user_msg = build_prompt_v3(query, chunks)
    else:
        system_msg, user_msg = build_prompt(query, chunks)

    if multistep_trace:
        user_msg = inject_computed_result(user_msg, multistep_trace)

    timings["prompt"] = (time.perf_counter() - t2) * 1000
    result.final_prompt = f"SYSTEM:\n{system_msg}\n\nUSER:\n{user_msg}"
    log_stage("prompt_build", timings["prompt"], query[:80], f"{len(result.final_prompt)} chars")

    # ── Stage 4: Generate ──────────────────────────────────────
    t3 = time.perf_counter()
    answer = generate(system_msg, user_msg)
    timings["generate"] = (time.perf_counter() - t3) * 1000
    log_stage("generate", timings["generate"], query[:80], answer[:120])

    result.answer = answer
    result.stage_timings = timings
    return result
