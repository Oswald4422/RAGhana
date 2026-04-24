"""
Hybrid retrieval with Reciprocal Rank Fusion (RRF).

Strategy:
  1. BM25 top-30 (keyword match, exact token recall)
  2. Vector top-30 (semantic similarity via cosine)
  3. Fuse both ranked lists with RRF: score = Σ 1/(60 + rank_i)
  4. Return top-k fused results

Why RRF over weighted sum:
- No calibration required (scores from BM25 and cosine are on different scales)
- Proven to outperform score normalisation in IR literature (Cormack et al., 2009)
- Simple, transparent, easy to explain in the assignment writeup

The 'mode' parameter lets experiments toggle to vector-only or bm25-only
for the ablation study in notebook 03.
"""

from typing import Literal
import numpy as np

from raghana.vectorstore import VectorStore
from raghana.bm25 import BM25
from raghana.embed import encode_query

_RRF_K = 60   # constant from Cormack et al.


def _rrf(ranks: list[int]) -> float:
    return sum(1.0 / (_RRF_K + r) for r in ranks)


def retrieve(
    query: str,
    vector_store: VectorStore,
    bm25: BM25,
    k: int = 5,
    fetch: int = 30,
    mode: Literal["hybrid", "vector", "bm25"] = "hybrid",
) -> list[dict]:
    """
    Main retrieval entry point.

    Parameters
    ----------
    query        : natural-language user query
    vector_store : loaded VectorStore instance
    bm25         : fitted BM25 instance
    k            : number of results to return
    fetch        : candidate pool size for each retriever (pre-fusion)
    mode         : "hybrid" | "vector" | "bm25"

    Returns
    -------
    List of chunk dicts with added fields:
      score        – fused RRF score (or raw score in single-mode)
      rrf_score    – same
      vector_score – cosine similarity (if vector was used)
      bm25_score   – BM25 score (if BM25 was used)
      rank         – final rank (1-indexed)
    """
    vector_hits: list[dict] = []
    bm25_hits: list[dict] = []

    # ── Vector retrieval ───────────────────────────────────────
    if mode in ("hybrid", "vector"):
        q_vec = encode_query(query)
        vector_hits = vector_store.search(q_vec, k=fetch)

    # ── BM25 retrieval ─────────────────────────────────────────
    if mode in ("hybrid", "bm25"):
        raw = bm25.search(query, k=fetch)
        bm25_hits = [
            {**vector_store.meta[idx], "bm25_score": float(score)}
            for idx, score in raw
        ]

    # ── Single-mode early return ───────────────────────────────
    if mode == "vector":
        for i, r in enumerate(vector_hits[:k], start=1):
            r.setdefault("vector_score", r.get("score", 0.0))
            r["rrf_score"] = r["score"]
            r["rank"] = i
        return vector_hits[:k]

    if mode == "bm25":
        for i, r in enumerate(bm25_hits[:k], start=1):
            r["score"] = r["bm25_score"]
            r["rrf_score"] = r["bm25_score"]
            r["rank"] = i
        return bm25_hits[:k]

    # ── Hybrid RRF fusion ──────────────────────────────────────
    seen: dict[str, dict] = {}

    for rank, hit in enumerate(vector_hits, start=1):
        cid = hit["chunk_id"]
        if cid not in seen:
            seen[cid] = dict(hit)
            seen[cid]["_vector_rank"] = None
            seen[cid]["_bm25_rank"] = None
        seen[cid]["_vector_rank"] = rank
        seen[cid]["vector_score"] = hit.get("score", 0.0)

    for rank, hit in enumerate(bm25_hits, start=1):
        cid = hit["chunk_id"]
        if cid not in seen:
            seen[cid] = dict(hit)
            seen[cid]["_vector_rank"] = None
            seen[cid]["_bm25_rank"] = None
        seen[cid]["_bm25_rank"] = rank
        seen[cid]["bm25_score"] = hit.get("bm25_score", 0.0)

    fused: list[dict] = []
    for cid, data in seen.items():
        ranks = [r for r in (data.pop("_vector_rank"), data.pop("_bm25_rank")) if r is not None]
        rrf = _rrf(ranks)
        data["rrf_score"] = rrf
        data["score"] = rrf
        fused.append(data)

    fused.sort(key=lambda x: -x["rrf_score"])
    for i, r in enumerate(fused, start=1):
        r["rank"] = i

    return fused[:k]
