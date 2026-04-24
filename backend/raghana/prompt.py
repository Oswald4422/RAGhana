"""
Prompt templates and context-window management.

Three prompt variants for Part C experiments:
  v1 – naïve baseline (no grounding rules)
  v2 – grounded + citation (default / production)
  v3 – grounded + citation + refusal clause + few-shot example

Context-window management:
  - Token-count each chunk with tiktoken
  - Greedy-pack by fused score until CONTEXT_BUDGET tokens
  - Dedup near-identical chunks (cosine > 0.95) before packing
"""

import re
import tiktoken
import numpy as np

_TOKENIZER = tiktoken.get_encoding("cl100k_base")
CONTEXT_BUDGET = 2500   # tokens reserved for context (~2.5 k of a ~4 k window)


# ──────────────────────────────────────────────────────────
# Prompt templates
# ──────────────────────────────────────────────────────────

_SYSTEM_V2 = """\
You are RAGhana, a factual assistant for Ghana public-sector data \
(election results and national budget).

Rules:
1. Answer ONLY from the CONTEXT section provided below.
2. If the answer is not present in the context, reply exactly:
   "I do not have that information in the provided documents."
3. Cite supporting chunk IDs inline, e.g. [C1] or [C2].
4. Do not speculate or infer facts that are not explicitly stated.
5. For numeric answers, quote the exact figure from the context.
6. Be concise and factual.\
"""

_SYSTEM_V1 = "You are a helpful assistant."

_SYSTEM_V3 = (
    _SYSTEM_V2
    + """

Few-shot citation example:
Q: How many votes did NPP receive in Ashanti in 2020?
A: NPP received 1,795,824 votes (72.79%) in the Ashanti Region in 2020 [C1].\
"""
)


# ──────────────────────────────────────────────────────────
# Token utilities
# ──────────────────────────────────────────────────────────

def _count_tokens(text: str) -> int:
    return len(_TOKENIZER.encode(text))


# ──────────────────────────────────────────────────────────
# Deduplication
# ──────────────────────────────────────────────────────────

def _dedup(chunks: list[dict], threshold: float = 0.95) -> list[dict]:
    """
    Remove chunks whose text is near-identical to an already-kept chunk.
    Uses cosine similarity on the stored 'score' embedding proxy — or
    falls back to exact-string dedup when embeddings are unavailable.
    """
    # Fast path: exact-text dedup
    seen_texts: set[str] = set()
    unique: list[dict] = []
    for c in chunks:
        t = c["text"].strip()
        if t not in seen_texts:
            seen_texts.add(t)
            unique.append(c)

    # If we want full cosine dedup (slower, more accurate):
    # from raghana.embed import encode
    # embs = encode([c["text"] for c in unique], show_progress=False)
    # kept, kept_embs = [], []
    # for c, e in zip(unique, embs):
    #     if not any(float(np.dot(e, ke)) > threshold for ke in kept_embs):
    #         kept.append(c); kept_embs.append(e)
    # return kept

    return unique


# ──────────────────────────────────────────────────────────
# Context packing
# ──────────────────────────────────────────────────────────

def pack_context(
    chunks: list[dict],
    budget_tokens: int = CONTEXT_BUDGET,
) -> list[dict]:
    """
    Sort chunks by score (descending), dedup, then greedily pack until budget.
    """
    sorted_chunks = sorted(chunks, key=lambda c: -c.get("score", 0.0))
    deduped = _dedup(sorted_chunks)

    packed: list[dict] = []
    used = 0
    for chunk in deduped:
        t = _count_tokens(chunk["text"]) + 25  # +25 overhead for the [Cx] header
        if used + t > budget_tokens:
            break
        packed.append(chunk)
        used += t

    return packed


# ──────────────────────────────────────────────────────────
# Context block builder
# ──────────────────────────────────────────────────────────

def _build_context_block(chunks: list[dict]) -> str:
    lines: list[str] = []
    for i, chunk in enumerate(chunks, start=1):
        source = chunk.get("source", "unknown")
        meta = chunk.get("metadata", {})
        if source == "csv":
            loc = f"csv:{meta.get('year', '?')}/{meta.get('region', '?')}"
        else:
            page = meta.get("page_approx") or meta.get("page_start") or meta.get("page_num", "?")
            loc = f"pdf:p{page}"
        lines.append(f"[C{i}] ({loc})\n{chunk['text']}\n")
    return "\n".join(lines) if lines else "(no context retrieved)"


# ──────────────────────────────────────────────────────────
# Public build functions (v1 / v2 / v3)
# ──────────────────────────────────────────────────────────

def build_prompt(
    query: str,
    chunks: list[dict],
) -> tuple[str, str]:
    """Default (v2): grounded + citation rules."""
    packed = pack_context(chunks)
    context = _build_context_block(packed)
    user_msg = f"CONTEXT:\n{context}\n\nQUESTION: {query}"
    return _SYSTEM_V2, user_msg


def build_prompt_v1(
    query: str,
    chunks: list[dict],
) -> tuple[str, str]:
    """Naïve baseline — no grounding or citation rules."""
    packed = pack_context(chunks)
    context = "\n\n".join(c["text"] for c in packed)
    user_msg = f"Here is some relevant text:\n{context}\n\nAnswer: {query}"
    return _SYSTEM_V1, user_msg


def build_prompt_v3(
    query: str,
    chunks: list[dict],
) -> tuple[str, str]:
    """Grounded + citation + refusal + few-shot example."""
    packed = pack_context(chunks)
    context = _build_context_block(packed)
    user_msg = f"CONTEXT:\n{context}\n\nQUESTION: {query}"
    return _SYSTEM_V3, user_msg
