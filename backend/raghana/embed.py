"""
Embedding pipeline using sentence-transformers/all-MiniLM-L6-v2.

Why this model:
- 384-d output → small, fast, CPU-friendly
- Trained for semantic similarity tasks
- Permissive Apache-2.0 licence
- This is NOT a RAG framework — it is a pure embedding model
"""

import numpy as np
from sentence_transformers import SentenceTransformer

_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
_model: SentenceTransformer | None = None


def _get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        _model = SentenceTransformer(_MODEL_NAME)
    return _model


def encode(
    texts: list[str],
    batch_size: int = 32,
    show_progress: bool = True,
) -> np.ndarray:
    """
    Encode a list of texts to L2-normalised 384-d float32 embeddings.

    normalize_embeddings=True ensures cosine similarity = dot product,
    which lets the vector store use a simple matrix multiply for retrieval.
    """
    model = _get_model()
    emb = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=show_progress,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    return emb.astype(np.float32)


def encode_query(query: str) -> np.ndarray:
    """Encode a single query string, no progress bar."""
    return encode([query], show_progress=False)[0]
