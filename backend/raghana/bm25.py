"""
Okapi BM25 implementation from scratch.

No NLTK, no rank_bm25 library — pure Python with a simple regex tokeniser.
Parameters: k1=1.5, b=0.75 (Robertson & Zaragoza, 2009 defaults).

Why BM25 alongside vector search:
- Exact keyword match is critical for this corpus:
  e.g. "NPP", "Ashanti", "2020", "Free SHS", "Debt-to-GDP"
- Dense embeddings sometimes lose specificity for proper nouns and numbers
- Hybrid fusion (BM25 + vector) outperforms either alone (Part B analysis)
"""

import re
import math
import pickle
from pathlib import Path
from collections import defaultdict


def _tokenise(text: str) -> list[str]:
    """Lowercase and split on non-alphanumeric characters."""
    return re.findall(r"[a-z0-9]+", text.lower())


class BM25:
    """Okapi BM25 with k1=1.5, b=0.75."""

    def __init__(self, k1: float = 1.5, b: float = 0.75) -> None:
        self.k1 = k1
        self.b = b
        self.corpus: list[list[str]] = []
        self.doc_ids: list[str] = []
        self.df: dict[str, int] = {}
        self.idf: dict[str, float] = {}
        self.doc_lengths: list[int] = []
        self.avgdl: float = 1.0
        self.N: int = 0

    # ── Fit ────────────────────────────────────────────────────

    def fit(self, texts: list[str], doc_ids: list[str]) -> None:
        """Build inverted index from corpus."""
        assert len(texts) == len(doc_ids)
        self.corpus = [_tokenise(t) for t in texts]
        self.doc_ids = list(doc_ids)
        self.N = len(texts)
        self.doc_lengths = [len(doc) for doc in self.corpus]
        self.avgdl = sum(self.doc_lengths) / self.N if self.N else 1.0

        # Document frequency per term
        df: dict[str, int] = defaultdict(int)
        for doc in self.corpus:
            for term in set(doc):
                df[term] += 1
        self.df = dict(df)

        # Robertson-Sparck Jones IDF (smoothed)
        self.idf = {
            term: math.log((self.N - freq + 0.5) / (freq + 0.5) + 1.0)
            for term, freq in self.df.items()
        }

    # ── Score ──────────────────────────────────────────────────

    def score(self, query: str, doc_idx: int) -> float:
        """BM25 score for a single (query, document) pair."""
        terms = _tokenise(query)
        doc = self.corpus[doc_idx]
        dl = self.doc_lengths[doc_idx]

        tf_map: dict[str, int] = defaultdict(int)
        for token in doc:
            tf_map[token] += 1

        total = 0.0
        for term in terms:
            if term not in self.idf:
                continue
            tf = tf_map.get(term, 0)
            numerator = tf * (self.k1 + 1)
            denominator = tf + self.k1 * (
                1 - self.b + self.b * dl / self.avgdl
            )
            total += self.idf[term] * numerator / denominator
        return total

    # ── Search ─────────────────────────────────────────────────

    def search(self, query: str, k: int = 30) -> list[tuple[int, float]]:
        """Return top-k (doc_idx, score) pairs, sorted descending."""
        scores = [(i, self.score(query, i)) for i in range(self.N)]
        scores.sort(key=lambda x: -x[1])
        return scores[:k]

    # ── Persistence ────────────────────────────────────────────

    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, path: str | Path) -> "BM25":
        with open(path, "rb") as f:
            return pickle.load(f)
