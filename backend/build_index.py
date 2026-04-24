"""
CLI: ingest Dataset/ -> clean -> chunk -> embed -> persist index.

Usage (from the backend/ directory):
    python build_index.py

Outputs (to backend/artifacts/):
    embeddings.npz   - L2-normalised chunk embeddings (float32)
    embeddings.json  - chunk metadata list
    bm25.pkl         - pickled BM25 index
    chunks.jsonl     - all chunks (convenience for notebook 02-03)

Expected runtime: ~1-2 minutes on CPU.
"""

import sys
import json
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from raghana.ingest import load_csv, load_pdf
from raghana.clean import clean_csv, clean_pdf
from raghana.chunk import chunk_csv, chunk_pdf_paragraph, Chunk
from raghana.embed import encode
from raghana.vectorstore import VectorStore
from raghana.bm25 import BM25

ARTIFACTS = Path(__file__).parent / "artifacts"
ARTIFACTS.mkdir(exist_ok=True)


def main() -> None:
    print("=" * 60)
    print("RAGhana Index Builder")
    print("=" * 60)
    t_total = time.perf_counter()

    # ── CSV ────────────────────────────────────────────────────
    print("\n[1/6] Loading CSV...")
    raw_df = load_csv()
    print(f"      {len(raw_df)} rows loaded")

    print("[2/6] Cleaning CSV...")
    df = clean_csv(raw_df)
    print(f"      {len(df)} rows after cleaning")
    print(f"      Years: {sorted(df['Year'].unique())}")
    print(f"      Regions: {df['Region'].nunique()} unique regions")

    print("[3/6] Chunking CSV (row-group, 1 chunk per year/region)...")
    csv_chunks = chunk_csv(df)
    print(f"      {len(csv_chunks)} CSV chunks")

    # ── PDF ────────────────────────────────────────────────────
    print("\n[4/6] Loading PDF...")
    raw_pages = load_pdf()
    print(f"      {len(raw_pages)} pages loaded")

    print("[5/6] Cleaning PDF & chunking (paragraph strategy, 512 tokens)...")
    pdf_pages = clean_pdf(raw_pages)
    print(f"      {len(pdf_pages)} pages kept after cleaning")
    pdf_chunks = chunk_pdf_paragraph(pdf_pages, max_tokens=512)
    print(f"      {len(pdf_chunks)} PDF chunks")

    # ── Combine & embed ────────────────────────────────────────
    all_chunks: list[Chunk] = csv_chunks + pdf_chunks
    print(f"\n[6/6] Embedding {len(all_chunks)} total chunks...")
    print("      (first run downloads the model ~90 MB; subsequent runs are fast)")
    texts = [c.text for c in all_chunks]
    t_emb = time.perf_counter()
    embeddings = encode(texts, batch_size=32, show_progress=True)
    print(f"      Embedding shape: {embeddings.shape}")
    print(f"      Embedding time:  {time.perf_counter() - t_emb:.1f}s")

    # ── Save vector store ──────────────────────────────────────
    print("\nSaving vector store...")
    vs = VectorStore()
    vs.add(all_chunks, embeddings)
    vs.save(ARTIFACTS / "embeddings")
    print(f"  -> {ARTIFACTS / 'embeddings.npz'}")
    print(f"  -> {ARTIFACTS / 'embeddings.json'}")

    # ── Save BM25 ──────────────────────────────────────────────
    print("Fitting & saving BM25...")
    bm25 = BM25()
    bm25.fit(texts, [c.chunk_id for c in all_chunks])
    bm25.save(ARTIFACTS / "bm25.pkl")
    print(f"  -> {ARTIFACTS / 'bm25.pkl'}")

    # ── Save chunks JSONL (for notebook exploration) ───────────
    chunks_path = ARTIFACTS / "chunks.jsonl"
    with open(chunks_path, "w", encoding="utf-8") as f:
        for c in all_chunks:
            f.write(json.dumps(c.to_dict(), ensure_ascii=False) + "\n")
    print(f"  -> {chunks_path}")

    elapsed = time.perf_counter() - t_total
    print(f"\nDone in {elapsed:.1f}s")
    print(f"\nSummary:")
    print(f"  CSV chunks    : {len(csv_chunks)}")
    print(f"  PDF chunks    : {len(pdf_chunks)}")
    print(f"  Total chunks  : {len(all_chunks)}")
    print(f"  Embedding dim : {embeddings.shape[1]}")
    print(f"\nNext step: uvicorn api:app --reload")


if __name__ == "__main__":
    main()
