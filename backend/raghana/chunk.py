"""
Chunking strategies for CSV and PDF data.

Three PDF strategies (compared in notebook 02):
  - Fixed-size token chunking (baseline)
  - Paragraph-boundary chunking (winner for this corpus)
  - Section-aware chunking

CSV strategy:
  - Row-group chunking: one chunk per (Year, Region) rendered as prose
"""

import re
import tiktoken
import pandas as pd
from dataclasses import dataclass, field

_TOKENIZER = tiktoken.get_encoding("cl100k_base")


def _count_tokens(text: str) -> int:
    return len(_TOKENIZER.encode(text))


def _decode_tokens(tokens: list[int]) -> str:
    return _TOKENIZER.decode(tokens)


# ──────────────────────────────────────────────────────────
# Chunk dataclass
# ──────────────────────────────────────────────────────────

@dataclass
class Chunk:
    chunk_id: str
    source: str          # "csv" | "pdf"
    text: str
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "chunk_id": self.chunk_id,
            "source": self.source,
            "text": self.text,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Chunk":
        return cls(
            chunk_id=d["chunk_id"],
            source=d["source"],
            text=d["text"],
            metadata=d.get("metadata", {}),
        )


# ──────────────────────────────────────────────────────────
# CSV: row-group chunking
# ──────────────────────────────────────────────────────────

def chunk_csv(df: pd.DataFrame) -> list[Chunk]:
    """
    One chunk per (Year, Region) group, rendered as a natural-language paragraph.

    Justification:
    - Keeps all candidates for one election/region together → self-sufficient context
    - ~13 rows per group → well under the 512-token MiniLM window
    - No overlap needed because row groups are semantically isolated units
    """
    chunks: list[Chunk] = []
    idx = 0
    for (year, region), group in df.groupby(["Year", "Region"], sort=True):
        lines = [f"In the {year} presidential election in {region}:"]
        for _, row in group.iterrows():
            lines.append(
                f"  {row['Candidate']} ({row['Party']}) received "
                f"{int(row['Votes']):,} votes ({row['Votes_pct']:.2f}%)"
            )
        text = "\n".join(lines)
        chunks.append(Chunk(
            chunk_id=f"csv_{idx:04d}",
            source="csv",
            text=text,
            metadata={
                "year": int(year),
                "region": str(region),
                "row_count": int(len(group)),
            },
        ))
        idx += 1
    return chunks


# ──────────────────────────────────────────────────────────
# PDF Strategy 1: Fixed-size token chunking
# ──────────────────────────────────────────────────────────

def chunk_pdf_fixed(
    pages: list[dict],
    max_tokens: int = 512,
    overlap_tokens: int = 50,
) -> list[Chunk]:
    """
    Tokenise the entire PDF as one stream, then slice into windows.

    Justification (baseline comparison):
    - Simple, reproducible
    - Overlap preserves sentences that straddle chunk boundaries
    - Weakness: splits mid-sentence, mid-table, losing semantic coherence
    """
    full_text = "\n\n".join(
        f"[Page {p['page_num']}] {p['text']}" for p in pages
    )
    tokens = _TOKENIZER.encode(full_text)

    chunks: list[Chunk] = []
    idx = 0
    start = 0
    while start < len(tokens):
        end = min(start + max_tokens, len(tokens))
        text = _decode_tokens(tokens[start:end]).strip()
        page_approx = _token_offset_to_page(pages, full_text, start)

        if text:
            chunks.append(Chunk(
                chunk_id=f"pdf_fixed_{idx:04d}",
                source="pdf",
                text=text,
                metadata={
                    "strategy": "fixed",
                    "max_tokens": max_tokens,
                    "overlap_tokens": overlap_tokens,
                    "page_approx": page_approx,
                },
            ))
            idx += 1

        if end >= len(tokens):
            break
        start = end - overlap_tokens

    return chunks


# ──────────────────────────────────────────────────────────
# PDF Strategy 2: Paragraph-boundary chunking  (recommended)
# ──────────────────────────────────────────────────────────

def chunk_pdf_paragraph(
    pages: list[dict],
    max_tokens: int = 512,
) -> list[Chunk]:
    """
    Split text on blank lines; greedily merge paragraphs until max_tokens.

    Justification (winner):
    - Preserves semantic units — paragraphs rarely split a thought
    - No overlap needed because merging only crosses blank-line boundaries
    - Variable chunk size stays within the embedding-model window
    """
    # Collect all paragraphs with page tags
    all_paragraphs: list[dict] = []
    for page in pages:
        for para in re.split(r"\n{2,}", page["text"]):
            para = para.strip()
            if para:
                all_paragraphs.append({"text": para, "page_num": page["page_num"]})

    chunks: list[Chunk] = []
    current_parts: list[dict] = []
    current_tokens = 0
    current_page_start = all_paragraphs[0]["page_num"] if all_paragraphs else 1
    idx = 0

    def _flush():
        nonlocal idx, current_parts, current_tokens, current_page_start
        if not current_parts:
            return
        text = "\n\n".join(p["text"] for p in current_parts)
        chunks.append(Chunk(
            chunk_id=f"pdf_para_{idx:04d}",
            source="pdf",
            text=text,
            metadata={
                "strategy": "paragraph",
                "max_tokens": max_tokens,
                "page_start": current_page_start,
                "page_end": current_parts[-1]["page_num"],
            },
        ))
        idx += 1
        current_parts = []
        current_tokens = 0

    for para in all_paragraphs:
        para_tokens = _count_tokens(para["text"])
        if current_tokens + para_tokens > max_tokens and current_parts:
            _flush()
            current_page_start = para["page_num"]
        current_parts.append(para)
        current_tokens += para_tokens

    _flush()
    return chunks


# ──────────────────────────────────────────────────────────
# PDF Strategy 3: Section-aware chunking
# ──────────────────────────────────────────────────────────

_HEADING_RE = re.compile(
    r"^(?:CHAPTER|SECTION|PART\s+\w+|"
    r"\d{1,2}\.\s+[A-Z][A-Za-z]|"
    r"[A-Z][A-Z\s\-]{4,}(?::|$))",
    re.MULTILINE,
)


def chunk_pdf_section(
    pages: list[dict],
    max_tokens: int = 512,
) -> list[Chunk]:
    """
    Split on heading-like lines, sub-chunk oversized sections with paragraph strategy.

    Justification:
    - Budget PDFs have clear section headings (REVENUE, EXPENDITURE, etc.)
    - Keeps thematic context together within a section
    - Falls back to paragraph splitting when a section is too large
    """
    # Flatten to (page_num, text) pairs
    sections: list[dict] = []
    current_heading = "Preamble"
    current_lines: list[str] = []
    current_page = pages[0]["page_num"] if pages else 1

    for page in pages:
        for line in page["text"].splitlines():
            stripped = line.strip()
            if _HEADING_RE.match(stripped) and len(stripped) > 4:
                if current_lines:
                    sections.append({
                        "heading": current_heading,
                        "text": "\n".join(current_lines).strip(),
                        "page_num": current_page,
                    })
                current_heading = stripped
                current_lines = []
                current_page = page["page_num"]
            else:
                current_lines.append(line)

    if current_lines:
        sections.append({
            "heading": current_heading,
            "text": "\n".join(current_lines).strip(),
            "page_num": current_page,
        })

    chunks: list[Chunk] = []
    idx = 0
    for sec in sections:
        header = f"[{sec['heading']}]\n"
        header_tokens = _count_tokens(header)
        body_tokens = _count_tokens(sec["text"])

        if body_tokens <= max_tokens - header_tokens:
            if sec["text"].strip():
                chunks.append(Chunk(
                    chunk_id=f"pdf_sec_{idx:04d}",
                    source="pdf",
                    text=header + sec["text"],
                    metadata={
                        "strategy": "section",
                        "heading": sec["heading"],
                        "page_num": sec["page_num"],
                    },
                ))
                idx += 1
        else:
            sub = chunk_pdf_paragraph(
                [{"page_num": sec["page_num"], "text": sec["text"]}],
                max_tokens=max(64, max_tokens - header_tokens),
            )
            for sc in sub:
                sc.chunk_id = f"pdf_sec_{idx:04d}"
                sc.text = header + sc.text
                sc.metadata["heading"] = sec["heading"]
                sc.metadata["strategy"] = "section"
                chunks.append(sc)
                idx += 1

    return chunks


# ──────────────────────────────────────────────────────────
# Helper
# ──────────────────────────────────────────────────────────

def _token_offset_to_page(pages: list[dict], full_text: str, token_start: int) -> int:
    """Map a token offset back to an approximate page number."""
    try:
        char_pos = len(_decode_tokens(_TOKENIZER.encode(full_text)[:token_start]))
    except Exception:
        return 1
    cumulative = 0
    for p in pages:
        segment = f"[Page {p['page_num']}] {p['text']}"
        cumulative += len(segment)
        if cumulative >= char_pos:
            return p["page_num"]
    return pages[-1]["page_num"] if pages else 1
