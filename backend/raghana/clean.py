"""Data cleaning for CSV (election results) and PDF (budget text)."""

import re
import pandas as pd
from collections import Counter


# ──────────────────────────────────────────────────────────
# CSV cleaning
# ──────────────────────────────────────────────────────────

def clean_csv(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalise the Ghana election results DataFrame:
    - Strip BOM from column names (handled by utf-8-sig, but defend anyway)
    - Use 'New Region' as the canonical region name
    - Cast Votes to int and Votes(%) to float
    - Uppercase party codes; strip whitespace throughout
    - Drop rows with missing/zero data
    """
    df = df.copy()

    # Strip BOM / whitespace from column names
    df.columns = [c.strip().lstrip("﻿") for c in df.columns]

    # Canonical region: prefer New Region column
    if "New Region" in df.columns:
        df["Region"] = df["New Region"].str.strip()
    elif "Old Region" in df.columns:
        df["Region"] = df["Old Region"].str.strip()
    else:
        raise KeyError("No region column found in CSV")

    # Year → int
    df["Year"] = pd.to_numeric(df["Year"], errors="coerce").fillna(0).astype(int)
    df = df[df["Year"] > 0]

    # Votes → int (remove commas, cast)
    df["Votes"] = (
        df["Votes"].astype(str).str.replace(",", "").str.strip()
    )
    df["Votes"] = pd.to_numeric(df["Votes"], errors="coerce").fillna(0).astype(int)

    # Votes(%) → float
    pct_col = "Votes(%)" if "Votes(%)" in df.columns else "Votes_pct"
    df["Votes_pct"] = (
        df[pct_col].astype(str).str.replace("%", "").str.strip()
    )
    df["Votes_pct"] = pd.to_numeric(df["Votes_pct"], errors="coerce").fillna(0.0)

    # Party: uppercase, normalise "Others" / "OTHERS" → "OTHERS"
    df["Party"] = df["Party"].astype(str).str.upper().str.strip()

    # Candidate: strip whitespace
    df["Candidate"] = df["Candidate"].astype(str).str.strip()

    # Drop invalid rows
    df = df[df["Candidate"].notna() & (df["Candidate"] != "nan")]
    df = df[df["Votes"] >= 0]

    return df.reset_index(drop=True)


# ──────────────────────────────────────────────────────────
# PDF cleaning
# ──────────────────────────────────────────────────────────

def _detect_boilerplate(pages: list[dict]) -> set[str]:
    """
    Identify lines that appear on >= 80% of pages — these are likely
    headers/footers and should be removed.
    """
    line_counts: Counter = Counter()
    total = len(pages)
    for page in pages:
        seen_lines = set()
        for line in page["text"].splitlines():
            stripped = line.strip()
            if 3 < len(stripped) < 120 and stripped not in seen_lines:
                line_counts[stripped] += 1
                seen_lines.add(stripped)
    threshold = max(3, int(total * 0.8))
    return {line for line, count in line_counts.items() if count >= threshold}


def _is_toc_page(text: str) -> bool:
    """
    Heuristic: a page is a table-of-contents page if more than 50% of
    non-empty lines consist of dot-leaders or bare page numbers.
    """
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    if not lines:
        return True
    dot_or_num = sum(
        1 for l in lines
        if re.search(r"\.{5,}", l) or re.fullmatch(r"\d{1,4}", l)
    )
    return dot_or_num / len(lines) > 0.5


def clean_pdf(pages: list[dict]) -> list[dict]:
    """
    Clean each PDF page:
    1. Remove boilerplate headers/footers
    2. Fix hyphenation at line breaks
    3. Collapse excessive whitespace
    4. Drop ToC / nearly-blank pages
    """
    boilerplate = _detect_boilerplate(pages)
    cleaned = []

    for page in pages:
        lines = page["text"].splitlines()

        # Remove boilerplate lines
        lines = [l for l in lines if l.strip() not in boilerplate]
        text = "\n".join(lines)

        # Fix soft hyphens at line breaks:  "eco-\nnomic" → "economic"
        text = re.sub(r"-\n(\S)", r"\1", text)

        # Collapse 3+ blank lines → 2
        text = re.sub(r"\n{3,}", "\n\n", text)

        # Collapse multiple spaces (but preserve newlines)
        text = re.sub(r"[ \t]{2,}", " ", text)
        text = text.strip()

        if text and not _is_toc_page(text):
            cleaned.append({"page_num": page["page_num"], "text": text})

    return cleaned
