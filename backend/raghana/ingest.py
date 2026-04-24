"""Load raw data from the Dataset/ folder."""

import pandas as pd
import pypdf
from pathlib import Path

DATASET_DIR = Path(__file__).parent.parent.parent / "Dataset"


def load_csv() -> pd.DataFrame:
    """Load Ghana election results CSV (handles UTF-8 BOM automatically)."""
    csv_path = DATASET_DIR / "Ghana_Election_Result.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")
    return pd.read_csv(csv_path, encoding="utf-8-sig")


def load_pdf() -> list[dict]:
    """
    Extract text from the budget PDF page-by-page.
    Returns list of {page_num: int, text: str}.
    """
    pdf_path = DATASET_DIR / "2025-Budget-Statement-and-Economic-Policy_v4.pdf"
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    pages = []
    with open(pdf_path, "rb") as f:
        reader = pypdf.PdfReader(f)
        for i, page in enumerate(reader.pages):
            text = page.extract_text() or ""
            pages.append({"page_num": i + 1, "text": text})
    return pages
