"""
Part G — Multi-step reasoning for numeric queries.

How it works:
  1. Router detects numeric intent via regex + keyword matching
  2. For numeric queries: extract structured rows from CSV chunks,
     compute the answer in Python (exact arithmetic), inject both
     the raw evidence AND the computed figure into the prompt
  3. For non-numeric queries: standard hybrid RAG path

Why this is valuable:
  - Pure LLM often rounds or misremembers vote totals
  - Dense retrieval returns prose; the model must then parse numbers
  - Pre-computing gives the LLM an exact figure to cite, not to invent
"""

import re
from typing import Optional

# ──────────────────────────────────────────────────────────
# Router
# ──────────────────────────────────────────────────────────

_NUMERIC_KW = re.compile(
    r"\b(total|sum|how many|average|mean|percentage|share|"
    r"difference|compare|vs\.?|versus|"
    r"more than|less than|highest|lowest|most|fewest|"
    r"who (?:got|received|had|won)|"
    r"what (?:percentage|share|fraction|number))\b",
    re.IGNORECASE,
)

_YEAR_RE = re.compile(r"\b(19|20)\d{2}\b")

_REGIONS = {
    "ahafo", "ashanti", "bono east", "bono", "brong ahafo",
    "central", "eastern", "greater accra", "north east", "northern",
    "oti", "savannah", "upper east", "upper west",
    "volta", "western north", "western",
}

_PARTIES = {"npp", "ndc", "cpp", "pnc", "ind", "lpg", "apc", "ppp", "gfp", "gum", "gcpp", "ndp"}


def is_numeric_query(query: str) -> bool:
    """Return True when the query is likely asking for a computed number."""
    q = query.lower()
    if _NUMERIC_KW.search(q):
        return True
    # Year + region/party combination suggests a tabular lookup
    if _YEAR_RE.search(q) and (
        any(r in q for r in _REGIONS) or any(p in q for p in _PARTIES)
    ):
        return True
    return False


# ──────────────────────────────────────────────────────────
# Row parser
# ──────────────────────────────────────────────────────────

_ROW_RE = re.compile(
    r"([\w][\w\s\-\.]+?)\s+\(([A-Z0-9]+)\)\s+received\s+([\d,]+)\s+votes\s+\(([\d.]+)%\)",
    re.IGNORECASE,
)

_HEADER_RE = re.compile(
    r"In the (\d{4}) presidential election in (.+?):",
    re.IGNORECASE,
)


def _parse_rows(chunk_text: str) -> list[dict]:
    """
    Extract structured rows from a CSV row-group chunk.

    Expected chunk format:
      In the 2020 presidential election in Ashanti Region:
        Nana Akufo-Addo (NPP) received 1,795,824 votes (72.79%)
        ...
    """
    rows: list[dict] = []
    header = _HEADER_RE.search(chunk_text)
    if not header:
        return rows
    year = int(header.group(1))
    region = header.group(2).strip()
    for m in _ROW_RE.finditer(chunk_text):
        rows.append({
            "year": year,
            "region": region,
            "candidate": m.group(1).strip(),
            "party": m.group(2).upper(),
            "votes": int(m.group(3).replace(",", "")),
            "pct": float(m.group(4)),
        })
    return rows


# ──────────────────────────────────────────────────────────
# Computation logic
# ──────────────────────────────────────────────────────────

def compute_numeric(query: str, chunks: list[dict]) -> Optional[dict]:
    """
    Attempt to answer a numeric query via direct calculation on CSV chunks.

    Returns {'computed_result': str, 'evidence_rows': list[dict]}
    or None if no computation can be performed.
    """
    csv_chunks = [c for c in chunks if c.get("source") == "csv"]
    if not csv_chunks:
        return None

    all_rows: list[dict] = []
    for chunk in csv_chunks:
        all_rows.extend(_parse_rows(chunk["text"]))

    if not all_rows:
        return None

    q = query.lower()

    # ── Who won / highest votes ────────────────────────────────
    if any(w in q for w in ("highest", "most votes", "winner", "who won",
                             "who received the most", "who got the most")):
        winner = max(all_rows, key=lambda r: r["votes"])
        return {
            "computed_result": (
                f"Highest single-region vote count: {winner['candidate']} ({winner['party']}) "
                f"with {winner['votes']:,} votes ({winner['pct']:.2f}%) "
                f"in {winner['region']} ({winner['year']})"
            ),
            "evidence_rows": [winner],
        }

    # ── Total votes for a party ────────────────────────────────
    party_m = re.search(r"\b(npp|ndc|cpp|pnc|ind|lpg)\b", q)
    if party_m and any(w in q for w in ("total", "sum", "how many", "votes")):
        party = party_m.group(1).upper()
        matching = [r for r in all_rows if r["party"] == party]
        if matching:
            total = sum(r["votes"] for r in matching)
            years = sorted({r["year"] for r in matching})
            return {
                "computed_result": (
                    f"Total {party} votes across all retrieved regions/years "
                    f"({', '.join(str(y) for y in years)}): {total:,}"
                ),
                "evidence_rows": matching[:15],
            }

    # ── Comparison between two parties ────────────────────────
    if any(w in q for w in ("compare", " vs ", " versus ")):
        parties_found = [p.upper() for p in _PARTIES if p in q]
        if len(parties_found) >= 2:
            p1, p2 = parties_found[:2]
            v1 = sum(r["votes"] for r in all_rows if r["party"] == p1)
            v2 = sum(r["votes"] for r in all_rows if r["party"] == p2)
            diff = abs(v1 - v2)
            leader = p1 if v1 >= v2 else p2
            return {
                "computed_result": (
                    f"{p1}: {v1:,} votes vs {p2}: {v2:,} votes "
                    f"(difference: {diff:,}; {leader} leads)"
                ),
                "evidence_rows": [
                    r for r in all_rows if r["party"] in (p1, p2)
                ][:15],
            }

    # ── Percentage share for a party ──────────────────────────
    if any(w in q for w in ("percentage", "share", "fraction")) and party_m:
        party = party_m.group(1).upper()
        total_all = sum(r["votes"] for r in all_rows)
        party_votes = sum(r["votes"] for r in all_rows if r["party"] == party)
        if total_all > 0:
            pct = party_votes / total_all * 100
            return {
                "computed_result": (
                    f"{party} share of total votes (retrieved set): "
                    f"{party_votes:,} / {total_all:,} = {pct:.2f}%"
                ),
                "evidence_rows": [r for r in all_rows if r["party"] == party][:10],
            }

    return None


# ──────────────────────────────────────────────────────────
# Prompt injection
# ──────────────────────────────────────────────────────────

def inject_computed_result(user_message: str, result: dict) -> str:
    """
    Prepend a COMPUTED_RESULT block to the user message.
    The system prompt instructs the model to quote this figure verbatim.
    """
    block = (
        "COMPUTED_RESULT "
        "(verified by direct arithmetic on the retrieved data — "
        "you MUST quote this figure verbatim in your answer):\n"
        f"{result['computed_result']}\n\n"
    )
    return block + user_message
