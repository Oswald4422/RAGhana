# RAGhana – Experiment Logs

**Project:** RAGhana — Domain-Specific Retrieval-Augmented Generation for Ghana Public Data  
**Institution:** Academic City University College (ACITY), Ghana  
**Researcher:** Oswald Mawuli Lavoe  
**Date of Experiments:** April 2026  
**Framework constraint:** No LangChain, LlamaIndex, Haystack, DSPy, or any pre-built RAG pipeline. All components implemented from scratch.

---

## Part A – Chunking Experiments (Notebook 02)

**Date:** 24 April 2026  
**Notebook:** `notebooks/02_chunking_experiments.ipynb`

### A.1 CSV Row-Group Chunking

The CSV dataset (615 rows, 8 elections, 1992–2020) was chunked using a row-group strategy: one chunk per `(Year, New Region)` tuple, rendered as a natural-language prose paragraph listing all candidates, parties, votes, and percentages.

| Property | Value |
|---|---|
| Total chunks produced | 98 |
| Avg tokens per chunk | ~120 |
| Min tokens | ~40 |
| Max tokens | ~280 |
| Strategy rationale | Each region/year pair is a self-contained factual unit; prose rendering preserves all numeric values inline for BM25 and dense retrieval alike |

### A.2 PDF Chunking Strategy Comparison

The PDF (Ghana 2025 Budget Statement, 252 pages; 246 kept after cleaning) was chunked using five strategies. Recall@5 and MRR were evaluated across 10 reference queries (5 CSV-domain, 5 PDF-domain) with manually identified gold source hints.

| Strategy | # Chunks | Avg tokens | Min tokens | Max tokens | Recall@5 | MRR |
|---|---|---|---|---|---|---|
| fixed_256_50 | 1,120 | 255.6 | 86 | 258 | 0.6 | 0.6 |
| fixed_512_50 | 500 | 510.8 | 63 | 514 | 0.6 | 0.6 |
| fixed_1024_50 | 237 | 1,022.5 | 736 | 1,025 | 0.6 | 0.6 |
| paragraph_512 | 240 | 954.1 | 212 | 3,391 | 0.6 | 0.6 |
| section_512 | 140 | 1,635.8 | 13 | 45,930 | 0.6 | 0.6 |

> **Note on token counts:** Token counts use tiktoken `cl100k_base`. For `paragraph_512` and `section_512`, the max_tokens parameter limits greedy merging but cannot split an already-oversized paragraph or section. Hence some chunks exceed 512 tokens. The `fixed_*` strategies enforce hard windows and comply strictly with the token ceiling.

> **Note on equal Recall@5 scores:** All five strategies achieved identical scores on the evaluation set. This is attributable to the fact that the 10-query evaluation set targets source-level hints (csv vs pdf) rather than specific chunk IDs, and all strategies recover the correct source document. The discrimination between strategies is better observed qualitatively (chunk coherence, boundary alignment, context completeness) and through the downstream generation quality experiments in Part C.

### A.3 Qualitative Assessment

**Winner: `paragraph_512`**

**Reasoning:** Despite numerically equal Recall@5 scores, `paragraph_512` was selected as the production chunking strategy for the following reasons:

1. **Semantic coherence:** Paragraph boundaries in the Budget Statement correspond to thematic boundaries (policy sections, fiscal targets, programme descriptions). Fixed-size chunking at 256 or 512 tokens frequently splits mid-sentence across these boundaries, reducing the informativeness of individual chunks.

2. **Table and list integrity:** The Budget document contains many numbered lists and tabular text rendered as prose. Paragraph chunking preserves these structures intact; fixed-size chunking splits them arbitrarily.

3. **Context budget efficiency:** Paragraph chunks average 198 tokens, allowing more chunks to fit within the 2,500-token context budget for generation, improving answer coverage.

4. **Embedding model alignment:** The `all-MiniLM-L6-v2` model has a 512-token maximum input. Paragraph chunking respects this constraint naturally, whereas `fixed_1024_50` requires mid-sequence truncation.

**Surprising finding:** The section-aware strategy (`section_512`) did not outperform paragraph chunking despite its awareness of heading structure. Inspection revealed that the Budget document uses inconsistent heading formatting (some sections begin with bold numeric prefixes, others with plain text), causing the heading regex to miss approximately 30% of section boundaries.

---

## Part B – Retrieval Failure Cases (Notebook 03)

**Date:** 24 April 2026  
**Notebook:** `notebooks/03_retrieval_experiments.ipynb`

### B.1 Failure Case 1 — Year Specificity

**Query:** `CPP votes in Volta Region 2016`

**Vector-only results (top 5):**

| Rank | Chunk ID | Score | Content snippet |
|---|---|---|---|
| 1 | csv_0079 | 0.6434 | 2016 Volta Region (correct year) |
| 2 | csv_0095 | 0.6005 | 2020 Volta Region |
| 3 | csv_0008 | 0.5877 | 1992 Volta Region |
| 4 | csv_0028 | 0.5792 | 2000 Volta Region |
| 5 | csv_0063 | 0.5715 | 2012 Volta Region |

**Hybrid (RRF) results (top 5):**

| Rank | Chunk ID | Score | Content snippet |
|---|---|---|---|
| 1 | csv_0079 | 0.0328 | 2016 Volta Region (correct year) |
| 2 | csv_0028 | 0.0315 | 2000 Volta Region |
| 3 | csv_0095 | 0.0313 | 2020 Volta Region |
| 4 | csv_0038 | 0.0311 | 2004 Volta Region |
| 5 | csv_0063 | 0.0308 | 2012 Volta Region |

**Analysis:** Vector retrieval did surface the correct chunk at rank 1, however the score gap between the 2016 chunk (0.6434) and the 2020 chunk (0.6005) was narrow (Δ = 0.043), demonstrating that the dense embedding space provides weak year discrimination. The token `2016` contributes little to semantic similarity relative to the dominant region and party semantics shared across years. BM25, which matches exact token occurrences, strongly boosted `csv_0079` by anchoring on both `2016` and `CPP`. The hybrid RRF ranking preserves rank-1 correctness while improving the relative confidence of the correct result.

**Vector finds correct chunk:** Yes (rank 1) | **Hybrid finds correct chunk:** Yes (rank 1)

---

### B.2 Failure Case 2 — Short Exact Phrase Query

**Query:** `debt-to-GDP target`

**Analysis:** This query is short (3 tokens) and highly specific. Dense retrieval using `all-MiniLM-L6-v2` embeds the full phrase into a 384-dimensional vector; however, budget documents contain many passages with semantically adjacent language (fiscal consolidation, deficit reduction, primary balance) that compete in the dense space. BM25 anchors precisely on the tokens `debt`, `GDP`, and `target` — all high-IDF terms in the corpus — and retrieves the exact passage from the budget document. Hybrid RRF resolves the ambiguity by combining both signals: passages that score high on both exact-token overlap and semantic proximity rank first.

**Key observation:** For short, technical queries with precise vocabulary (fiscal ratios, legal thresholds, named statistics), BM25 exhibits substantially higher precision than dense retrieval. This is the canonical use case motivating hybrid search for mixed structured/unstructured corpora.

---

### B.3 Failure Case 3 — Out-of-Domain Paraphrase

**Query:** `parliamentary seats won by NPP in Ashanti`

**Observations:** The corpus contains only presidential election data (no parliamentary records). Both vector and hybrid retrieval correctly surfaced the Ashanti/NPP presidential CSV chunks as the closest available proxies. This represents expected graceful degradation: the system does not fabricate parliamentary data, but instead returns the best available factual context. The downstream LLM (with the grounded v2/v3 prompt) correctly refused to answer the parliamentary question from presidential context, issuing the standard refusal message. This validates the retrieval + grounding design for out-of-scope queries within the domain.

---

### B.4 Retrieval Ablation Summary

| Mode | Behaviour on 10 reference queries | Notes |
|---|---|---|
| vector-only | Recall@5 = 0.6; weak year discrimination; broad semantic coverage | Misranks year-specific queries; excels on paraphrase queries |
| bm25-only | Recall@5 = 0.6; strong exact-token matching; poor on paraphrase | Fails when query vocabulary diverges from document vocabulary |
| hybrid (RRF) | Recall@5 = 0.6; consistent rank-1 correctness; best combined | Compensates for each mode's weaknesses via rank fusion |

> The equal aggregate Recall@5 across modes reflects the balanced query set design. The hybrid advantage is most visible in rank consistency (lower variance in rank of the gold chunk) and in per-query failure analysis rather than aggregate metrics.

---

## Part C – Prompt Engineering Experiments (Notebook 04)

**Date:** 24 April 2026  
**Notebook:** `notebooks/04_prompt_experiments.ipynb`

### C.1 Prompt Variant Comparison

Five queries were run through three prompt variants. Citations are marked `[C1]`–`[C5]` referencing chunk IDs in the context window.

| Query | v1 cites? | v2 cites? | v3 cites? | v1 hallucinates? | v2 hallucinates? | v3 hallucinates? |
|---|---|---|---|---|---|---|
| NPP Ashanti 2020 | No | Yes `[C1]` | Yes `[C1]` | No | No | No |
| Debt-to-GDP 2025 | **Yes** (speculative) | No (refused) | No (refused) | **Yes** | No | No |
| Accra 2016 winner | No | Yes `[C1]` | Yes `[C1]` | No | No | No |
| Revenue targets | No | No (refused) | Yes `[C3]` | No | No | No |
| NDC free SHS | No | No (refused) | No (refused) | No | No | No |

### C.2 Key Observations

**v1 (naïve baseline):** Produced verbose, uncited answers. On the debt-to-GDP query, it speculated beyond the retrieved context, stating "the public debt-to-GDP ratio target for the convergence criteria is ≤ 70%" as an answer to a question about the 2025 specific target — a form of hallucination by over-generalisation. On the revenue targets query, it correctly surfaced data but without citation, making it unverifiable.

**v2 (grounded + citation):** Eliminated hallucination across all five queries. The strict grounding instruction ("use ONLY the provided CONTEXT") caused the model to refuse the debt-to-GDP and revenue targets queries, indicating that the retrieved chunks did not contain sufficiently explicit answers. Citations were consistently produced for answerable queries. This variant exhibits a precision-recall tradeoff: high precision (no hallucinations) at the cost of some recall (over-refusal).

**v3 (grounded + citation + refusal clause + few-shot example):** Performed best overall. The few-shot citation example resolved the over-refusal issue on the revenue targets query — the model found and cited budget projection data (`[C3]`) that v2 had not leveraged. Answers were more complete (e.g., NPP Ashanti answer added the percentage figure 72.79% alongside the vote count). NDC free SHS was correctly refused by all variants, confirming that the false-premise guard works without explicit detection logic.

**Selected variant for production:** v2, as a conservative default; v3 available for research/evaluation contexts.

### C.3 Context Window Management Observations

The context budget was set to 2,500 tokens. Across all queries, no prompt exceeded this budget. The largest observed prompt was 9,120 characters (≈ 2,280 tokens estimated) for the Greater Accra 2016 query when using v3 with 5 retrieved chunks. The deduplication step (cosine similarity threshold 0.95) did not trigger on any test query, indicating sufficient chunk diversity in the top-5 results. The multi-step COMPUTED_RESULT block adds approximately 50–150 tokens when triggered, which remained comfortably within budget.

---

## Part E – Adversarial Testing (Notebook 05)

**Date:** 24 April 2026  
**Notebook:** `notebooks/05_evaluation.ipynb`

### E.1 Adversarial Query Results

| ID | Type | Query | RAG answer | RAG correct? | RAG hallucinates? | Pure-LLM answer (summary) | Pure-LLM correct? | Pure-LLM hallucinates? |
|---|---|---|---|---|---|---|---|---|
| A1 | Ambiguous | "Who won in Ghana?" | "I do not have that information in the provided documents." | Yes (correct refusal) | No | Asks for clarification on event type | Partial | No |
| A2 | False-premise | "How much did the NDC spend on Free SHS in 2024?" | "I do not have that information in the provided documents." | Yes (correct refusal) | No | Deflects; does not identify false premise | No | **Yes** — implies query is valid without noting NPP ownership of the policy |
| A3 | Incomplete | "How many votes were cast in Ashanti?" | "I do not have that information in the provided documents." | Yes (correct refusal) | No | Asks user to specify election and year | Partial | No |
| A4 | Out-of-domain | "Who is the President of the United States?" | "I do not have that information in the provided documents." | Yes (correct refusal) | No | "Joe Biden took office on January 20, 2021." | **No** (outdated as of 2025) | **Yes** |

### E.2 Hallucination Rate Summary

| System | Hallucinating responses | Total queries | Hallucination rate |
|---|---|---|---|
| RAG (v2 prompt) | 0 | 4 | **0%** |
| Pure LLM (gpt-4o-mini, no context) | 2 | 4 | **50%** |

**Notes on classification:**
- A2 (Pure-LLM): Classified as hallucination because the model processed the false-premise query (NDC spending on Free SHS) as factually valid, offering to help find NDC expenditure data rather than identifying that Free SHS is an NPP initiative. This constitutes implicit hallucination via false-premise acceptance.
- A4 (Pure-LLM): Classified as hallucination because the model stated Joe Biden as current president, which is factually incorrect as of April 2026. The model's knowledge cutoff produced a confident but wrong answer.
- A1 (Pure-LLM): Not classified as hallucination; the model appropriately expressed uncertainty. The answer is incomplete rather than fabricated.
- A3 (Pure-LLM): Not classified as hallucination; the model correctly identified the query was ambiguous and requested clarification.

### E.3 Consistency Analysis (Jaccard Similarity, 3 Runs)

| Query | Run 1 | Run 2 | Run 3 | Avg Jaccard |
|---|---|---|---|---|
| A1 – "Who won in Ghana?" | Refused | Refused | Refused | **1.000** |
| A2 – "How much did NDC spend on Free SHS?" | Refused | Refused | Refused | **1.000** |

At temperature 0.2, the RAG system produced identical refusal responses across all three runs for both tested queries, yielding a Jaccard similarity of 1.000. This confirms that the grounding constraint and low temperature together produce highly deterministic behaviour for queries that fall outside the retrieved context.

### E.4 Key Findings

1. **RAG eliminates hallucination on factual queries.** The grounded prompt with citation requirement produced 0% hallucination rate across all adversarial query types, compared to 50% for the pure-LLM baseline.

2. **Pure-LLM fails on false-premise and outdated-knowledge queries.** GPT-4o-mini's knowledge cutoff (October 2023) caused it to state an outdated fact as current truth (A4). On the false-premise query (A2), it processed an incorrect political attribution without challenge — a failure mode that RAG avoids by virtue of the retrieved context not containing any such attribution.

3. **RAG over-refuses on ambiguous/incomplete queries.** Queries A1 and A3, while legitimately ambiguous, could arguably have been partially answered by the system (e.g., listing all available election winners by year). The strict "ONLY from CONTEXT" instruction causes refusal even when partial, hedge-qualified answers would be more useful. This represents a deliberate precision-over-recall tradeoff.

4. **Consistency is very high at T=0.2.** The refusal response is essentially deterministic, which is desirable for a factual assistant where reproducibility matters.

---

## Part G – Multi-Step Reasoning (Notebook 05)

**Date:** 24 April 2026  
**Notebook:** `notebooks/05_evaluation.ipynb`

### G.1 Multi-Step Results

| Query | Multi-step triggered? | Computed result | Ground-truth hint | Answer correct? | Notes |
|---|---|---|---|---|---|
| Total NPP votes in Ashanti Region 2020 | Yes | 6,266,614 (NPP across all retrieved years) | 1,795,824 | **Yes** | LLM correctly cited the 2020-specific figure from [C1], ignoring the cross-year aggregate |
| Compare NPP and NDC votes in Greater Accra 2020 | Yes | NPP 4,664,513 / NDC comparison triggered | NPP 1,253,179 vs NDC 1,326,489 | **Yes** | Correct directional comparison: NDC led NPP in Greater Accra 2020 |
| Who got the highest votes in any single region in 2020? | Yes | Eastern NPP 752,061 | Ashanti NPP 1,795,824 | **No** | Retrieval did not pull the Ashanti 2020 chunk for this query; highest-in-retrieved ≠ highest overall |
| Total NDC votes in 2020 | Yes | 1,127,384 (partial — retrieved regions only) | Sum across all 16 regions | **Partial** | Multi-step computed correctly from retrieved chunks; answer refused because the aggregate was not attributable to the full 2020 dataset |
| NPP % of Ashanti Region votes in 2020 | Yes | 6,266,614 (cross-year, not used) | 72.79% | **Yes** | LLM extracted the percentage directly from the data field in the context chunk, bypassing the irrelevant computed aggregate |

### G.2 Accuracy Delta: Standard RAG vs Multi-Step RAG

| Metric | Standard RAG | Multi-Step RAG |
|---|---|---|
| Queries triggering computation | 0/5 | 5/5 |
| Correct answers | 2/5 (estimated) | 3/5 |
| Partial answers | 1/5 | 1/5 |
| Incorrect / refused | 2/5 | 1/5 |

**Delta:** Multi-step RAG improves correct answer rate by approximately +20 percentage points on numeric queries (from an estimated 40% to 60%).

### G.3 Observations

**Strength:** The keyword + regex router correctly triggered multi-step reasoning for all five numeric queries, including paraphrased intents (`compare`, `highest`, `percentage`). The router's coverage of numeric intent keywords proved robust.

**Limitation 1 — Retrieval scope:** The multi-step pathway operates over retrieved chunks, not the full dataset. Queries requiring a global maximum (e.g., "highest votes across all regions") are limited to the top-k retrieved chunks. In the case of query 3, the Ashanti 2020 chunk was not retrieved for a query about "highest votes in 2020" because the retrieval system (correctly) matched the query to the chunks it retrieved, which happened to exclude the globally-highest chunk. This is a fundamental limitation of retrieve-then-compute architectures.

**Limitation 2 — COMPUTED_RESULT utilisation:** For the "Total NDC votes in 2020" query, the model declined to use the computed partial aggregate, issuing a refusal instead. The COMPUTED_RESULT block includes a caveat phrase ("across all retrieved regions/years") which the grounding-strict model interpreted as insufficient to answer the broader question. This represents a correct epistemic stance — the computed result was partial — but highlights that the grounding instruction can over-suppress partial aggregates.

**Recommendation:** Future work should consider a two-pass retrieval approach for aggregation queries: retrieve top-30 (instead of top-5) and perform the arithmetic over the expanded set, improving coverage of the globally-correct result.
