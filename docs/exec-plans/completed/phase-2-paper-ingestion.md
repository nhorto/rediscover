# Phase 2a: Paper Ingestion

> Goal: Populate the Chroma knowledge base with pre-Dec 2023 ML papers from arXiv.
> Status: IN PROGRESS
> Started: 2026-03-16

## Progress

- [ ] Create scripts/ingest_papers.py — CLI script to run paper ingestion
- [ ] Run initial ingestion: "attention" papers from cs.LG, cs.CL (2017-2023)
- [ ] Run second ingestion: "transformer" papers
- [ ] Run third ingestion: broader terms (efficiency, architecture, optimization)
- [ ] Verify paper count and search quality
- [ ] Update docs

## Approach

### Ingestion Script
Simple CLI script that calls LiteratureService.ingest_papers() with different queries.
Runs sequentially (arXiv rate limit: 1 request per 3 seconds).

### Queries to Run (in order)
1. "attention mechanism" — core topic
2. "transformer architecture" — broader architecture papers
3. "efficient attention" — efficiency-focused papers
4. "self-attention" — fundamental attention papers
5. "multi-head attention" — specific mechanism papers
6. "linear attention" — sub-quadratic approaches
7. "sparse attention" — sparsity-based approaches
8. "positional encoding" — PE research
9. "flash attention" — IO-aware attention
10. "language model training" — broader training papers

### Expected Corpus
~3,000-8,000 unique papers (many will overlap across queries, dedup handled by Chroma).

### Time Estimate
- Each query: ~30-60 seconds (arXiv API with rate limiting)
- Embedding: ~2-5 minutes for 1,000 papers with SPECTER
- Total: ~15-30 minutes for full ingestion

## Exit Criteria
- 3,000+ papers in Chroma DB
- `literature.search("KV cache compression")` returns relevant results
- All papers have year <= 2023
