# Step 4: arXiv Paper Ingestion Pipeline

> Goal: Ingest pre-cutoff ML papers into a searchable vector database.
> Status: IN PROGRESS
> Started: 2026-03-16

## Progress

- [x] Create src/providers/arxiv.py — arXiv API client with date filtering, cutoff enforcement
- [x] Create src/domains/literature/__init__.py
- [x] Create src/domains/literature/types.py — Paper, SearchResult types
- [x] Create src/domains/literature/config.py — categories, cutoff date, embedding model
- [x] Create src/domains/literature/service.py — ingest, embed (SPECTER), retrieve via Chroma
- [x] Create tests/unit/test_literature.py — 4 tests pass (mocked arXiv API)
- [x] Ruff check passes
- [x] Pytest 31/31 pass
- [x] Update docs

## Status: COMPLETE

## Approach

### src/providers/arxiv.py
- Wraps the `arxiv` Python library (v2.4.1)
- `search_papers(query, categories, before_date, max_results)` → list of paper metadata
- Date filtering via `submittedDate:[YYYYMMDD TO YYYYMMDD]` in query
- Rate limiting: 3s delay between requests (arxiv library handles this)
- Returns: list of dicts with id, title, abstract, authors, published, categories

### src/domains/literature/types.py
- Paper dataclass: id, title, abstract, authors, published, categories, pdf_url
- SearchResult dataclass: paper, score, chunk_text

### src/domains/literature/config.py
- CUTOFF_DATE = "2023-12-31"
- CATEGORIES = ["cs.LG", "cs.CL", "cs.AI", "cs.NE", "stat.ML"]
- EMBEDDING_MODEL = "allenai-specter" (sentence-transformers, trained on scientific papers)
- CHROMA_PATH = "./data/chroma_db"
- DEFAULT_QUERY_RESULTS = 10

### src/domains/literature/service.py
- `ingest_papers(query, max_results)` — fetch from arXiv, embed abstracts, store in Chroma
- `search(topic, n_results)` → list of SearchResult
- `get_paper(arxiv_id)` → Paper
- Uses Chroma with metadata filtering (year <= 2023)

## Exit Criteria
- Can call `search("attention mechanism efficiency")` and get relevant papers
- Papers are filtered to before cutoff date
- Chroma DB persists between runs
- Tests pass with mocked arXiv responses
