# arXiv API & Paper Ingestion Guide

> Everything needed to build the paper ingestion pipeline for Rediscover.
> Last updated: 2026-03-16

## Two Protocols

### REST API (Primary for Search)
- **URL:** `https://export.arxiv.org/api/query`
- Returns Atom XML
- Max 2,000 results per call, 30,000 total
- Rate limit: 1 request per 3 seconds

### OAI-PMH (Bulk Metadata)
- **URL:** `https://oaipmh.arxiv.org/oai`
- Better for complete harvests with date filtering
- Returns resumptionTokens for pagination (1,000 records per page)

## Date Filtering (CRITICAL)

### REST API
Embed in `search_query` using:
```
submittedDate:[YYYYMMDD TO YYYYMMDD]
```

Example — all papers before Dec 31, 2023:
```python
query = '(ti:attention OR ti:transformer) AND (cat:cs.LG OR cat:cs.CL) AND submittedDate:[20170101 TO 20231231]'
```

### OAI-PMH
First-class parameters:
```
&from=2017-01-01&until=2023-12-31
```

## Python Library: `arxiv` (v2.4.1)

```python
import arxiv

client = arxiv.Client(page_size=2000, delay_seconds=3.0, num_retries=5)

search = arxiv.Search(
    query='(ti:attention OR ti:transformer) AND (cat:cs.LG OR cat:cs.CL) AND submittedDate:[20170101 TO 20231231]',
    max_results=50000,
    sort_by=arxiv.SortCriterion.SubmittedDate,
    sort_order=arxiv.SortOrder.Descending,
)

for paper in client.results(search):
    paper.entry_id       # https://arxiv.org/abs/XXXX.XXXXX
    paper.title          # title
    paper.summary        # abstract
    paper.published      # datetime
    paper.categories     # ['cs.LG', 'cs.CL']
    paper.pdf_url        # PDF link
    paper.download_pdf(dirpath="./pdfs/")
    paper.download_source(dirpath="./sources/")  # LaTeX tar.gz
```

## Relevant arXiv Categories

| Category | Covers |
|----------|--------|
| `cs.LG` | Machine Learning — primary venue |
| `cs.CL` | Computation and Language — NLP, transformers for text |
| `cs.AI` | Artificial Intelligence — reasoning, planning |
| `cs.NE` | Neural and Evolutionary Computing — architectures |
| `cs.CV` | Computer Vision — ViT, DETR |
| `stat.ML` | Statistical ML — theory, Bayesian |

## Full Text Access

### Method A: LaTeX Source (Best)
```python
# Download source tarball
url = f"https://arxiv.org/e-print/{arxiv_id}"
# Returns tar.gz with .tex files

# Parse LaTeX to text
from pylatexenc.latex2text import LatexNodes2Text
text = LatexNodes2Text().latex_to_text(latex_content)
```

### Method B: PDF
```python
from pdfminer.high_level import extract_text
text = extract_text("paper.pdf")
```

### Method C: HTML (experimental)
```
https://ar5iv.labs.arxiv.org/html/{arxiv_id}
```

## Semantic Scholar (Complement to arXiv)

**URL:** `https://api.semanticscholar.org/graph/v1`

Adds citation graphs (2.49B citations) that arXiv lacks.

```python
# Lookup by arXiv ID
GET /paper/arXiv:1706.03762?fields=title,abstract,citations,citationCount

# Date filtering (native, clean)
params = {"publicationDateOrYear": ":2023-12-31"}

# Batch lookup
POST /paper/batch
{"ids": ["arXiv:1706.03762", "arXiv:1810.04805"]}
```

Rate limit: 1 req/sec with API key (free, request at semanticscholar.org).

## Building the Knowledge Base

### Recommended Architecture
```
arXiv API → Metadata + IDs → SQLite
                 ↓
    LaTeX Source Download → Text Extraction
                 ↓
    Section-Aware Chunking (512-1024 tokens)
                 ↓
    Embedding (allenai-specter2 for scientific papers)
                 ↓
    Vector DB (Chroma for dev, Qdrant for prod)
                 ↓
    LLM Agent retrieval with date filters
```

### Embedding Model
Use `allenai-specter2` — specifically fine-tuned on scientific citation relationships. Significantly outperforms general embedders on paper retrieval.

```python
from sentence_transformers import SentenceTransformer
model = SentenceTransformer("allenai-specter")
embeddings = model.encode(abstracts, batch_size=64)
```

### Vector Storage
```python
import chromadb
client = chromadb.PersistentClient(path="./chroma_db")
collection = client.create_collection("arxiv_papers")

# Query with date filter
results = collection.query(
    query_embeddings=[query_embedding],
    n_results=20,
    where={"year": {"$lte": 2023}},
)
```

## Corpus Size Estimates

| Scope | Estimated Papers |
|-------|-----------------|
| Title: "attention" in cs.LG | ~4,600 |
| Title: "transformer" in cs.LG | ~7,100 |
| Title: attention/transformer/LM across cs.LG/CL/AI/NE/stat.ML | ~41,600 |
| Abstract: "transformer" AND "attention" across 5 categories | ~8,700 |

**Recommended scope:** Title-based queries across ML categories = ~40-50K papers.

**Feasibility:**
- 50K abstracts via API: ~25 minutes (page_size=2000, 25 calls)
- Embedding 50K abstracts (SPECTER2): ~10 min on GPU, $0.30 via OpenAI
- Full LaTeX for top-cited papers only (Semantic Scholar citationCount filter)

## Rate Limits Summary

| API | Limit | Notes |
|-----|-------|-------|
| arXiv REST | 1 req / 3 sec | No API key needed |
| arXiv OAI-PMH | 1 req / 3 sec | Sequential (resumption tokens) |
| Semantic Scholar | 1 req / sec | Free API key required |
| arXiv S3 | No limit | Requester-pays AWS costs |

## Key Dependencies

```
arxiv>=2.0          # arXiv API client
sentence-transformers  # SPECTER2 embeddings
chromadb            # Vector storage
pylatexenc          # LaTeX → text
pdfminer.six        # PDF → text (fallback)
semanticscholar     # Citation data (optional)
```
