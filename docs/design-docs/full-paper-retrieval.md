# Full Paper Text Retrieval — Detailed Design

> Detailed plan for ingesting, chunking, storing, and retrieving full paper text.
> Status: PLANNED — not yet implemented.
> Last updated: 2026-03-17

## The Problem

Currently we only embed paper abstracts (~100-300 words). This gives the agent enough to know a paper exists and what it's about, but not enough to understand the actual method, math, implementation details, or experimental results. An abstract says "we propose linear attention" — the full paper tells you *how*, what the complexity is, what trade-offs were made, and what the results looked like.

The challenge: full papers are 5,000-15,000 words each. We can't pass 10 full papers (50,000-150,000 words) to a model with a 32K context window. We need a retrieval strategy.

## Architecture

```
                                        ┌─────────────┐
                                        │  arXiv API  │
                                        └──────┬──────┘
                                               │
                                        ┌──────▼──────┐
                                        │  LaTeX/PDF  │
                                        │  Download   │
                                        └──────┬──────┘
                                               │
                                        ┌──────▼──────┐
                                        │   Parser    │
                                        │ (pylatexenc │
                                        │  or pandoc) │
                                        └──────┬──────┘
                                               │
                                        ┌──────▼──────┐
                                        │  Section    │
                                        │  Splitter   │
                                        └──────┬──────┘
                                               │
                                    ┌──────────┼──────────┐
                                    │          │          │
                              ┌─────▼────┐ ┌──▼───┐ ┌───▼────┐
                              │ Abstract │ │ Intro│ │Methods │ ...
                              │  Chunk   │ │Chunk │ │ Chunks │
                              └─────┬────┘ └──┬───┘ └───┬────┘
                                    │          │          │
                                    └──────────┼──────────┘
                                               │
                                        ┌──────▼──────┐
                                        │  SPECTER    │
                                        │  Embedding  │
                                        └──────┬──────┘
                                               │
                                        ┌──────▼──────┐
                                        │   Chroma    │
                                        │  (chunks    │
                                        │ collection) │
                                        └──────┬──────┘
                                               │
                                        ┌──────▼──────┐
                                        │   Agent     │
                                        │  Retrieval  │
                                        └─────────────┘
```

## Step 1: Download Full Paper Text

### Source: LaTeX from arXiv e-print endpoint

Each arXiv paper has source files available at:
```
https://arxiv.org/e-print/{arxiv_id}
```

This returns a tar.gz containing `.tex` files, figures, `.bib`, etc.

### Download strategy

- **Rate limit:** Same 1 request per 3 seconds as the API
- **Storage:** `data/papers/{arxiv_id}/` — one directory per paper
- **Selective download:** Don't download all 5,832 papers. Download on demand when a paper appears in search results and the agent wants to read more. Cache locally after first download.
- **Fallback:** If LaTeX source isn't available (some papers are PDF-only), use PDF extraction via `pdfminer.six`

### Prioritized download

Not all papers need full text. Download full text for:
1. Papers that appear in search results (the agent asked for them)
2. Papers with high citation counts (use Semantic Scholar API to rank)
3. Papers matching the current research direction closely (high similarity score from initial abstract search)

This keeps download volume manageable — maybe 200-500 papers get full text, not 5,832.

### Implementation

```python
# src/providers/paper_downloader.py

class PaperDownloader:
    """Downloads and caches full paper text from arXiv."""

    def __init__(self, cache_dir: str = "data/papers"):
        self.cache_dir = Path(cache_dir)

    def get_full_text(self, arxiv_id: str) -> str | None:
        """Get full paper text. Downloads and caches if not already local."""
        cached = self._check_cache(arxiv_id)
        if cached:
            return cached

        # Try LaTeX source first
        text = self._download_latex(arxiv_id)
        if text:
            self._save_cache(arxiv_id, text)
            return text

        # Fallback to PDF
        text = self._download_pdf(arxiv_id)
        if text:
            self._save_cache(arxiv_id, text)
            return text

        return None

    def _download_latex(self, arxiv_id: str) -> str | None:
        """Download LaTeX source, extract, parse to plain text."""
        url = f"https://arxiv.org/e-print/{arxiv_id}"
        # Download tar.gz → extract .tex files → parse with pylatexenc
        ...

    def _download_pdf(self, arxiv_id: str) -> str | None:
        """Download PDF, extract text."""
        url = f"https://arxiv.org/pdf/{arxiv_id}"
        # Download PDF → extract with pdfminer.six
        ...
```

## Step 2: Parse and Section-Split

### LaTeX parsing

Use `pylatexenc` (already in our dependencies) to convert LaTeX to plain text:
```python
from pylatexenc.latex2text import LatexNodes2Text
converter = LatexNodes2Text()
plain_text = converter.latex_to_text(latex_content)
```

This handles:
- Math environments (`\begin{equation}`) → converted to readable text
- Macros and commands → expanded or stripped
- Citations, references → stripped
- Figures, tables → captions preserved, content stripped

### Section splitting

Scientific papers have consistent structure. Split on section headers:
```python
import re

def split_into_sections(text: str, arxiv_id: str) -> list[dict]:
    """Split paper text into sections."""
    # Match common LaTeX/text section patterns
    section_pattern = r'\n(?:#{1,3}\s+|\d+\.?\s+)([A-Z][^\n]+)\n'

    sections = []
    parts = re.split(section_pattern, text)

    # Standard paper sections to recognize
    KNOWN_SECTIONS = [
        "abstract", "introduction", "related work", "background",
        "method", "methodology", "approach", "model", "architecture",
        "experiments", "results", "evaluation",
        "discussion", "analysis", "ablation",
        "conclusion", "future work",
    ]

    for i, part in enumerate(parts):
        section_name = identify_section(part, KNOWN_SECTIONS)
        sections.append({
            "arxiv_id": arxiv_id,
            "section": section_name,
            "text": part.strip(),
        })

    return sections
```

### Why section-aware splitting matters

Not all sections are equally useful for the agent:
- **Abstract/Introduction:** Good for understanding the paper's contribution at a high level (we already have this via abstracts)
- **Method/Architecture:** **Most valuable** — this is where the actual technical details live
- **Related Work:** Useful for understanding the landscape but often redundant with our paper DB
- **Experiments/Results:** Important for understanding what works and what doesn't
- **Conclusion:** Usually just restates the abstract

The agent should be able to request specific sections: "give me the methods section from paper X."

## Step 3: Chunking Strategy

### Chunk size: 512 tokens with 25% overlap

Why 512:
- SPECTER embedding model has a 512 token limit
- Large enough to contain a complete paragraph or subsection
- Small enough that 10-15 chunks fit in a model's context alongside the prompt

Why 25% overlap:
- Prevents splitting a concept across two chunks where neither chunk has the full picture
- Each chunk shares 128 tokens with its neighbor

### Section-aware chunking

Don't chunk blindly across section boundaries. Each chunk should be tagged with:
- `arxiv_id`: which paper
- `title`: paper title
- `section`: which section (introduction, methods, results, etc.)
- `chunk_index`: position within the section
- `total_chunks`: how many chunks in this section

```python
def chunk_section(section_text: str, arxiv_id: str, title: str,
                  section_name: str, chunk_size: int = 512,
                  overlap: int = 128) -> list[dict]:
    """Chunk a section with overlap, preserving metadata."""
    words = section_text.split()
    chunks = []

    for i in range(0, len(words), chunk_size - overlap):
        chunk_words = words[i:i + chunk_size]
        if len(chunk_words) < 50:  # skip tiny trailing fragments
            continue

        chunks.append({
            "id": f"{arxiv_id}__{section_name}__{len(chunks)}",
            "text": " ".join(chunk_words),
            "metadata": {
                "arxiv_id": arxiv_id,
                "title": title,
                "section": section_name,
                "chunk_index": len(chunks),
                "year": int(arxiv_id[:2]) + 2000 if arxiv_id[:2].isdigit() else 0,
            }
        })

    # Update total_chunks count
    for chunk in chunks:
        chunk["metadata"]["total_chunks_in_section"] = len(chunks)

    return chunks
```

## Step 4: Storage in Chroma

### Separate collection for chunks

Don't mix abstract embeddings and chunk embeddings in the same collection. Use two collections:
- `arxiv_papers` (existing): abstract-level embeddings for broad discovery search
- `arxiv_chunks`: chunk-level embeddings for deep reading

```python
# Two-tier retrieval
abstract_collection = client.get_collection("arxiv_papers")     # broad search
chunk_collection = client.get_collection("arxiv_chunks")         # deep reading
```

### Metadata for filtering

Each chunk stored with:
```python
{
    "arxiv_id": "2205.14135",
    "title": "FlashAttention: Fast and Memory-Efficient...",
    "section": "methods",
    "chunk_index": 3,
    "total_chunks_in_section": 7,
    "year": 2022,
}
```

This enables:
- `where={"arxiv_id": "2205.14135"}` → get all chunks from one paper
- `where={"section": "methods"}` → get only methods sections
- `where={"year": {"$lte": 2023}}` → enforce cutoff date
- Semantic search across all chunks

## Step 5: Two-Tier Retrieval

### How the agent uses this

The agent gets two levels of retrieval:

**Tier 1: Broad discovery (existing — abstract embeddings)**
```
Agent: "What papers exist about KV cache compression?"
→ Search abstract collection
→ Returns: 10 papers with titles, abstracts, similarity scores
→ Agent picks 2-3 papers that look most relevant
```

**Tier 2: Deep reading (new — chunk embeddings)**
```
Agent: "Give me the methods section from paper 2205.14135"
→ Filter chunk collection by arxiv_id + section="methods"
→ Returns: 5-7 chunks covering the full methods section
→ Agent reads the actual implementation details
```

Or the agent can do a targeted deep search:
```
Agent: "Find detailed descriptions of kernel-based attention approximations"
→ Search chunk collection (semantic search)
→ Returns: 10 chunks from various papers' methods sections
→ Agent gets implementation-level detail from multiple papers
```

### Context budget per council step

| Council Step | Tier 1 (abstracts) | Tier 2 (chunks) | Max context from papers |
|-------------|-------------------|-----------------|----------------------|
| Scan | Not used (just generates queries) | Not used | 0 |
| Propose | 5 paper abstracts (~1,500 tokens) | 5 deep chunks from top 2 papers (~2,500 tokens) | ~4,000 tokens |
| Critique | Proposal text only | Not used | 0 |
| Refine | Not used | Not used | 0 |
| Implement | Not used | Not used | 0 |

Total paper context per experiment cycle: ~4,000 tokens. At $0.60/M tokens for Mixtral, that's $0.0024 per cycle. Negligible.

### The "drill down" flow

```
1. SCAN generates search queries
2. Tier 1 search returns 10 paper abstracts
3. PROPOSE reads abstracts, identifies 2-3 papers it wants to read deeper
4. Tier 2 retrieval gets methods/results chunks from those 2-3 papers
5. PROPOSE reads the chunks and forms a specific, informed hypothesis
6. Rest of pipeline continues as before
```

This means the propose step makes TWO LLM calls:
1. First call: read abstracts, decide which papers to read deeply, and what sections to request
2. Second call: read the deep chunks, form the final hypothesis

Or we could do this in one call by always including the top 5 chunks alongside the abstracts. Simpler, slightly more tokens, but avoids the round-trip.

## Step 6: On-Demand Download Pipeline

### Don't download everything upfront

5,832 papers × ~1MB each = ~6GB of LaTeX sources. Downloading all of them takes ~5 hours at arXiv's rate limit. Not practical.

Instead, download on demand:

```
1. Abstract search returns paper X with high relevance
2. Agent wants to read paper X deeply
3. Check cache: data/papers/{arxiv_id}/text.txt
4. If not cached: download LaTeX → parse → chunk → embed → store in Chroma chunks collection
5. Return chunks to agent
6. Paper is now cached for future requests
```

Over time, the most relevant papers get cached. After a few experiment sessions, the most important 200-500 papers will be locally available.

### Cache structure

```
data/
├── chroma_db/                    # Abstract embeddings (existing)
├── chroma_chunks_db/             # Chunk embeddings (new)
└── papers/                       # Raw text cache
    ├── 2205.14135/
    │   ├── source.tar.gz         # Original download
    │   ├── text.txt              # Parsed full text
    │   └── sections.json         # Section-split metadata
    ├── 1706.03762/
    │   ├── source.tar.gz
    │   ├── text.txt
    │   └── sections.json
    └── ...
```

## Step 7: Implementation Order

1. **PaperDownloader** (`src/providers/paper_downloader.py`)
   - Download LaTeX/PDF from arXiv
   - Parse to plain text
   - Cache locally

2. **Chunker** (add to `src/domains/literature/service.py` or new file)
   - Section splitting
   - Section-aware chunking with overlap
   - Metadata tagging

3. **Chunk collection** (extend `LiteratureService`)
   - New Chroma collection for chunks
   - `ingest_full_paper(arxiv_id)` — download, parse, chunk, embed, store
   - `search_chunks(query, arxiv_id=None, section=None)` — search with filters
   - `get_paper_sections(arxiv_id)` — get all chunks for one paper

4. **Council integration** (update `src/domains/council/service.py`)
   - Propose step: after abstract search, select papers for deep reading
   - Retrieve chunks for selected papers
   - Include chunks in propose prompt

5. **Tests**
   - Mock downloads (don't hit arXiv in tests)
   - Test chunking logic with sample text
   - Test two-tier retrieval flow

## Cost and Performance Estimates

### Download time
- Per paper: 3-5 seconds (rate limit + download + parse)
- First session (cache cold): ~10-20 papers downloaded = 30-100 seconds
- Subsequent sessions: cached, instant

### Storage
- Full text per paper: ~30-50KB
- Chunks per paper: ~10-20 chunks × 512 tokens = ~2-4KB text + embeddings
- 500 cached papers: ~25MB text + ~50MB embeddings
- Chroma DB: ~100MB total for chunks collection

### Token cost
- ~4,000 extra tokens per experiment cycle for paper context
- At Mixtral pricing ($0.60/M): $0.0024 per cycle
- 100 experiments: $0.24 additional cost
- Negligible impact on budget

## Risks

| Risk | Impact | Mitigation |
|------|--------|------------|
| LaTeX parsing fails on some papers | Missing papers | Fall back to PDF extraction |
| arXiv rate limiting during experiments | Slow first few experiments | Pre-cache top-cited papers before running loop |
| Chunks too large for SPECTER 512 limit | Bad embeddings | Enforce 512 token max, truncate if needed |
| Section detection fails | Bad chunk boundaries | Fallback to fixed-size chunking without section awareness |
| Too many chunks returned | Context overflow | Hard cap at 10 chunks per retrieval call |

## Dependencies

Already installed:
- `pylatexenc` — LaTeX to text
- `pdfminer.six` — PDF to text (add to pyproject.toml if not present)
- `sentence-transformers` — SPECTER embeddings
- `chromadb` — vector storage
- `requests` — HTTP downloads

New:
- None needed

## Not Doing (Deliberate Omissions)

- **No citation graph traversal.** We could follow citations to find related papers, but this adds complexity and Semantic Scholar API dependency. Keep it simple — the agent's search queries are good enough.
- **No figure/table extraction.** Figures and tables are lost in text conversion. For ML architecture papers, the architecture diagrams would be valuable, but multimodal embeddings add significant complexity.
- **No summarization step.** We discussed using a model to summarize papers, but the contamination risk is real. Pass raw chunks and let the cutoff-safe council models read them directly.
