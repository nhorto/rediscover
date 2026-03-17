"""Ingest older landmark ML papers (2017-2022) from arXiv.

The initial ingestion grabbed mostly 2023 papers (arXiv sorts by most recent).
This script targets earlier years to ensure foundational papers are included.

Usage:
    uv run scripts/ingest_older_papers.py
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.domains.literature.config import CATEGORIES
from src.domains.literature.service import LiteratureService
from src.providers.arxiv import search_papers

CHROMA_PATH = str(Path(__file__).parent.parent / "data" / "chroma_db")

# Year-specific queries to get foundational papers from each era
YEAR_QUERIES = [
    # 2017-2018: Transformer era begins
    ("2017-01-01", "2018-12-31", [
        "attention is all you need transformer",
        "self-attention mechanism neural",
        "transformer language model",
        "multi-head attention",
        "neural machine translation attention",
    ]),
    # 2019-2020: Scaling and efficiency
    ("2019-01-01", "2020-12-31", [
        "transformer efficiency",
        "sparse attention long sequence",
        "language model pretraining",
        "attention mechanism efficient",
        "positional encoding transformer",
        "mixture of experts transformer",
    ]),
    # 2021-2022: Modern techniques
    ("2021-01-01", "2022-12-31", [
        "efficient transformer attention",
        "flash attention io aware",
        "linear attention transformer",
        "state space model sequence",
        "rotary position embedding",
        "multi query attention grouped",
        "low rank adaptation language model",
        "scaling laws language model",
    ]),
]

MAX_RESULTS_PER_QUERY = 300


def run():
    print("Initializing literature service...")
    t0 = time.time()
    service = LiteratureService(chroma_path=CHROMA_PATH)
    print(f"  Model loaded in {time.time() - t0:.1f}s")
    print(f"  Current papers in DB: {service.paper_count}")
    print()

    total_new = 0

    for start_date, end_date, queries in YEAR_QUERIES:
        print(f"{'=' * 50}")
        print(f"Period: {start_date} to {end_date}")
        print(f"{'=' * 50}")

        for query in queries:
            print(f"  Querying: \"{query}\"...")
            t_query = time.time()
            try:
                # Use the arxiv provider directly with custom date range
                papers = search_papers(
                    query=query,
                    categories=CATEGORIES,
                    before_date=end_date,
                    max_results=MAX_RESULTS_PER_QUERY,
                )

                # Filter to only papers within the target date range
                from datetime import UTC, datetime
                start_dt = datetime.strptime(start_date, "%Y-%m-%d").replace(tzinfo=UTC)
                filtered = [p for p in papers if p.published and p.published >= start_dt.isoformat()]

                if not filtered:
                    print("    No papers found in date range")
                    continue

                # Check which are already in the DB
                existing_ids = set()
                if service.paper_count > 0:
                    all_ids = service._collection.get()["ids"]
                    existing_ids = set(all_ids)

                new_papers = [p for p in filtered if p.arxiv_id not in existing_ids]

                if not new_papers:
                    print(f"    All {len(filtered)} papers already in DB")
                    continue

                # Embed and store
                texts = [f"{p.title}. {p.abstract}" for p in new_papers]
                embeddings = service.embedder.encode(texts, show_progress_bar=False, batch_size=64)

                def get_year(published):
                    try:
                        return int(published[:4])
                    except (ValueError, IndexError):
                        return 0

                service._collection.add(
                    ids=[p.arxiv_id for p in new_papers],
                    embeddings=[emb.tolist() for emb in embeddings],
                    documents=texts,
                    metadatas=[
                        {
                            "title": p.title,
                            "abstract": p.abstract[:4000],
                            "year": get_year(p.published),
                            "primary_category": p.primary_category,
                            "published": p.published,
                            "authors": ", ".join(p.authors[:5]),
                        }
                        for p in new_papers
                    ],
                )

                elapsed = time.time() - t_query
                total_new += len(new_papers)
                print(f"    +{len(new_papers)} new papers ({elapsed:.1f}s) | Total: {service.paper_count}")

            except Exception as e:
                print(f"    ERROR: {e}")
                continue

    print()
    print(f"{'=' * 50}")
    print("Older paper ingestion complete!")
    print(f"  New papers added: {total_new}")
    print(f"  Total papers in DB: {service.paper_count}")
    print(f"{'=' * 50}")

    # Verify year distribution
    results = service._collection.get(include=["metadatas"])
    from collections import Counter
    year_counts = Counter(m.get("year", 0) for m in results["metadatas"])
    print()
    print("Year distribution:")
    for year in sorted(year_counts.keys()):
        print(f"  {year}: {year_counts[year]} papers")


if __name__ == "__main__":
    run()
