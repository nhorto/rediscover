"""Ingest pre-cutoff ML papers from arXiv into the Chroma knowledge base.

Usage:
    uv run scripts/ingest_papers.py                    # run all queries
    uv run scripts/ingest_papers.py --query "attention" # run single query
    uv run scripts/ingest_papers.py --status            # show current DB stats
"""

import argparse
import sys
import time
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.domains.literature.config import CATEGORIES, CUTOFF_DATE
from src.domains.literature.service import LiteratureService

# Queries to ingest, ordered by priority
QUERIES = [
    "attention mechanism",
    "transformer architecture",
    "efficient attention",
    "self-attention",
    "multi-head attention",
    "linear attention",
    "sparse attention",
    "positional encoding",
    "flash attention",
    "language model training",
    "key value cache",
    "grouped query attention",
    "mixture of experts",
    "state space model",
    "recurrent neural network efficient",
]

CHROMA_PATH = str(Path(__file__).parent.parent / "data" / "chroma_db")
MAX_RESULTS_PER_QUERY = 500


def run_ingestion(queries: list[str], max_results: int = MAX_RESULTS_PER_QUERY) -> None:
    """Run paper ingestion for a list of queries."""
    print(f"Initializing literature service (loading SPECTER embedding model)...")
    t0 = time.time()
    service = LiteratureService(chroma_path=CHROMA_PATH)
    print(f"  Model loaded in {time.time() - t0:.1f}s")
    print(f"  Current papers in DB: {service.paper_count}")
    print(f"  Cutoff date: {CUTOFF_DATE}")
    print(f"  Categories: {CATEGORIES}")
    print(f"  Max results per query: {max_results}")
    print()

    total_new = 0
    for i, query in enumerate(queries, 1):
        print(f"[{i}/{len(queries)}] Querying arXiv: \"{query}\"...")
        t_query = time.time()
        try:
            new_count = service.ingest_papers(
                query=query,
                max_results=max_results,
                categories=CATEGORIES,
                cutoff_date=CUTOFF_DATE,
            )
            elapsed = time.time() - t_query
            total_new += new_count
            print(f"  +{new_count} new papers ({elapsed:.1f}s) | Total in DB: {service.paper_count}")
        except Exception as e:
            print(f"  ERROR: {e}")
            continue

    print()
    print(f"{'=' * 50}")
    print(f"Ingestion complete!")
    print(f"  New papers added: {total_new}")
    print(f"  Total papers in DB: {service.paper_count}")
    print(f"  DB location: {CHROMA_PATH}")

    # Quick search test
    print()
    print("Search test: 'improving attention efficiency'")
    results = service.search("improving attention efficiency", n_results=5)
    for r in results:
        print(f"  [{r.score:.3f}] {r.paper.title[:80]}")


def show_status() -> None:
    """Show current database stats."""
    service = LiteratureService(chroma_path=CHROMA_PATH)
    print(f"Papers in DB: {service.paper_count}")
    print(f"DB location: {CHROMA_PATH}")

    if service.paper_count > 0:
        print()
        print("Sample search: 'attention mechanism'")
        results = service.search("attention mechanism", n_results=5)
        for r in results:
            print(f"  [{r.score:.3f}] {r.paper.title[:80]}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ingest arXiv papers into Rediscover knowledge base")
    parser.add_argument("--query", type=str, help="Run a single query instead of all")
    parser.add_argument("--max-results", type=int, default=MAX_RESULTS_PER_QUERY, help="Max papers per query")
    parser.add_argument("--status", action="store_true", help="Show current DB stats")
    args = parser.parse_args()

    if args.status:
        show_status()
    elif args.query:
        run_ingestion([args.query], max_results=args.max_results)
    else:
        run_ingestion(QUERIES, max_results=args.max_results)
