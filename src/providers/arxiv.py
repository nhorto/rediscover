"""arXiv API client: date-filtered paper retrieval."""

from datetime import UTC, datetime

import arxiv

from src.types import Paper

# Default rate limiting is handled by the arxiv library (3s delay)
DEFAULT_PAGE_SIZE = 500
DEFAULT_MAX_RESULTS = 5000


def search_papers(
    query: str,
    categories: list[str] | None = None,
    before_date: str = "2023-12-31",
    max_results: int = DEFAULT_MAX_RESULTS,
) -> list[Paper]:
    """Search arXiv for papers matching query, filtered by date and category.

    Args:
        query: Search terms (searches title and abstract)
        categories: arXiv categories to filter (e.g., ["cs.LG", "cs.CL"])
        before_date: Only return papers submitted before this date (YYYY-MM-DD)
        max_results: Maximum number of papers to return
    """
    # Build query string with category and date filters
    parts = [f"(ti:{query} OR abs:{query})"]

    if categories:
        cat_filter = " OR ".join(f"cat:{cat}" for cat in categories)
        parts.append(f"({cat_filter})")

    # Date filter: submittedDate range
    before_clean = before_date.replace("-", "")
    parts.append(f"submittedDate:[20170101 TO {before_clean}]")

    full_query = " AND ".join(parts)

    client = arxiv.Client(
        page_size=min(DEFAULT_PAGE_SIZE, max_results),
        delay_seconds=3.0,
        num_retries=3,
    )

    search = arxiv.Search(
        query=full_query,
        max_results=max_results,
        sort_by=arxiv.SortCriterion.SubmittedDate,
        sort_order=arxiv.SortOrder.Descending,
    )

    cutoff = datetime.strptime(before_date, "%Y-%m-%d").replace(tzinfo=UTC)

    papers = []
    for result in client.results(search):
        # Extra safety: skip papers after cutoff (in case query filter is imprecise)
        if result.published and result.published > cutoff:
            continue

        papers.append(
            Paper(
                arxiv_id=result.entry_id.split("/")[-1],
                title=result.title.replace("\n", " ").strip(),
                abstract=result.summary.replace("\n", " ").strip(),
                authors=[a.name for a in result.authors],
                published=result.published.isoformat() if result.published else "",
                categories=result.categories,
                primary_category=result.primary_category,
                pdf_url=result.pdf_url or "",
            )
        )

    return papers


def get_paper_by_id(arxiv_id: str) -> Paper | None:
    """Fetch a single paper by arXiv ID."""
    client = arxiv.Client(delay_seconds=3.0, num_retries=3)
    search = arxiv.Search(id_list=[arxiv_id])

    for result in client.results(search):
        return Paper(
            arxiv_id=result.entry_id.split("/")[-1],
            title=result.title.replace("\n", " ").strip(),
            abstract=result.summary.replace("\n", " ").strip(),
            authors=[a.name for a in result.authors],
            published=result.published.isoformat() if result.published else "",
            categories=result.categories,
            primary_category=result.primary_category,
            pdf_url=result.pdf_url or "",
        )
    return None
