"""Tests for literature domain (mocked arXiv API — no real network calls)."""

from datetime import UTC
from unittest.mock import patch

import pytest

from src.domains.literature.types import SearchResult
from src.types import Paper


def _make_paper(arxiv_id="2301.00001", title="Test Paper", abstract="Test abstract about attention."):
    return Paper(
        arxiv_id=arxiv_id,
        title=title,
        abstract=abstract,
        authors=["Author One", "Author Two"],
        published="2023-01-15T00:00:00+00:00",
        categories=["cs.LG", "cs.CL"],
        primary_category="cs.LG",
        pdf_url=f"https://arxiv.org/pdf/{arxiv_id}",
    )


@pytest.mark.unit
class TestPaperTypes:
    def test_paper_creation(self):
        paper = _make_paper()
        assert paper.arxiv_id == "2301.00001"
        assert paper.title == "Test Paper"
        assert len(paper.authors) == 2

    def test_search_result(self):
        paper = _make_paper()
        result = SearchResult(paper=paper, score=0.95)
        assert result.score == 0.95
        assert result.paper.title == "Test Paper"


@pytest.mark.unit
class TestArxivProvider:
    @patch("src.providers.arxiv.arxiv.Client")
    def test_search_papers_returns_papers(self, mock_client_cls):
        """Test that search_papers converts arxiv results to Paper objects."""
        from datetime import datetime

        from src.providers.arxiv import search_papers

        # Mock a single arxiv result
        mock_result = type("Result", (), {
            "entry_id": "http://arxiv.org/abs/2301.00001v1",
            "title": "Attention Is Improved",
            "summary": "We improve attention mechanisms.",
            "authors": [type("Author", (), {"name": "Jane Doe"})()],
            "published": datetime(2023, 1, 15, tzinfo=UTC),
            "categories": ["cs.LG"],
            "primary_category": "cs.LG",
            "pdf_url": "https://arxiv.org/pdf/2301.00001",
        })()

        mock_client = mock_client_cls.return_value
        mock_client.results.return_value = iter([mock_result])

        papers = search_papers("attention", categories=["cs.LG"], before_date="2023-12-31", max_results=10)

        assert len(papers) == 1
        assert papers[0].arxiv_id == "2301.00001v1"
        assert papers[0].title == "Attention Is Improved"
        assert papers[0].authors == ["Jane Doe"]

    @patch("src.providers.arxiv.arxiv.Client")
    def test_search_papers_filters_by_date(self, mock_client_cls):
        """Papers after the cutoff date should be excluded."""
        from datetime import datetime

        from src.providers.arxiv import search_papers

        old_paper = type("Result", (), {
            "entry_id": "http://arxiv.org/abs/2301.00001v1",
            "title": "Old Paper",
            "summary": "Old research.",
            "authors": [],
            "published": datetime(2023, 1, 1, tzinfo=UTC),
            "categories": ["cs.LG"],
            "primary_category": "cs.LG",
            "pdf_url": "",
        })()

        new_paper = type("Result", (), {
            "entry_id": "http://arxiv.org/abs/2401.00001v1",
            "title": "New Paper",
            "summary": "Post-cutoff research.",
            "authors": [],
            "published": datetime(2024, 1, 15, tzinfo=UTC),
            "categories": ["cs.LG"],
            "primary_category": "cs.LG",
            "pdf_url": "",
        })()

        mock_client = mock_client_cls.return_value
        mock_client.results.return_value = iter([old_paper, new_paper])

        papers = search_papers("attention", before_date="2023-12-31", max_results=10)

        assert len(papers) == 1
        assert papers[0].title == "Old Paper"
