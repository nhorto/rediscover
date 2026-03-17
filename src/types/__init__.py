"""Shared type definitions for Rediscover. Zero runtime imports from other src/ modules."""

from dataclasses import dataclass


@dataclass
class Paper:
    """An arXiv paper with metadata."""

    arxiv_id: str
    title: str
    abstract: str
    authors: list[str]
    published: str  # ISO 8601 datetime string
    categories: list[str]
    primary_category: str
    pdf_url: str
