"""Type definitions for the literature domain."""

from dataclasses import dataclass

from src.types import Paper

# Re-export Paper so existing imports from this module still work
__all__ = ["Paper", "SearchResult"]


@dataclass
class SearchResult:
    """A paper returned from a similarity search."""

    paper: Paper
    score: float  # similarity score (higher = more relevant)
