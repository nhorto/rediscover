"""Parsing helpers for council LLM response extraction."""

import re

from src.domains.council.config import MAX_SEARCH_QUERIES
from src.domains.council.types import SearchQuery


def extract_field(text: str, field_name: str) -> str:
    """Extract a named field value from structured LLM output."""
    pattern = rf"{field_name}:\s*(.+?)(?=\n[A-Z_]+:|$)"
    match = re.search(pattern, text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return text.strip()  # Fallback: return entire text if field not found


def extract_list(text: str, section_name: str) -> list[str]:
    """Extract a bulleted list under a section header."""
    pattern = rf"{section_name}:\s*\n((?:\s*-\s*.+\n?)+)"
    match = re.search(pattern, text)
    if match:
        items = re.findall(r"-\s*(.+)", match.group(1))
        return [item.strip() for item in items]
    return []


def parse_search_queries(text: str) -> list[SearchQuery]:
    """Parse search queries from scan step output."""
    queries = []
    parts = re.split(r"---+", text)
    for part in parts:
        query_match = re.search(r"QUERY:\s*(.+)", part)
        rationale_match = re.search(r"RATIONALE:\s*(.+)", part)
        if query_match:
            queries.append(SearchQuery(
                query=query_match.group(1).strip(),
                rationale=rationale_match.group(1).strip() if rationale_match else "",
            ))
    # Fallback: if no structured format found, treat each line as a query
    if not queries:
        for line in text.strip().split("\n"):
            line = line.strip().lstrip("0123456789.-) ")
            if line and len(line) > 5:
                queries.append(SearchQuery(query=line, rationale=""))
    return queries[:MAX_SEARCH_QUERIES]


def clean_code_response(text: str) -> str:
    """Strip markdown code fences from LLM response."""
    text = re.sub(r"^```(?:python)?\s*\n", "", text.strip())
    text = re.sub(r"\n```\s*$", "", text.strip())
    return text.strip()
