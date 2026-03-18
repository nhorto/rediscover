"""Types for the validation domain: retrodiction scoring of agent proposals."""

from dataclasses import dataclass, field
from enum import Enum


class SimilarityLevel(str, Enum):
    """How closely an agent proposal matches a post-cutoff breakthrough."""

    DIRECT_HIT = "direct_hit"           # Essentially the same mechanism
    ADJACENT = "adjacent"               # Same problem, different valid approach
    DIRECTIONALLY_CORRECT = "directional"  # Right problem space, weaker solution
    NOVEL = "novel"                     # Not in post-cutoff literature
    MISS = "miss"                       # Unrelated to any advance


class BreakthroughCategory(str, Enum):
    """Category of ML breakthrough."""

    ATTENTION = "attention"
    SSM = "ssm"
    MOE = "moe"
    REASONING = "reasoning"


@dataclass
class Breakthrough:
    """A known post-cutoff breakthrough to compare agent proposals against."""

    name: str
    category: BreakthroughCategory
    arxiv_date: str  # YYYY-MM-DD
    venue: str
    description: str
    key_mechanisms: list[str]
    keywords: list[str]


@dataclass
class SimilarityScore:
    """Score for how similar a proposal is to a single breakthrough."""

    breakthrough_name: str
    embedding_similarity: float  # 0.0 to 1.0, cosine similarity
    keyword_overlap: float       # 0.0 to 1.0, fraction of keywords matched
    combined_score: float        # Weighted combination
    level: SimilarityLevel       # Categorical classification


@dataclass
class ProposalComparison:
    """Full comparison of one agent proposal against all breakthroughs."""

    proposal_hypothesis: str
    proposal_approach: str
    scores: list[SimilarityScore] = field(default_factory=list)
    best_match: SimilarityScore | None = None
    overall_level: SimilarityLevel = SimilarityLevel.MISS


@dataclass
class ValidationReport:
    """Summary report across multiple proposals."""

    comparisons: list[ProposalComparison] = field(default_factory=list)
    total_proposals: int = 0
    direct_hits: int = 0
    adjacent: int = 0
    directional: int = 0
    novel: int = 0
    misses: int = 0
