"""Validation service: retrodiction scoring of agent proposals against breakthroughs."""

import re

import numpy as np

from src.domains.validation.config import (
    ADJACENT_THRESHOLD,
    BREAKTHROUGH_REGISTRY,
    DIRECT_HIT_THRESHOLD,
    DIRECTIONAL_THRESHOLD,
    EMBEDDING_WEIGHT,
    KEYWORD_WEIGHT,
)
from src.domains.validation.types import (
    Breakthrough,
    ProposalComparison,
    SimilarityLevel,
    SimilarityScore,
    ValidationReport,
)


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine similarity between two vectors."""
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


def keyword_overlap(proposal_text: str, keywords: list[str]) -> float:
    """Compute fraction of breakthrough keywords found in proposal text."""
    if not keywords:
        return 0.0
    text_lower = proposal_text.lower()
    matches = sum(1 for kw in keywords if kw.lower() in text_lower)
    return matches / len(keywords)


def classify_level(score: float) -> SimilarityLevel:
    """Map a combined score to a SimilarityLevel."""
    if score >= DIRECT_HIT_THRESHOLD:
        return SimilarityLevel.DIRECT_HIT
    if score >= ADJACENT_THRESHOLD:
        return SimilarityLevel.ADJACENT
    if score >= DIRECTIONAL_THRESHOLD:
        return SimilarityLevel.DIRECTIONALLY_CORRECT
    return SimilarityLevel.MISS


class ValidationService:
    """Compares agent proposals against the breakthrough registry."""

    def __init__(self, embedder=None):
        """Initialize with an optional sentence embedder.

        If embedder is None, only keyword-based scoring is used.
        The embedder should have an .encode(list[str]) method returning np.ndarray.
        """
        self.embedder = embedder
        self._breakthrough_embeddings: dict[str, np.ndarray] = {}

    def _get_breakthrough_embedding(self, bt: Breakthrough) -> np.ndarray | None:
        """Get or compute embedding for a breakthrough's description + mechanisms."""
        if self.embedder is None:
            return None
        if bt.name not in self._breakthrough_embeddings:
            text = f"{bt.description} {' '.join(bt.key_mechanisms)}"
            self._breakthrough_embeddings[bt.name] = self.embedder.encode([text])[0]
        return self._breakthrough_embeddings[bt.name]

    def score_proposal(
        self,
        hypothesis: str,
        approach: str,
        breakthroughs: list[Breakthrough] | None = None,
    ) -> ProposalComparison:
        """Score a single proposal against all breakthroughs."""
        if breakthroughs is None:
            breakthroughs = BREAKTHROUGH_REGISTRY

        proposal_text = f"{hypothesis} {approach}"
        proposal_embedding = None
        if self.embedder is not None:
            proposal_embedding = self.embedder.encode([proposal_text])[0]

        scores: list[SimilarityScore] = []
        for bt in breakthroughs:
            # Embedding similarity
            emb_sim = 0.0
            if proposal_embedding is not None:
                bt_emb = self._get_breakthrough_embedding(bt)
                if bt_emb is not None:
                    emb_sim = cosine_similarity(proposal_embedding, bt_emb)

            # Keyword overlap
            kw_score = keyword_overlap(proposal_text, bt.keywords)

            # Combined score
            if self.embedder is not None:
                combined = EMBEDDING_WEIGHT * emb_sim + KEYWORD_WEIGHT * kw_score
            else:
                combined = kw_score  # keyword-only fallback

            level = classify_level(combined)
            scores.append(SimilarityScore(
                breakthrough_name=bt.name,
                embedding_similarity=emb_sim,
                keyword_overlap=kw_score,
                combined_score=combined,
                level=level,
            ))

        # Sort by combined score descending
        scores.sort(key=lambda s: s.combined_score, reverse=True)
        best = scores[0] if scores else None

        # Determine overall level — if best match is MISS, check for novelty
        overall = best.level if best else SimilarityLevel.MISS
        if overall == SimilarityLevel.MISS and self._looks_novel(proposal_text):
            overall = SimilarityLevel.NOVEL

        return ProposalComparison(
            proposal_hypothesis=hypothesis,
            proposal_approach=approach,
            scores=scores,
            best_match=best,
            overall_level=overall,
        )

    def score_multiple(
        self,
        proposals: list[tuple[str, str]],
    ) -> ValidationReport:
        """Score multiple proposals. Each tuple is (hypothesis, approach)."""
        report = ValidationReport(total_proposals=len(proposals))

        for hypothesis, approach in proposals:
            comparison = self.score_proposal(hypothesis, approach)
            report.comparisons.append(comparison)

            match comparison.overall_level:
                case SimilarityLevel.DIRECT_HIT:
                    report.direct_hits += 1
                case SimilarityLevel.ADJACENT:
                    report.adjacent += 1
                case SimilarityLevel.DIRECTIONALLY_CORRECT:
                    report.directional += 1
                case SimilarityLevel.NOVEL:
                    report.novel += 1
                case SimilarityLevel.MISS:
                    report.misses += 1

        return report

    def generate_report(self, report: ValidationReport) -> str:
        """Generate a human-readable comparison report."""
        lines = ["# Retrodiction Validation Report", ""]
        lines.append(f"**Total proposals scored:** {report.total_proposals}")
        lines.append(f"- Direct hits: {report.direct_hits}")
        lines.append(f"- Adjacent: {report.adjacent}")
        lines.append(f"- Directionally correct: {report.directional}")
        lines.append(f"- Novel: {report.novel}")
        lines.append(f"- Misses: {report.misses}")
        lines.append("")

        for i, comp in enumerate(report.comparisons, 1):
            lines.append(f"## Proposal {i}")
            lines.append(f"**Hypothesis:** {comp.proposal_hypothesis[:200]}")
            lines.append(f"**Overall:** {comp.overall_level.value}")
            if comp.best_match:
                bm = comp.best_match
                lines.append(
                    f"**Best match:** {bm.breakthrough_name} "
                    f"(combined={bm.combined_score:.3f}, "
                    f"embedding={bm.embedding_similarity:.3f}, "
                    f"keywords={bm.keyword_overlap:.3f})"
                )
            # Show top 3 scores
            for s in comp.scores[:3]:
                lines.append(
                    f"  - {s.breakthrough_name}: {s.combined_score:.3f} "
                    f"({s.level.value})"
                )
            lines.append("")

        return "\n".join(lines)

    @staticmethod
    def _looks_novel(proposal_text: str) -> bool:
        """Heuristic: does the proposal seem to contain novel ideas?

        Checks for indicators of creative proposals vs generic/trivial ones.
        """
        novelty_indicators = [
            r"novel", r"new\s+(?:approach|method|mechanism|technique)",
            r"propose", r"introduce", r"combine.*with",
            r"hybrid", r"reformulat",
        ]
        text_lower = proposal_text.lower()
        return any(re.search(p, text_lower) for p in novelty_indicators)
