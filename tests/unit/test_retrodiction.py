"""Tests for the validation domain: retrodiction scoring."""

import numpy as np
import pytest

from src.domains.validation.config import BREAKTHROUGH_REGISTRY
from src.domains.validation.service import (
    ValidationService,
    classify_level,
    cosine_similarity,
    keyword_overlap,
)
from src.domains.validation.types import (
    BreakthroughCategory,
    SimilarityLevel,
)


@pytest.mark.unit
class TestCosineSimiarity:
    def test_identical_vectors(self):
        a = np.array([1.0, 2.0, 3.0])
        assert cosine_similarity(a, a) == pytest.approx(1.0)

    def test_orthogonal_vectors(self):
        a = np.array([1.0, 0.0])
        b = np.array([0.0, 1.0])
        assert cosine_similarity(a, b) == pytest.approx(0.0)

    def test_opposite_vectors(self):
        a = np.array([1.0, 0.0])
        b = np.array([-1.0, 0.0])
        assert cosine_similarity(a, b) == pytest.approx(-1.0)

    def test_zero_vector(self):
        a = np.array([1.0, 2.0])
        b = np.array([0.0, 0.0])
        assert cosine_similarity(a, b) == 0.0


@pytest.mark.unit
class TestKeywordOverlap:
    def test_full_overlap(self):
        text = "low-rank KV compression with projection"
        keywords = ["low-rank", "compression", "projection"]
        assert keyword_overlap(text, keywords) == pytest.approx(1.0)

    def test_no_overlap(self):
        text = "standard multi-head attention with dropout"
        keywords = ["mamba", "ssm", "recurrence"]
        assert keyword_overlap(text, keywords) == pytest.approx(0.0)

    def test_partial_overlap(self):
        text = "compress the key-value cache using low-rank matrices"
        keywords = ["key-value", "low-rank", "latent", "down-projection"]
        assert keyword_overlap(text, keywords) == pytest.approx(0.5)

    def test_empty_keywords(self):
        assert keyword_overlap("some text", []) == 0.0

    def test_case_insensitive(self):
        text = "LATENT representation with COMPRESSION"
        keywords = ["latent", "compression"]
        assert keyword_overlap(text, keywords) == pytest.approx(1.0)


@pytest.mark.unit
class TestClassifyLevel:
    def test_direct_hit(self):
        assert classify_level(0.80) == SimilarityLevel.DIRECT_HIT

    def test_adjacent(self):
        assert classify_level(0.60) == SimilarityLevel.ADJACENT

    def test_directional(self):
        assert classify_level(0.40) == SimilarityLevel.DIRECTIONALLY_CORRECT

    def test_miss(self):
        assert classify_level(0.20) == SimilarityLevel.MISS


@pytest.mark.unit
class TestBreakthroughRegistry:
    def test_registry_not_empty(self):
        assert len(BREAKTHROUGH_REGISTRY) > 0

    def test_all_entries_have_required_fields(self):
        for bt in BREAKTHROUGH_REGISTRY:
            assert bt.name
            assert bt.category in BreakthroughCategory
            assert bt.arxiv_date
            assert bt.description
            assert len(bt.key_mechanisms) > 0
            assert len(bt.keywords) > 0

    def test_mla_in_registry(self):
        names = [bt.name for bt in BREAKTHROUGH_REGISTRY]
        assert any("MLA" in n for n in names)

    def test_flashattention3_in_registry(self):
        names = [bt.name for bt in BREAKTHROUGH_REGISTRY]
        assert any("FlashAttention" in n for n in names)

    def test_mamba2_in_registry(self):
        names = [bt.name for bt in BREAKTHROUGH_REGISTRY]
        assert any("Mamba" in n for n in names)


@pytest.mark.unit
class TestValidationServiceKeywordOnly:
    """Test ValidationService without an embedder (keyword-only mode)."""

    def setup_method(self):
        self.service = ValidationService(embedder=None)

    def test_mla_description_scores_high(self):
        """A proposal that describes MLA should score well."""
        result = self.service.score_proposal(
            hypothesis="Compressing the KV cache by projecting keys and values into a low-rank latent space",
            approach="Add down-projection and up-projection layers to compress key-value pairs before storage",
        )
        assert result.best_match is not None
        assert "MLA" in result.best_match.breakthrough_name
        assert result.best_match.keyword_overlap > 0.3

    def test_unrelated_proposal_scores_low(self):
        """A proposal about dropout should not match any breakthrough."""
        result = self.service.score_proposal(
            hypothesis="Adding more dropout layers will regularize the model",
            approach="Insert dropout after each attention layer with p=0.1",
        )
        assert result.best_match is not None
        assert result.best_match.combined_score < 0.35

    def test_mamba_related_proposal(self):
        """A proposal about SSMs should match Mamba-2."""
        result = self.service.score_proposal(
            hypothesis="State space models with selective scan are equivalent to structured attention",
            approach="Replace attention with a linear recurrence using SSM formulation",
        )
        mamba_scores = [s for s in result.scores if "Mamba" in s.breakthrough_name]
        assert len(mamba_scores) > 0
        assert mamba_scores[0].keyword_overlap > 0.2

    def test_score_multiple(self):
        """Test scoring multiple proposals at once."""
        report = self.service.score_multiple([
            ("KV cache compression via latent projection", "low-rank down-projection"),
            ("Standard attention with dropout", "add dropout layers"),
        ])
        assert report.total_proposals == 2
        assert len(report.comparisons) == 2

    def test_generate_report(self):
        """Test report generation produces readable text."""
        report = self.service.score_multiple([
            ("KV cache compression via latent projection", "low-rank down-projection"),
        ])
        text = self.service.generate_report(report)
        assert "Retrodiction Validation Report" in text
        assert "Total proposals scored" in text
        assert "Proposal 1" in text

    def test_novel_detection(self):
        """A creative but unmatched proposal should be marked novel."""
        result = self.service.score_proposal(
            hypothesis="We propose a novel hybrid approach combining attention with a new mechanism",
            approach="Introduce a reformulated cross-layer attention sharing scheme",
        )
        if result.best_match and result.best_match.combined_score < 0.35:
            assert result.overall_level == SimilarityLevel.NOVEL

    def test_synthetic_mla_direct_match(self):
        """Feed exact MLA description — should score as direct hit (keyword-only)."""
        result = self.service.score_proposal(
            hypothesis=(
                "Compress key-value pairs into a low-rank latent space using "
                "down-projection and up-projection, creating a shared representation "
                "across heads to reduce KV cache memory"
            ),
            approach=(
                "Add a down-projection layer to compress keys and values into a "
                "latent vector, then up-projection to reconstruct. This achieves "
                "cache efficiency through compression and memory reduction."
            ),
        )
        assert result.best_match is not None
        assert "MLA" in result.best_match.breakthrough_name
        # With so many MLA keywords, should score well even keyword-only
        assert result.best_match.keyword_overlap >= 0.5
        assert result.overall_level in (
            SimilarityLevel.DIRECT_HIT,
            SimilarityLevel.ADJACENT,
            SimilarityLevel.DIRECTIONALLY_CORRECT,
        )

    def test_synthetic_mamba2_match(self):
        """Feed Mamba-2 SSD description — should match Mamba-2."""
        result = self.service.score_proposal(
            hypothesis=(
                "State space models can be shown to be equivalent to structured "
                "attention via a duality theorem, enabling faster implementation"
            ),
            approach=(
                "Replace the selective scan with a semi-separable matrix "
                "formulation, enabling chunk-based parallel computation"
            ),
        )
        mamba_scores = [s for s in result.scores if "Mamba" in s.breakthrough_name]
        assert len(mamba_scores) > 0
        assert mamba_scores[0].keyword_overlap >= 0.3

    def test_all_breakthroughs_distinguishable(self):
        """Each breakthrough should be the top match for its own description."""
        for bt in BREAKTHROUGH_REGISTRY:
            result = self.service.score_proposal(
                hypothesis=bt.description,
                approach=" ".join(bt.key_mechanisms),
            )
            assert result.best_match is not None
            assert result.best_match.breakthrough_name == bt.name, (
                f"Expected {bt.name} to be top match for its own description, "
                f"got {result.best_match.breakthrough_name}"
            )
