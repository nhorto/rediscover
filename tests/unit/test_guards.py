"""Tests for loop guards."""

import numpy as np
import pytest

from src.app.guards import LoopGuards, cosine_similarity
from src.utils.costs import CostTracker


@pytest.mark.unit
class TestCosineSimilarity:
    def test_identical_vectors(self):
        a = np.array([1.0, 2.0, 3.0])
        assert cosine_similarity(a, a) == pytest.approx(1.0)

    def test_orthogonal_vectors(self):
        a = np.array([1.0, 0.0])
        b = np.array([0.0, 1.0])
        assert cosine_similarity(a, b) == pytest.approx(0.0)

    def test_opposite_vectors(self):
        a = np.array([1.0, 2.0])
        b = np.array([-1.0, -2.0])
        assert cosine_similarity(a, b) == pytest.approx(-1.0)

    def test_zero_vector(self):
        a = np.array([0.0, 0.0])
        b = np.array([1.0, 2.0])
        assert cosine_similarity(a, b) == 0.0


@pytest.mark.unit
class TestLoopGuards:
    def test_initial_state(self):
        guards = LoopGuards()
        assert guards.iteration == 0
        assert guards.best_val_bpb == float("inf")
        assert guards.consecutive_errors == 0

    def test_record_improvement(self):
        guards = LoopGuards()
        guards.record_result(1.5, "keep", "test hypothesis")
        assert guards.best_val_bpb == 1.5
        assert guards.experiments_since_improvement == 0
        assert guards.iteration == 1

    def test_record_no_improvement(self):
        guards = LoopGuards()
        guards.record_result(1.5, "keep", "good one")
        guards.record_result(1.6, "discard", "worse one")
        assert guards.best_val_bpb == 1.5
        assert guards.experiments_since_improvement == 1

    def test_record_crash(self):
        guards = LoopGuards()
        guards.record_result(None, "crash", "broken code")
        assert guards.consecutive_errors == 1
        guards.record_result(None, "crash", "still broken")
        assert guards.consecutive_errors == 2
        guards.record_result(1.5, "keep", "fixed it")
        assert guards.consecutive_errors == 0

    def test_max_iterations_guard(self):
        guards = LoopGuards(max_iterations=3)
        tracker = CostTracker()
        for i in range(3):
            guards.record_result(1.5 - i * 0.01, "keep", f"exp {i}")
        status = guards.check(tracker)
        assert status.should_stop is True
        assert "Max iterations" in status.reason

    def test_budget_guard(self):
        guards = LoopGuards(budget_limit=1.0)
        tracker = CostTracker(budget_limit=1.0)
        tracker.total_cost = 1.5  # over budget
        status = guards.check(tracker)
        assert status.should_stop is True
        assert "Budget" in status.reason

    def test_stuck_detection(self):
        guards = LoopGuards(stuck_threshold=3)
        tracker = CostTracker()
        guards.best_val_bpb = 1.5
        for i in range(3):
            guards.record_result(1.6, "discard", f"bad exp {i}")
        status = guards.check(tracker)
        assert status.should_stop is False
        assert status.should_force_novelty is True
        assert "FUNDAMENTALLY different" in status.novelty_message

    def test_error_cascade(self):
        guards = LoopGuards(error_cascade_limit=2)
        tracker = CostTracker()
        guards.record_result(None, "crash", "broken 1")
        guards.record_result(None, "crash", "broken 2")
        status = guards.check(tracker)
        assert status.should_stop is False
        assert status.should_force_novelty is True
        assert "SIMPLER" in status.novelty_message

    def test_error_hard_stop(self):
        guards = LoopGuards(error_hard_stop=5)
        tracker = CostTracker()
        for i in range(5):
            guards.record_result(None, "crash", f"broken {i}")
        status = guards.check(tracker)
        assert status.should_stop is True
        assert "cascade" in status.reason.lower()

    def test_no_guard_triggered(self):
        guards = LoopGuards()
        tracker = CostTracker()
        guards.record_result(1.5, "keep", "good exp")
        status = guards.check(tracker)
        assert status.should_stop is False
        assert status.should_force_novelty is False

    def test_recent_hypotheses_capped(self):
        guards = LoopGuards(max_hypothesis_history=5)
        for i in range(10):
            guards.record_result(1.5, "keep", f"hypothesis {i}")
        assert len(guards.recent_hypotheses) == 5

    def test_summary(self):
        guards = LoopGuards()
        guards.record_result(1.5, "keep", "test")
        summary = guards.summary()
        assert "Iteration 1" in summary
        assert "1.500000" in summary


@pytest.mark.unit
class TestHypothesisSimilarity:
    def test_no_history_not_similar(self):
        guards = LoopGuards()
        emb = np.random.randn(768).astype(np.float32)
        result = guards.check_similarity(emb)
        assert result.is_too_similar is False
        assert result.most_similar_score == 0.0

    def test_identical_hypothesis_detected(self):
        guards = LoopGuards(similarity_threshold=0.9)
        emb = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
        guards.record_result(1.5, "keep", "reduce head dim to 64")
        guards.record_hypothesis_embedding(emb)

        result = guards.check_similarity(emb)
        assert result.is_too_similar is True
        assert result.most_similar_score == pytest.approx(1.0)

    def test_different_hypothesis_passes(self):
        guards = LoopGuards(similarity_threshold=0.9)
        emb1 = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        emb2 = np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float32)
        guards.record_result(1.5, "keep", "reduce head dim")
        guards.record_hypothesis_embedding(emb1)

        result = guards.check_similarity(emb2)
        assert result.is_too_similar is False

    def test_embeddings_capped(self):
        guards = LoopGuards(max_hypothesis_history=3)
        for i in range(5):
            emb = np.random.randn(768).astype(np.float32)
            guards.record_result(1.5, "keep", f"hyp {i}")
            guards.record_hypothesis_embedding(emb)
        assert len(guards.hypothesis_embeddings) == 3

    def test_similar_hypothesis_returns_text(self):
        guards = LoopGuards(similarity_threshold=0.9)
        emb = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        guards.record_result(1.5, "keep", "try linear attention with ELU kernel")
        guards.record_hypothesis_embedding(emb)

        result = guards.check_similarity(emb)
        assert "linear attention" in result.most_similar_hypothesis
