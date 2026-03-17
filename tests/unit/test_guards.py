"""Tests for loop guards."""

import pytest

from src.app.guards import LoopGuards
from src.utils.costs import CostTracker


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

    def test_no_guard_triggered(self):
        guards = LoopGuards()
        tracker = CostTracker()
        guards.record_result(1.5, "keep", "good exp")
        status = guards.check(tracker)
        assert status.should_stop is False
        assert status.should_force_novelty is False

    def test_recent_hypotheses_capped(self):
        guards = LoopGuards()
        for i in range(10):
            guards.record_result(1.5, "keep", f"hypothesis {i}")
        assert len(guards.recent_hypotheses) == 5

    def test_summary(self):
        guards = LoopGuards()
        guards.record_result(1.5, "keep", "test")
        summary = guards.summary()
        assert "Iteration 1" in summary
        assert "1.500000" in summary
