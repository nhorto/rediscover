"""Tests for cost tracking and budget enforcement."""

import pytest

from src.utils.costs import BudgetExceededError, CostTracker, estimate_cost


@pytest.mark.unit
class TestEstimateCost:
    def test_known_model(self):
        # Mixtral 8x7B via OpenRouter: $0.60/1M input, $0.60/1M output
        cost = estimate_cost("openrouter/mistralai/mixtral-8x7b-instruct", input_tokens=1000, output_tokens=500)
        expected = (1000 * 0.60 + 500 * 0.60) / 1_000_000
        assert abs(cost - expected) < 1e-10

    def test_unknown_model_uses_default(self):
        cost = estimate_cost("some-unknown-model", input_tokens=1000, output_tokens=500)
        expected = (1000 * 1.0 + 500 * 3.0) / 1_000_000
        assert abs(cost - expected) < 1e-10

    def test_zero_tokens(self):
        assert estimate_cost("openrouter/mistralai/mixtral-8x7b-instruct", 0, 0) == 0.0


@pytest.mark.unit
class TestCostTracker:
    def test_record_accumulates(self):
        tracker = CostTracker(budget_limit=100.0)
        tracker.record("openrouter/mistralai/mixtral-8x7b-instruct", 1000, 500)
        tracker.record("openrouter/mistralai/mixtral-8x7b-instruct", 2000, 1000)
        assert tracker.call_count == 2
        assert tracker.total_input_tokens == 3000
        assert tracker.total_output_tokens == 1500
        assert tracker.total_cost > 0

    def test_budget_enforcement(self):
        tracker = CostTracker(budget_limit=0.001)
        tracker.record("openrouter/mistralai/mixtral-8x7b-instruct", 1_000_000, 500_000)  # way over budget
        with pytest.raises(BudgetExceededError):
            tracker.check_budget()

    def test_under_budget_no_raise(self):
        tracker = CostTracker(budget_limit=100.0)
        tracker.record("openrouter/mistralai/mixtral-8x7b-instruct", 100, 50)
        tracker.check_budget()  # should not raise

    def test_remaining(self):
        tracker = CostTracker(budget_limit=10.0)
        tracker.record("openrouter/mistralai/mixtral-8x7b-instruct", 100, 50)
        assert tracker.remaining() < 10.0
        assert tracker.remaining() > 0.0

    def test_summary_format(self):
        tracker = CostTracker(budget_limit=50.0)
        tracker.record("openrouter/mistralai/mixtral-8x7b-instruct", 1000, 500)
        summary = tracker.summary()
        assert "Cost:" in summary
        assert "1 calls" in summary

    def test_history_recorded(self):
        tracker = CostTracker()
        tracker.record("openrouter/mistralai/mixtral-8x7b-instruct", 1000, 500)
        assert len(tracker.history) == 1
        assert tracker.history[0]["model"] == "openrouter/mistralai/mixtral-8x7b-instruct"
        assert "cumulative_cost" in tracker.history[0]
