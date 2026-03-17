"""Tests for LLM provider (mocked — no real API calls)."""

from unittest.mock import MagicMock, patch

import pytest

from src.providers.llm import DEFAULT_MODEL_MAP, LLMProvider, LLMResponse
from src.utils.costs import BudgetExceededError


def _mock_completion_response(content="test response", input_tokens=100, output_tokens=50):
    """Create a mock litellm completion response."""
    mock = MagicMock()
    mock.choices = [MagicMock()]
    mock.choices[0].message.content = content
    mock.usage = MagicMock()
    mock.usage.prompt_tokens = input_tokens
    mock.usage.completion_tokens = output_tokens
    return mock


@pytest.mark.unit
class TestLLMProvider:
    @patch("src.providers.llm.litellm.completion")
    def test_complete_returns_response(self, mock_completion):
        mock_completion.return_value = _mock_completion_response("Hello from GPT")
        provider = LLMProvider()
        resp = provider.complete(role="propose", prompt="What should we try?")

        assert isinstance(resp, LLMResponse)
        assert resp.content == "Hello from GPT"
        assert resp.role == "propose"
        assert resp.model == DEFAULT_MODEL_MAP["propose"]
        assert resp.input_tokens == 100
        assert resp.output_tokens == 50
        assert resp.cost > 0

    @patch("src.providers.llm.litellm.completion")
    def test_role_routing(self, mock_completion):
        mock_completion.return_value = _mock_completion_response()
        provider = LLMProvider()

        for role, expected_model in DEFAULT_MODEL_MAP.items():
            provider.complete(role=role, prompt="test")
            call_args = mock_completion.call_args
            assert call_args.kwargs["model"] == expected_model

    def test_unknown_role_raises(self):
        provider = LLMProvider()
        with pytest.raises(ValueError, match="Unknown role"):
            provider.complete(role="nonexistent", prompt="test")

    @patch("src.providers.llm.litellm.completion")
    def test_budget_enforcement(self, mock_completion):
        mock_completion.return_value = _mock_completion_response(input_tokens=1_000_000, output_tokens=500_000)
        provider = LLMProvider()
        provider.cost_tracker.budget_limit = 0.001

        with pytest.raises(BudgetExceededError):
            provider.complete(role="propose", prompt="test")

    @patch("src.providers.llm.litellm.completion")
    def test_cost_tracking(self, mock_completion):
        mock_completion.return_value = _mock_completion_response()
        provider = LLMProvider()
        provider.complete(role="scan", prompt="test1")
        provider.complete(role="scan", prompt="test2")

        assert provider.cost_tracker.call_count == 2
        assert provider.cost_tracker.total_cost > 0
        assert "2 calls" in provider.budget_summary()

    @patch("src.providers.llm.litellm.completion")
    def test_system_message(self, mock_completion):
        mock_completion.return_value = _mock_completion_response()
        provider = LLMProvider()
        provider.complete(role="propose", prompt="test", system="You are a researcher.")

        call_args = mock_completion.call_args
        messages = call_args.kwargs["messages"]
        assert messages[0]["role"] == "system"
        assert messages[0]["content"] == "You are a researcher."
        assert messages[1]["role"] == "user"

    @patch("src.providers.llm.litellm.completion")
    def test_complete_raw(self, mock_completion):
        mock_completion.return_value = _mock_completion_response("raw response")
        provider = LLMProvider()
        resp = provider.complete_raw(model="gpt-4-turbo", prompt="test")

        assert resp.content == "raw response"
        assert resp.role == "raw"
        assert resp.model == "gpt-4-turbo"
