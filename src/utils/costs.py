"""Cost tracking and budget enforcement for LLM API calls."""

from dataclasses import dataclass, field

# Prices per 1M tokens (input, output) as of March 2026
# OpenRouter adds 5.5% platform fee on top of provider costs
MODEL_PRICES: dict[str, tuple[float, float]] = {
    # October 2023 cutoff council (via OpenRouter)
    "openrouter/openai/o1": (15.0, 60.0),
    "openrouter/openai/o1-pro": (150.0, 600.0),
    "openrouter/openai/gpt-4o-2024-05-13": (5.0, 15.0),
    "openrouter/openai/gpt-4": (30.0, 60.0),
    # Open-source alternatives (Dec 2023 / Nov 2023 cutoff)
    "openrouter/mistralai/mistral-7b-instruct-v0.2": (0.10, 0.10),
    "openrouter/mistralai/mixtral-8x7b-instruct": (0.60, 0.60),
    "openrouter/deepseek/deepseek-coder-33b-instruct": (0.80, 0.80),
    # Direct API
    "o1": (15.0, 60.0),
    "gpt-4o-2024-05-13": (5.0, 15.0),
    # Anthropic
    "claude-sonnet-4-5-20250514": (3.0, 15.0),
    "openrouter/anthropic/claude-sonnet-4-5-20250514": (3.0, 15.0),
    # Local (Ollama) — free
    "ollama/deepseek-coder:33b": (0.0, 0.0),
    "ollama/mixtral:8x7b": (0.0, 0.0),
}

# Default: assume $1/$3 per 1M tokens if model not in price list
DEFAULT_PRICE = (1.0, 3.0)


def estimate_cost(model: str, input_tokens: int, output_tokens: int) -> float:
    """Estimate cost in dollars for a single LLM call."""
    input_price, output_price = MODEL_PRICES.get(model, DEFAULT_PRICE)
    return (input_tokens * input_price + output_tokens * output_price) / 1_000_000


@dataclass
class CostTracker:
    """Accumulates LLM API costs and enforces budget caps."""

    budget_limit: float = 50.0  # dollars
    total_cost: float = 0.0
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    call_count: int = 0
    history: list[dict] = field(default_factory=list)

    def record(self, model: str, input_tokens: int, output_tokens: int) -> float:
        """Record a call's cost. Returns the cost. Raises if budget exceeded."""
        cost = estimate_cost(model, input_tokens, output_tokens)
        self.total_cost += cost
        self.total_input_tokens += input_tokens
        self.total_output_tokens += output_tokens
        self.call_count += 1
        self.history.append({
            "model": model,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "cost": cost,
            "cumulative_cost": self.total_cost,
        })
        return cost

    def check_budget(self) -> None:
        """Raise if total cost exceeds budget limit."""
        if self.total_cost >= self.budget_limit:
            raise BudgetExceededError(
                f"Budget exceeded: ${self.total_cost:.2f} >= ${self.budget_limit:.2f} "
                f"after {self.call_count} calls"
            )

    def remaining(self) -> float:
        """Dollars remaining in budget."""
        return max(0.0, self.budget_limit - self.total_cost)

    def summary(self) -> str:
        """Human-readable cost summary."""
        return (
            f"Cost: ${self.total_cost:.4f} / ${self.budget_limit:.2f} "
            f"({self.call_count} calls, "
            f"{self.total_input_tokens:,} in / {self.total_output_tokens:,} out)"
        )


class BudgetExceededError(Exception):
    """Raised when LLM spending exceeds the configured budget."""
