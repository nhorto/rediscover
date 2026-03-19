"""LLM provider: unified interface for all model calls via litellm."""

import signal
from dataclasses import dataclass, field

import litellm

from src.utils.costs import CostTracker


class APITimeoutError(Exception):
    """Raised when an API call exceeds the hard timeout."""

# Suppress litellm's verbose logging
litellm.suppress_debug_info = True

# Role → model mapping
# Council roles use knowledge-limited models (Oct 2023 cutoff) for retrodiction.
# Implement uses Sonnet 4.5 with full-file context for code quality.
# TO REVERT: change implement back to "gpt-4o-2024-11-20"
DEFAULT_MODEL_MAP: dict[str, str] = {
    "scan": "gpt-4o-mini",                        # Oct 2023 cutoff
    "propose": "gpt-4o-mini",                      # Oct 2023 cutoff
    "critique": "gpt-4o-mini",                     # Oct 2023 cutoff
    "refine": "gpt-4o-mini",                       # Oct 2023 cutoff
    "implement": "openrouter/anthropic/claude-sonnet-4-5-20250514",  # Full-file code quality
}


@dataclass
class LLMResponse:
    """Response from an LLM call."""

    content: str
    model: str
    role: str
    input_tokens: int
    output_tokens: int
    cost: float


@dataclass
class LLMProvider:
    """Unified LLM interface with model routing, cost tracking, and budget enforcement."""

    model_map: dict[str, str] = field(default_factory=lambda: dict(DEFAULT_MODEL_MAP))
    cost_tracker: CostTracker = field(default_factory=CostTracker)
    temperature: float = 0.7
    max_tokens: int = 2048

    def _call(
        self,
        model: str,
        role: str,
        prompt: str,
        system: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> LLMResponse:
        """Internal: call a model and track costs."""
        self.cost_tracker.check_budget()

        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        # Hard timeout via signal (litellm's timeout doesn't always fire)
        def _timeout_handler(signum, frame):
            raise APITimeoutError("API call exceeded 120s hard timeout")

        old_handler = signal.signal(signal.SIGALRM, _timeout_handler)
        signal.alarm(120)  # 2-minute hard timeout
        try:
            response = litellm.completion(
                model=model,
                messages=messages,
                temperature=temperature if temperature is not None else self.temperature,
                max_tokens=max_tokens if max_tokens is not None else self.max_tokens,
                timeout=90,  # litellm soft timeout
            )
        finally:
            signal.alarm(0)  # cancel alarm
            signal.signal(signal.SIGALRM, old_handler)

        content = response.choices[0].message.content or ""
        usage = response.usage
        input_tokens = usage.prompt_tokens if usage else 0
        output_tokens = usage.completion_tokens if usage else 0

        cost = self.cost_tracker.record(model, input_tokens, output_tokens)
        self.cost_tracker.check_budget()

        return LLMResponse(
            content=content,
            model=model,
            role=role,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost=cost,
        )

    def complete(
        self,
        role: str,
        prompt: str,
        system: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> LLMResponse:
        """Call an LLM based on the role. Routes to the appropriate model."""
        model = self.model_map.get(role)
        if model is None:
            raise ValueError(f"Unknown role: {role}. Available roles: {list(self.model_map.keys())}")
        return self._call(model, role, prompt, system, temperature, max_tokens)

    def complete_raw(
        self,
        model: str,
        prompt: str,
        system: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> LLMResponse:
        """Call a specific model directly (bypasses role routing)."""
        return self._call(model, "raw", prompt, system, temperature, max_tokens)

    def budget_summary(self) -> str:
        """Return cost tracking summary."""
        return self.cost_tracker.summary()
