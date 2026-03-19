"""LLM provider: unified interface for all model calls via litellm."""

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
    "implement": "anthropic/claude-sonnet-4-5-20250929",  # Direct Anthropic API
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

        # Scale timeout for large outputs (full-file ~700 lines needs more time)
        effective_max_tokens = max_tokens if max_tokens is not None else self.max_tokens
        api_timeout = 300 if effective_max_tokens > 4096 else 120

        # Use threading-based timeout (signal doesn't work with C-level HTTP calls)
        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(
                litellm.completion,
                model=model,
                messages=messages,
                temperature=temperature if temperature is not None else self.temperature,
                max_tokens=effective_max_tokens,
                timeout=api_timeout,
            )
            try:
                response = future.result(timeout=api_timeout + 30)  # extra buffer
            except concurrent.futures.TimeoutError:
                future.cancel()
                raise APITimeoutError(f"API call exceeded {api_timeout}s timeout")

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
