# Step 3: LLM Provider (litellm wrapper)

> Goal: Unified LLM interface with model routing, cost tracking, and budget enforcement.
> Status: IN PROGRESS
> Started: 2026-03-16

## Progress

- [x] Create src/providers/__init__.py
- [x] Create src/providers/llm.py — litellm wrapper with model routing, caching, cost tracking
- [x] Create src/utils/__init__.py
- [x] Create src/utils/costs.py — cost accumulation and budget enforcement
- [x] Create src/types/__init__.py — shared types
- [x] Create tests/unit/test_llm_provider.py — 7 tests pass (mocked)
- [x] Create tests/unit/test_costs.py — 6 tests pass
- [x] Ruff check passes
- [x] Pytest 31/31 pass
- [x] Update docs

## Status: COMPLETE

## Approach

### src/providers/llm.py
- Thin wrapper around litellm.completion()
- Model routing config: which model for which role (scan, propose, critique, implement)
- Token counting per call
- Cost estimation per call (input + output tokens × model price)
- Cumulative cost tracking
- Budget cap: raise if total spend exceeds limit
- Prompt caching hint support (for Anthropic models)

### src/utils/costs.py
- MODEL_PRICES dict: cost per 1M tokens for each model
- CostTracker class: accumulate costs, check budget, report totals
- Pure functions, no I/O

### Model Assignments (defaults)
- scan: gemini-2.0-flash (cheapest)
- propose: gpt-4-turbo (Dec 2023 cutoff)
- critique: groq/llama-3.1-70b-versatile (Dec 2023 cutoff, cheap via Groq)
- implement: claude-3-5-sonnet-20240620 (April 2024 cutoff, best at code)

## Exit Criteria
- Can call `llm.complete(role="propose", prompt="...")` and get a response type
- Cost tracking accumulates correctly
- Budget cap raises when exceeded
- All tests pass (mocked — no real API calls in tests)
