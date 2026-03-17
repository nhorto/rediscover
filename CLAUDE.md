Default to using UV instead of pip or conda.

- Use `uv run <file>` instead of `python <file>`
- Use `uv pip install` instead of `pip install`
- Use `uv sync` to install dependencies from `pyproject.toml`
- Use `uv add <package>` to add dependencies
- Use `uv run pytest` to run tests

## Agent Workflow (MANDATORY)

Every non-trivial change follows this sequence. No exceptions.

1. **Plan** → Create exec plan in `docs/exec-plans/active/`, update `docs/PLANS.md`
2. **Implement** → Follow the plan. If the plan changes, update the document first.
3. **Code Review** → Self-review: correctness, edge cases, test coverage
4. **Test** → `uv run pytest` must pass with zero failures
5. **Update Docs** → Update ARCHITECTURE.md if structure changed, move plan to `completed/`

## Repository Knowledge System

All project knowledge lives in the repo. Read the relevant doc before working in that area.

```
ARCHITECTURE.md              Code map — where things are and architectural constraints
docs/
├── PLANS.md                 Index of all execution plans
├── DESIGN.md                System design patterns and research loop
├── PRODUCT_SENSE.md         Product vision, beliefs, north star metric
├── QUALITY_SCORE.md         Quality grades per domain
├── RELIABILITY.md           Long-running operation, error recovery, loop guards
├── SECURITY.md              API keys, sandboxing, cost protection
├── design-docs/             Design decisions and methodology
│   └── retrodiction-validation.md  ← READ THIS for validation approach
├── exec-plans/              Active and completed implementation plans
├── references/              Research and external reference material
│   ├── model-cutoff-dates.md       ← LLM cutoffs and council groupings
│   ├── architecture-breakthroughs.md ← Post-cutoff validation set
│   ├── prior-art.md                ← AI Scientist, autoresearch, etc.
│   ├── hardware-constraints.md     ← M4 Mac feasibility and datasets
│   └── cloud-gpu-options.md        ← Modal/Vast.ai/RunPod comparison (DEFERRED)
└── generated/               Auto-generated docs
```

### Documentation Rules

- If you change the project structure → update `ARCHITECTURE.md`
- If you add tech debt → log in `docs/exec-plans/tech-debt-tracker.md`
- New features need an exec plan in `docs/exec-plans/active/` before implementation
- Completed plans move to `docs/exec-plans/completed/`

## Architectural Constraints

See `ARCHITECTURE.md` for the full code map, dependency flow, and layer model.

**The short version:**
- Dependencies flow: `Utils → Types → Providers → Domains → App`
- Each domain is self-contained: types, config, service
- External dependencies (LLM APIs, arXiv, file I/O) enter through `src/providers/` only
- Evaluation functions are pure — no API calls, no I/O
- No frameworks (no LangChain, no CrewAI) — `litellm` + plain Python only
- Don't hardcode API keys — use `.env`
