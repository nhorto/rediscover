# Rediscover

## How to Orient in This Repo (Read This First)

> This is a map, not a manual. Read what you need for your current task.

| If you need to... | Read this |
|-------------------|-----------|
| Understand the project purpose | docs/PRODUCT_SENSE.md |
| See the code map and architecture | ARCHITECTURE.md |
| See what's been built and what's in progress | docs/PLANS.md |
| See active work | docs/exec-plans/active/ |
| See completed work and past decisions | docs/exec-plans/completed/ |
| Understand a design decision | docs/design-docs/index.md |
| Understand the validation methodology | docs/design-docs/retrodiction-validation.md |
| Check quality standards | docs/QUALITY_SCORE.md |
| Know model cutoff dates and council groupings | docs/references/model-cutoff-dates.md |

## Agent Workflow (MANDATORY — No Exceptions)

Every non-trivial change follows this sequence. These are not suggestions.

### Before You Start
1. Read ARCHITECTURE.md to understand the code map
2. Read docs/PLANS.md to see what's done and what's in progress
3. If your task relates to an existing plan, read that plan

### Plan
4. Create or update an exec plan in `docs/exec-plans/active/`
5. Update `docs/PLANS.md` with the plan entry (status: ACTIVE)

### Implement
6. Follow the plan. If the plan needs to change, update the plan document FIRST, then implement.

### Verify
7. Self-review: correctness, edge cases, test coverage
8. Run `uv run pytest` — zero failures required
9. Run architecture tests: `uv run pytest tests/architecture/` — zero failures required

### Update Documentation (DO NOT SKIP)
10. If the plan is complete:
    - Move the plan file from `docs/exec-plans/active/` to `docs/exec-plans/completed/`
    - Update `docs/PLANS.md`: move entry from Active to Completed table
11. If you changed project structure → update `ARCHITECTURE.md`
12. If you changed capabilities → update "Current Guarantees" and "What Does NOT Exist" in ARCHITECTURE.md
13. If quality changed → update `docs/QUALITY_SCORE.md`
14. If you introduced tech debt → log in `docs/exec-plans/tech-debt-tracker.md`

## Tools & Package Management

- Use `uv run <file>` instead of `python <file>`
- Use `uv pip install` instead of `pip install`
- Use `uv sync` to install dependencies from `pyproject.toml`
- Use `uv add <package>` to add dependencies
- Use `uv run pytest` to run tests

## Repository Knowledge System

All project knowledge lives in the repo. If it's not here, it doesn't exist for agents.

```
ARCHITECTURE.md              Code map — system design, layers, what exists and what doesn't
docs/
├── PLANS.md                 Index of all execution plans (active + completed)
├── DESIGN.md                Research loop design, council architecture, cost model
├── PRODUCT_SENSE.md         Product vision, core beliefs, north star metric
├── QUALITY_SCORE.md         Quality grades per domain and provider
├── RELIABILITY.md           Error recovery, loop guards, health checks
├── SECURITY.md              API keys, sandboxing, budget caps
├── design-docs/             Design decisions and methodology
│   └── retrodiction-validation.md  ← Validation approach (READ THIS)
├── exec-plans/              Implementation plans (active + completed)
│   ├── active/              Work in progress
│   ├── completed/           Finished work (with decision rationale)
│   └── tech-debt-tracker.md Known tech debt
├── references/              Research and external reference material
│   ├── model-cutoff-dates.md       LLM cutoffs and council groupings
│   ├── architecture-breakthroughs.md Post-cutoff validation set
│   └── hardware-constraints.md     M4 Mac feasibility
└── generated/               Auto-generated docs
```

### Documentation Rules

- If you change project structure → update `ARCHITECTURE.md`
- If you add tech debt → log in `docs/exec-plans/tech-debt-tracker.md`
- New features need an exec plan in `docs/exec-plans/active/` before implementation
- **Completed plans MUST move to `docs/exec-plans/completed/` and PLANS.md MUST be updated**

## Architectural Constraints

### Dependency Flow
```
Utils → Types → Providers → Domains → App
```

### Layer Model (within each domain)
```
types.py → config.py → service.py
                ↑
          Providers (llm, arxiv, git)
```

- Each domain is self-contained: types, config, service
- External dependencies (LLM APIs, arXiv, file I/O) enter through `src/providers/` only
- Evaluation functions are pure — no API calls, no I/O
- No frameworks (no LangChain, no CrewAI) — `litellm` + plain Python only
- Don't hardcode API keys — use `.env`

### Golden Principles

These opinionated rules prevent drift. Enforce them in code reviews.

1. **Providers are the only door to the outside** — If it talks to an API, a database, or the filesystem, it goes through src/providers/
2. **Evaluation functions are pure** — No imports from providers, no I/O, no side effects. Accept data as parameters.
3. **One domain, one directory** — Don't scatter a domain's logic across the codebase.
4. **Plans are living documents** — Update the plan before deviating from it. Move to completed/ when done.
5. **Tests run fast** — Unit tests mock providers. No network calls in unit tests.

### Mechanical Enforcement

Architecture and convention rules are enforced by tests in `tests/architecture/`:
- `test_imports.py` — Dependency direction, provider isolation, no circular deps
- `test_conventions.py` — File size, naming, pure function checks

Run with `uv run pytest tests/architecture/`. If a test fails, the error includes VIOLATION, FIX, and RULE.
