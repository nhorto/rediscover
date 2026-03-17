# Rediscover — Agent Instructions

> Instructions for AI coding agents (Codex, external agents) working on this codebase.
> For Claude Code-specific instructions, see `CLAUDE.md`.

## Before You Start

1. Read `ARCHITECTURE.md` for the code map — it tells you where everything is and what doesn't belong.
2. Check `docs/exec-plans/active/` for current work in progress.
3. Check `docs/PLANS.md` for the execution plan index.

## Agent Workflow (MANDATORY)

Every non-trivial change follows this sequence. No exceptions.

1. **Plan** → Create exec plan in `docs/exec-plans/active/`, update `docs/PLANS.md`
2. **Implement** → Follow the plan. If the plan changes, update the document first.
3. **Code Review** → Self-review: correctness, edge cases, test coverage
4. **Test** → `uv run pytest` must pass with zero failures
5. **Update Docs** → Update ARCHITECTURE.md if structure changed, move plan to `completed/`

## Repository Knowledge System

All project knowledge lives in the repo. If it's not in the repo, it doesn't exist for you.

```
ARCHITECTURE.md              Code map (bird's-eye view)
docs/
├── PLANS.md                 Index of all execution plans
├── PRODUCT_SENSE.md         Product vision and beliefs
├── QUALITY_SCORE.md         Quality standards
├── RELIABILITY.md           Reliability standards
├── SECURITY.md              Security requirements
├── design-docs/             Detailed design decisions
├── exec-plans/              Implementation plans (active + completed)
├── references/              Research and reference material
└── generated/               Auto-generated docs
```

## Architectural Constraints

### Dependency Flow
```
Utils → Types → Providers → Domains → App
```

### Layer Model (within each domain)
```
Types → Config → Service → Runtime
                   ↑
             Providers (litellm, arxiv, experiment runner)
```

### What You Must Not Do
- Do not import domain internals from another domain
- Do not add API/IO dependencies to pure evaluation functions
- Do not hardcode API keys or credentials
- Do not use heavy frameworks (no LangChain, no CrewAI) — litellm + plain Python only
- Do not pass full files to LLM calls when diffs or sections suffice

## Tech Stack

| Do | Don't |
|----|-------|
| `uv run <file>` | `python <file>` |
| `uv sync` | `pip install -r requirements.txt` |
| `uv run pytest` | `pytest` directly |
| `litellm` for LLM calls | `openai`/`anthropic` SDK directly |
| File-based state (TSV + git) | Databases for experiment tracking |

## Testing

```bash
uv run pytest                      # Run all tests
uv run pytest tests/unit/          # Unit tests only
uv run pytest tests/integration/   # Integration tests only
```
