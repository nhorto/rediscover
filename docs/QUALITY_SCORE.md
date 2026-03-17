# Quality Score

> Grades each domain and architectural layer. Updated as the project evolves.

## Domain Grades

| Domain | Types | Config | Service | Tests | Overall |
|--------|-------|--------|---------|-------|---------|
| literature | A | A | A | B | Types, config, service (Chroma + SPECTER), arXiv provider — 4 tests |
| council | A | A | A | A | Types, config (5 prompt templates), service (full pipeline), 18 tests |
| experiments | B | B | - | C | Harness working — train.py + prepare.py running on MPS, baseline val_bpb=1.763539 |
| validation | - | - | - | - | Not started |

## Provider Grades

| Provider | Tests | Status |
|----------|-------|--------|
| llm.py | 7 pass | Complete — litellm wrapper, model routing, cost tracking, budget enforcement |
| arxiv.py | 2 pass | Complete — date-filtered search, single paper lookup |
| git.py | 8 pass | Complete — commit, reset, log, diff, has_changes |
| costs.py | 6 pass | Complete — estimate_cost, CostTracker, BudgetExceededError |
| guards.py | 11 pass | Complete — LoopGuards with all guard conditions |
| loop.py | 6 pass | Complete — run_loop orchestrator, parse_val_bpb, append helpers |

## Scoring Criteria

- **A**: Complete, tested, documented, enforced
- **B**: Functional, mostly tested, some gaps
- **C**: Works but has known issues or missing tests
- **D**: Incomplete or has architectural violations
- **F**: Missing or broken
