# Quality Score

> Grades each domain and architectural layer. Updated as the project evolves.

## Domain Grades

| Domain | Types | Config | Service | Tests | Overall |
|--------|-------|--------|---------|-------|---------|
| literature | A | A | A | B | Types, config, service (Chroma + SPECTER), arXiv provider — 4 tests |
| council | A | A | A | A | Types, config (5 prompt templates + working examples), service (full pipeline), parsing helpers, 18 tests |
| experiments | B | B | - | C | Harness working — train.py + prepare.py running on MPS, baseline val_bpb=1.763539 |
| validation | A | A | A | A | Types, config (7 breakthroughs), service (embedding + keyword scoring, reports), 27 tests |

## Provider Grades

| Provider | Tests | Status |
|----------|-------|--------|
| llm.py | 7 pass | Complete — litellm wrapper, model routing, cost tracking, budget enforcement |
| arxiv.py | 2 pass | Complete — date-filtered search, single paper lookup |
| git.py | 8 pass | Complete — commit, reset, log, diff, has_changes |
| costs.py | 6 pass | Complete — estimate_cost, CostTracker, BudgetExceededError |
| guards.py | 16 pass | Complete — LoopGuards with all guards + hypothesis similarity (cosine, SPECTER embeddings) |
| loop.py | 6 pass | Complete — run_loop orchestrator, 10-attempt pre-training fix loop, 3-attempt post-training fix loop |
| runner.py | 12 pass | Complete — RunnerProvider protocol, LocalRunner (subprocess), ModalRunner (A10G cloud GPU) |

## Scoring Criteria

- **A**: Complete, tested, documented, enforced
- **B**: Functional, mostly tested, some gaps
- **C**: Works but has known issues or missing tests
- **D**: Incomplete or has architectural violations
- **F**: Missing or broken
