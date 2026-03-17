# Step 5: Council Pipeline

> Goal: Multi-agent deliberation pipeline that produces code changes for train.py.
> Status: IN PROGRESS
> Started: 2026-03-16

## Progress

- [x] Create src/domains/council/__init__.py
- [x] Create src/domains/council/types.py — SearchQuery, Proposal, Critique, ExperimentPlan, CouncilResult
- [x] Create src/domains/council/config.py — prompt templates for all 5 roles, context helpers (extract_hyperparams, format_results_history, format_papers_summary)
- [x] Create src/domains/council/service.py — CouncilService with scan → propose → critique → refine → implement pipeline
- [x] Create tests/unit/test_council.py — 18 tests: config helpers (6), types (3), parsing (6), pipeline (3)
- [x] Code review — found 5 issues: O(n²) dedup (fixed), test log count (fixed), DOTALL regex (correct), fallback behavior (acceptable), variable shadowing (minor)
- [x] Ruff check passes
- [x] Pytest 49/49 pass
- [x] Update docs — ARCHITECTURE.md, QUALITY_SCORE.md, PLANS.md

## Status: COMPLETE

## Design Decisions (from discussion)

### Pipeline Flow
```
1. SCAN — agent generates 1-3 search queries, retrieves papers from literature service
2. PROPOSE — given papers + results history + train.py hyperparams, proposes hypothesis + approach
3. CRITIQUE — independent model reviews proposal, raises concerns (NO VETO POWER)
4. REFINE — addresses critique, produces specific implementation plan
5. IMPLEMENT — given plan + full train.py, writes complete replacement train.py
```

### Key Decisions
- **Critique has no veto.** Always run the experiment. Let val_bpb decide.
- **Full replacement train.py** from implement step (not diffs). Simpler, always works.
- **Agent-driven search.** The propose step generates its own search queries based on what it thinks is promising. This lets the agent explore the literature creatively.
- **Tiered context per role.** Each step gets only what it needs:

| Role | Context It Receives |
|------|-------------------|
| Scan | program.md research direction only |
| Propose | Paper summaries + last 10 results + train.py hyperparams section |
| Critique | The proposal text + last 5 results |
| Refine | Proposal + critique + train.py hyperparams section |
| Implement | Refined plan + FULL current train.py (no papers, no research context) |

### Models (Dec 2023 cutoff council)
- scan: openrouter/mistralai/mistral-7b-instruct-v0.2
- propose: openrouter/mistralai/mixtral-8x7b-instruct
- critique: openrouter/mistralai/mistral-7b-instruct-v0.2
- refine: openrouter/mistralai/mixtral-8x7b-instruct
- implement: openrouter/deepseek/deepseek-coder-33b-instruct

See docs/design-docs/model-selection-strategy.md for full rationale.

### Logging
Every prompt, response, search query, and model used is logged. The full chain is auditable.

## Approach

### src/domains/council/types.py
```python
SearchQuery: query string + rationale
Proposal: hypothesis, approach, search_queries used, papers found
Critique: concerns list, suggestions list, confidence (0-1)
ExperimentPlan: description, code_changes summary, expected_impact
CouncilResult: all of the above + final train.py code
```

### src/domains/council/config.py
- Prompt templates for each role (scan, propose, critique, refine, implement)
- Context extraction helpers (get hyperparams section from train.py, format results history)
- MAX_RESULTS_HISTORY = 10 (last 10 experiments shown to propose/critique)
- MAX_PAPERS = 5 (top 5 papers per search query)

### src/domains/council/service.py
- `CouncilService.__init__(llm_provider, literature_service)`
- `run_council(train_py: str, results_tsv: str, program_md: str) -> CouncilResult`
  - Calls scan → propose → critique → refine → implement in sequence
  - Returns CouncilResult with the new train.py and full deliberation log
- Each step is a separate private method for testability

## Exit Criteria
- `run_council()` produces a valid modified train.py given mocked LLM responses
- Each pipeline step (scan, propose, critique, refine, implement) can be tested independently
- Full deliberation is logged in the CouncilResult
- All tests pass with mocked LLM calls
