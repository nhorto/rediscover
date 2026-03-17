# Step 7: The Research Loop

> Goal: Wire everything together into an autonomous research loop that can run overnight.
> Status: IN PROGRESS
> Started: 2026-03-16

## Progress

- [x] Create src/app/guards.py — LoopGuards with max_iterations, budget_cap, stuck_detection, error_cascade
- [x] Create src/app/loop.py — run_loop() orchestrator with council + training subprocess + git + guards + logging
- [x] Create tests/unit/test_guards.py — 11 tests (all guard conditions, state tracking, summary)
- [x] Create tests/unit/test_loop.py — 6 tests (parse_val_bpb, append_results_tsv)
- [x] Code review — found 6 issues: baseline parsing bug (fixed — now takes min, not last), crash output not logged (fixed), sys.executable correct for local, commit_hash safe, type annotations missing (minor)
- [x] Ruff check passes
- [x] Pytest 66/66 pass
- [x] Update docs — ARCHITECTURE.md, QUALITY_SCORE.md, PLANS.md

## Status: COMPLETE

## Design

### The Loop (src/app/loop.py)

```
def run_loop(config):
    Initialize providers (llm, literature, git)
    Initialize guards (max_iter, budget, stuck, similarity, error_cascade)

    while True:
        Check all guards → break if any triggered

        1. Read current train.py, results.tsv, program.md
        2. Run council deliberation → get CouncilResult with new train.py
        3. Write new train.py to disk
        4. Git commit the change
        5. Run training subprocess (uv run experiments/train.py)
           - Parse val_bpb from stdout
           - Handle crash/timeout
        6. Evaluate result:
           - If improved (val_bpb < best): keep commit, log "keep"
           - If worse or equal: git reset, log "discard"
           - If crash/timeout: git reset, log "crash"
        7. Append to results.tsv
        8. Append to experiment_log.md (narrative)
        9. Update guards (iteration count, check stuck, check similarity)
        10. Print status summary
```

### Guards (src/app/guards.py)

| Guard | Default | Trigger | Action |
|-------|---------|---------|--------|
| max_iterations | 500 | iteration count >= limit | Stop loop, produce summary |
| budget_cap | $50 | cost_tracker.total_cost >= limit | Stop loop |
| stuck_detection | 20 | no improvement in last N experiments | Force message: "try something fundamentally different" |
| error_cascade | 3 | N consecutive crashes | Skip to next idea (inject "avoid similar approaches" into next prompt) |
| similarity_check | 0.9 | last 3 hypotheses cosine sim > threshold | Reject, demand different approach |

### Training Runner

- Runs `uv run experiments/train.py` as subprocess
- Hard timeout: TIME_BUDGET (300s) + 600s overhead = 900s max
- Captures stdout, parses `val_bpb:` line from output
- Returns: val_bpb float, or None on crash/timeout

### Experiment Logging

**results.tsv** — append one row per experiment:
```
commit\tval_bpb\tmemory_gb\tstatus\tdescription
```

**experiment_log.md** — append narrative block per experiment:
```markdown
## Experiment N — [timestamp]
**Hypothesis:** ...
**Approach:** ...
**Papers consulted:** ...
**Critique summary:** ...
**Result:** val_bpb=X.XXX (keep/discard/crash)
**Cost this cycle:** $X.XX
**Cumulative cost:** $X.XX
---
```

### Entry Point

```bash
uv run src/app/loop.py
```

Or with config overrides:
```bash
uv run src/app/loop.py --max-iterations 50 --budget 10.0
```

## Dependencies

- src/providers/llm.py ✅
- src/providers/git.py ✅
- src/providers/arxiv.py ✅
- src/domains/council/service.py ✅
- src/domains/literature/service.py ✅
- experiments/train.py ✅
- experiments/prepare.py ✅

## Exit Criteria

- Loop runs 2-3 iterations end-to-end with mocked LLM calls in tests
- Guards correctly trigger on their conditions
- results.tsv and experiment_log.md are written correctly
- Crash recovery works (git reset on failure)
- Budget tracking accumulates across iterations
- All tests pass
