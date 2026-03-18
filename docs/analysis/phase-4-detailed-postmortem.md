# Phase 4 Detailed Post-Mortem

> A complete account of what happened during the Phase 4 experiment runs on March 18, 2026.
> Written to help understand what's working, what's broken, and what needs to change.

## Timeline of Events

### 5:29 AM — First Run Attempt (gpt-4o via OpenRouter)
- Started loop with `--max-iterations 120 --budget 30.0`
- **Result:** OpenRouter hung on API calls. The gpt-4o-2024-05-13 model took >5 minutes per call, sometimes hanging indefinitely.
- Loop appeared stuck — no output for hours due to Python output buffering.
- 3 experiments eventually completed (all crashed), discovered via checking results.tsv directly.
- **Root cause:** OpenRouter's routing to gpt-4o was extremely slow/unreliable.

### 8:44 AM — Second Run (reduced timeout)
- Added 120s signal-based hard timeout to kill hung API calls.
- **Result:** Timeouts fired correctly, but experiments still failed because fix_code() calls weren't wrapped in try/except. The loop crashed entirely when a timeout hit during a fix attempt.
- **Root cause:** Unhandled exception in fix_code path.

### 11:03 AM — Third Run (fixed exception handling)
- Wrapped all fix_code() calls in safe_fix_code() with try/except.
- Increased API timeout to 300s.
- **Result:** Loop ran but OpenRouter still hung — experiment 2 deliberated for over an hour with 0% CPU usage.
- **Root cause:** litellm's timeout parameter didn't actually work for OpenRouter connections.

### 1:25 PM — Fourth Run (gpt-4o-mini + signal timeout)
- Switched all models to gpt-4o-mini (10x faster, 6x cheaper).
- Added SIGALRM-based hard timeout at 120s.
- **Result:** API calls were fast! Council deliberation took seconds. But code quality was terrible — 88% crash rate.
- 10 experiments ran before hitting error hard stop (10 consecutive failures).
- Total cost: $0.32

### 3:46 PM — Fifth Run (MuonAdamW fix)
- Fixed the None gradient crash in MuonAdamW optimizer.
- **Result:** First experiment to complete training! val_bpb=1.913 (discard — regression).
- However, most experiments still crashed or timed out during training.
- Training timeouts dominated: generated code was too slow for 5-min budget.
- Loop continued running until ~5 PM.

### 5:30 PM — Sixth Run (Claude 3.5 Sonnet + Modal GPU)
- Switched implement model to Claude 3.5 Sonnet for better code quality.
- Added Modal cloud GPU runner.
- **Result:** Modal not deployed — "Function has not been hydrated" error. All training calls failed instantly.
- Model config also got reverted by linter back to OpenRouter.

## What the Experiments Actually Looked Like

### Total Across All Runs: 26 Experiments Logged

| Outcome | Count | % |
|---------|-------|---|
| Crashed before training | 9 | 35% |
| Crashed during training | 6 | 23% |
| Training timed out | 5 | 19% |
| Completed but worse (discard) | 2 | 8% |
| Completed and better (keep) | 4 | 15% |

### The 4 Successful Experiments (all from Phase 2)
1. **Nystrom attention** — val_bpb=1.717026 (2.6% improvement)
2. **Performer FAVOR+** — val_bpb=1.677103 (4.9% improvement)
3. **Dynamic sparse attention** — val_bpb=1.675790 (5.0% improvement)
4. **Performer FAVOR+ v2** — val_bpb=1.675352 (5.0% improvement, best)

All 4 successes came from Phase 2 when using gpt-4o-2024-05-13 (the smarter model). **Zero successful experiments came from the Phase 4 runs using gpt-4o-mini.**

## The Core Problem: Code Generation Quality

This is the fundamental bottleneck. Here's what goes wrong:

### Problem 1: Missing Function Signature Parameters
The model generates `forward(self, x)` instead of `forward(self, x, ve, cos_sin, window_size)`. This happens because:
- The zone code is ~100 lines, and the model "simplifies" the interface
- Even when explicitly told the signature in the prompt, gpt-4o-mini drops parameters
- Fix attempts often produce the same mistake

### Problem 2: Missing Helper Functions
The model drops the `norm()` helper function that exists in the zone. The frozen code calls `norm(x)` from the Block class, so removing it crashes everything.

### Problem 3: Parameters Without Gradients
The model creates conditional parameters (`if config.use_x: self.param = ...`) but the config defaults don't activate them. The parameters exist in the model but have `grad=None`, crashing MuonAdamW. **This was fixed mid-run.**

### Problem 4: Compute-Heavy Code
The model generates attention mechanisms with O(N³) or worse complexity. The 5-minute training budget isn't enough for these implementations, causing timeouts. Examples:
- Full N×N attention matrix manipulation (defeats the purpose of efficient attention)
- Nested loops over sequence length
- Multiple redundant attention computations

### Problem 5: Shape Mismatches
The model changes tensor shapes in ways that don't match the frozen code's expectations. The frozen code expects `[B, T, C]` output but gets `[B, T, 2*C]` or similar.

## Why gpt-4o-mini vs gpt-4o Made Such a Difference

| Metric | gpt-4o (Phase 2) | gpt-4o-mini (Phase 4) |
|--------|-------------------|----------------------|
| Crash rate | 23% (post-zone) | 88% |
| Successful experiments | 4 | 0 |
| Cost per experiment | $0.44 | $0.07 |
| Code correctness | Good | Poor |
| Proposal originality | Low (paper replication) | Higher (novel ideas) |

**gpt-4o-mini is creative but can't code.** It proposes interesting ideas (hierarchical attention, gating mechanisms, dual-layer attention) but can't implement them correctly within the constraints. gpt-4o was boring (replicated papers) but could actually write working code.

## What the Proposals Looked Like

### Phase 2 Proposals (gpt-4o) — Paper Replications
- "Implement Performer FAVOR+"
- "Implement Linformer-style attention"
- "Implement FLuRKA"
- "Implement Reformer LSH"

### Phase 4 Proposals (gpt-4o-mini) — More Original But Broken
- "Hierarchical attention with global + local components"
- "Dynamic content-dependent sliding window"
- "Hybrid content-based attention with learned positional bias"
- "Spatial-temporal gating mechanism"
- "Dual-context attention separating global and local"
- "Learned attention sparsity patterns per head"
- "Dynamic thresholding in softmax attention"

The Phase 4 proposals are genuinely more interesting and closer to real post-cutoff innovations. "Hierarchical attention with global + local" is conceptually adjacent to FlashAttention-3. "Spatial-temporal gating" is novel. But none could be implemented.

## Retrodiction Validation Results

| Level | Phase 2 | Phase 4 |
|-------|---------|---------|
| Direct hit | 0 | 0 |
| Adjacent | 9/10 (90%) | 7/8 (88%) |
| Directional | 1/10 (10%) | 1/8 (12%) |

The "adjacent" rate is misleading — it just means the proposals are about attention (same semantic neighborhood). No proposals independently converged on a specific post-cutoff mechanism (MLA, Mamba-2, etc.).

## Infrastructure Issues Encountered

1. **OpenRouter API reliability** — Calls hung for hours, litellm timeouts didn't fire. Fixed by switching to direct OpenAI API.
2. **Python output buffering** — Background processes produced no visible output. Fixed with `-u` flag and line-buffered IO.
3. **Git state collisions** — Development commits during loop runs caused git commit failures. Known tech debt.
4. **OpenRouter credit limits** — Weekly credit limit too low for 100+ experiments. Wasted ~120 iterations hitting 402 errors.
5. **Modal deployment** — The Modal app needs to be deployed (`modal deploy`) before the loop can use it.

## Cost Analysis

| Run | Model | Experiments | API Cost | Compute Cost |
|-----|-------|-------------|----------|-------------|
| Phase 2 | gpt-4o (OpenRouter) | 10 | $4.44 | $0 (local) |
| Phase 4 run 1 | gpt-4o (OpenRouter) | 3 | ~$0.30 | $0 (local) |
| Phase 4 run 2 | gpt-4o-mini (OpenRouter) | 10 | $0.32 | $0 (local) |
| Phase 4 run 3 | gpt-4o-mini (OpenRouter) | 5 | ~$0.17 | $0 (local) |
| Phase 4 run 4 | gpt-4o-mini/Sonnet (OpenRouter) | 8 | ~$0.50 | $0 (Modal failed) |
| **Total** | | **36** | **~$5.73** | **$0** |

At ~$0.04/experiment with gpt-4o-mini, 100 experiments would cost ~$4. At ~$0.44/experiment with gpt-4o, 100 experiments would cost ~$44. The budget isn't the issue — reliability is.

## What Needs to Change

### Option A: Better Code Model (Recommended)
Use gpt-4o-2024-11-20 for the implement step (already configured). This model should produce better code than gpt-4o-mini while still having the Oct 2023 cutoff. Cost is moderate (~$0.10-0.20 per implement call).

### Option B: Simpler Code Generation Approach
Instead of generating free-form zone code, provide the model with a more constrained template:
- Pre-define the class skeleton with all required methods and signatures
- Only let the model fill in the attention computation (a much smaller scope)
- This would reduce the surface area for mistakes

### Option C: Multi-Shot Code Generation
- Generate code, validate, and if it fails, show the error and the code to the model
- This is what we do now (fix attempts), but it could be more structured:
  - Include the passing baseline code alongside the error
  - Show exactly which line failed and why

### Option D: Test-Driven Generation
- Before generating, create a test that the code must pass
- Run the test in the validation step
- This is essentially what quick_validate_code does, but could be more comprehensive

## Immediate Next Steps

1. **Deploy Modal** — `modal deploy modal_app.py` to enable cloud GPU training
2. **Fix model config** — Use direct OpenAI API (gpt-4o-2024-11-20 for implement)
3. **Run with local training first** — Don't use Modal until it's tested separately
4. **Monitor first 5 experiments** — Watch for the crash pattern before letting it run overnight
5. **Consider Option B** — If gpt-4o-2024-11-20 still crashes >50%, constrain the code template further
