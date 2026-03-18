# Phase 4 Results Analysis

> Analysis of experiments run during Phase 2 validation + Phase 4 attempt.
> Note: Phase 4 was blocked by insufficient OpenRouter credits (402 error).
> This analysis covers the 17 experiments available from Phase 2 runs.
> Date: 2026-03-18

## Overview

| Metric | Value |
|--------|-------|
| Total experiments | 17 (excluding baseline) |
| Unique proposals | 10 (some experiments were pre-zone duplicates) |
| Successful (keep) | 4 (24%) |
| Crashed | 7 (41%) |
| Discarded (ran but worse) | 4 (24%) |
| Baseline only | 1 |
| Best val_bpb | 1.675352 (5.0% improvement over baseline 1.763539) |
| Total LLM cost | ~$4.44 |

## Top 5 Best val_bpb Scores

| Rank | val_bpb | Improvement | Experiment |
|------|---------|-------------|------------|
| 1 | 1.675352 | 5.00% | Performer FAVOR+ linear attention |
| 2 | 1.675790 | 4.97% | Dynamic sparse attention (adaptive k) |
| 3 | 1.676400 | 4.94% | Reformer-style LSH attention (discarded — marginal) |
| 4 | 1.677103 | 4.90% | Performer FAVOR+ attention (earlier run) |
| 5 | 1.680573 | 4.70% | FLuRKA low-rank + kernel (discarded — marginal) |

Note: All successful experiments cluster within a narrow band (1.675-1.681), suggesting diminishing returns from attention-only changes at this model scale.

## Hypotheses by Theme

### 1. Linear Attention via Feature Maps (4 proposals, all crashed)
- Linear log-normal attention (2 attempts)
- ELU+1 feature map linear attention (2 attempts)
- All failed due to code structure issues (pre-zone approach didn't integrate with the existing model correctly)

### 2. Kernel/Random Feature Approximation (2 proposals, 1 keep + 1 crash)
- Performer FAVOR+ (keep — val_bpb=1.675352, best result)
- FLuRKA low-rank + kernel (crash — optimizer error on 1D params)

### 3. Sparse/Structured Attention (2 proposals, 1 keep + 1 discard)
- Dynamic sparse attention with adaptive k (keep — val_bpb=1.675790)
- Reformer LSH attention (discard — val_bpb=1.676400, didn't beat best)

### 4. Low-Rank Projection (2 proposals, 1 crash + 1 discard)
- Linformer-style low-rank (crash)
- FLuRKA second attempt (discard — val_bpb=1.680573)

### 5. Other (1 proposal)
- Nystrom attention approximation (keep — val_bpb=1.717026, early improvement)
- Nystrom variant with structured sampling (discard — val_bpb=1.842817, regression)
- Kernelized attention (discard — val_bpb=2.347710, major regression)

## Search Query Diversity

The council used a narrow set of literature queries, mostly focused on:
- "Linear attention mechanisms" and variants
- "Efficient transformer attention"
- Papers from the arXiv attention/transformers corpus

**Assessment: Low diversity.** Most proposals came from the same cluster of papers (Performer, Linformer, FLuRKA, Reformer). The council did not explore:
- Gating mechanisms
- Cross-head information sharing
- Position encoding innovations
- Value projection alternatives
- Architectural decomposition (global + local)

## Crash Analysis

| Crash Cause | Count | Description |
|-------------|-------|-------------|
| Pre-zone code structure | 4 | Early experiments before zone-based generation was implemented — generated code didn't preserve the existing model interface |
| MuonAdamW 1D param error | 2 | New code added nn.Linear with bias=True or 1D parameters, which the MuonAdamW optimizer can't handle |
| General code errors | 1 | Linformer implementation had shape mismatches |

**Crash rate improved from 80% (pre-zone) to 33% (post-zone).** The zone-based code generation and structure validation fixes from Phase 2 significantly reduced crashes.

## Patterns in Success vs Failure

**Successful experiments:**
- Used simpler modifications that preserved the overall attention structure
- Performer FAVOR+ and dynamic sparse attention both worked because they modified *how* attention is computed without changing tensor shapes or adding new parameter types
- Kept bias=False on all nn.Linear layers

**Failed experiments:**
- Attempted to rewrite the entire attention mechanism from scratch
- Added new parameter types (biases) incompatible with the optimizer
- Didn't preserve the forward(self, x, ve, cos_sin, window_size) interface

## Cost Breakdown

| Category | Cost |
|----------|------|
| LLM calls (council pipeline) | ~$4.44 |
| Training compute (local MPS) | $0.00 (local hardware) |
| Total | ~$4.44 |

Average cost per experiment: ~$0.26 (post-zone), ~$1.27 (pre-zone, before the council was optimized)

## Blocker: OpenRouter Credits

Phase 4b (100+ experiment run) was blocked by insufficient OpenRouter credits:
```
APIError: OpenrouterException - "This request requires more credits...
You requested up to 4096 tokens, but can only afford 2463."
```

**Action needed:** Add credits at https://openrouter.ai/settings/keys to continue.
