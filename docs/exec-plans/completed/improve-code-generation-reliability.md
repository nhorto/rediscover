# Improve Code Generation Reliability

> Goal: Reduce crash rate from ~70% to <30% by giving the implement step working examples and a persistent fix loop.
> Status: COMPLETE
> Started: 2026-03-18
> Completed: 2026-03-18

## Problem

Out of ~30 experiments, ~21 crashed (~70% crash rate). The implement step produces code with:
- Shape mismatches in forward pass
- 1D nn.Parameter (crashes MuonAdamW)
- Missing or broken helper functions
- Incorrect use of attention patterns

Previous fix loop: max 3 attempts for pre-training validation, then 1 fix attempt if training crashes.

## Changes Implemented

### 1. Working examples in IMPLEMENT_PROMPT ✅
- Created `src/domains/council/examples.py` with 2 proven-working examples:
  - Baseline SDPA attention (always works)
  - Nyström approximation (val_bpb=1.717)
- Added patterns note highlighting mandatory patterns (bias=False, ve_gate, rotary, transpose, etc.)
- Examples are injected into the prompt at runtime

### 2. Better error feedback ✅
- IMPLEMENT_SYSTEM now includes CRITICAL RULES listing common crash causes and fixes
- IMPLEMENT_FIX_SYSTEM includes COMMON MISTAKES AND FIXES section
- IMPLEMENT_FIX_PROMPT tells model to "Read it carefully and fix the SPECIFIC issue"

### 3. Fix attempts increased ✅
- Pre-training validation: 10 attempts (from 3)
- Post-training crash fix: 3 attempts in a loop (from 1 single attempt)
- Each fix attempt passes the specific error from the previous attempt
- Total: up to 13 code generations per experiment before giving up

### 4. Model routing — DEFERRED
- Implement step already uses gpt-4o-2024-11-20 (not cutoff-restricted)
- Can upgrade later if crash rate doesn't improve enough

## Files Changed

| File | Change |
|------|--------|
| `src/domains/council/examples.py` | NEW — working code examples for implement prompt |
| `src/domains/council/config.py` | Added crash rules to IMPLEMENT_SYSTEM, improved IMPLEMENT_FIX_SYSTEM/PROMPT, examples placeholder |
| `src/domains/council/service.py` | Injects examples into IMPLEMENT_PROMPT, increased max_tokens to 4096 |
| `src/app/loop.py` | max_fix_attempts=10, post-training fix loop (3 attempts) |
| `tests/unit/test_council.py` | Updated max_tokens assertion |

## Exit Criteria — All Met

- [x] max_fix_attempts raised to 10 in pre-training validation loop
- [x] Post-training crash fix loop tries up to 3 times (from 1)
- [x] IMPLEMENT_PROMPT includes 2 working examples
- [x] IMPLEMENT_FIX_PROMPT includes better error context
- [x] 198 tests pass, 0 failures
