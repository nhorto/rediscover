# Fix Code Generation Quality

> Goal: Reduce crash rate by improving context, validation, and error feedback for the implement step.
> Status: SUPERSEDED by targeted-code-generation.md (zone-based approach)
> Started: 2026-03-17

## Problem

All 3 test experiments crashed — the implement step (o1) produces broken train.py files.
- Experiment 1: Runtime error (Performer-style attention)
- Experiment 2: Runtime error (FLuRKA-style attention)
- Experiment 3: Syntax error (unclosed parenthesis on line 633)

## Progress

- [x] 1. More context — extract_code_structure() shows imports, classes, functions, constants as a preamble
- [x] 2. Code validation — quick_validate_code() runs in subprocess with 60s timeout, catches import/init errors
- [x] 3. Error feedback — fix_code() method + retry loop (max 2 fixes per experiment, catches both validation and training crashes)
- [x] 4. Tests — 11 new tests (quick_validate, extract_code_structure, error feedback integration)
- [x] 5. Code review — quick_validate appends sys.exit(0) but timeout handles it; cost per experiment may increase with fixes
- [x] 6. Ruff + pytest 133/133 pass
- [x] 7. Update docs

## Status: COMPLETE

## Design

### 1. More Context for Implement Step

Current: implement gets only the plan text + full train.py.
Problem: The model has to understand 500 lines and rewrite the whole file correctly.

New approach: Add a structured preamble to the implement prompt:
- List of all imports in the current train.py
- List of all class names and their signatures
- List of all top-level functions
- The hyperparameters section
- Explicit note: "Only modify the CausalSelfAttention class and related code. Keep everything else EXACTLY as-is."

### 2. Code Validation Step

Before running the full 5-minute training:
1. Syntax check: `compile(code, "train.py", "exec")` (already done)
2. Import check: Write the file, try to import it as a module in a subprocess with a 30-second timeout
3. Quick forward pass: Run a minimal test — create model, do one forward pass with dummy data

This catches:
- Missing imports
- Undefined variables
- Shape mismatches
- Device errors

### 3. Error Feedback Loop

When code crashes (validation or training):
1. Capture the full error traceback
2. Send it back to the implement model: "Your code produced this error: [traceback]. Fix the code. Return the complete train.py."
3. Max 2 fix attempts per experiment
4. If all attempts fail, log as crash and move on

## Exit Criteria

- At least 1 out of 3 experiments produces a training run that completes (not crash)
- Error feedback successfully fixes at least one broken implementation
- Validation step catches obvious errors before wasting 5 minutes of training
