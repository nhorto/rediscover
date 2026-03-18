# Targeted Code Generation — Patch Zone Approach

> Goal: Replace full-file rewrite with targeted zone editing to eliminate syntax errors.
> Status: IN PROGRESS
> Started: 2026-03-17
> Branch: feature/targeted-code-generation

## Problem

The implement step asks the model to rewrite all ~500 lines of train.py. The model consistently
produces syntax errors (unclosed parentheses) around line 350-370. 100% crash rate across 8+ attempts.

## Solution: Modifiable Zone

Instead of rewriting the whole file, the model only outputs the "modifiable zone" — the parts of
train.py it's allowed to change. We extract that zone, send it to the model, get back the modified
version, and splice it back into the unchanged surrounding code.

## What the Model CAN Modify

```
MODIFIABLE ZONE (extracted and sent to the model):
├── GPTConfig dataclass (can add new fields like feature_dim, rank, etc.)
├── Helper functions (norm, has_ve, apply_rotary_emb — can modify or add new ones)
└── CausalSelfAttention class (can change internals, must preserve interface)
```

This covers everything the model needs to innovate on attention:
- Add new helper functions (feature maps, kernel functions, etc.)
- Add new config parameters (feature_dim, rank, num_random_features, etc.)
- Rewrite the attention forward pass (linear attention, sparse, etc.)
- Modify how rotary embeddings or value embeddings are applied
- Change the head structure (different KV sharing schemes, etc.)

## What the Model CANNOT Modify

```
FROZEN (kept exactly as-is, never sent to the model):
├── Imports
├── MLP class
├── Block class
├── GPT class (model init, forward, optimizer setup)
├── MuonAdamW optimizer
├── Hyperparameters section
├── Training loop
├── Evaluation and output
└── Setup code (device detection, data loading, etc.)
```

## Progress

- [x] Create src/utils/code_splicing.py — extract/replace modifiable zone from train.py
- [x] Update council config — zone-only implement prompt with frozen context
- [x] Update council service — _implement extracts zone, splices result back
- [x] fix_code also uses zone extraction for targeted fixes
- [x] Tests for extraction and splicing (8 tests)
- [x] Fix eval() false positive (model.eval() is standard PyTorch)
- [x] Code review — moved code_splicing to utils (architecture compliance), service under 300 lines
- [x] Ruff + pytest 143 pass

## Test Results

### Run 1 (pre-fix): Full file rewrite
- 100% syntax error rate (unclosed parens around line 350-370)
- Cost: ~$0.15 per council cycle

### Run 2 (zone approach): Targeted zone editing
- 0% syntax errors (zone is only ~60-100 lines)
- New issue: runtime errors during model init (shape mismatches in forward pass)
- Error feedback catches tracebacks but can't resolve them in 2 attempts
- Cost: ~$0.08 per council cycle (50% cheaper — less output tokens)

## Status: COMPLETE — zone approach working, 66% success rate in production runs

### Run 3 (improved prompts + quick_validate + fix_code): Production
- ~66% success rate (4 keeps out of ~14 experiments that got past validation)
- Best val_bpb: 1.675352 (5% improvement over baseline 1.763539)
- Quick validate catches shape mismatches before 5-min training
- Fix_code with frozen context resolves some runtime errors
- Cost: ~$0.07-0.25 per council cycle

## Design: Code Splicing

### Zone Boundaries

The modifiable zone starts at `@dataclass` (GPTConfig) and ends at `class MLP`.
Everything between these markers is extractable and replaceable.

```python
# --- FROZEN: imports and env setup ---
import os
...
from prepare import MAX_SEQ_LEN, TIME_BUDGET, Tokenizer, make_dataloader, evaluate_bpb

# ======= MODIFIABLE ZONE START =======

@dataclass
class GPTConfig:
    ...

def norm(x):
    ...

def has_ve(layer_idx, n_layer):
    ...

def apply_rotary_emb(x, cos, sin):
    ...

class CausalSelfAttention(nn.Module):
    ...

# ======= MODIFIABLE ZONE END =======

class MLP(nn.Module):
    ...
# --- FROZEN: everything after MLP ---
```

### Implementation

```python
def extract_modifiable_zone(train_py: str) -> tuple[str, str, str]:
    """Split train.py into (before_zone, zone, after_zone).

    Returns three strings that concatenate to the original file.
    """
    ...

def replace_modifiable_zone(train_py: str, new_zone: str) -> str:
    """Replace the modifiable zone in train.py with new code."""
    before, _, after = extract_modifiable_zone(train_py)
    return before + new_zone + after
```

## Exit Criteria

- Model only generates ~60-100 lines (the zone) instead of ~500
- Syntax errors eliminated (zone is small enough for reliable generation)
- At least 1 experiment produces a successful training run
- Model can add new helper functions and config params within the zone
