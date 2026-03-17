# Step 1+2: Project Setup & Experiment Harness

> Goal: pyproject.toml with linting/testing, then a working train.py on the M4 Mac.
> Status: IN PROGRESS
> Started: 2026-03-16

## Progress

- [x] Step 1a: Create pyproject.toml with all dependencies (112 packages installed)
- [x] Step 1b: Set up ruff (linter + formatter) — passes clean
- [x] Step 1c: Set up pytest with conftest.py — 3/3 tests pass, MPS confirmed
- [x] Step 1d: Create .env.example
- [x] Step 1e: Verify `uv sync`, `uv run ruff check .`, `uv run pytest` all work
- [x] Step 2a: Adapt prepare.py from autoresearch-macos into experiments/
- [x] Step 2b: Adapt train.py from autoresearch-macos into experiments/
- [x] Step 2c: Create experiments/program.md
- [x] Step 2d: Create experiments/results.tsv with header
- [x] Step 2e: Run `uv run experiments/prepare.py` — 5 shards downloaded, tokenizer trained (18.9s)
- [x] Step 2f: Run `uv run experiments/train.py` — PASSED. val_bpb=1.763539, 89 steps, ~17K tok/sec, 303.8s training
- [x] Step 2g: Record baseline val_bpb — 1.763539 (11.5M params, 4 layers, 256 dim)
- [x] Code review: ruff check passes, pytest 3/3 pass
- [x] Update docs: progress file, QUALITY_SCORE.md

## Results

- **Baseline val_bpb: 1.763539**
- Model: 11.5M params, 4 layers, 256 dim, 2 heads
- Throughput: ~17,000 tokens/sec on M4 MPS
- Training: 89 steps in 303.8s (5 min budget)
- Total wall clock: 802s (~13 min incl startup + eval)
- Loss trajectory: 9.01 → 4.97

## Status: COMPLETE

## Approach

### Step 1: Project Setup

**pyproject.toml dependencies:**
- Core: `torch`, `tiktoken`, `pyarrow`, `requests`
- Research: `litellm`, `arxiv`, `chromadb`, `sentence-transformers`
- Tokenizer: `rustbpe` (used by autoresearch's prepare.py)
- Dev: `ruff`, `pytest`

**Ruff config (in pyproject.toml):**
- Line length: 120
- Target: Python 3.11
- Rules: E (pycodestyle), F (pyflakes), I (isort), UP (pyupgrade)
- Format: on save

**Pytest config:**
- testpaths: tests/
- markers: unit, integration

### Step 2: Experiment Harness

**Source:** `miolini/autoresearch-macos` (master branch)
- prepare.py: Use mostly as-is, make cache dir configurable
- train.py: Use the MPS-compatible version as-is — this is the agent's canvas
- Both files already handle MPS device detection, bfloat16 casting, SDPA attention

**Key defaults (already Mac-tuned):**
- DEPTH = 4 (4 transformer layers)
- DEVICE_BATCH_SIZE = 16
- TOTAL_BATCH_SIZE = 65K tokens
- TIME_BUDGET = 300s (5 minutes)
- WINDOW_PATTERN = "L" (full context)

## Risks

| Risk | Mitigation |
|------|------------|
| rustbpe may not compile on macOS | Fall back to pure-Python BPE or pre-trained tokenizer |
| MPS bfloat16 issues on M4 | The fork already handles this with nullcontext autocast |
| Data download slow (HuggingFace shards) | Start with fewer shards (--num-shards 4) |

## Exit Criteria

1. `uv sync` installs all dependencies
2. `uv run ruff check .` passes with zero errors
3. `uv run pytest` runs (even with minimal tests)
4. `uv run experiments/prepare.py --num-shards 4` downloads data and trains tokenizer
5. `uv run experiments/train.py` completes a 5-minute training run on MPS
6. Baseline val_bpb is recorded
