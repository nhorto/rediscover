# Cloud GPU Integration: Modal

> Move experiment training from local MPS subprocess to Modal cloud GPU.
> Status: **COMPLETE**
> Created: 2026-03-17
> Completed: 2026-03-18
> Reference: docs/references/cloud-gpu-options.md

## Goal

Replace the inline `run_training()` subprocess call in `loop.py` with a pluggable `RunnerProvider` that supports both local execution (current behavior) and Modal cloud GPU execution. The Mac continues to orchestrate (LLM calls, literature, git, logging). Only the 5-minute training step moves to the cloud.

## Architecture

```
BEFORE:
  loop.py → subprocess.run([python, train.py]) → parse stdout → val_bpb

AFTER:
  loop.py → RunnerProvider.run(train_code, prepare_code) → TrainingResult(val_bpb, output)
               │
               ├── LocalRunner (subprocess, same as today)
               └── ModalRunner (modal remote call → A10G GPU)
```

### What Stays Local (Mac)
- Literature scanning (arXiv API + Chroma)
- Council deliberation (5 LLM API calls via OpenRouter)
- Git operations (commit, reset, log)
- Results logging (results.tsv, experiment_log.md)
- Guard evaluation (budget, stuck detection, similarity)
- Loop orchestration

### What Moves to Cloud (Modal)
- `python train.py` execution (the 5-minute training run)
- `prepare.py` runtime utilities (imported by train.py)
- Data shards + tokenizer (cached in Modal Volume)

## Prerequisites (User Actions on the Web)

These steps must be done by the user before implementation begins:

1. **Create a Modal account** at https://modal.com (GitHub login works)
2. **Check free tier status** — $30/month in credits is included
3. **Optional: Add payment method** for usage beyond free tier
4. **Run `modal token set`** locally after signup — Modal's web dashboard provides the token command
   - This writes credentials to `~/.modal/token` (not in the repo, not committed)

## Implementation Steps

### Step 1: Add Modal Dependency

**File:** `pyproject.toml`

Add `modal` to the dependencies list:
```toml
dependencies = [
    # ... existing deps ...
    "modal>=0.64.0",
]
```

Then run `uv sync` to install.

**Verification:** `uv run python -c "import modal; print(modal.__version__)"` succeeds.

---

### Step 2: Create the Modal App Definition

**New file:** `modal_app.py` (repo root)

This is the Modal-side function that runs on cloud GPU. It receives `train.py` and `prepare.py` as strings, writes them to the container filesystem, runs training, and returns the result.

```python
"""Modal app for Rediscover cloud GPU training."""

import modal
import subprocess
import sys
import re

app = modal.App("rediscover")

# Persistent volume for data shards + tokenizer (~500MB, downloaded once)
vol = modal.Volume.from_name("rediscover-data", create_if_missing=True)

# Container image with PyTorch + dependencies
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch>=2.2.0",
        "tiktoken>=0.6.0",
        "pyarrow>=15.0.0",
        "requests>=2.31.0",
        "rustbpe>=0.1.0",
    )
)

CACHE_DIR = "/data/autoresearch"
TRAINING_TIMEOUT = 900  # match local timeout


@app.function(
    gpu="A10G",
    timeout=TRAINING_TIMEOUT + 120,  # extra buffer for container startup
    volumes={"/data": vol},
    image=image,
    memory=16384,  # 16GB RAM
)
def run_experiment(train_code: str, prepare_code: str) -> dict:
    """Run a single training experiment on cloud GPU.

    Args:
        train_code: Contents of train.py (modified by council)
        prepare_code: Contents of prepare.py (fixed, never modified)

    Returns:
        dict with keys: val_bpb (float|None), output (str), success (bool)
    """
    import os
    import tempfile

    # Create experiment directory
    exp_dir = tempfile.mkdtemp(prefix="rediscover_")
    train_path = os.path.join(exp_dir, "train.py")
    prepare_path = os.path.join(exp_dir, "prepare.py")

    # Write experiment files
    with open(train_path, "w") as f:
        f.write(train_code)
    with open(prepare_path, "w") as f:
        # Patch CACHE_DIR to use Modal volume path
        patched = prepare_code.replace(
            'os.path.join(os.path.expanduser("~"), ".cache", "autoresearch")',
            f'"{CACHE_DIR}"'
        )
        f.write(patched)

    # Ensure data exists on volume (first run downloads, subsequent runs use cache)
    data_dir = os.path.join(CACHE_DIR, "data")
    tokenizer_dir = os.path.join(CACHE_DIR, "tokenizer")
    if not os.path.exists(data_dir) or len(os.listdir(data_dir)) == 0:
        # Run prepare.py to download data + train tokenizer
        result = subprocess.run(
            [sys.executable, prepare_path, "--num-shards", "4"],
            capture_output=True, text=True, timeout=600,
            cwd=exp_dir,
            env={**os.environ, "HOME": "/data"}  # so ~/.cache -> /data/.cache
        )
        if result.returncode != 0:
            return {
                "val_bpb": None,
                "output": f"PREPARE FAILED:\n{result.stdout}\n{result.stderr}",
                "success": False,
            }
        vol.commit()  # persist data to volume

    # Run training
    try:
        result = subprocess.run(
            [sys.executable, train_path],
            capture_output=True, text=True,
            timeout=TRAINING_TIMEOUT,
            cwd=exp_dir,
        )
        output = result.stdout + result.stderr

        if result.returncode != 0:
            return {"val_bpb": None, "output": output, "success": False}

        # Parse val_bpb
        match = re.search(r"val_bpb:\s+([\d.]+)", output)
        val_bpb = float(match.group(1)) if match else None

        return {"val_bpb": val_bpb, "output": output, "success": val_bpb is not None}

    except subprocess.TimeoutExpired:
        return {
            "val_bpb": None,
            "output": "TIMEOUT: Training exceeded time limit on Modal",
            "success": False,
        }
```

**Key design decisions:**
- `prepare.py`'s `CACHE_DIR` is patched at runtime to point to the Modal Volume (`/data/autoresearch/`) instead of `~/.cache/autoresearch/`. This is the simplest approach — no changes to `prepare.py` itself.
- Data downloads happen on first run only, then persist in the Volume.
- `vol.commit()` after data prep ensures the volume is persisted.
- The container image pins the same PyTorch + data deps as the local environment.
- `timeout` on `@app.function` is set to `TRAINING_TIMEOUT + 120` to account for container cold start.

**Verification:**
- `modal deploy modal_app.py` succeeds
- `uv run python -c "from modal_app import run_experiment; print('import ok')"` works

---

### Step 3: Create RunnerProvider

**New file:** `src/providers/runner.py`

This is the provider abstraction that lets the loop switch between local and Modal execution.

```python
"""Experiment runner provider: local subprocess or Modal cloud GPU."""

from __future__ import annotations

import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol


@dataclass
class TrainingResult:
    """Result from a single training run."""
    val_bpb: float | None
    output: str
    success: bool


class RunnerProvider(Protocol):
    """Protocol for experiment execution backends."""

    def run(self, train_code: str, prepare_code: str) -> TrainingResult:
        """Run an experiment and return the result."""
        ...


class LocalRunner:
    """Run experiments as a local subprocess (current behavior)."""

    def __init__(self, project_root: Path, timeout: int = 900):
        self.project_root = project_root
        self.train_py = project_root / "experiments" / "train.py"
        self.timeout = timeout

    def run(self, train_code: str, prepare_code: str) -> TrainingResult:
        """Run training locally via subprocess.

        Note: train_code has already been written to experiments/train.py by the loop.
        prepare_code is unused (it's already on disk). Both args are accepted
        to match the RunnerProvider protocol.
        """
        try:
            result = subprocess.run(
                [sys.executable, str(self.train_py)],
                capture_output=True,
                text=True,
                timeout=self.timeout,
                cwd=str(self.project_root),
            )
            output = result.stdout + result.stderr
            if result.returncode != 0:
                return TrainingResult(val_bpb=None, output=output, success=False)
            val_bpb = _parse_val_bpb(output)
            return TrainingResult(val_bpb=val_bpb, output=output, success=val_bpb is not None)
        except subprocess.TimeoutExpired:
            return TrainingResult(
                val_bpb=None,
                output="TIMEOUT: Training exceeded time limit",
                success=False,
            )
        except Exception as e:
            return TrainingResult(val_bpb=None, output=f"ERROR: {e}", success=False)


class ModalRunner:
    """Run experiments on Modal cloud GPU."""

    def __init__(self):
        # Import modal lazily to avoid ImportError when not installed
        pass

    def run(self, train_code: str, prepare_code: str) -> TrainingResult:
        """Send code to Modal for cloud GPU training."""
        try:
            from modal_app import run_experiment

            result = run_experiment.remote(train_code, prepare_code)
            return TrainingResult(
                val_bpb=result["val_bpb"],
                output=result["output"],
                success=result["success"],
            )
        except ImportError:
            return TrainingResult(
                val_bpb=None,
                output="ERROR: modal not installed. Run: uv add modal",
                success=False,
            )
        except Exception as e:
            return TrainingResult(
                val_bpb=None,
                output=f"MODAL ERROR: {e}",
                success=False,
            )


def _parse_val_bpb(output: str) -> float | None:
    """Parse val_bpb from training script output."""
    match = re.search(r"val_bpb:\s+([\d.]+)", output)
    return float(match.group(1)) if match else None
```

**Key design decisions:**
- `RunnerProvider` is a Protocol (structural typing), not an ABC. This matches the project's plain-Python style.
- `LocalRunner` replicates the exact behavior of the current `run_training()` function in `loop.py`.
- `ModalRunner` imports `modal_app` lazily — if Modal isn't installed, the import fails gracefully.
- Both runners return `TrainingResult` (a simple dataclass) instead of a raw tuple.
- `_parse_val_bpb` is extracted here and reused by both runners (Modal also parses server-side, but this is the local fallback).

**Verification:** `uv run pytest tests/unit/test_runner.py` (new test file, see Step 5).

---

### Step 4: Modify loop.py to Use RunnerProvider

**File:** `src/app/loop.py`

Changes required:

1. **Remove** the `run_training()` function and `parse_val_bpb()` function (they move to `src/providers/runner.py`).

2. **Add** a `--runner` CLI argument:
   ```python
   parser.add_argument(
       "--runner", choices=["local", "modal"], default="local",
       help="Training backend: local subprocess or Modal cloud GPU"
   )
   ```

3. **Add** a `runner` parameter to `run_loop()`:
   ```python
   def run_loop(
       max_iterations: int = 500,
       budget: float = 50.0,
       stuck_threshold: int = 20,
       runner: RunnerProvider | None = None,
   ) -> None:
   ```

4. **Initialize** the runner based on the argument:
   ```python
   from src.providers.runner import LocalRunner, ModalRunner, RunnerProvider

   if runner is None:
       runner = LocalRunner(project_root=PROJECT_ROOT)
   ```

5. **Replace** the training call (currently lines 248-250):
   ```python
   # BEFORE:
   val_bpb, training_output = run_training()

   # AFTER:
   prepare_code = read_file(EXPERIMENTS_DIR / "prepare.py")
   train_code = read_file(TRAIN_PY)  # read what was just written
   training_result = runner.run(train_code, prepare_code)
   val_bpb = training_result.val_bpb
   training_output = training_result.output
   ```

6. **Update** the `__main__` block:
   ```python
   if args.runner == "modal":
       runner = ModalRunner()
   else:
       runner = LocalRunner(project_root=PROJECT_ROOT)

   run_loop(
       max_iterations=args.max_iterations,
       budget=args.budget,
       stuck_threshold=args.stuck_threshold,
       runner=runner,
   )
   ```

**What does NOT change:**
- Guard logic
- Council pipeline
- Git operations (still local)
- Results logging
- Experiment log format
- The rest of the loop flow

**Verification:**
- `uv run python src/app/loop.py --help` shows the `--runner` flag
- `uv run python src/app/loop.py --runner local --max-iterations 1 --budget 0.01` still works identically to today
- `uv run python src/app/loop.py --runner modal --max-iterations 1 --budget 0.50` sends training to Modal

---

### Step 5: Write Tests

**New file:** `tests/unit/test_runner.py`

```python
"""Tests for the runner provider."""

from pathlib import Path
from src.providers.runner import LocalRunner, TrainingResult, _parse_val_bpb


class TestParseValBpb:
    def test_parses_valid_output(self):
        output = "some output\nval_bpb:          1.763539\nmore output"
        assert _parse_val_bpb(output) == 1.763539

    def test_returns_none_on_missing(self):
        assert _parse_val_bpb("no bpb here") is None

    def test_returns_none_on_empty(self):
        assert _parse_val_bpb("") is None


class TestTrainingResult:
    def test_success_result(self):
        r = TrainingResult(val_bpb=1.5, output="ok", success=True)
        assert r.val_bpb == 1.5
        assert r.success is True

    def test_crash_result(self):
        r = TrainingResult(val_bpb=None, output="CRASH", success=False)
        assert r.val_bpb is None
        assert r.success is False


class TestLocalRunner:
    def test_init(self, tmp_path):
        runner = LocalRunner(project_root=tmp_path)
        assert runner.project_root == tmp_path
        assert runner.timeout == 900
```

**Update existing tests:** Any tests that mock `run_training()` or `parse_val_bpb()` in `loop.py` need their import paths updated to `src.providers.runner`.

**Verification:** `uv run pytest tests/unit/test_runner.py` passes.

---

### Step 6: Seed Modal Volume with Data

Before the first Modal run, the data needs to exist on the Volume. Two options:

**Option A (Automatic — recommended):** The `run_experiment()` function in `modal_app.py` already includes a check: if data doesn't exist on the Volume, it runs `prepare.py --num-shards 4` to download it. The first experiment will take ~5 extra minutes for data download, then all subsequent runs use the cached Volume.

**Option B (Manual — faster first run):** Create a separate Modal function to pre-seed the data:

```python
@app.function(volumes={"/data": vol}, image=image, timeout=600)
def seed_data():
    """Pre-download data shards and tokenizer to the Modal Volume."""
    # ... run prepare.py, then vol.commit()
```

Run it once: `modal run modal_app.py::seed_data`

**Recommendation:** Start with Option A. If the first-run delay is annoying, add Option B later.

---

### Step 7: Update Providers __init__.py

**File:** `src/providers/__init__.py`

Add the RunnerProvider to the docstring/imports:

```python
"""Cross-cutting providers for Rediscover. LLM, arXiv, Git, experiment runner."""

from src.providers.runner import LocalRunner, ModalRunner, RunnerProvider, TrainingResult

__all__ = [
    "LocalRunner",
    "ModalRunner",
    "RunnerProvider",
    "TrainingResult",
]
```

---

### Step 8: Update Architecture Documentation

**Files to update:**
- `ARCHITECTURE.md` — Add `runner.py` to the providers listing, update "What Does NOT Exist" section (remove "No distributed training" / "No multi-GPU support", replace with "Cloud GPU via Modal for training only")
- `docs/references/cloud-gpu-options.md` — Change status from "DEFERRED" to "IMPLEMENTED (Modal)"

---

### Step 9: Smoke Test End-to-End

Run the full loop with Modal for exactly 1 experiment:

```bash
# Ensure Modal token is set
modal token set  # if not already done

# Deploy the Modal app
modal deploy modal_app.py

# Run one experiment on Modal
uv run python src/app/loop.py --runner modal --max-iterations 1 --budget 1.00
```

**Expected behavior:**
1. Council deliberates locally (LLM API calls via OpenRouter) — ~2-5 minutes
2. Training code is sent to Modal — container spins up on A10G — ~30s cold start
3. Training runs for ~1-2 minutes (faster than MPS) — `torch.compile` activates on CUDA
4. Result returns to local machine
5. val_bpb is parsed, experiment is kept/discarded, logged to TSV + experiment_log.md

**Compare results:** Run the same experiment locally (`--runner local`) and verify that val_bpb values are in the same ballpark (they won't be identical due to CUDA vs MPS numerical differences, but should be within ~5%).

---

### Step 10: Performance Validation

Run 5 experiments on Modal and verify:

```bash
uv run python src/app/loop.py --runner modal --max-iterations 5 --budget 3.00
```

Check:
- [ ] All 5 experiments complete without crashes
- [ ] val_bpb values are reasonable (within range of local baseline)
- [ ] Total wall time per experiment is ~8-10 min (vs ~13 min local)
- [ ] Modal GPU billing shows ~1-2 min GPU time per experiment
- [ ] No data re-download on experiments 2-5 (Volume persists)
- [ ] results.tsv and experiment_log.md are populated correctly

---

## File Inventory

| File | Action | Description |
|------|--------|-------------|
| `pyproject.toml` | MODIFY | Add `modal>=0.64.0` dependency |
| `modal_app.py` | CREATE | Modal app definition with `run_experiment()` function |
| `src/providers/runner.py` | CREATE | RunnerProvider protocol + LocalRunner + ModalRunner |
| `src/providers/__init__.py` | MODIFY | Export runner types |
| `src/app/loop.py` | MODIFY | Replace inline `run_training()` with RunnerProvider, add `--runner` CLI flag |
| `tests/unit/test_runner.py` | CREATE | Unit tests for runner provider |
| `ARCHITECTURE.md` | MODIFY | Add runner.py, update capabilities |
| `docs/references/cloud-gpu-options.md` | MODIFY | Update status to IMPLEMENTED |

## Risk Register

| Risk | Mitigation |
|------|------------|
| Modal cold start adds latency (~30-60s) | Acceptable — amortized over 5-min training. Use `keep_warm=1` if it becomes a problem. |
| `prepare.py` CACHE_DIR patching is fragile | The string replacement targets a specific line. If `prepare.py` changes, the patch breaks loudly (data not found → clear error). |
| CUDA vs MPS numerical differences | Expected and acceptable. val_bpb should be within ~5%. Document this. |
| Modal Volume data corruption | Volume is append-only for our use case (download once). If corrupted, delete and re-seed. |
| `torch.compile` issues on Modal's CUDA version | train.py already handles this (compile only on CUDA, skip on MPS). If compilation fails, it falls back gracefully. |
| Modal free tier runs out mid-experiment | CostTracker only tracks LLM costs, not Modal costs. Add a note to the user about checking Modal dashboard. Future: add Modal cost tracking. |
| Network failure during Modal call | ModalRunner catches exceptions and returns TrainingResult with success=False. Loop treats this as a crash and continues. |

## Cost Estimate

| Component | Per Experiment | Per 100 Experiments | Per 24h (~150 exp) |
|-----------|---------------|--------------------|--------------------|
| Modal A10G GPU time | ~$0.04 (2 min @ $1.10/hr) | ~$4 | ~$6 |
| OpenRouter LLM calls | ~$0.02-0.05 | ~$2-5 | ~$3-7 |
| **Total** | **~$0.06-0.09** | **~$6-9** | **~$9-13** |

Free tier ($30/month) covers ~3-5 full 24-hour runs.

## Decision Log

| Decision | Rationale |
|----------|-----------|
| Protocol, not ABC | Matches project style (plain Python, no frameworks). Structural typing is sufficient. |
| Patch CACHE_DIR at runtime, don't modify prepare.py | prepare.py is "fixed, never modified by agents". Patching respects this contract. |
| Lazy modal import | Allows `--runner local` to work even if Modal isn't installed. |
| TrainingResult dataclass instead of tuple | The existing `tuple[float | None, str]` return type loses the success signal. A dataclass is clearer. |
| `modal_app.py` at repo root | Modal convention. The file needs to be importable from the project root for `ModalRunner`. |
| Download data inside `run_experiment()` on first call | Simplest approach. Avoids requiring a separate setup step. |
