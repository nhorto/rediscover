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
            output = result.stderr + "\n" + result.stdout if result.stderr else result.stdout
            if result.returncode != 0:
                return TrainingResult(val_bpb=None, output=output, success=False)
            val_bpb = _parse_val_bpb(output)
            return TrainingResult(val_bpb=val_bpb, output=output, success=val_bpb is not None)
        except subprocess.TimeoutExpired:
            return TrainingResult(val_bpb=None, output="TIMEOUT: Training exceeded time limit", success=False)
        except Exception as e:
            return TrainingResult(val_bpb=None, output=f"ERROR: {e}", success=False)


class ModalRunner:
    """Run experiments on Modal cloud GPU."""

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
                output="ERROR: modal not installed or modal_app.py not found. Run: uv add modal",
                success=False,
            )
        except Exception as e:
            return TrainingResult(val_bpb=None, output=f"MODAL ERROR: {e}", success=False)


def _parse_val_bpb(output: str) -> float | None:
    """Parse val_bpb from training script output."""
    match = re.search(r"val_bpb:\s+([\d.]+)", output)
    return float(match.group(1)) if match else None
