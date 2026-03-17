"""Main research loop: ties council + training + git + guards into an autonomous system."""

import argparse
import re
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

from src.app.guards import LoopGuards
from src.domains.council.service import CouncilService
from src.domains.literature.service import LiteratureService
from src.providers.git import GitProvider
from src.providers.llm import LLMProvider
from src.utils.costs import CostTracker

# Paths relative to project root
PROJECT_ROOT = Path(__file__).parent.parent.parent
EXPERIMENTS_DIR = PROJECT_ROOT / "experiments"
TRAIN_PY = EXPERIMENTS_DIR / "train.py"
RESULTS_TSV = EXPERIMENTS_DIR / "results.tsv"
PROGRAM_MD = EXPERIMENTS_DIR / "program.md"
EXPERIMENT_LOG = EXPERIMENTS_DIR / "experiment_log.md"

# Training timeout: TIME_BUDGET (300s) + 600s for startup/eval overhead
TRAINING_TIMEOUT = 900


def read_file(path: Path) -> str:
    """Read a file and return its contents."""
    return path.read_text()


def parse_val_bpb(output: str) -> float | None:
    """Parse val_bpb from training script output."""
    match = re.search(r"val_bpb:\s+([\d.]+)", output)
    if match:
        return float(match.group(1))
    return None


def run_training() -> tuple[float | None, str]:
    """Run experiments/train.py and return (val_bpb, full_output).

    Returns (None, output) if training crashes or times out.
    """
    try:
        result = subprocess.run(
            [sys.executable, str(TRAIN_PY)],
            capture_output=True,
            text=True,
            timeout=TRAINING_TIMEOUT,
            cwd=str(PROJECT_ROOT),
        )
        output = result.stdout + result.stderr
        if result.returncode != 0:
            return None, output
        val_bpb = parse_val_bpb(output)
        return val_bpb, output
    except subprocess.TimeoutExpired:
        return None, "TIMEOUT: Training exceeded time limit"
    except Exception as e:
        return None, f"ERROR: {e}"


def append_results_tsv(commit: str, val_bpb: float | None, status: str, description: str) -> None:
    """Append a row to results.tsv."""
    bpb_str = f"{val_bpb:.6f}" if val_bpb is not None else "N/A"
    row = f"{commit}\t{bpb_str}\t0.0\t{status}\t{description}\n"
    with open(RESULTS_TSV, "a") as f:
        f.write(row)


def append_experiment_log(
    iteration: int,
    proposal,
    critique,
    plan,
    val_bpb: float | None,
    status: str,
    cost_this_cycle: float,
    cumulative_cost: float,
) -> None:
    """Append a narrative block to experiment_log.md."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    bpb_str = f"{val_bpb:.6f}" if val_bpb is not None else "CRASH"

    papers_str = ""
    if proposal.papers_found:
        papers_str = ", ".join(p.title[:60] for p in proposal.papers_found[:3])

    entry = f"""
## Experiment {iteration} — {timestamp}
**Hypothesis:** {proposal.hypothesis}
**Approach:** {proposal.approach}
**Papers consulted:** {papers_str or 'None'}
**Critique:** {critique.overall_assessment[:200]}
**Plan:** {plan.description}
**Result:** val_bpb={bpb_str} ({status})
**Cost this cycle:** ${cost_this_cycle:.4f}
**Cumulative cost:** ${cumulative_cost:.4f}
---
"""
    with open(EXPERIMENT_LOG, "a") as f:
        f.write(entry)


def run_loop(
    max_iterations: int = 500,
    budget: float = 50.0,
    stuck_threshold: int = 20,
) -> None:
    """Run the autonomous research loop."""
    print("=" * 60)
    print("REDISCOVER — Autonomous ML Research Loop")
    print("=" * 60)

    # Initialize providers
    cost_tracker = CostTracker(budget_limit=budget)
    llm = LLMProvider(cost_tracker=cost_tracker)
    git = GitProvider(repo_path=str(PROJECT_ROOT))

    # Initialize literature service (if chroma DB exists)
    literature = None
    chroma_path = PROJECT_ROOT / "data" / "chroma_db"
    if chroma_path.exists():
        literature = LiteratureService(chroma_path=str(chroma_path))
        print(f"Literature: {literature.paper_count} papers in knowledge base")
    else:
        print("Literature: No knowledge base found (run paper ingestion first)")

    # Initialize council and guards
    council = CouncilService(llm=llm, literature=literature)
    guards = LoopGuards(
        max_iterations=max_iterations,
        budget_limit=budget,
        stuck_threshold=stuck_threshold,
    )

    # Read best val_bpb from results.tsv (minimum across all "keep" results)
    results_content = read_file(RESULTS_TSV)
    lines = results_content.strip().split("\n")
    if len(lines) > 1:
        best_keep = float("inf")
        for line in lines[1:]:
            parts = line.split("\t")
            if len(parts) >= 4 and parts[3] == "keep":
                try:
                    val = float(parts[1])
                    best_keep = min(best_keep, val)
                except ValueError:
                    pass
        if best_keep < float("inf"):
            guards.best_val_bpb = best_keep
            print(f"Best val_bpb from history: {best_keep:.6f}")

    print(f"Budget: ${budget:.2f}")
    print(f"Max iterations: {max_iterations}")
    print(f"Models: {llm.model_map}")
    print("=" * 60)
    print()

    # Main loop
    while True:
        # Check guards
        guard_status = guards.check(cost_tracker)
        if guard_status.should_stop:
            print(f"\n{'=' * 60}")
            print(f"LOOP STOPPED: {guard_status.reason}")
            print(f"{'=' * 60}")
            break

        iteration = guards.iteration + 1
        print(f"\n--- Experiment {iteration} ---")

        # Read current state
        train_py = read_file(TRAIN_PY)
        results_tsv = read_file(RESULTS_TSV)
        program_md = read_file(PROGRAM_MD)

        # Inject novelty message if guards demand it
        if guard_status.should_force_novelty:
            print(f"GUARD: {guard_status.novelty_message}")
            program_md += f"\n\n## IMPORTANT\n{guard_status.novelty_message}"

        # Track cost for this cycle
        cost_before = cost_tracker.total_cost

        # Run council
        print("  Council deliberating...")
        try:
            result = council.run_council(train_py, results_tsv, program_md)
        except Exception as e:
            print(f"  Council FAILED: {e}")
            guards.record_result(None, "crash", f"Council error: {e}")
            continue

        cost_this_cycle = cost_tracker.total_cost - cost_before
        print(f"  Hypothesis: {result.proposal.hypothesis[:100]}")
        print(f"  Critique: {result.critique.overall_assessment[:100]}")
        print(f"  Plan: {result.plan.description[:100]}")
        print(f"  Council cost: ${cost_this_cycle:.4f}")

        # Write new train.py
        TRAIN_PY.write_text(result.new_train_py)

        # Git commit
        try:
            commit_hash = git.commit(
                f"Experiment {iteration}: {result.plan.description[:80]}",
                files=[str(TRAIN_PY)],
            )
        except Exception as e:
            print(f"  Git commit failed: {e}")
            TRAIN_PY.write_text(train_py)  # restore original
            guards.record_result(None, "crash", f"Git error: {e}")
            continue

        # Run training
        print("  Training (5 min budget)...")
        t0 = time.time()
        val_bpb, training_output = run_training()
        training_time = time.time() - t0
        print(f"  Training completed in {training_time:.0f}s")

        # Evaluate
        if val_bpb is None:
            status = "crash"
            print("  CRASH — reverting")
            print(f"  Output: {training_output[:200]}")
            git.reset_last()
            description = f"CRASH: {result.plan.description[:60]}"
        elif val_bpb < guards.best_val_bpb:
            status = "keep"
            print(f"  KEEP — val_bpb={val_bpb:.6f} (improved from {guards.best_val_bpb:.6f})")
            description = f"KEEP: {result.plan.description[:60]} (val_bpb={val_bpb:.6f})"
        else:
            status = "discard"
            print(f"  DISCARD — val_bpb={val_bpb:.6f} (best={guards.best_val_bpb:.6f})")
            git.reset_last()
            description = f"DISCARD: {result.plan.description[:60]} (val_bpb={val_bpb:.6f})"

        # Log
        append_results_tsv(commit_hash if status == "keep" else "reverted", val_bpb, status, description)
        append_experiment_log(
            iteration, result.proposal, result.critique, result.plan,
            val_bpb, status, cost_this_cycle, cost_tracker.total_cost,
        )

        # Update guards
        guards.record_result(val_bpb, status, result.proposal.hypothesis)

        # Status
        print(f"  {guards.summary()}")
        print(f"  {cost_tracker.summary()}")

    # Final summary
    print(f"\nFinal: {guards.summary()}")
    print(f"Final: {cost_tracker.summary()}")
    print(f"Results saved to: {RESULTS_TSV}")
    print(f"Log saved to: {EXPERIMENT_LOG}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Rediscover autonomous research loop")
    parser.add_argument("--max-iterations", type=int, default=500, help="Maximum experiments to run")
    parser.add_argument("--budget", type=float, default=50.0, help="Maximum LLM spend in dollars")
    parser.add_argument("--stuck-threshold", type=int, default=20, help="Experiments without improvement before forcing novelty")
    args = parser.parse_args()

    run_loop(
        max_iterations=args.max_iterations,
        budget=args.budget,
        stuck_threshold=args.stuck_threshold,
    )
