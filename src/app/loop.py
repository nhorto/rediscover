"""Main research loop: ties council + training + git + guards into an autonomous system."""

import argparse
import re
import stat
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
PREPARE_PY = EXPERIMENTS_DIR / "prepare.py"
RESULTS_TSV = EXPERIMENTS_DIR / "results.tsv"
PROGRAM_MD = EXPERIMENTS_DIR / "program.md"
EXPERIMENT_LOG = EXPERIMENTS_DIR / "experiment_log.md"

# Files that should be read-only during training (prevent agent cheating)
PROTECTED_FILES = [PREPARE_PY, RESULTS_TSV, PROGRAM_MD, EXPERIMENT_LOG]

# Files to preserve during git reset (don't lose results when reverting experiments)
PRESERVE_ON_RESET = [str(RESULTS_TSV), str(EXPERIMENT_LOG)]

# Training timeout: TIME_BUDGET (300s) + 600s for startup/eval overhead
TRAINING_TIMEOUT = 900

# Patterns that indicate dangerous code in train.py
DANGEROUS_PATTERNS = [
    r'open\s*\([^)]*["\'](?:\.\.|\bprepare\b|\bresults\b|\bprogram\b)',  # writing to protected files
    r'os\.(?:system|popen|remove|unlink|rmdir)',  # shell commands, file deletion
    r'subprocess\.',  # subprocess calls
    r'shutil\.',  # file operations
    r'__import__',  # dynamic imports
    r'(?<!\w)exec\s*\(',  # dynamic code execution (not .exec_something)
    r'(?<!\.)eval\s*\(',  # dynamic evaluation (not model.eval())
]


def safe_fix_code(council, code: str, error: str, log: list) -> tuple[str, bool]:
    """Try to fix code via council. Returns (fixed_code, success).

    If the API call fails (timeout, etc.), returns the original code unchanged.
    """
    try:
        fixed, _ = council.fix_code(code, error, log)
        return fixed, True
    except Exception as e:
        print(f"  Fix API call failed: {type(e).__name__}: {str(e)[:100]}")
        return code, False


def read_file(path: Path) -> str:
    """Read a file and return its contents."""
    return path.read_text()


def protect_files() -> None:
    """Make protected files read-only to prevent agent tampering during training."""
    for f in PROTECTED_FILES:
        if f.exists():
            f.chmod(stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH)  # 444


def unprotect_files() -> None:
    """Restore write permissions on protected files after training."""
    for f in PROTECTED_FILES:
        if f.exists():
            f.chmod(stat.S_IRUSR | stat.S_IWUSR | stat.S_IRGRP | stat.S_IROTH)  # 644


def validate_train_py(code: str) -> tuple[bool, str]:
    """Check the generated train.py for dangerous patterns.

    Returns (is_safe, reason). If not safe, reason explains why.
    """
    for pattern in DANGEROUS_PATTERNS:
        match = re.search(pattern, code)
        if match:
            return False, f"Dangerous pattern detected: {match.group(0)}"

    # Check it's valid Python syntax
    try:
        compile(code, "train.py", "exec")
    except SyntaxError as e:
        return False, f"Syntax error: {e}"

    return True, ""


def validate_zone_structure(code: str) -> tuple[bool, str]:
    """Check that the generated code preserves required structure.

    Catches common model mistakes before expensive subprocess validation:
    - norm() helper must exist
    - No bias=True on nn.Linear (MuonAdamW crashes on 1D params)
    - CausalSelfAttention.forward signature must match interface
    """
    # Check norm() function exists
    if "def norm(" not in code:
        return False, "Missing norm() helper function — it must be preserved"

    # Check for bias=True on nn.Linear (causes MuonAdamW IndexError)
    bias_match = re.search(r'nn\.Linear\([^)]*bias\s*=\s*True', code)
    if bias_match:
        return False, "nn.Linear with bias=True will crash MuonAdamW optimizer — use bias=False"

    # Check for 1D nn.Parameter (MuonAdamW requires all params to be 2D+)
    param_1d = re.search(r'nn\.Parameter\(torch\.\w+\(\s*\d+\s*\)', code)
    if param_1d:
        return False, f"1D nn.Parameter found ({param_1d.group(0)}) — MuonAdamW crashes on <2D params. Use 2D shape like (1, n) or register as buffer"

    # Check CausalSelfAttention.forward signature
    forward_match = re.search(r'def forward\(self,\s*(\w+)', code)
    if forward_match:
        # Must accept (self, x, ve, cos_sin, window_size)
        full_sig = re.search(r'def forward\(self,[^)]+\)', code)
        if full_sig:
            sig = full_sig.group(0)
            for param in ["ve", "cos_sin", "window_size"]:
                if param not in sig:
                    return False, f"forward() missing required parameter '{param}' — signature must be forward(self, x, ve, cos_sin, window_size)"

    return True, ""


def quick_validate_code(code: str) -> tuple[bool, str]:
    """Write code to a temp file and validate model init + one forward pass.

    This catches: missing imports, undefined names, shape mismatches.
    Inserts an exit point after model construction and a dummy forward pass,
    before the training loop starts. Runs in a subprocess with a 60-second timeout.
    """
    import tempfile

    # Insert exit-after-init: find the training loop start and replace it
    # with a quick forward pass test followed by sys.exit(0)
    validation_snippet = (
        "\n# --- QUICK VALIDATION: test model init + one forward pass ---\n"
        "print('Quick validation: testing forward pass...')\n"
        "try:\n"
        "    _test_x = torch.randint(0, vocab_size, (1, 64), device=device)\n"
        "    _test_y = torch.randint(0, vocab_size, (1, 64), device=device)\n"
        "    with torch.no_grad():\n"
        "        _test_loss = model(_test_x, _test_y)\n"
        "    print(f'Quick validation PASSED: loss={_test_loss.item():.4f}')\n"
        "    import sys; sys.exit(0)\n"
        "except Exception as e:\n"
        "    print(f'Quick validation FAILED: {e}', file=__import__('sys').stderr)\n"
        "    import traceback; traceback.print_exc(file=__import__('sys').stderr)\n"
        "    import sys; sys.exit(1)\n"
        "# --- END QUICK VALIDATION ---\n"
    )

    # Insert the validation snippet before the training loop
    # The training loop starts with "t_start_training = time.time()"
    marker = "t_start_training = time.time()"
    if marker in code:
        test_code = code.replace(marker, validation_snippet + marker, 1)
    else:
        # Fallback: append sys.exit(0) at the end
        test_code = code + "\nimport sys; sys.exit(0)\n"

    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False, dir=str(EXPERIMENTS_DIR)) as f:
        f.write(test_code)
        temp_path = f.name

    try:
        result = subprocess.run(
            [sys.executable, temp_path],
            capture_output=True,
            text=True,
            timeout=60,
            cwd=str(PROJECT_ROOT),
        )
        if result.returncode != 0:
            error = result.stderr[-1500:] if result.stderr else result.stdout[-1000:]
            return False, f"Model init/forward error: {error}"
        return True, ""
    except subprocess.TimeoutExpired:
        return False, "Code validation timed out (>60s)"
    finally:
        Path(temp_path).unlink(missing_ok=True)


def validate_diff_is_attention_related(old_code: str, new_code: str) -> tuple[bool, str]:
    """Check that changes are primarily in attention-related code, not just hyperparameter gaming.

    Returns (is_valid, reason). Allows attention changes + minor supporting changes.
    Rejects changes that ONLY modify hyperparameters without touching attention.
    """
    old_lines = old_code.strip().split("\n")
    new_lines = new_code.strip().split("\n")

    # Find changed line numbers (simple line-by-line diff)
    changed_in_attention = False
    changed_in_hyperparams_only = True

    # Sections we consider "attention-related"
    attention_markers = [
        "class CausalSelfAttention",
        "def forward(self, x, ve, cos_sin",
        "apply_rotary_emb",
        "scaled_dot_product_attention",
        "self.c_q", "self.c_k", "self.c_v", "self.c_proj",
        "n_head", "n_kv_head", "head_dim",
        "window_size", "window_pattern",
        "ve_gate", "value_embeds",
        "norm(q)", "norm(k)",
    ]

    # Hyperparameter-only markers
    hyperparam_markers = [
        "ASPECT_RATIO", "HEAD_DIM", "WINDOW_PATTERN",
        "TOTAL_BATCH_SIZE", "EMBEDDING_LR", "UNEMBEDDING_LR",
        "MATRIX_LR", "SCALAR_LR", "WEIGHT_DECAY",
        "ADAM_BETAS", "WARMUP_RATIO", "WARMDOWN_RATIO",
        "FINAL_LR_FRAC", "DEPTH", "DEVICE_BATCH_SIZE",
    ]

    # Compare lines to find changes
    has_any_changes = False
    max_lines = max(len(old_lines), len(new_lines))
    for i in range(max_lines):
        old_line = old_lines[i] if i < len(old_lines) else ""
        new_line = new_lines[i] if i < len(new_lines) else ""

        if old_line != new_line:
            has_any_changes = True
            line_content = new_line + old_line
            # Check if this change touches attention code
            if any(marker in line_content for marker in attention_markers):
                changed_in_attention = True
                changed_in_hyperparams_only = False
            # Check if this is ONLY a hyperparameter change
            elif any(marker in line_content for marker in hyperparam_markers):
                pass  # hyperparameter changes are fine IF attention also changed
            else:
                changed_in_hyperparams_only = False

    # No changes at all is fine (nothing wrong happened)
    if not has_any_changes:
        return True, ""

    if not changed_in_attention and changed_in_hyperparams_only:
        return False, "Changes only modify hyperparameters without touching attention code"

    return True, ""


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
        # Combine output, prioritizing stderr (where Python tracebacks go)
        output = result.stderr + "\n" + result.stdout if result.stderr else result.stdout
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
    embedder = None
    chroma_path = PROJECT_ROOT / "data" / "chroma_db"
    if chroma_path.exists():
        literature = LiteratureService(chroma_path=str(chroma_path))
        embedder = literature.embedder  # reuse for hypothesis similarity
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

        # Run council (with similarity-based retry)
        max_retries = 3
        result = None
        for attempt in range(max_retries):
            print(f"  Council deliberating{' (retry ' + str(attempt) + ')' if attempt > 0 else ''}...")
            try:
                result = council.run_council(train_py, results_tsv, program_md)
            except Exception as e:
                print(f"  Council FAILED: {e}")
                guards.record_result(None, "crash", f"Council error: {e}")
                break

            # Check hypothesis similarity against recent experiments
            if embedder is not None:
                hyp_embedding = embedder.encode([result.proposal.hypothesis])[0]
                sim_result = guards.check_similarity(hyp_embedding)
                if sim_result.is_too_similar:
                    print(f"  SIMILAR ({sim_result.most_similar_score:.2f}) to: {sim_result.most_similar_hypothesis[:80]}")
                    if attempt < max_retries - 1:
                        program_md += (
                            f"\n\n## IMPORTANT: AVOID REPETITION\n"
                            f"Your last proposal was too similar (score={sim_result.most_similar_score:.2f}) to a previous one: "
                            f'"{sim_result.most_similar_hypothesis[:200]}". '
                            f"Propose something DIFFERENT."
                        )
                        continue
                    else:
                        print("  Max retries reached — proceeding with this hypothesis anyway")
            break

        if result is None:
            continue

        cost_this_cycle = cost_tracker.total_cost - cost_before
        print(f"  Hypothesis: {result.proposal.hypothesis[:100]}")
        print(f"  Critique: {result.critique.overall_assessment[:100]}")
        print(f"  Plan: {result.plan.description[:100]}")
        print(f"  Council cost: ${cost_this_cycle:.4f}")

        # Validate and run with error feedback (max 3 fix attempts)
        current_code = result.new_train_py
        max_fix_attempts = 3
        val_bpb = None
        training_output = ""
        code_accepted = False

        for fix_attempt in range(max_fix_attempts + 1):
            # Safety check
            is_safe, safety_reason = validate_train_py(current_code)
            if not is_safe:
                if fix_attempt < max_fix_attempts:
                    print(f"  FIX ATTEMPT {fix_attempt + 1}: {safety_reason}")
                    current_code, ok = safe_fix_code(council, current_code, safety_reason, result.log)
                    if not ok:
                        break
                    cost_this_cycle = cost_tracker.total_cost - cost_before
                    continue
                print(f"  REJECTED (unsafe after {max_fix_attempts} fixes): {safety_reason}")
                guards.record_result(None, "crash", f"Unsafe code: {safety_reason}")
                break

            # Zone structure check (cheap string matching — catches norm/bias/signature issues)
            is_structured, struct_reason = validate_zone_structure(current_code)
            if not is_structured:
                if fix_attempt < max_fix_attempts:
                    print(f"  FIX ATTEMPT {fix_attempt + 1}: {struct_reason}")
                    current_code, ok = safe_fix_code(council, current_code, struct_reason, result.log)
                    if not ok:
                        break
                    cost_this_cycle = cost_tracker.total_cost - cost_before
                    continue
                print(f"  REJECTED (structure invalid after {max_fix_attempts} fixes): {struct_reason}")
                guards.record_result(None, "crash", f"Structure: {struct_reason}")
                break

            # Topic check
            is_on_topic, topic_reason = validate_diff_is_attention_related(train_py, current_code)
            if not is_on_topic:
                print(f"  REJECTED (off-topic): {topic_reason}")
                guards.record_result(None, "discard", f"Off-topic: {topic_reason}")
                break

            # Quick validation (import + init)
            print(f"  Validating code{' (attempt ' + str(fix_attempt + 1) + ')' if fix_attempt > 0 else ''}...")
            is_valid, valid_reason = quick_validate_code(current_code)
            if not is_valid:
                if fix_attempt < max_fix_attempts:
                    print(f"  FIX ATTEMPT {fix_attempt + 1}: {valid_reason[:150]}")
                    current_code, ok = safe_fix_code(council, current_code, valid_reason, result.log)
                    if not ok:
                        break
                    cost_this_cycle = cost_tracker.total_cost - cost_before
                    continue
                print(f"  REJECTED (validation failed after {max_fix_attempts} fixes): {valid_reason[:150]}")
                guards.record_result(None, "crash", f"Validation failed: {valid_reason[:100]}")
                break

            # Code passed all checks — write, commit, train
            code_accepted = True
            break

        if not code_accepted:
            continue

        # Write new train.py
        TRAIN_PY.write_text(current_code)

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

        # Protect files and run training
        print("  Training (5 min budget)...")
        protect_files()
        t0 = time.time()
        try:
            val_bpb, training_output = run_training()
        finally:
            unprotect_files()
        training_time = time.time() - t0
        print(f"  Training completed in {training_time:.0f}s")

        # If training crashed, try error feedback
        if val_bpb is None and max_fix_attempts > 0:
            # Save failing code for debugging
            debug_path = EXPERIMENTS_DIR / f"debug_crash_{iteration}.py"
            debug_path.write_text(current_code)
            print("  Training CRASHED — attempting fix...")
            print(f"  Failing code saved to: {debug_path}")
            print(f"  STDERR: {training_output[-800:]}")
            git.reset_last(preserve_files=PRESERVE_ON_RESET)

            # Try to fix the code based on the training error
            fixed_code, fix_ok = safe_fix_code(council, current_code, training_output[:2000], result.log)
            cost_this_cycle = cost_tracker.total_cost - cost_before
            if not fix_ok:
                print("  Fix API call failed — giving up on this experiment")
                fixed_code = current_code  # ensure we don't train broken code

            # Validate the fix
            is_safe_fix, _ = validate_train_py(fixed_code)
            is_valid_fix, _ = quick_validate_code(fixed_code) if is_safe_fix else (False, "")

            if is_safe_fix and is_valid_fix:
                print("  Fix passed validation — retraining...")
                TRAIN_PY.write_text(fixed_code)
                try:
                    commit_hash = git.commit(
                        f"Experiment {iteration} (fixed): {result.plan.description[:70]}",
                        files=[str(TRAIN_PY)],
                    )
                except Exception:
                    TRAIN_PY.write_text(train_py)
                    guards.record_result(None, "crash", "Git error on fix")
                    continue

                protect_files()
                t0 = time.time()
                try:
                    val_bpb, training_output = run_training()
                finally:
                    unprotect_files()
                training_time = time.time() - t0
                print(f"  Fixed training completed in {training_time:.0f}s")
                current_code = fixed_code
            else:
                print("  Fix failed validation — giving up on this experiment")

        # Evaluate
        if val_bpb is None:
            status = "crash"
            print("  CRASH — reverting")
            print(f"  Output: {training_output[:200]}")
            git.reset_last(preserve_files=PRESERVE_ON_RESET)
            description = f"CRASH: {result.plan.description[:60]}"
        elif val_bpb < guards.best_val_bpb:
            status = "keep"
            print(f"  KEEP — val_bpb={val_bpb:.6f} (improved from {guards.best_val_bpb:.6f})")
            description = f"KEEP: {result.plan.description[:60]} (val_bpb={val_bpb:.6f})"
        else:
            status = "discard"
            print(f"  DISCARD — val_bpb={val_bpb:.6f} (best={guards.best_val_bpb:.6f})")
            git.reset_last(preserve_files=PRESERVE_ON_RESET)
            description = f"DISCARD: {result.plan.description[:60]} (val_bpb={val_bpb:.6f})"

        # Log
        append_results_tsv(commit_hash if status == "keep" else "reverted", val_bpb, status, description)
        append_experiment_log(
            iteration, result.proposal, result.critique, result.plan,
            val_bpb, status, cost_this_cycle, cost_tracker.total_cost,
        )

        # Update guards
        guards.record_result(val_bpb, status, result.proposal.hypothesis)
        if embedder is not None:
            hyp_embedding = embedder.encode([result.proposal.hypothesis])[0]
            guards.record_hypothesis_embedding(hyp_embedding)

        # Status
        print(f"  {guards.summary()}")
        print(f"  {cost_tracker.summary()}")

    # Final summary
    print(f"\nFinal: {guards.summary()}")
    print(f"Final: {cost_tracker.summary()}")
    print(f"Results saved to: {RESULTS_TSV}")
    print(f"Log saved to: {EXPERIMENT_LOG}")


if __name__ == "__main__":
    # Force unbuffered output so logs appear in real-time when redirected
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, line_buffering=True)
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, line_buffering=True)

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
