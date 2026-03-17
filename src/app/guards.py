"""Loop guards: safety checks that control when the research loop stops or adapts."""

from dataclasses import dataclass, field

from src.utils.costs import CostTracker


@dataclass
class GuardStatus:
    """Result of checking all guards."""

    should_stop: bool
    reason: str
    should_force_novelty: bool = False
    novelty_message: str = ""


@dataclass
class LoopGuards:
    """Monitors loop health and enforces limits."""

    max_iterations: int = 500
    budget_limit: float = 50.0
    stuck_threshold: int = 20  # no improvement in this many experiments → force novelty
    error_cascade_limit: int = 3  # consecutive crashes → force different approach
    similarity_threshold: float = 0.9  # cosine sim between hypotheses → reject

    # Internal state
    iteration: int = 0
    best_val_bpb: float = float("inf")
    experiments_since_improvement: int = 0
    consecutive_errors: int = 0
    recent_hypotheses: list[str] = field(default_factory=list)
    results_history: list[dict] = field(default_factory=list)

    def record_result(self, val_bpb: float | None, status: str, hypothesis: str) -> None:
        """Record an experiment result and update internal state."""
        self.iteration += 1
        self.recent_hypotheses.append(hypothesis)
        if len(self.recent_hypotheses) > 5:
            self.recent_hypotheses.pop(0)

        self.results_history.append({
            "iteration": self.iteration,
            "val_bpb": val_bpb,
            "status": status,
            "hypothesis": hypothesis,
        })

        if status == "crash":
            self.consecutive_errors += 1
        else:
            self.consecutive_errors = 0

        if val_bpb is not None and val_bpb < self.best_val_bpb:
            self.best_val_bpb = val_bpb
            self.experiments_since_improvement = 0
        else:
            self.experiments_since_improvement += 1

    def check(self, cost_tracker: CostTracker) -> GuardStatus:
        """Check all guards. Returns a GuardStatus."""
        # Max iterations
        if self.iteration >= self.max_iterations:
            return GuardStatus(
                should_stop=True,
                reason=f"Max iterations reached ({self.max_iterations})",
            )

        # Budget cap
        if cost_tracker.total_cost >= self.budget_limit:
            return GuardStatus(
                should_stop=True,
                reason=f"Budget exceeded: ${cost_tracker.total_cost:.2f} >= ${self.budget_limit:.2f}",
            )

        # Stuck detection
        if self.experiments_since_improvement >= self.stuck_threshold:
            return GuardStatus(
                should_stop=False,
                reason="",
                should_force_novelty=True,
                novelty_message=(
                    f"WARNING: No improvement in {self.experiments_since_improvement} experiments. "
                    f"Best val_bpb remains {self.best_val_bpb:.6f}. "
                    "Try something FUNDAMENTALLY different — a new architectural idea, "
                    "not a variation of what you've been trying."
                ),
            )

        # Error cascade
        if self.consecutive_errors >= self.error_cascade_limit:
            return GuardStatus(
                should_stop=False,
                reason="",
                should_force_novelty=True,
                novelty_message=(
                    f"WARNING: {self.consecutive_errors} consecutive crashes. "
                    "The recent approaches are producing broken code. "
                    "Try a SIMPLER change — a small hyperparameter adjustment or "
                    "a minimal architectural tweak. Avoid complex refactors."
                ),
            )

        return GuardStatus(should_stop=False, reason="")

    def summary(self) -> str:
        """Human-readable guard status."""
        return (
            f"Iteration {self.iteration}/{self.max_iterations} | "
            f"Best val_bpb: {self.best_val_bpb:.6f} | "
            f"Since improvement: {self.experiments_since_improvement} | "
            f"Consecutive errors: {self.consecutive_errors}"
        )
