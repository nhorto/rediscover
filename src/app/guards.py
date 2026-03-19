"""Loop guards: safety checks that control when the research loop stops or adapts."""

from dataclasses import dataclass, field

import numpy as np

from src.utils.costs import CostTracker


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine similarity between two vectors."""
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


@dataclass
class GuardStatus:
    """Result of checking all guards."""

    should_stop: bool
    reason: str
    should_force_novelty: bool = False
    novelty_message: str = ""


@dataclass
class SimilarityResult:
    """Result of checking a hypothesis against previous ones."""

    is_too_similar: bool
    most_similar_score: float
    most_similar_hypothesis: str


@dataclass
class LoopGuards:
    """Monitors loop health and enforces limits."""

    max_iterations: int = 500
    budget_limit: float = 50.0
    stuck_threshold: int = 20  # no improvement in this many experiments → force novelty
    error_cascade_limit: int = 3  # consecutive crashes → force different approach
    error_hard_stop: int = 10  # consecutive crashes → stop the loop entirely
    similarity_threshold: float = 0.8  # cosine sim between hypotheses → reject
    max_hypothesis_history: int = 50  # how many past hypotheses to check against

    # Internal state
    iteration: int = 0
    best_val_bpb: float = float("inf")
    experiments_since_improvement: int = 0
    consecutive_errors: int = 0
    recent_hypotheses: list[str] = field(default_factory=list)
    hypothesis_embeddings: list[np.ndarray] = field(default_factory=list)
    results_history: list[dict] = field(default_factory=list)

    def record_result(self, val_bpb: float | None, status: str, hypothesis: str) -> None:
        """Record an experiment result and update internal state."""
        self.iteration += 1
        self.recent_hypotheses.append(hypothesis)
        if len(self.recent_hypotheses) > self.max_hypothesis_history:
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

    def record_hypothesis_embedding(self, embedding: np.ndarray) -> None:
        """Store the embedding for a hypothesis (called after recording result)."""
        self.hypothesis_embeddings.append(embedding)
        if len(self.hypothesis_embeddings) > self.max_hypothesis_history:
            self.hypothesis_embeddings.pop(0)

    def check_similarity(self, new_embedding: np.ndarray) -> SimilarityResult:
        """Check if a new hypothesis is too similar to recent ones."""
        if not self.hypothesis_embeddings:
            return SimilarityResult(is_too_similar=False, most_similar_score=0.0, most_similar_hypothesis="")

        best_score = 0.0
        best_hypothesis = ""

        for i, stored_emb in enumerate(self.hypothesis_embeddings):
            score = cosine_similarity(new_embedding, stored_emb)
            if score > best_score:
                best_score = score
                # Get the corresponding hypothesis text
                offset = max(0, len(self.recent_hypotheses) - len(self.hypothesis_embeddings))
                if i + offset < len(self.recent_hypotheses):
                    best_hypothesis = self.recent_hypotheses[i + offset]

        return SimilarityResult(
            is_too_similar=best_score >= self.similarity_threshold,
            most_similar_score=best_score,
            most_similar_hypothesis=best_hypothesis,
        )

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

        # Error hard stop — if everything fails repeatedly, something is fundamentally broken
        if self.consecutive_errors >= self.error_hard_stop:
            return GuardStatus(
                should_stop=True,
                reason=f"Error cascade: {self.consecutive_errors} consecutive failures — stopping loop",
            )

        # Error cascade — force different approach
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
