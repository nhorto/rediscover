"""Git provider: programmatic git operations for the experiment loop."""

import subprocess
from dataclasses import dataclass
from pathlib import Path


@dataclass
class GitProvider:
    """Git operations via subprocess. All commands run in the specified working directory."""

    repo_path: str

    def _run(self, *args: str, check: bool = True) -> str:
        """Run a git command and return stdout."""
        result = subprocess.run(
            ["git", *args],
            cwd=self.repo_path,
            capture_output=True,
            text=True,
            check=check,
        )
        return result.stdout.strip()

    def init_repo(self) -> None:
        """Initialize a git repo if not already initialized."""
        git_dir = Path(self.repo_path) / ".git"
        if not git_dir.exists():
            self._run("init")
            self._run("checkout", "-b", "main")

    def commit(self, message: str, files: list[str] | None = None) -> str:
        """Stage files and commit. Returns the commit hash."""
        if files:
            for f in files:
                self._run("add", f)
        else:
            self._run("add", "-A")
        self._run("commit", "-m", message)
        return self.current_hash()

    def reset_last(self, preserve_files: list[str] | None = None) -> None:
        """Revert the last commit, preserving specified files.

        Uses git reset --hard HEAD~1 to revert train.py changes,
        but saves and restores any files that should not be affected
        (e.g., results.tsv, experiment_log.md).
        """
        # Save contents of files to preserve
        saved: dict[str, str] = {}
        if preserve_files:
            for f in preserve_files:
                path = Path(self.repo_path) / f
                if path.exists():
                    saved[f] = path.read_text()

        self._run("reset", "--hard", "HEAD~1")

        # Restore preserved files
        for f, content in saved.items():
            path = Path(self.repo_path) / f
            path.write_text(content)

    def current_hash(self) -> str:
        """Return the short hash of HEAD."""
        return self._run("rev-parse", "--short", "HEAD")

    def log(self, n: int = 10) -> list[tuple[str, str]]:
        """Return last n commits as (hash, message) tuples."""
        output = self._run("log", f"--max-count={n}", "--format=%h\t%s")
        if not output:
            return []
        entries = []
        for line in output.split("\n"):
            if "\t" in line:
                hash_str, msg = line.split("\t", 1)
                entries.append((hash_str, msg))
        return entries

    def diff(self) -> str:
        """Return current uncommitted changes as a diff string."""
        return self._run("diff")

    def has_changes(self) -> bool:
        """Check if there are uncommitted changes."""
        status = self._run("status", "--porcelain")
        return len(status) > 0
