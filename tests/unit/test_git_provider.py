"""Tests for git provider (uses real git in temp directories)."""

import pytest

from src.providers.git import GitProvider


@pytest.mark.unit
class TestGitProvider:
    @pytest.fixture
    def git_repo(self, tmp_path):
        """Create a temporary git repo with an initial commit."""
        git = GitProvider(repo_path=str(tmp_path))
        git.init_repo()
        # Create initial file and commit
        (tmp_path / "test.txt").write_text("initial content")
        git.commit("Initial commit", files=["test.txt"])
        return git, tmp_path

    def test_init_repo(self, tmp_path):
        git = GitProvider(repo_path=str(tmp_path))
        git.init_repo()
        assert (tmp_path / ".git").exists()

    def test_init_repo_idempotent(self, tmp_path):
        git = GitProvider(repo_path=str(tmp_path))
        git.init_repo()
        git.init_repo()  # should not raise
        assert (tmp_path / ".git").exists()

    def test_commit_and_hash(self, git_repo):
        git, tmp_path = git_repo
        (tmp_path / "new.txt").write_text("new content")
        hash_str = git.commit("Add new file", files=["new.txt"])
        assert len(hash_str) >= 7  # short hash

    def test_current_hash(self, git_repo):
        git, _ = git_repo
        hash_str = git.current_hash()
        assert len(hash_str) >= 7

    def test_log(self, git_repo):
        git, tmp_path = git_repo
        (tmp_path / "second.txt").write_text("second")
        git.commit("Second commit", files=["second.txt"])

        entries = git.log(n=5)
        assert len(entries) == 2
        assert entries[0][1] == "Second commit"
        assert entries[1][1] == "Initial commit"

    def test_reset_last(self, git_repo):
        git, tmp_path = git_repo
        initial_hash = git.current_hash()

        (tmp_path / "temp.txt").write_text("temporary")
        git.commit("Temporary commit", files=["temp.txt"])
        assert git.current_hash() != initial_hash

        git.reset_last()
        assert git.current_hash() == initial_hash
        assert not (tmp_path / "temp.txt").exists()

    def test_reset_last_preserves_files(self, git_repo):
        git, tmp_path = git_repo

        # Create files (simulate initial repo state)
        results = tmp_path / "results.tsv"
        results.write_text("header\nrow1\n")
        train = tmp_path / "train.py"
        train.write_text("original code")
        git.commit("Add results and train", files=["results.tsv", "train.py"])

        # Make a new commit (simulate experiment — modifies train.py)
        train.write_text("modified code")
        git.commit("Experiment", files=["train.py"])

        # Append to results (this happens outside git, after commit)
        results.write_text("header\nrow1\nrow2\n")

        # Reset last commit but preserve results
        git.reset_last(preserve_files=["results.tsv"])

        # train.py should be reverted, but results.tsv should keep the appended data
        assert train.read_text() == "original code"
        assert results.read_text() == "header\nrow1\nrow2\n"

    def test_diff_empty(self, git_repo):
        git, _ = git_repo
        assert git.diff() == ""

    def test_has_changes(self, git_repo):
        git, tmp_path = git_repo
        assert not git.has_changes()

        (tmp_path / "untracked.txt").write_text("new")
        assert git.has_changes()
