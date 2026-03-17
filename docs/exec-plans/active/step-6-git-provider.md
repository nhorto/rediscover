# Step 6: Git Provider

> Goal: Programmatic git operations for the experiment loop.
> Status: IN PROGRESS
> Started: 2026-03-16

## Progress

- [x] Create src/providers/git.py — commit, reset, log, diff, has_changes, init_repo
- [x] Create tests/unit/test_git_provider.py — 8 tests pass (real git in tmp dirs)
- [x] Ruff check passes
- [x] Pytest 31/31 pass
- [x] Update docs

## Status: COMPLETE

## Approach

### src/providers/git.py
- `commit(message, files)` — stage specific files and commit
- `reset_last()` — `git reset --hard HEAD~1` (revert failed experiment)
- `log(n)` → list of (hash, message) tuples
- `current_hash()` → short hash of HEAD
- `diff()` → string of current uncommitted changes
- `init_repo(path)` — initialize git repo if not already initialized

All operations use subprocess.run with cwd parameter.
No external git libraries — just subprocess calls to git CLI.

## Exit Criteria
- Can commit, log, reset in a test directory
- current_hash() returns valid short hash
- reset_last() reverts to previous commit
- All tests pass
