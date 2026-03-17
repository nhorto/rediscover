"""Shared test fixtures for Rediscover."""

import pytest


@pytest.fixture
def experiments_dir(tmp_path):
    """Temporary directory mimicking experiments/ layout."""
    (tmp_path / "train.py").touch()
    (tmp_path / "prepare.py").touch()
    (tmp_path / "results.tsv").write_text("commit\tval_bpb\tmemory_gb\tstatus\tdescription\n")
    return tmp_path
