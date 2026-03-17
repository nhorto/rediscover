"""Smoke tests to verify project setup."""

import pytest


@pytest.mark.unit
def test_imports():
    """Verify core dependencies are importable."""
    import torch
    assert torch.__version__


@pytest.mark.unit
def test_mps_available():
    """Verify MPS (Metal) is available on this machine."""
    import torch
    assert torch.backends.mps.is_available(), "MPS not available — Apple Silicon required"


@pytest.mark.unit
def test_experiments_dir_fixture(experiments_dir):
    """Verify test fixture creates expected layout."""
    assert (experiments_dir / "train.py").exists()
    assert (experiments_dir / "results.tsv").exists()
