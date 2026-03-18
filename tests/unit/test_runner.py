"""Tests for the runner provider."""

import pytest

from src.providers.runner import LocalRunner, ModalRunner, TrainingResult, _parse_val_bpb


@pytest.mark.unit
class TestParseValBpb:
    def test_parses_valid_output(self):
        output = "some output\nval_bpb:          1.763539\nmore output"
        assert _parse_val_bpb(output) == 1.763539

    def test_returns_none_on_missing(self):
        assert _parse_val_bpb("no bpb here") is None

    def test_returns_none_on_empty(self):
        assert _parse_val_bpb("") is None

    def test_parses_different_values(self):
        assert _parse_val_bpb("val_bpb: 2.345678") == 2.345678


@pytest.mark.unit
class TestTrainingResult:
    def test_success_result(self):
        r = TrainingResult(val_bpb=1.5, output="ok", success=True)
        assert r.val_bpb == 1.5
        assert r.success is True

    def test_crash_result(self):
        r = TrainingResult(val_bpb=None, output="CRASH", success=False)
        assert r.val_bpb is None
        assert r.success is False

    def test_timeout_result(self):
        r = TrainingResult(val_bpb=None, output="TIMEOUT", success=False)
        assert "TIMEOUT" in r.output


@pytest.mark.unit
class TestLocalRunner:
    def test_init(self, tmp_path):
        runner = LocalRunner(project_root=tmp_path)
        assert runner.project_root == tmp_path
        assert runner.timeout == 900

    def test_custom_timeout(self, tmp_path):
        runner = LocalRunner(project_root=tmp_path, timeout=300)
        assert runner.timeout == 300

    def test_train_py_path(self, tmp_path):
        runner = LocalRunner(project_root=tmp_path)
        assert runner.train_py == tmp_path / "experiments" / "train.py"


@pytest.mark.unit
class TestModalRunner:
    def test_init(self):
        runner = ModalRunner()
        assert isinstance(runner, ModalRunner)

    def test_run_without_modal_app(self):
        """ModalRunner should handle missing modal_app gracefully."""
        runner = ModalRunner()
        # This will fail because modal_app.run_experiment isn't deployed,
        # but it shouldn't crash — it should return a TrainingResult with an error
        result = runner.run("print('hello')", "# prepare")
        assert isinstance(result, TrainingResult)
        assert result.success is False
        assert result.val_bpb is None
