"""Tests for the research loop orchestration (mocked — no real training or API calls)."""

import pytest

from src.app.loop import append_results_tsv, parse_val_bpb


@pytest.mark.unit
class TestParseValBpb:
    def test_parse_from_training_output(self):
        output = """step 00088 (100.0%) | loss: 4.973049
---
val_bpb:          1.763539
training_seconds: 303.8
total_seconds:    802.0"""
        assert parse_val_bpb(output) == pytest.approx(1.763539)

    def test_parse_none_on_crash(self):
        output = "FAIL\nTraceback: some error"
        assert parse_val_bpb(output) is None

    def test_parse_none_on_empty(self):
        assert parse_val_bpb("") is None

    def test_parse_with_extra_whitespace(self):
        output = "val_bpb:   1.500000  \n"
        assert parse_val_bpb(output) == pytest.approx(1.5)


@pytest.mark.unit
class TestAppendResultsTsv:
    def test_append_keep(self, tmp_path):
        tsv = tmp_path / "results.tsv"
        tsv.write_text("commit\tval_bpb\tmemory_gb\tstatus\tdescription\n")

        # Monkey-patch the module-level path for testing
        import src.app.loop as loop_mod
        original = loop_mod.RESULTS_TSV
        loop_mod.RESULTS_TSV = tsv
        try:
            append_results_tsv("abc123", 1.5, "keep", "test experiment")
            content = tsv.read_text()
            assert "abc123" in content
            assert "1.500000" in content
            assert "keep" in content
        finally:
            loop_mod.RESULTS_TSV = original

    def test_append_crash(self, tmp_path):
        tsv = tmp_path / "results.tsv"
        tsv.write_text("commit\tval_bpb\tmemory_gb\tstatus\tdescription\n")

        import src.app.loop as loop_mod
        original = loop_mod.RESULTS_TSV
        loop_mod.RESULTS_TSV = tsv
        try:
            append_results_tsv("reverted", None, "crash", "broken code")
            content = tsv.read_text()
            assert "N/A" in content
            assert "crash" in content
        finally:
            loop_mod.RESULTS_TSV = original
