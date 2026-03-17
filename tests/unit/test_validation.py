"""Tests for code validation and sandboxing in the research loop."""

import pytest

from src.app.loop import validate_diff_is_attention_related, validate_train_py


@pytest.mark.unit
class TestValidateTrainPy:
    def test_valid_python_passes(self):
        code = "import torch\nprint('hello')\n"
        is_safe, reason = validate_train_py(code)
        assert is_safe is True

    def test_syntax_error_rejected(self):
        code = "def broken(\n  no closing"
        is_safe, reason = validate_train_py(code)
        assert is_safe is False
        assert "Syntax error" in reason

    def test_os_system_rejected(self):
        code = "import os\nos.system('rm -rf /')\n"
        is_safe, reason = validate_train_py(code)
        assert is_safe is False
        assert "Dangerous" in reason

    def test_subprocess_rejected(self):
        code = "import subprocess\nsubprocess.run(['ls'])\n"
        is_safe, reason = validate_train_py(code)
        assert is_safe is False
        assert "Dangerous" in reason

    def test_open_prepare_py_rejected(self):
        code = "open('../prepare.py', 'w').write('hacked')\n"
        is_safe, reason = validate_train_py(code)
        assert is_safe is False

    def test_exec_rejected(self):
        code = "exec('import os')\n"
        is_safe, reason = validate_train_py(code)
        assert is_safe is False

    def test_eval_rejected(self):
        code = "x = eval('1+1')\n"
        is_safe, reason = validate_train_py(code)
        assert is_safe is False

    def test_shutil_rejected(self):
        code = "import shutil\nshutil.rmtree('/tmp')\n"
        is_safe, reason = validate_train_py(code)
        assert is_safe is False

    def test_dunder_import_rejected(self):
        code = "__import__('os').system('ls')\n"
        is_safe, reason = validate_train_py(code)
        assert is_safe is False

    def test_normal_torch_code_passes(self):
        code = """
import torch
import torch.nn as nn
import torch.nn.functional as F

class Attention(nn.Module):
    def __init__(self):
        super().__init__()
        self.q = nn.Linear(256, 256)

    def forward(self, x):
        return F.scaled_dot_product_attention(x, x, x)
"""
        is_safe, reason = validate_train_py(code)
        assert is_safe is True

    def test_empty_code_passes_safety_fails_syntax(self):
        # Empty string is valid Python (compiles fine)
        is_safe, reason = validate_train_py("")
        assert is_safe is True


@pytest.mark.unit
class TestValidateDiffAttentionRelated:
    BASELINE = """
# Hyperparams
DEPTH = 4
DEVICE_BATCH_SIZE = 16

class CausalSelfAttention(nn.Module):
    def __init__(self):
        self.c_q = nn.Linear(256, 256)
        self.c_k = nn.Linear(256, 256)
        self.n_head = 2
"""

    def test_attention_change_passes(self):
        new_code = self.BASELINE.replace("self.n_head = 2", "self.n_head = 4")
        is_valid, reason = validate_diff_is_attention_related(self.BASELINE, new_code)
        assert is_valid is True

    def test_hyperparameter_only_change_rejected(self):
        new_code = self.BASELINE.replace("DEPTH = 4", "DEPTH = 8")
        is_valid, reason = validate_diff_is_attention_related(self.BASELINE, new_code)
        assert is_valid is False
        assert "hyperparameters" in reason.lower()

    def test_batch_size_only_change_rejected(self):
        new_code = self.BASELINE.replace("DEVICE_BATCH_SIZE = 16", "DEVICE_BATCH_SIZE = 32")
        is_valid, reason = validate_diff_is_attention_related(self.BASELINE, new_code)
        assert is_valid is False

    def test_attention_plus_hyperparameter_passes(self):
        new_code = self.BASELINE.replace("self.n_head = 2", "self.n_head = 4").replace("DEPTH = 4", "DEPTH = 4  # adjusted for new heads")
        is_valid, reason = validate_diff_is_attention_related(self.BASELINE, new_code)
        assert is_valid is True

    def test_identical_code_passes(self):
        """No changes at all should pass (nothing wrong happened)."""
        is_valid, reason = validate_diff_is_attention_related(self.BASELINE, self.BASELINE)
        assert is_valid is True

    def test_adding_new_attention_class_passes(self):
        new_code = self.BASELINE + "\nclass LinearAttention(nn.Module):\n    pass\n"
        is_valid, reason = validate_diff_is_attention_related(self.BASELINE, new_code)
        # New code that doesn't match any markers but also isn't hyperparam-only
        assert is_valid is True


@pytest.mark.unit
class TestCouncilParsingAdversarial:
    """Test council parsing with malformed LLM responses."""

    def test_extract_field_no_format(self):
        from src.domains.council.parsing import extract_field

        # LLM returns freeform text with no structured fields
        text = "I think we should try reducing the number of heads because it would save memory."
        result = extract_field(text, "HYPOTHESIS")
        # Should return the full text as fallback
        assert "reducing the number of heads" in result

    def test_extract_list_empty_section(self):
        from src.domains.council.parsing import extract_list

        text = "CONCERNS:\n\nOVERALL: Looks fine"
        result = extract_list(text, "CONCERNS")
        assert result == []

    def test_parse_search_queries_garbage(self):
        from src.domains.council.parsing import parse_search_queries

        text = "Here are some random thoughts about attention that don't follow any format whatsoever."
        queries = parse_search_queries(text)
        # Should fall back to line-by-line parsing
        assert len(queries) >= 1

    def test_clean_code_response_nested_fences(self):
        from src.domains.council.parsing import clean_code_response

        text = "```python\n```python\nprint('hello')\n```\n```"
        result = clean_code_response(text)
        assert "print" in result

    def test_extract_field_multiline_value(self):
        from src.domains.council.parsing import extract_field

        text = "HYPOTHESIS: This is a long hypothesis\nthat spans multiple lines\nand keeps going\nAPPROACH: Do the thing"
        result = extract_field(text, "HYPOTHESIS")
        assert "long hypothesis" in result
        assert "APPROACH" not in result
