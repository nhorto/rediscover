"""Tests for code validation: quick_validate_code and error feedback flow."""

import pytest

from src.app.loop import quick_validate_code, validate_train_py
from src.domains.council.config import extract_code_structure


@pytest.mark.unit
class TestQuickValidateCode:
    def test_valid_code_passes(self):
        code = "import torch\nprint('hello')\n"
        is_valid, reason = quick_validate_code(code)
        assert is_valid is True

    def test_import_error_caught(self):
        code = "import nonexistent_module_xyz\nprint('hello')\n"
        is_valid, reason = quick_validate_code(code)
        assert is_valid is False
        assert "error" in reason.lower() or "Error" in reason

    def test_undefined_name_caught(self):
        code = "x = undefined_variable_xyz + 1\n"
        is_valid, reason = quick_validate_code(code)
        assert is_valid is False

    def test_syntax_error_caught_by_safety_first(self):
        code = "def broken(\n  no closing"
        is_safe, reason = validate_train_py(code)
        assert is_safe is False
        assert "Syntax error" in reason


@pytest.mark.unit
class TestExtractCodeStructure:
    SAMPLE_CODE = """import torch
import torch.nn as nn
from prepare import MAX_SEQ_LEN

DEPTH = 4
DEVICE_BATCH_SIZE = 16
HEAD_DIM = 128

class GPTConfig:
    pass

class CausalSelfAttention(nn.Module):
    def __init__(self):
        pass

class GPT(nn.Module):
    def __init__(self):
        pass

def build_model_config(depth):
    pass

def get_lr_multiplier(progress):
    pass
"""

    def test_extracts_imports(self):
        result = extract_code_structure(self.SAMPLE_CODE)
        assert "import torch" in result
        assert "import torch.nn" in result

    def test_extracts_classes(self):
        result = extract_code_structure(self.SAMPLE_CODE)
        assert "GPTConfig" in result
        assert "CausalSelfAttention" in result
        assert "GPT" in result

    def test_extracts_functions(self):
        result = extract_code_structure(self.SAMPLE_CODE)
        assert "build_model_config" in result
        assert "get_lr_multiplier" in result

    def test_extracts_constants(self):
        result = extract_code_structure(self.SAMPLE_CODE)
        assert "DEPTH = 4" in result
        assert "HEAD_DIM = 128" in result

    def test_includes_line_count(self):
        result = extract_code_structure(self.SAMPLE_CODE)
        assert "File length:" in result


@pytest.mark.unit
class TestErrorFeedbackIntegration:
    """Test that the fix_code method exists and has the right interface."""

    def test_council_has_fix_code_method(self):
        from src.domains.council.service import CouncilService
        assert hasattr(CouncilService, "fix_code")

    def test_fix_prompts_exist(self):
        from src.domains.council.config import IMPLEMENT_FIX_PROMPT, IMPLEMENT_FIX_SYSTEM
        assert "Error" in IMPLEMENT_FIX_PROMPT or "error" in IMPLEMENT_FIX_PROMPT
        assert "fix" in IMPLEMENT_FIX_SYSTEM.lower()
