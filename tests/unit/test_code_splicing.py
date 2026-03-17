"""Tests for code splicing: extract and replace the modifiable zone."""

import pytest

from src.utils.code_splicing import extract_modifiable_zone, get_frozen_context, replace_modifiable_zone

# Minimal train.py structure for testing
SAMPLE_TRAIN_PY = '''"""Training script."""

import torch
import torch.nn as nn

from prepare import MAX_SEQ_LEN

@dataclass
class GPTConfig:
    n_head: int = 2
    n_embd: int = 256

def norm(x):
    return F.rms_norm(x, (x.size(-1),))

def has_ve(layer_idx, n_layer):
    return layer_idx % 2 == 0

def apply_rotary_emb(x, cos, sin):
    return x

class CausalSelfAttention(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.c_q = nn.Linear(config.n_embd, config.n_embd)

    def forward(self, x, ve, cos_sin, window_size):
        return self.c_q(x)

class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()

    def forward(self, x):
        return x

class Block(nn.Module):
    pass

class GPT(nn.Module):
    pass

# Training loop
while True:
    pass
'''


@pytest.mark.unit
class TestExtractModifiableZone:
    def test_extracts_zone(self):
        before, zone, after = extract_modifiable_zone(SAMPLE_TRAIN_PY)
        assert "import torch" in before
        assert "GPTConfig" in zone
        assert "CausalSelfAttention" in zone
        assert "norm" in zone
        assert "has_ve" in zone
        assert "apply_rotary_emb" in zone
        assert "class MLP" in after
        assert "class Block" in after
        assert "class GPT" in after

    def test_zone_does_not_contain_frozen_code(self):
        _, zone, _ = extract_modifiable_zone(SAMPLE_TRAIN_PY)
        assert "import torch" not in zone
        assert "class MLP" not in zone
        assert "class Block" not in zone
        assert "Training loop" not in zone

    def test_concatenation_preserves_original(self):
        before, zone, after = extract_modifiable_zone(SAMPLE_TRAIN_PY)
        reconstructed = before + zone + after
        # Should contain all key elements from the original
        assert "import torch" in reconstructed
        assert "GPTConfig" in reconstructed
        assert "CausalSelfAttention" in reconstructed
        assert "class MLP" in reconstructed
        assert "Training loop" in reconstructed

    def test_raises_on_missing_start_marker(self):
        bad_code = "import torch\nclass MLP:\n    pass\n"
        with pytest.raises(ValueError, match="zone start"):
            extract_modifiable_zone(bad_code)

    def test_raises_on_missing_end_marker(self):
        bad_code = "@dataclass\nclass GPTConfig:\n    pass\n# no MLP here\n"
        with pytest.raises(ValueError, match="zone end"):
            extract_modifiable_zone(bad_code)


@pytest.mark.unit
class TestReplaceModifiableZone:
    def test_replaces_zone(self):
        new_zone = """@dataclass
class GPTConfig:
    n_head: int = 4
    n_embd: int = 512

def norm(x):
    return x

class CausalSelfAttention(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()

    def forward(self, x, ve, cos_sin, window_size):
        return x * 2
"""
        result = replace_modifiable_zone(SAMPLE_TRAIN_PY, new_zone)
        assert "n_head: int = 4" in result  # new config
        assert "n_head: int = 2" not in result  # old config gone
        assert "return x * 2" in result  # new attention
        assert "class MLP" in result  # frozen code preserved
        assert "import torch" in result  # imports preserved

    def test_preserves_frozen_code(self):
        _, zone, _ = extract_modifiable_zone(SAMPLE_TRAIN_PY)
        result = replace_modifiable_zone(SAMPLE_TRAIN_PY, zone)
        assert "class MLP" in result
        assert "class Block" in result
        assert "Training loop" in result


@pytest.mark.unit
class TestGetFrozenContext:
    def test_includes_key_interfaces(self):
        context = get_frozen_context(SAMPLE_TRAIN_PY)
        assert "MLP" in context
        assert "Block" in context
        assert "GPT" in context
        assert "CausalSelfAttention" in context
        assert "forward" in context
