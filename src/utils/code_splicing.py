"""Code splicing: extract and replace the modifiable zone in train.py.

The modifiable zone contains GPTConfig, helper functions, and CausalSelfAttention.
Everything else (imports, MLP, Block, GPT, optimizer, training loop) is frozen.
"""

# Markers for zone boundaries
ZONE_START_MARKER = "@dataclass"  # GPTConfig starts with @dataclass
ZONE_END_MARKER = "class MLP"  # MLP class is the first frozen class after the zone


def extract_modifiable_zone(train_py: str) -> tuple[str, str, str]:
    """Split train.py into (before_zone, zone, after_zone).

    The zone contains: GPTConfig, norm(), has_ve(), apply_rotary_emb(), CausalSelfAttention.
    Before: imports, env setup.
    After: MLP, Block, GPT, optimizer, training loop, evaluation.

    Returns three strings that concatenate to the original file.
    """
    lines = train_py.split("\n")

    zone_start = None
    zone_end = None

    for i, line in enumerate(lines):
        if zone_start is None and line.strip().startswith(ZONE_START_MARKER):
            zone_start = i
        elif zone_start is not None and zone_end is None and line.strip().startswith(ZONE_END_MARKER):
            zone_end = i
            break

    if zone_start is None:
        raise ValueError(f"Could not find zone start marker: {ZONE_START_MARKER}")
    if zone_end is None:
        raise ValueError(f"Could not find zone end marker: {ZONE_END_MARKER}")

    before = "\n".join(lines[:zone_start])
    zone = "\n".join(lines[zone_start:zone_end])
    after = "\n".join(lines[zone_end:])

    # Ensure before ends with newlines for clean concatenation
    if not before.endswith("\n"):
        before += "\n"

    # Ensure zone ends with newlines
    if not zone.endswith("\n"):
        zone += "\n\n"

    return before, zone, after


def replace_modifiable_zone(train_py: str, new_zone: str) -> str:
    """Replace the modifiable zone in train.py with new code.

    The new_zone should contain GPTConfig, helper functions, and CausalSelfAttention.
    """
    before, _, after = extract_modifiable_zone(train_py)

    # Ensure clean boundaries
    new_zone = new_zone.strip() + "\n\n"

    return before + new_zone + after


def get_frozen_context(train_py: str) -> str:
    """Get a summary of the frozen parts of train.py for the model's reference.

    This tells the model what interfaces it must be compatible with.
    """
    _, _, after = extract_modifiable_zone(train_py)

    # Extract key interfaces from the frozen code
    context_parts = []

    context_parts.append("## Frozen Code Interfaces (your zone must be compatible with these)")
    context_parts.append("")
    context_parts.append("### MLP class (frozen — do not redefine)")
    context_parts.append("class MLP(nn.Module):")
    context_parts.append("    def __init__(self, config): ...")
    context_parts.append("    def forward(self, x): ...")
    context_parts.append("")
    context_parts.append("### Block class (frozen — calls your CausalSelfAttention)")
    context_parts.append("class Block(nn.Module):")
    context_parts.append("    def __init__(self, config, layer_idx):")
    context_parts.append("        self.attn = CausalSelfAttention(config, layer_idx)")
    context_parts.append("        self.mlp = MLP(config)")
    context_parts.append("    def forward(self, x, ve, cos_sin, window_size):")
    context_parts.append("        x = x + self.attn(norm(x), ve, cos_sin, window_size)")
    context_parts.append("        x = x + self.mlp(norm(x))")
    context_parts.append("")
    context_parts.append("### GPT class (frozen — uses your GPTConfig and CausalSelfAttention)")
    context_parts.append("- Creates CausalSelfAttention via Block(config, layer_idx)")
    context_parts.append("- Calls block(x, ve, cos_sin, window_sizes[i]) in forward()")
    context_parts.append("- Uses config.n_head, config.n_kv_head, config.n_embd, config.sequence_len")
    context_parts.append("- Uses config.window_pattern, config.vocab_size, config.n_layer")
    context_parts.append("- Value embeds: self.value_embeds[str(i)](idx) → passed as ve to block")
    context_parts.append("- Rotary: self.cos[:, :T], self.sin[:, :T] → passed as cos_sin to block")

    return "\n".join(context_parts)
