# Post-Cutoff ML Architecture Breakthroughs

> The validation set. These breakthroughs were published AFTER December 2023.
> Agent outputs are compared against these to measure discovery capability.
> Last updated: 2026-03-16

## Attention Mechanism Innovations

| Innovation | arXiv Date | Venue | Key Contribution |
|-----------|-----------|-------|-----------------|
| **GLA (Gated Linear Attention)** | Dec 11, 2023 | ICML 2024 | Hardware-efficient linear attention with gating; competitive with Transformers at linear cost |
| **DeepSeek-V2 / MLA (Multi-Head Latent Attention)** | May 7, 2024 | — | Compresses KV cache into latent vector, drastically reducing KV cache memory |
| **FlashAttention-3** | Jul 11, 2024 | NeurIPS 2024 | Async + low-precision attention for H100; 1.5-2x over FA2 |
| **xLSTM** | May 7, 2024 | NeurIPS 2024 | Extended LSTM with exponential gating + matrix memory; competitive with Transformers |
| **RWKV Eagle/Finch (v5/v6)** | Apr 9, 2024 | COLM 2024 | Matrix-valued states + dynamic recurrence; closes gap with attention |

## State Space Model (SSM) Innovations

| Innovation | arXiv Date | Venue | Key Contribution |
|-----------|-----------|-------|-----------------|
| **Mamba-2 / SSD** | May 31, 2024 | ICML 2024 | Proves SSM = structured attention (SSD theorem); 2-8x faster than Mamba-1 |
| **Falcon Mamba 7B** | Aug 2024 | — | First 7B attention-free model to beat same-size Transformers |

## Mixture of Experts (MoE) Innovations

| Innovation | arXiv Date | Key Contribution |
|-----------|-----------|-----------------|
| **DeepSeekMoE** | Jan 11, 2024 | Fine-grained expert specialization |
| **Mixtral 8x7B** | Jan 2024 (paper) | Popularized sparse MoE with 8 experts, 2 active |
| **DeepSeek-V3** | Dec 27, 2024 | Combined MLA + MoE at massive scale; shocked industry with cost efficiency |

## Reasoning / Training Paradigm

| Innovation | Date | Key Contribution |
|-----------|------|-----------------|
| **OpenAI o1** | Sep 2024 | Chain-of-thought reasoning via RL |
| **DeepSeek-R1** | Jan 20, 2025 | Open-source RL reasoning; GRPO training |

## How to Use This Document

When evaluating agent outputs, compare against these breakthroughs:

1. **Conceptual similarity**: Does the agent's proposal address the same problem space? (e.g., KV cache compression → related to MLA)
2. **Mechanism overlap**: Does the agent propose a similar mechanism? (e.g., low-rank projection of keys/values)
3. **Empirical validation**: Does the agent's implementation show improvement on the same metrics?

### Scoring Framework

- **Direct hit**: Agent proposes essentially the same mechanism as a post-cutoff breakthrough
- **Adjacent**: Agent addresses the same problem with a different but valid approach
- **Directionally correct**: Agent identifies the right problem space but proposes a less effective solution
- **Novel**: Agent proposes something not in the post-cutoff literature (may be genuinely new, or may be impractical)
- **Miss**: Agent's proposals don't relate to any post-cutoff advances

## Pre-Cutoff State of the Art (What Agents WILL Know)

As of December 2023, agents have knowledge of:
- Standard multi-head attention (Vaswani et al., 2017)
- FlashAttention 1 & 2 (Dao et al., 2022-2023)
- Grouped Query Attention (GQA)
- Multi-Query Attention (MQA)
- Mamba-1 / S4 / SSMs (Gu et al., 2023) — borderline, Dec 1 2023
- Rotary Position Embeddings (RoPE)
- ALiBi positional encoding
- Sliding window attention
- Sparse attention patterns (Longformer, BigBird)
- LoRA and parameter-efficient fine-tuning
- Basic MoE concepts (Switch Transformer, 2021)
- Chinchilla scaling laws
