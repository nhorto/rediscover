# Phase 4 Retrodiction Validation Report

> Validation of agent proposals against post-cutoff breakthroughs.
> Scoring: SPECTER embeddings (70%) + keyword overlap (30%)
> Date: 2026-03-18

## Summary

| Level | Count | Percentage |
|-------|-------|------------|
| Direct hits | 0 | 0% |
| Adjacent | 9 | 90% |
| Directionally correct | 1 | 10% |
| Novel | 0 | 0% |
| Misses | 0 | 0% |

**Hit rate (direct + adjacent):** 90%
**Near-miss rate (directional):** 10%
**Novel ideas rate:** 0%

## Interpretation

The high "adjacent" rate (90%) is misleading — it reflects that all proposals are about efficient attention, which naturally has high embedding similarity with post-cutoff attention breakthroughs. This is a limitation of the scoring methodology at small scale: all proposals and all breakthroughs live in the same semantic neighborhood ("efficient attention mechanisms"), so embedding similarity is high by default.

**No proposals achieved "direct hit" status**, meaning none independently converged on the specific mechanisms of MLA, FlashAttention-3, Mamba-2, etc. This is expected given:
1. Only 10 unique proposals were generated (need 100+ for meaningful signal)
2. Most proposals replicated pre-cutoff methods (Performer, Linformer, Reformer) rather than inventing new ones
3. The prompt rewrites to encourage originality happened AFTER these experiments

## Per-Breakthrough Scores

### Multi-Head Latent Attention (MLA)
Best proposal match: Linformer-style linear attention (score=0.618)
- The Linformer proposal's use of low-rank key/value projection is conceptually adjacent to MLA's latent KV compression
- However, the agent proposed projecting sequence length (Linformer), not embedding dimension (MLA) — a critical distinction

### RWKV Eagle/Finch (v5/v6)
Most frequently matched breakthrough (best match for 7/10 proposals)
- This is because RWKV's linear attention formulation is semantically close to many of the proposals (ELU+1, log-normal, Performer)
- Score range: 0.540-0.631

### Gated Linear Attention (GLA)
Score range: 0.100-0.574
- No proposals explicitly used gating in their linear attention formulation
- GLA's key insight (hardware-efficient gating) was not explored

### Mamba-2 / SSD Theorem
Score range: 0.517-0.576
- The SSM-attention equivalence theorem was not approached by any proposal
- No proposals explored state-space formulations

### FlashAttention-3
Score range: 0.000-0.529
- Lowest similarity overall — expected since FA3 is an implementation optimization (async + low-precision), not an algorithmic change
- No proposals addressed hardware-level attention optimization

### xLSTM
Score range: 0.487-0.572
- No proposals explored recurrent alternatives to attention
- Exponential gating was not proposed

### DeepSeek-V3
Score range: 0.091-0.521
- Lowest keyword overlap — proposals didn't touch MoE concepts
- Some embedding similarity from shared attention efficiency language

## Per-Proposal Detail

| # | Result | Best Match | Score | Level |
|---|--------|------------|-------|-------|
| 1 | CRASH | RWKV Eagle/Finch | 0.596 | adjacent |
| 2 | CRASH | RWKV Eagle/Finch | 0.629 | adjacent |
| 3 | CRASH | RWKV Eagle/Finch | 0.611 | adjacent |
| 4 | CRASH | RWKV Eagle/Finch | 0.631 | adjacent |
| 5 | KEEP (1.675790) | RWKV Eagle/Finch | 0.604 | adjacent |
| 6 | CRASH | MLA | 0.618 | adjacent |
| 7 | CRASH | RWKV Eagle/Finch | 0.614 | adjacent |
| 8 | KEEP (1.675352) | RWKV Eagle/Finch | 0.588 | adjacent |
| 9 | DISCARD (1.676400) | RWKV Eagle/Finch | 0.540 | directional |
| 10 | DISCARD (1.680573) | MLA | 0.597 | adjacent |

## Methodology Notes

- Embeddings: SPECTER (allenai-specter) — 768-dim academic text embeddings
- Combined score = 0.7 * embedding_similarity + 0.3 * keyword_overlap
- Thresholds: direct_hit >= 0.75, adjacent >= 0.55, directional >= 0.35
- The "adjacent" threshold may be too low for this domain — proposals about attention naturally score 0.5+ against attention breakthroughs regardless of conceptual novelty
- Consider raising thresholds or using more discriminative embeddings for Phase 5
