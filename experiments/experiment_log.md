# Experiment Log

> Narrative record of all experiments. Each entry documents what was tried, why, and what was learned.

---

## Experiment 1 — 2026-03-17 19:01:08
**Hypothesis:** Replacing the standard softmax self-attention with linear log-normal attention (from [2311.13541v4]) will reduce the O(N²) memory cost to roughly O(N·head_dim) while preserving or slightly improving val_bpb.
**Approach:** In CausalSelfAttention, remove the QK^T softmax computation and instead compute attention as (φ(Q))·(φ(K))^T·V, where φ includes a log-normal feature mapping (following Sec. 3 of [2311.13541v4]). Keep the same embedding/head dimensions but replace the attention forward pass with this linearized kernel approach.
**Papers consulted:** Which Transformer to Favor: A Comparative Analysis of Effici, Having Second Thoughts? Let's hear it, Reproduction Report on "Learn to Pay Attention"
**Critique:** The proposal to replace standard softmax self-attention with a log-normal attention mechanism is theoretically intriguing and could offer substantial memory savings. However, the practical challenges,
**Plan:** This experiment replaces the standard softmax in causal self-attention with a log-normal feature mapping for approximate linear attention, tested first on a smaller ablation to validate approximation quality and memory savings before full-scale integration.
**Result:** val_bpb=CRASH (crash)
**Cost this cycle:** $2.5412
**Cumulative cost:** $2.5412
---
