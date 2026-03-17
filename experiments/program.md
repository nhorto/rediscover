# Rediscover Research Program

## Research Direction: Reduce Attention Memory Complexity

You are an autonomous ML research agent. Your goal is to reduce the memory cost of the attention mechanism in `train.py` while maintaining or improving validation performance (val_bpb).

**The specific problem:** Standard multi-head attention computes a full N×N attention matrix, requiring O(N²) memory in sequence length. For long sequences, this is the primary bottleneck. Find ways to reduce this cost without sacrificing model quality.

## Current Model

The baseline model is a 4-layer GPT with:
- 256-dim embeddings, 2 attention heads, 128-dim per head
- RoPE positional encoding
- Grouped Query Attention (n_kv_head configurable)
- Value embeddings with input-dependent gating (ResFormer pattern)
- RMSNorm, ReluSquared MLP activation
- MuonAdamW optimizer (Muon for matrices, AdamW for embeddings)
- Sliding window attention pattern (configurable)
- Baseline val_bpb: 1.763539

## What to Explore

Search the literature for existing approaches to reducing attention memory cost. The papers in the knowledge base cover work published up to December 2023. Read them, understand the trade-offs, and propose experiments based on what you learn.

**Your goal is to reduce the memory footprint of the attention mechanism while maintaining or improving val_bpb.**

Start by understanding the current attention implementation in train.py (the CausalSelfAttention class), then search the literature for ideas that could improve it. Let the papers guide your approach — don't invent from scratch when prior work exists.

### General directions to consider
- Can the attention computation be made cheaper without losing quality?
- Are all heads doing useful work, or could some be shared or removed?
- Is the full sequence length needed for every layer, or could some layers attend to less?
- Are there mathematical reformulations of attention that avoid the quadratic cost?
- Could the model get the same information from attention using fewer parameters?

### Important
- Focus your changes on the attention mechanism (CausalSelfAttention class and related code)
- Do not simply change hyperparameters like DEPTH, TOTAL_BATCH_SIZE, or learning rates to game the metric
- The goal is architectural innovation in attention, not hyperparameter tuning

## Constraints

- Only modify `train.py` — never touch `prepare.py` or its `evaluate_bpb`
- No new package imports beyond what's already installed
- Training budget is fixed at 5 minutes — you cannot change TIME_BUDGET
- The metric is `val_bpb` (validation bits per byte) — **lower is better**
- You may change: model architecture, attention mechanism, optimizer settings, hyperparameters, batch size, model dimensions
- Do NOT change the output format (the loop parses `val_bpb:` from stdout)

## Strategy Guidance

1. **Start with the simplest change that could work.** A hyperparameter tweak that improves val_bpb is more valuable than a complex architectural change that crashes.
2. **Read the results history.** If something similar was tried and failed, try a different direction.
3. **One change at a time.** Don't combine multiple ideas in one experiment — you won't know which one helped or hurt.
4. **Be specific.** "Try linear attention" is too vague. "Replace softmax(QK^T)V with (phi(Q))(phi(K)^T V) using ELU+1 feature map" is actionable.
5. **Think about the 5-minute budget.** Your changes must train meaningfully in 5 minutes. A model that's 3x slower per step gets fewer training steps, which may hurt val_bpb even if the architecture is theoretically better.

## Recording Results

Results are recorded automatically by the research loop. Each experiment logs:
- Hypothesis, approach, papers consulted
- Critique from the independent reviewer
- val_bpb result and keep/discard/crash status
- Full deliberation chain in experiment_log.md
