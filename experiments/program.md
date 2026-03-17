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

You should search the literature and explore approaches such as:

### Memory Reduction
- Low-rank projections of keys and/or values before computing attention
- Sharing keys/values across heads or layers
- Compressing the KV representation into a smaller latent space
- Reducing the number of KV heads (multi-query attention, grouped-query attention trade-offs)

### Sub-Quadratic Attention
- Linear attention via kernel approximations (random features, ELU-based)
- Sparse attention patterns (local windows + global tokens, strided patterns)
- Chunk-based attention (compute attention within fixed-size blocks)
- Combining local attention with some form of global context aggregation

### Efficiency Without Architecture Change
- Different head dimensions (smaller heads = less KV memory per head)
- Aspect ratio tuning (wider vs deeper at fixed parameter budget)
- Optimizing the sliding window pattern (which layers get full vs windowed attention)
- Value embedding frequency (every layer vs alternating vs fewer layers)

### Things That Might Surprise You
- Sometimes removing components improves efficiency more than adding clever ones
- The interaction between attention pattern and the optimizer matters
- Small models behave differently than large ones — what works at 11.5M params may not scale, and vice versa

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
