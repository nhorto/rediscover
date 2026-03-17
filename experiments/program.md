# Rediscover Research Program

## Research Direction: Attention Mechanism Improvements

You are an autonomous ML research agent. Your goal is to improve the transformer architecture in `train.py`, focusing on attention mechanism efficiency and quality.

## Setup

1. Read `prepare.py` to understand the fixed evaluation infrastructure (DO NOT MODIFY prepare.py)
2. Read `train.py` to understand the current model architecture
3. Read `results.tsv` to see what has been tried and what worked

## The Loop

1. **Propose** a hypothesis for improving the model (attention, architecture, optimization)
2. **Edit** `train.py` to implement your idea
3. **Run** `uv run experiments/train.py` — training runs for exactly 5 minutes
4. **Evaluate** the `val_bpb` output (lower is better)
5. If improved → keep the change, record "keep" in results.tsv
6. If worse → revert the change, record "discard" in results.tsv
7. **Repeat** — never stop until interrupted

## Constraints

- Only modify `train.py` — never touch `prepare.py` or its `evaluate_bpb`
- No new package imports beyond what's already installed
- Training budget is fixed at 5 minutes — you cannot change TIME_BUDGET
- The metric is `val_bpb` (validation bits per byte) — lower is better
- You may change: model architecture, attention mechanism, optimizer, hyperparameters, batch size, model dimensions

## What to Try

Focus areas for attention improvements:
- KV cache efficiency (compression, sharing, low-rank projections)
- Sub-quadratic attention (linear attention, sparse patterns)
- Positional encoding alternatives (different RoPE configurations, ALiBi)
- Multi-head vs grouped-query vs multi-query attention trade-offs
- Attention pattern variations (local + global, dilated, etc.)
- Value embedding strategies
- Activation functions in MLP

## Recording Results

Append to `results.tsv` (tab-separated):
```
commit	val_bpb	memory_gb	status	description
```

Status is one of: `keep`, `discard`, `crash`
