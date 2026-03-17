# Hardware Constraints & Feasibility

> Analysis of what's feasible on the development machine.
> Last updated: 2026-03-16

## Development Machine

- **Chip:** Apple M4
- **RAM:** 32GB unified memory
- **GPU cores:** 10-core GPU
- **Metal:** Metal 4
- **OS:** macOS (Darwin 25.4.0)

## What Can We Train?

| Model Size | Memory Required | Feasibility | Training Time (WikiText-2) |
|------------|----------------|-------------|---------------------------|
| 1M-10M params | ~1-2 GB | Easy | Minutes per epoch |
| 10M-60M params | ~4-8 GB | **Sweet spot** | Minutes to hours |
| 60M-125M params | ~8-16 GB | Feasible | Hours |
| 125M-350M params | ~12-20 GB | Possible, tight | Many hours |
| 500M+ params | 20+ GB | Not practical for iteration | Days |

**Recommended operating range: 10-60M parameters.** This allows rapid iteration (dozens of experiments per day) while being large enough to show meaningful architectural differences.

## Why Small Scale Is Scientifically Valid

Important papers validated at small scale before scaling:

| Paper | Validation Scale | What It Showed |
|-------|-----------------|----------------|
| FlashAttention | GPT-2 small (124M) | IO-awareness insight demonstrated via profiling |
| RoPE | 125M params | Faster convergence vs learned positional embeddings |
| Chinchilla scaling laws | 70M-16B sweep | Small end of sweep was the critical data |
| Mechanistic interpretability | GPT-2 small (117M) | Entire field runs at this scale |
| NanoGPT speedrun | 124M params | All algorithmic improvements validated small first |

**Key insight:** Properties like memory efficiency, algorithmic complexity, and convergence behavior transfer across scales. A 20% memory improvement at 30M params will be a ~20% improvement at 30B params.

## Framework Choice

| Framework | Use For | Speed |
|-----------|---------|-------|
| **PyTorch MPS** | Training experiments | ~3,000-5,000 tokens/sec for GPT-2 scale |
| **MLX** | Inference-heavy analysis (interpretability) | 10-30x faster than MPS for generation |

**Recommendation:** PyTorch MPS for training (all research code is PyTorch). MLX for inference analysis.

## Datasets for Small-Scale Research

| Dataset | Size | Good For | Training Time (30M model) |
|---------|------|----------|--------------------------|
| **TinyStories** | ~470MB | 1M-10M params, coherent text at tiny scale | Minutes |
| **WikiText-2** | 2M tokens | 10M-100M params, standard benchmark | 30-60 min |
| **WikiText-103** | 103M tokens | 50M-500M params, more demanding | Hours |
| **Penn Treebank** | 1M tokens | Rapid iteration, fast ablations | Minutes |

**Primary benchmark: WikiText-2.** Standard, well-understood, fast enough for rapid iteration.

## Cloud Burst Options

For occasional validation at larger scale:

| Provider | GPU | Price/hr | Notes |
|----------|-----|----------|-------|
| **Vast.ai** | RTX 4090 | ~$0.34/hr | Cheapest option |
| **RunPod Community** | RTX 4090 | ~$0.34/hr | Better reliability |
| **Kaggle Free** | P100/T4 | Free | 30 hrs/week |
| **Lambda Labs** | H100 | ~$2.99/hr | For serious validation |

**Target cost per validation run:** $1-5 (RTX 4090 for 2-4 hours).

## Research Directions Feasible at This Scale

| Direction | Why It Works Small | Metric |
|-----------|-------------------|--------|
| Attention mechanism variants | Complexity is scale-independent | Perplexity, memory, throughput |
| Positional encoding | Directional gains hold small | Validation perplexity |
| Optimizer research | Convergence properties visible at any scale | Loss curves |
| Efficient attention profiling | No training needed | FLOPs, memory, wall-clock |
| Scaling law verification | Specifically requires small models | Power-law fit |

## Non-Training Experiments

Many experiments require no training at all:
- Profile attention implementations at varying sequence lengths (512 → 8192)
- Compare algorithmic complexity empirically
- Analyze numerical stability, gradient flow, attention entropy
- Run mechanistic interpretability on downloaded GPT-2
- Test memory/FLOPS characteristics of new attention kernels
