# Product Vision

## What Is Rediscover?

Rediscover tests a fundamental question: **Can AI agents independently discover ML/AI breakthroughs if given only the prior research?**

It extends Karpathy's [autoresearch](https://github.com/karpathy/autoresearch) concept from "optimize training code" to "do actual ML research" — agents read existing literature, understand the state of the art, and experiment with novel improvements across any domain in ML/AI.

**This is domain-agnostic.** Attention mechanisms are the first test case, but the system is designed to work for any ML/AI research problem where you can:
1. Define a clear metric to optimize
2. Give the agent code to modify and run
3. Evaluate results empirically

Examples of potential research domains:
- Attention mechanism efficiency and quality
- Training optimization (optimizers, learning rate schedules, curriculum)
- Model architecture (MoE routing, layer design, activation functions)
- Inference efficiency (quantization, pruning, distillation)
- Data efficiency (few-shot learning, data augmentation, synthetic data)
- Positional encoding schemes
- State space models vs. attention trade-offs

## Who Is This For?

- **ML researchers** exploring whether AI can accelerate scientific discovery
- **AI infrastructure builders** interested in autonomous agent systems
- **Anyone curious** about whether LLMs can think creatively about architecture

## The Core Insight: Retrodiction as Validation

The hardest problem in "AI doing research" is evaluation. How do you know if an AI's research output is actually good?

**Our answer:** Use models with known knowledge cutoffs, feed them only pre-cutoff papers, let them innovate, then compare against real post-cutoff breakthroughs. This creates a ground truth — like a holdout set, but for *ideas* instead of data.

If the agent independently converges on something like Multi-Head Latent Attention (published May 2024) using only pre-December-2023 research, that's a real, measurable result.

## North Star Metric

**Discovery Rate:** What percentage of post-cutoff breakthroughs does the agent independently approach (measured by both conceptual similarity and empirical validation on small models)?

## Core Beliefs

1. **Small-scale experiments are scientifically valid.** FlashAttention, RoPE, and Chinchilla scaling laws were all validated at 60-125M parameters before being applied at scale. Properties like memory efficiency and algorithmic complexity transfer across scales.

2. **No frameworks, just loops.** The AI Scientist v2 tried to do everything and 42% of experiments failed from coding errors. A simple Python loop with litellm handles 80% of agent use cases. Complexity is the enemy.

3. **The council is a pipeline, not a roundtable.** Agents have different roles (propose, critique, implement) — not everyone answering the same question. This is more efficient and cheaper than full deliberation.

4. **Git is the experiment log.** Every experiment is a commit. Every failed idea is a reverted commit. `git log` shows the complete research history. No database needed.

5. **Cost discipline is a feature.** Smart model routing (cheap models for scanning, expensive for reasoning) can cut costs 94% while improving success rate (BudgetMLAgent, 2024). Target: $15-30/day for the full system.

6. **Start simple, iterate.** The first version should be a working loop that can run overnight. Add sophistication only when the simple version reveals specific bottlenecks.
