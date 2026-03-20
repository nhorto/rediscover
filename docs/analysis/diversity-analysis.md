# Diversity Analysis: Phase 4 Cloud Experiments

**Date:** 2026-03-20
**Experiments analyzed:** 71 total (38 pre-cloud + 33 cloud)
**Best val_bpb:** 1.157934 (cloud experiment 15)

## Executive Summary

Your hypothesis is correct. There is very low diversity in both the proposals and the papers cited. The system has converged on a narrow band of ideas — all variations of "dynamically gate/weight attention based on token relevance" — and is drawing from a tiny, mostly irrelevant paper pool. This is the single biggest bottleneck and explains why improvements have plateaued since experiment 15.

---

## Paper Diversity: Very Low

### The Numbers
- **~25 unique papers cited** across 71 experiments (some entries are fragments of the same paper)
- The top 6 papers account for **~75% of all citations**
- Many cited papers are not even about attention mechanisms

### Most Cited Papers (with citation counts)

| Citations | Paper | Relevant? |
|-----------|-------|-----------|
| 36 | "Reproduction Report on 'Learn to Pay Attention'" | Marginal — it's a reproduction report, not a novel method |
| 24 | "Faster Causal Attention Over Large Sequences Through Sparse..." | Yes — directly relevant |
| 21 | "One Pass Streaming Algorithm for Super Long Token Attention" | Yes — relevant |
| 19 | "The Curse of Dense Low-Dimensional Information Retrieval..." | No — about information retrieval, not attention |
| 17 | "Gated recurrent neural networks discover attention" | Marginal — about RNNs, not transformers |
| 16 | "Which Transformer to Favor: A Comparative Analysis..." | Yes — comparative study |

### Problems
1. **The #1 cited paper is a reproduction report** — not a source of novel ideas. It gets cited in 50% of experiments.
2. **Several papers are completely off-topic:** "Settling the Reward Hypothesis" (RL theory), "Progressive Data Science" (data science methodology), "Reversible Recurrent Neural Networks" (RNNs), "Some recent advances in reasoning based on analogical proportion" (analogy reasoning).
3. **Key efficient attention papers are missing entirely.** No citations of: Reformer (LSH), Longformer, BigBird, cosFormer, Random Feature Attention, Linformer, Luna, Nyströmformer (the actual paper, not just our early experiments).
4. **The search queries are likely too narrow.** The scan step generates queries that keep returning the same small set of papers from our SPECTER knowledge base, suggesting the knowledge base itself may be too small or the queries too repetitive.

### Root Cause
The scan step uses gpt-4o-mini which has an Oct 2023 cutoff. It generates generic queries like "attention mechanism efficiency" which keep retrieving the same papers. The knowledge base has 5,832 papers but the system only ever surfaces ~25 of them.

---

## Proposal Diversity: Very Low

### Thematic Analysis
I categorized all 71 hypotheses by recurring themes (a hypothesis can match multiple themes):

| Theme | Count (out of 71) | % |
|-------|-------------------|---|
| "Reduce redundancy" | **39** | 55% |
| "Dynamic/adaptive" mechanism | **31** | 44% |
| "Token relevance/importance" scoring | **27** | 38% |
| "Local + global" attention combo | **19** | 27% |
| "Hierarchical/multi-level" structure | **15** | 21% |
| Gating mechanism | 7 | 10% |
| Linear attention variants | 7 | 10% |
| Sparse attention | 6 | 8% |

### The Pattern
Almost every proposal after experiment 10 follows the same template:
> "Current attention treats all tokens equally. By introducing a [dynamic/learned/adaptive] [gating/relevance/importance] mechanism that [selectively focuses on/dynamically adjusts] attention based on [token relevance/contextual importance], we can reduce redundancy."

This is essentially **one idea rephrased 40+ different ways.** The similarity guard catches this at the embedding level (0.80-0.96 cosine similarity every time) but can't force genuinely different ideas — it just retries 3 times and proceeds anyway.

### What's Missing
Ideas the system has **never tried** despite being well-known pre-cutoff techniques:

| Approach | Why it's interesting | Status |
|----------|---------------------|--------|
| **Grouped Query Attention (GQA)** | Reduces KV heads, real-world proven | Never proposed |
| **Sliding window + global tokens** (Longformer-style) | Simple, effective sparse pattern | Never proposed |
| **Low-rank key/value projections** | Direct parameter reduction | Tried once (early crash), never revisited |
| **Mixture of Experts in attention** | Different heads for different patterns | Never proposed |
| **RoPE modifications** (frequency scaling, NTK-aware) | Position embedding improvements | Never proposed |
| **Attention head pruning** | Remove redundant heads entirely | Never proposed |
| **KV cache compression** | Quantize/compress stored keys/values | Never proposed |
| **Differential attention** (attend to differences) | Novel attention formulation | Never proposed |
| **Cross-layer attention sharing** | Reuse attention patterns across layers | Never proposed |

### Early vs Late Diversity
- **Experiments 1-10 (pre-cloud):** More diverse — linear attention, Performer FAVOR+, Nyström, LSH, FLuRKA. These produced the best pre-cloud results.
- **Experiments 11-71 (cloud):** Monotonically converged on "dynamic relevance gating" variations. The 4 KEEPs are all minor variations of this same idea.

---

## What's Working vs What's Not

### Working
- **The infrastructure is solid** — code generation, validation, cloud training, git integration all work
- **The council pipeline produces runnable code** — 89% success rate on cloud
- **Incremental improvement is happening** — 1.174 → 1.157 over 33 cloud experiments
- **Cost efficiency is good** — ~$0.09/experiment

### Not Working
- **Idea generation has converged** — the propose model is stuck in a local optimum of "relevance gating"
- **Paper retrieval is broken** — same 6 papers surface repeatedly, many irrelevant
- **Similarity guard is counterproductive** — fires 100% of the time, wastes 2 extra council calls per experiment, doesn't actually force novel ideas
- **Throughput is too low** — ~2.5 experiments/hour due to similarity retries and fix loops

---

## Recommendations for Next Phase

### High Impact (do these first)

1. **Overhaul the scan step / search queries**
   - The current scan generates the same generic queries every time
   - Option A: Pre-define a diverse set of search query categories (sparse patterns, position embeddings, head management, KV optimization, activation functions, normalization approaches)
   - Option B: Include the list of "unexplored approaches" in the program.md to directly guide the proposal model toward new territory
   - Option C: Rotate through different research themes each N experiments

2. **Raise or remove the similarity threshold**
   - Current: 0.80 threshold, catches everything, forces 3 retries that don't help
   - Recommended: Either raise to 0.95 (only catch near-duplicates) or remove entirely
   - The prompt-level "don't repeat" instruction is more effective than embedding similarity for this domain

3. **Cap fix attempts at 3 instead of 10**
   - The `ve` parameter bug and similar structural issues are never fixed after 3 attempts
   - Attempts 4-10 are pure waste (~$0.01 each + time)

### Medium Impact

4. **Enrich the paper knowledge base**
   - Current: 5,832 papers but only ~25 ever surface
   - Add targeted papers on: GQA, sliding window attention, attention head pruning, KV cache techniques, RoPE scaling
   - Or: bypass the knowledge base for some experiments and let the model reason from first principles

5. **Add approach categories to the proposal prompt**
   - Instead of open-ended "propose an experiment," give a menu:
     - "This experiment should explore: [sparse patterns / head management / position encodings / activation alternatives / normalization changes / KV optimization]"
   - Rotate categories to force diversity

6. **Run a baseline on cloud GPU**
   - We still don't have a cloud GPU baseline for the unmodified train.py
   - All comparisons are cloud-vs-cloud which is fine, but we should know if 1.157 is actually better than what the baseline gets on CUDA

### Lower Priority

7. **Parallelize experiments** — run 2-3 experiments concurrently on Modal to improve throughput
8. **Add experiment deduplication at the code level** — check if the generated code diff is meaningfully different from previous experiments, not just the hypothesis text
9. **Consider a different propose model** — gpt-4o-mini may lack the creative capacity for genuinely novel ideas. A larger model (gpt-4o) for the propose step only could improve diversity at modest cost increase

---

## Bottom Line

The system is **operationally successful** — it runs autonomously, generates code, trains on cloud GPU, and finds improvements. But it's **scientifically stuck** — it found one decent idea (relevance gating) in the first few experiments and has been generating variations of it ever since. The paper retrieval is essentially broken (same 6 papers on repeat), and the similarity guard is making things worse, not better.

The biggest lever for improvement is forcing genuine diversity in what gets proposed. This is a prompt engineering and knowledge base problem, not an infrastructure problem.
