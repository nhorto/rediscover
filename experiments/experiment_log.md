# Experiment Log

> Narrative record of all experiments. Each entry documents what was tried, why, and what was learned.

---

## Experiment 2 — 2026-03-17 18:43:07
**Hypothesis:** Adopting a unified low-rank + kernel approximation of QK^T (inspired by FLuRKA [2306.15799v2]) in each attention head will reduce memory usage (by avoiding the full N×N matrix) and preserve or improve val_bpb by retaining most of the informative attention signal.
**Approach:** 1. In CausalSelfAttention.forward(), replace the standard softmax(QK^T)V with a two-step approximation:  
   a) Low-rank factorization: Compute rank-r factors FQ, FK of Q and K (e.g., via a small linear projection).  
   b) Kernel feature map: Transform FQ, FK with a simple positive feature map φ(x) = ELU(x) + 1, then approximate softmax(QK^T) as φ(FQ)·[φ(FK)^T] / normalizer.  
2. Set r = 64 (or another small rank) to keep compute overhead manageable within 5 minutes.  
3. Keep all other hyperparameters (e.g., LR schedules, HEAD_DIM) unchanged.
**Papers consulted:** Long Short-Term Attention, Gated recurrent neural networks discover attention, Which Transformer to Favor: A Comparative Analysis of Effici
**Critique:** The hypothesis is theoretically sound and aligns well with known methods to reduce memory usage in attention mechanisms. However, significant attention needs to be given to the choice of rank r and en
**Plan:** Implement a low-rank + kernel-based attention variant with adjustable ranks (32, 64, 128) and tuned hyperparameters to ensure stability and compatibility.
**Result:** val_bpb=CRASH (crash)
**Cost this cycle:** $1.2873
**Cumulative cost:** $2.5682
---
