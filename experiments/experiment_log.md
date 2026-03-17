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

## Experiment 3 — 2026-03-17 19:20:55
**Hypothesis:** Using linear attention with the ELU+1 feature map will reduce the memory complexity of the attention mechanism while maintaining or improving val_bpb.
**Approach:** Replace the standard softmax operation in the attention mechanism with a linear attention mechanism using the ELU+1 feature map. Specifically, modify the CausalSelfAttention class in `train.py` to compute attention using the following steps:
1. Compute feature maps for queries and keys using the ELU+1 activation function.
2. Use these feature maps to perform linear attention by multiplying the query and key feature maps, followed by the value matrices.

```python
import torch.nn.functional as F

class CausalSelfAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.query = nn.Linear(embed_dim, embed_dim)
        self.key = nn.Linear(embed_dim, embed_dim)
        self.value = nn.Linear(embed_dim, embed_dim)
        self.out = nn.Linear(embed_dim, embed_dim)

    def forward(self, x):
        B, T, C = x.size()
        q = self.query(x)
        k = self.key(x)
        v = self.value(x)

        q = q.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)

        feature_map_q = F.elu(q) + 1
        feature_map_k = F.elu(k) + 1

        attention_weights = torch.einsum('bhqd,bhkd->bhqk', feature_map_q, feature_map_k)
        attention_weights = attention_weights / self.scale

        out = torch.einsum('bhqk,bhvd->bhqd', attention_weights, v)
        out = out.transpose(1, 2).contiguous().view(B, T, C)

        return self.out(out)
```

EXPECTED IMPACT: Linear attention reduces the memory complexity from O(N²) to O(N), which should significantly lower the memory footprint. The use of the ELU+1 feature map is expected to maintain the quality of attention, thereby preserving or potentially improving val_bpb. This approach is grounded in recent research indicating that linear attention mechanisms can be both efficient and effective for long sequence modeling tasks.
**Papers consulted:** Long Short-Term Attention, Reproduction Report on "Learn to Pay Attention", Which Transformer to Favor: A Comparative Analysis of Effici
**Critique:** The proposal presents a sound hypothesis based on recent ML research suggesting the benefits of linear attention mechanisms. However, given the recent experimental results indicating crashes with simi
**Plan:** This experiment aims to test the stability and effectiveness of using linear attention with the ELU+1 feature map by conducting smaller scale unit tests and refining the mathematical operations to ensure numerical stability and efficiency.
**Result:** val_bpb=CRASH (crash)
**Cost this cycle:** $0.2454
**Cumulative cost:** $0.8091
---
