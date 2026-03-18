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

## Experiment 2 — 2026-03-17 19:27:54
**Hypothesis:** Implementing Linear Log-Normal Attention will reduce the memory complexity of the attention mechanism while maintaining or slightly improving val_bpb.
**Approach:** 1. Replace the standard softmax attention mechanism in the CausalSelfAttention class with Linear Log-Normal Attention as described in the paper "Linear Log-Normal Attention with Unbiased Concentration".
2. Specifically, modify the attention calculation to use the log-normal distribution for attention weights, which can be computed in linear time.
3. Update the forward pass of the CausalSelfAttention class to implement the log-normal attention mechanism.
4. Ensure the new attention mechanism integrates seamlessly with the rest of the model components like RoPE positional encoding, RMSNorm, and the ReLU squared MLP activation.
**Papers consulted:** Long Short-Term Attention, FLuRKA: Fast and accurate unified Low-Rank & Kernel Attentio, Linear Log-Normal Attention with Unbiased Concentration
**Critique:** The proposal to implement Linear Log-Normal Attention has potential merit, especially for reducing memory complexity in larger models. However, the recent experimental crashes and the small scale of t
**Plan:** This experiment implements Linear Log-Normal Attention in the CausalSelfAttention class to reduce memory complexity while maintaining or slightly improving val_bpb, and addresses stability concerns and integration within the existing training framework.
**Result:** val_bpb=CRASH (crash)
**Cost this cycle:** $0.3874
**Cumulative cost:** $0.6182
---

## Experiment 1 — 2026-03-17 19:33:22
**Hypothesis:** Implementing a linear attention mechanism using kernel-based feature maps, specifically the ELU+1 feature map, will reduce the memory complexity of the attention mechanism from O(N²) to O(N) and maintain or improve the validation bits-per-byte (val_bpb).
**Approach:** 1. Modify the `CausalSelfAttention` class in `train.py` to replace the standard multi-head self-attention mechanism with a linear attention mechanism.
2. Use the ELU+1 feature map to approximate the softmax attention, as described in the literature [2311.13541v4] Linear Log-Normal Attention with Unbiased Concentration.
3. Specifically, replace the attention computation:
   ```python
   # Original attention computation
   attn = torch.softmax((q @ k.transpose(-2, -1)) / sqrt(d_k), dim=-1)
   attn = attn @ v

   # Modified linear attention computation with ELU+1 feature map
   q_prime = elu(q) + 1
   k_prime = elu(k.transpose(-2, -1)) + 1
   v_prime = v

   attn = (q_prime @ k_prime) @ v_prime
   ```
4. Ensure that the dimensions and weight initialization are consistent with the baseline model.
5. Validate that the modified attention mechanism integrates correctly with the existing model architecture and training loop.
**Papers consulted:** Reproduction Report on "Learn to Pay Attention", Reversible Recurrent Neural Networks, Long Short-Term Attention
**Critique:** The hypothesis of using kernel-based feature maps to reduce memory complexity in attention mechanisms is promising, especially for handling longer sequences efficiently. However, the recent crashes in
**Plan:** This experiment refines the implementation of a linear attention mechanism using the ELU+1 feature map to reduce memory complexity, while ensuring stability and integration within the model.
**Result:** val_bpb=CRASH (crash)
**Cost this cycle:** $0.2374
**Cumulative cost:** $0.2374
---

## Experiment 4 — 2026-03-17 21:32:23
**Hypothesis:** Implementing a sparse attention mechanism could reduce memory complexity while maintaining or potentially improving val_bpb by focusing only on the most relevant parts of the sequence.
**Approach:** 1. Modify the CausalSelfAttention class to use a sparse attention mechanism.
2. Specifically, implement the Sparse Transformer approach, which restricts attention to a fixed number of nearest neighbors in the sequence. Use a fixed number of k-nearest neighbors for each token.
3. Set k to 64, meaning each token only attends to its 64 nearest neighbors, reducing the memory requirement from O(N²) to O(Nk).
**Papers consulted:** Attention Visualizer Package: Revealing Word Importance for , Reproduction Report on "Learn to Pay Attention", Emergence and Function of Abstract Representations in Self-S
**Critique:** The hypothesis of using sparse attention to reduce memory complexity while maintaining or improving val_bpb is theoretically sound and aligns well with recent trends in efficient transformer architect
**Plan:** Implement a dynamic sparse attention mechanism that adapts k based on sequence length and token importance, and evaluate its impact on tasks requiring long-range dependencies.
**Result:** val_bpb=1.675790 (keep)
**Cost this cycle:** $0.0735
**Cumulative cost:** $0.4049
---

## Experiment 5 — 2026-03-17 21:33:34
**Hypothesis:** Implementing Linformer-style linear attention, which reduces the memory complexity of the attention mechanism from O(N²) to O(N), will lower validation bits-per-byte (val_bpb) while maintaining or improving model performance. This approach approximates the attention matrix using low-rank matrices, which can decrease the memory footprint without significantly sacrificing the quality of the attention mechanism.
**Approach:** 1. Modify the `CausalSelfAttention` class in `train.py` to implement the Linformer linear attention mechanism. Specifically, project the sequence length dimension of the key and value matrices to a lower dimension (k < N), where k is a configurable parameter.
2. Implement the following steps:
   - Project the input sequence length dimension (N) of the key (K) and value (V) matrices to a smaller dimension (k) using learned projection matrices.
   - Perform the attention computation using these projected key and value matrices.
3. Adjust the attention computation as follows:
   - Instead of computing the full NxN attention matrix, compute an Nxk attention matrix.
   - Use the projected matrices in the attention calculation: `Attention(Q, K, V) = softmax(Q @ K.T) @ V`, where K and V are the projected matrices.
4. Set the projection dimension k to be 64 (or another reasonable value based on memory constraints).
**Papers consulted:** Long Short-Term Attention, Gated recurrent neural networks discover attention, Which Transformer to Favor: A Comparative Analysis of Effici
**Critique:** The hypothesis is sound based on existing ML theory, as Linformer-style linear attention is known to reduce memory complexity. However, the effectiveness of this approach at the proposed small scale r
**Plan:** This experiment implements Linformer-style linear attention in the `CausalSelfAttention` class, dynamically handling different sequence lengths and comparing its performance against other linear attention mechanisms.
**Result:** val_bpb=CRASH (crash)
**Cost this cycle:** $0.2562
**Cumulative cost:** $0.6612
---

## Experiment 6 — 2026-03-17 21:34:40
**Hypothesis:** Implementing a combination of low-rank and kernel approximation for the attention mechanism, specifically using the FLuRKA approach, will reduce memory complexity while maintaining or improving the val_bpb. This approach leverages the strengths of both low-rank and kernel methods, potentially leading to more efficient and effective attention computation.
**Approach:** 1. **Modify the CausalSelfAttention class** in `train.py` to implement the FLuRKA mechanism:
    - **Low-Rank Approximation**: Decompose the attention matrix using a low-rank approximation.
    - **Kernel Approximation**: Use kernel methods to approximate the softmax attention mechanism.
    - Combine these two approximations in a synergistic manner as described in the FLuRKA paper.
2. **Ensure compatibility** with the existing model architecture, particularly with RoPE positional encoding and Grouped Query Attention.
3. **Adjust hyperparameters** as needed to stabilize training, but keep the core architectural changes focused on the attention mechanism.

EXPECTED IMPACT: 
- **Memory Efficiency**: The combined low-rank and kernel approximation should significantly reduce the memory footprint of the attention mechanism from O(N²) to a more manageable complexity.
- **Validation Performance**: By maintaining the effectiveness of the attention mechanism while reducing memory usage, the model should be able to process longer sequences within the same memory budget, potentially leading to improved val_bpb due to better context understanding.
- **Training Stability**: Previous crashes with individual low-rank or kernel methods suggest that a combined approach may offer the robustness needed for stable training within the 5-minute budget.

The hypothesis is grounded in the strengths observed in the FLuRKA paper and aims to build on prior successful experiments with Performer FAVOR+ and dynamic sparse attention, which also focused on efficient attention mechanisms.
**Papers consulted:** Long Short-Term Attention, Which Transformer to Favor: A Comparative Analysis of Effici, How Smooth Is Attention?
**Critique:** The hypothesis is grounded in sound ML theory and has the potential to improve memory efficiency and validation performance. However, given the history of instability with similar methods, the propose
**Plan:** This experiment modifies the CausalSelfAttention class in `train.py` to implement FLuRKA, combining low-rank and kernel approximations, and performs thorough testing on a small subset of data with enhanced monitoring for stability.
**Result:** val_bpb=CRASH (crash)
**Cost this cycle:** $0.1346
**Cumulative cost:** $0.7958
---

## Experiment 9 — 2026-03-17 21:45:55
**Hypothesis:** Implementing linear attention using the Performer FAVOR+ mechanism will reduce the memory complexity of the attention mechanism while maintaining or improving validation performance (val_bpb).
**Approach:** 1. Modify the `CausalSelfAttention` class to replace the standard multi-head attention mechanism with the Performer FAVOR+ mechanism. 
2. In the `CausalSelfAttention` class, implement the random feature mapping for the FAVOR+ linear attention, which approximates the softmax kernel using random projections.
3. Ensure that the random feature mapping is applied to both the queries and keys.
4. Adjust the attention computation to use the linearized approximation, which will reduce the memory complexity from O(N²) to O(N) for sequence length N.
5. Keep the rest of the model architecture and hyperparameters unchanged to isolate the impact of the attention mechanism change.
**Papers consulted:** Long Short-Term Attention, Which Transformer to Favor: A Comparative Analysis of Effici, Gated recurrent neural networks discover attention
**Critique:** The proposal to implement the Performer FAVOR+ mechanism in the `CausalSelfAttention` class is theoretically sound and aligns with recent successful experiments. However, the small scale of the model 
**Plan:** Implement the Performer FAVOR+ mechanism in the `CausalSelfAttention` class, ensuring thorough testing of the random feature mapping and linearized attention approximation components to reduce memory complexity while maintaining or improving validation performance.
**Result:** val_bpb=1.675352 (keep)
**Cost this cycle:** $0.2283
**Cumulative cost:** $1.2986
---

## Experiment 13 — 2026-03-17 21:59:07
**Hypothesis:** Implementing the Reformer-style locality-sensitive hashing (LSH) attention will reduce the memory complexity of the attention mechanism while maintaining or improving validation performance (val_bpb).
**Approach:** 1. Modify the `CausalSelfAttention` class in `train.py` to implement LSH attention.
2. Replace the standard attention mechanism with LSH attention, which reduces the quadratic memory complexity to O(N log N) by hashing sequences into buckets and only computing attention within each bucket.
3. Ensure that the LSH implementation includes multi-round hashing to handle cases where relevant tokens may be split across different buckets.
4. Maintain the existing hyperparameters and optimizer settings to isolate the impact of the LSH attention mechanism change.

EXPECTED IMPACT: 
Implementing LSH attention should significantly reduce the memory footprint of the attention mechanism due to its sub-quadratic complexity. This reduction in memory complexity is expected to allow the model to handle longer sequences more efficiently, potentially improving val_bpb due to better utilization of the available information in longer sequences. Given the successful precedent of Reformer models in handling large sequences with reduced memory, this approach is anticipated to lower val_bpb while maintaining or improving model quality.
**Papers consulted:** Reversible Recurrent Neural Networks, Which Transformer to Favor: A Comparative Analysis of Effici, Reproduction Report on "Learn to Pay Attention"
**Critique:** The proposal is grounded in sound ML theory, leveraging the Reformer-style LSH attention to reduce memory complexity. However, given the previous challenges with integrating complex attention mechanis
**Plan:** Implement a prototype of Reformer-style LSH attention in the `CausalSelfAttention` class to test its feasibility and performance impact, ensuring correctness and efficiency, and validate it through comprehensive testing with various hyperparameter settings.
**Result:** val_bpb=1.676400 (discard)
**Cost this cycle:** $0.2476
**Cumulative cost:** $2.2620
---

## Experiment 14 — 2026-03-17 22:08:32
**Hypothesis:** Implementing the FLuRKA (Fast and accurate unified Low-Rank & Kernel Attention) mechanism from [2306.15799v2] will reduce the memory footprint of the attention mechanism while maintaining or improving validation performance (val_bpb).
**Approach:** 1. Modify the `CausalSelfAttention` class in `train.py` to replace the current attention mechanism with the FLuRKA mechanism.
2. The FLuRKA approach combines low-rank and kernel methods for attention computation:
   - Replace the standard attention computation with a low-rank approximation for the QK^T term.
   - Use a kernel-based method to approximate the softmax operation.
3. Specifically, implement the following steps:
   - Project the queries (Q) and keys (K) into a lower-dimensional space using a learned projection matrix.
   - Compute the attention scores using the low-rank approximation of QK^T.
   - Apply a kernel-based approximation to the softmax of the attention scores.
   - Multiply the resulting attention weights with the values (V) to get the final attended values.
4. Update the forward pass of the `CausalSelfAttention` class to use the above steps.

EXPECTED IMPACT: 
- The FLuRKA mechanism is expected to reduce the memory complexity from O(N²) to O(N) by leveraging low-rank approximations and kernel methods.
- This reduction in memory footprint should allow the model to handle longer sequences more efficiently.
- Since FLuRKA combines both low-rank and kernel methods, it is expected to maintain or even improve the model's performance (val_bpb) compared to the baseline by preserving important attention relationships while reducing computational cost.
**Papers consulted:** Reproduction Report on "Learn to Pay Attention", Which Transformer to Favor: A Comparative Analysis of Effici, Long Short-Term Attention
**Critique:** The hypothesis is theoretically sound as it aims to reduce memory complexity while maintaining performance, which is a critical challenge in scaling attention mechanisms. However, given the mixed resu
**Plan:** This experiment modifies the `CausalSelfAttention` class to implement the FLuRKA mechanism, testing low-rank approximations and kernel-based softmax separately before combining them to reduce memory footprint while maintaining or improving validation performance.
**Result:** val_bpb=1.680573 (discard)
**Cost this cycle:** $0.0851
**Cumulative cost:** $2.3471
---
