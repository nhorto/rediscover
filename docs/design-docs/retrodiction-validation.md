# Retrodiction Validation Methodology

## The Problem

How do you evaluate whether AI-generated research is genuinely novel and useful? Most systems (like AI Scientist v2) use LLM-as-judge, which is circular — the AI evaluates its own novelty.

## The Solution: Retrodiction

**Retrodiction** = using a model to "predict" events that have already happened, where you control what the model knows.

### How It Works

1. **Select models with known knowledge cutoffs.** We use GPT-4 Turbo and Llama 3.1 70B, both with December 2023 cutoffs.

2. **Feed only pre-cutoff research.** The arXiv API filters papers by date. Agents receive ONLY papers published before December 31, 2023.

3. **Let agents innovate.** Given the state of the art as of December 2023, what improvements can the agent propose and validate?

4. **Compare against post-cutoff ground truth.** We know what actually happened in 2024-2025 (MLA, FlashAttention-3, Mamba-2, xLSTM, etc.). Did the agent independently converge on similar ideas?

### Why This Works

This is the ML research equivalent of a **holdout test set**:
- Training data = pre-cutoff papers (what the agent knows)
- Test data = post-cutoff breakthroughs (what the agent doesn't know)
- Metric = did the agent's proposals resemble the test data?

### Scoring Framework

| Score | Meaning | Example |
|-------|---------|---------|
| **Direct hit** | Agent proposes essentially the same mechanism | Agent proposes low-rank KV projection ≈ MLA |
| **Adjacent** | Same problem, different valid approach | Agent proposes KV quantization (not MLA, but addresses same problem) |
| **Directionally correct** | Right problem space, weaker solution | Agent identifies KV cache as bottleneck but proposes simple pruning |
| **Novel** | Not in post-cutoff literature | Agent proposes something we haven't seen (may be genuinely new or impractical) |
| **Miss** | Unrelated to any post-cutoff advance | Agent optimizes something already well-optimized |

### Limitations

1. **Knowledge contamination.** Models may have absorbed hints from informal sources (blogs, tweets) before formal papers. This is unavoidable but doesn't invalidate the approach — human researchers also build on informal knowledge.

2. **Metric scope.** Retrodiction only validates against *known* breakthroughs. An agent could discover something genuinely novel that we can't validate this way.

3. **Scale gap.** Small-scale experiments (10-60M params) may not capture all properties of innovations designed for 70B+ scale. Mitigation: focus on properties that transfer (algorithmic complexity, memory efficiency).

### Future Extensions

- Run the same experiment with different cutoff dates to see if more recent models discover more
- Run with multiple research domains (not just attention) to test generalizability
- Use post-cutoff models as evaluators to assess conceptual similarity
- Apply the methodology to other fields beyond ML (materials science, drug discovery?)
