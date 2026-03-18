"""Configuration for validation domain: breakthrough registry and scoring thresholds."""

from src.domains.validation.types import Breakthrough, BreakthroughCategory

# Scoring thresholds — map combined score to similarity level
DIRECT_HIT_THRESHOLD = 0.75
ADJACENT_THRESHOLD = 0.55
DIRECTIONAL_THRESHOLD = 0.35

# Weight for embedding vs keyword score in combined score
EMBEDDING_WEIGHT = 0.7
KEYWORD_WEIGHT = 0.3

# Post-cutoff breakthrough registry
# Each entry represents a known breakthrough published AFTER Dec 2023
BREAKTHROUGH_REGISTRY: list[Breakthrough] = [
    Breakthrough(
        name="Multi-Head Latent Attention (MLA)",
        category=BreakthroughCategory.ATTENTION,
        arxiv_date="2024-05-07",
        venue="DeepSeek-V2",
        description=(
            "Compresses key-value pairs into a low-dimensional latent space before "
            "projecting back to full dimensions. Reduces KV cache memory by 5-13x "
            "while maintaining or improving model quality."
        ),
        key_mechanisms=[
            "Low-rank KV compression via down-projection and up-projection",
            "Shared latent vector across heads replaces per-head KV storage",
            "Decoupled rotary position encoding applied after up-projection",
            "Joint compression of keys and values into single latent",
        ],
        keywords=[
            "latent", "kv cache", "compression", "low-rank", "projection",
            "key-value", "memory reduction", "down-projection", "up-projection",
            "shared representation", "cache efficiency",
        ],
    ),
    Breakthrough(
        name="FlashAttention-3",
        category=BreakthroughCategory.ATTENTION,
        arxiv_date="2024-07-11",
        venue="NeurIPS 2024",
        description=(
            "Exploits asynchronous execution and low-precision computation on H100 "
            "GPUs. Uses warp-specialization for overlapping GEMM and softmax, "
            "FP8 quantization of KV, and incoherent processing for accuracy."
        ),
        key_mechanisms=[
            "Asynchronous GEMM and softmax overlap via warp specialization",
            "FP8 quantization of K and V matrices for throughput",
            "Incoherent processing to maintain accuracy with low precision",
            "Hardware-aware scheduling for H100 TMA units",
        ],
        keywords=[
            "flash attention", "io-aware", "hardware-efficient", "warp",
            "asynchronous", "fp8", "quantization", "tiling", "fused kernel",
            "memory-efficient attention", "gpu optimization",
        ],
    ),
    Breakthrough(
        name="Mamba-2 / SSD Theorem",
        category=BreakthroughCategory.SSM,
        arxiv_date="2024-05-31",
        venue="ICML 2024",
        description=(
            "Proves that state space models are equivalent to a specific form of "
            "structured attention (Structured State Space Duality). Enables 2-8x "
            "faster implementation using matrix multiplication instead of scans."
        ),
        key_mechanisms=[
            "Structured State Space Duality: SSM = structured masked attention",
            "Semi-separable matrix representation of SSM computation",
            "Block decomposition enabling chunked parallel computation",
            "Replacing sequential scan with matrix multiply on chunks",
        ],
        keywords=[
            "state space", "ssm", "mamba", "structured attention", "duality",
            "recurrence", "linear recurrence", "scan", "selective", "sequence model",
            "semi-separable", "chunk",
        ],
    ),
    Breakthrough(
        name="xLSTM",
        category=BreakthroughCategory.SSM,
        arxiv_date="2024-05-07",
        venue="NeurIPS 2024",
        description=(
            "Extends LSTM with exponential gating (sLSTM) and matrix-valued memory "
            "(mLSTM). The mLSTM variant uses a covariance update rule that is "
            "competitive with Transformers while maintaining linear complexity."
        ),
        key_mechanisms=[
            "Exponential gating for improved gradient flow",
            "Matrix-valued memory state (mLSTM) instead of vector",
            "Covariance-based memory update rule",
            "Parallelizable computation despite recurrent structure",
        ],
        keywords=[
            "lstm", "gating", "exponential", "matrix memory", "recurrent",
            "covariance", "memory cell", "forget gate", "linear complexity",
            "sequence modeling",
        ],
    ),
    Breakthrough(
        name="RWKV Eagle/Finch (v5/v6)",
        category=BreakthroughCategory.SSM,
        arxiv_date="2024-04-09",
        venue="COLM 2024",
        description=(
            "Linear attention with matrix-valued states and dynamic recurrence. "
            "Uses multi-headed matrix-valued states with data-dependent decay, "
            "closing the quality gap with standard attention."
        ),
        key_mechanisms=[
            "Matrix-valued hidden state per head",
            "Data-dependent decay (dynamic recurrence weights)",
            "Linear attention formulation with WKV operator",
            "Receptance-weighted key-value attention mechanism",
        ],
        keywords=[
            "rwkv", "linear attention", "recurrent", "wkv", "receptance",
            "matrix state", "dynamic decay", "data-dependent", "linear complexity",
            "token mixing",
        ],
    ),
    Breakthrough(
        name="Gated Linear Attention (GLA)",
        category=BreakthroughCategory.ATTENTION,
        arxiv_date="2023-12-11",
        venue="ICML 2024",
        description=(
            "Hardware-efficient linear attention with data-dependent gating. "
            "Uses a gated recurrence that can be parallelized via a chunk-wise "
            "algorithm, achieving competitive quality with linear cost."
        ),
        key_mechanisms=[
            "Data-dependent gating on linear attention",
            "Chunk-wise parallel computation of gated recurrence",
            "Hardware-efficient implementation via tiling",
            "Gated retention mechanism",
        ],
        keywords=[
            "gated", "linear attention", "gating", "retention", "chunk",
            "hardware-efficient", "recurrence", "parallel", "sub-quadratic",
            "efficient attention",
        ],
    ),
    Breakthrough(
        name="DeepSeek-V3",
        category=BreakthroughCategory.MOE,
        arxiv_date="2024-12-27",
        venue="—",
        description=(
            "Combines MLA with auxiliary-loss-free MoE at 671B parameter scale. "
            "Uses multi-token prediction and FP8 training. Achieved GPT-4 level "
            "performance at a fraction of the training cost."
        ),
        key_mechanisms=[
            "MLA + MoE combination at massive scale",
            "Auxiliary-loss-free load balancing for MoE",
            "Multi-token prediction training objective",
            "FP8 mixed-precision training pipeline",
        ],
        keywords=[
            "mixture of experts", "moe", "load balancing", "multi-token",
            "fp8", "cost efficient", "routing", "expert", "sparse",
            "deepseek", "scaling",
        ],
    ),
]
