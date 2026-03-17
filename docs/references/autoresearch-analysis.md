# Autoresearch Code Analysis

> Detailed analysis of Karpathy's autoresearch source code for adaptation into Rediscover.
> Source: https://github.com/karpathy/autoresearch
> Last updated: 2026-03-16

## Architecture: Three Files

### 1. prepare.py (Fixed, ~300 lines)

**Constants (DO NOT MODIFY in original):**
- `MAX_SEQ_LEN = 2048` — context length
- `TIME_BUDGET = 300` — 5 minutes wall-clock training time
- `EVAL_TOKENS = 40 * 524288` — tokens for validation eval

**What it provides:**
- Downloads data shards from HuggingFace (`karpathy/climbmix-400b-shuffle`)
- Trains BPE tokenizer (8192 vocab, GPT-4 style split pattern)
- `Tokenizer` class — wraps tiktoken with BOS prepend
- `make_dataloader()` — BOS-aligned dataloader with best-fit packing
- `evaluate_bpb()` — THE fixed metric (bits per byte, vocab-agnostic)

**evaluate_bpb details:**
```python
# Sums per-token cross-entropy (nats), sums target byte lengths,
# converts nats/byte → bits/byte. Special tokens excluded.
# Uses fixed MAX_SEQ_LEN so results are comparable across configs.
total_nats / (math.log(2) * total_bytes)
```

**Data storage:** `~/.cache/autoresearch/` (data shards + tokenizer)

### 2. train.py (Agent modifies this, ~500 lines)

**Model: GPT with modern tricks**
- RMSNorm (via `F.rms_norm`)
- Rotary Position Embeddings (RoPE)
- Grouped Query Attention (n_kv_head configurable)
- Value Embeddings (ResFormer) — alternating layers with input-dependent gating
- Sliding window attention pattern (configurable SSSL = short/short/short/long)
- FlashAttention 3 kernel (Hopper-only, falls back to community kernel)
- Logit soft-capping at 15
- Per-layer residual lambdas and x0 lambdas (deep signal propagation)
- ReluSquared activation in MLP

**Optimizer: MuonAdamW (single GPU)**
- Muon for 2D matrix parameters (polar express orthogonalization + NorMuon variance)
- AdamW for embeddings, unembeddings, scalars
- Separate learning rates per param group
- LR scales as 1/sqrt(dmodel/768)
- Warmup + warmdown schedule based on wall-clock progress (not steps)

**Default hyperparameters:**
```python
ASPECT_RATIO = 64       # model_dim = depth * 64
HEAD_DIM = 128
WINDOW_PATTERN = "SSSL"
TOTAL_BATCH_SIZE = 2**19  # ~524K tokens
DEPTH = 8               # 8 layers
DEVICE_BATCH_SIZE = 128
```

**Default model size:** 8 layers × 64 = 512 dim, 4 heads, ~30M params

**Training loop:**
- Time-based (not step-based) — runs until `TIME_BUDGET` reached
- First 10 steps excluded from timing (compilation warmup)
- Progress = training_time / TIME_BUDGET
- LR warmup → constant → warmdown schedule
- Fast fail: abort if loss NaN or > 100
- GC disabled after step 0 (avoids 500ms stalls)

### 3. program.md (Agent instructions)

**Key instructions to the agent:**
1. Read README.md, prepare.py, train.py first
2. Check `~/.cache/autoresearch/` has data (halt if not)
3. Create `results.tsv` with header on first run
4. Run baseline first, record it
5. Enter loop: propose → edit train.py → run → record → keep/discard → repeat
6. NEVER STOP once loop begins
7. 10-minute human timeout → classify as failure

**results.tsv format:** `commit | val_bpb | memory_gb | status | description`
- status: `keep`, `discard`, or `crash`

## macOS Forks (CRITICAL — Solves the Adaptation Problem)

### miolini/autoresearch-macos (PyTorch MPS)
- **Repo:** https://github.com/miolini/autoresearch-macos
- **Approach:** Pure PyTorch, replaces FA3 with `F.scaled_dot_product_attention` + manual sliding window masks
- **All CUDA→MPS changes already done:**
  - `verify_macos_env()` checks for MPS at startup
  - Device detection: `"cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"`
  - `sync_device()` function handles `torch.mps.synchronize()` vs `torch.cuda.synchronize()`
  - `torch.compile` disabled on MPS (only used on CUDA)
  - `autocast_ctx = contextlib.nullcontext()` on MPS (no autocast)
  - Optimizer scalar tensors explicitly cast to device with `.to(device=p.device, dtype=p.dtype)`
  - GQA head expansion done manually with `repeat_interleave` instead of FA3
  - Peak VRAM tracking gracefully returns 0 on non-CUDA
- **Default hyperparams tuned for Mac:**
  - `DEPTH = 4` (vs 8 on CUDA) — smaller model
  - `DEVICE_BATCH_SIZE = 16` (vs 128 on CUDA) — fits in 32GB
  - `TOTAL_BATCH_SIZE = 2**16` (vs 2**19 on CUDA) — 65K vs 524K tokens
  - `WINDOW_PATTERN = "L"` (vs "SSSL") — full context only
- **Muon optimizer preserved** — works on MPS with manual dtype casting

### trevin-creator/autoresearch-mlx (MLX Native)
- **Repo:** https://github.com/trevin-creator/autoresearch-mlx
- **Approach:** Complete rewrite in MLX (Apple's ML framework)
- **Performance results on M4 Max:**
  - Started: 2.667 val_bpb → Final: 1.294 val_bpb
  - 6-7 minutes per experiment
  - AdamW-only default (Muon not ported)
- **Key insight:** Different hardware finds different winning recipes — Mac Mini discovered different optima than M4 Max

## What This Means for Rediscover

**The hard part is already done.** The macOS MPS fork gives us a working train.py that runs on your M4 Mac with 32GB. We can use it almost directly as our experiment harness.

### What We Can Use As-Is (from miolini/autoresearch-macos)
1. Complete model architecture (GPT, RoPE, RMSNorm, GQA, value embeddings)
2. MPS-compatible attention (SDPA with sliding window masks)
3. MuonAdamW optimizer (with MPS dtype casting)
4. Device detection and sync
5. Time-based training loop
6. evaluate_bpb metric
7. Tokenizer system
8. Mac-tuned default hyperparameters (DEPTH=4, BATCH=16, etc.)

### What We Still Need to Add (for Rediscover)
1. **Council deliberation** before each code edit (propose → critique → refine)
2. **Literature retrieval** from knowledge base before proposing
3. **experiment_log.md** — narrative alongside TSV
4. **Loop guards** — max iterations, budget cap, stuck detection
5. **Cost tracking** — LLM API spend per experiment
6. **Similarity detection** — prevent repeating the same hypothesis

### What We Should Consider (MLX vs MPS)
The MLX fork achieved val_bpb of 1.294 on M4 Max — but it's a complete rewrite. The MPS fork keeps PyTorch compatibility (all research code is PyTorch). **Recommendation: Start with MPS fork for ecosystem compatibility, consider MLX later if performance is a bottleneck.**

## Key Insight for Adaptation

The autoresearch loop is elegantly simple:
```
while time_remaining:
    hypothesis = llm("Given results.tsv and train.py, what to try?")
    edit(train.py, hypothesis)
    git_commit()
    result = run("uv run train.py")
    if result.val_bpb < best:
        best = result.val_bpb
        log(results.tsv, "keep")
    else:
        git_reset()
        log(results.tsv, "discard")
```

The entire system is ~630 lines across 3 files. We should aim for similar simplicity in our core loop, with the council and literature layers as clean additions around this same pattern.
