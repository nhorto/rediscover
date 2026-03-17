# Cloud GPU Options for Rediscover

> Reference for running experiments in the cloud instead of locally on the M4 Mac.
> Status: DEFERRED — using local MPS for now. Revisit when ready to scale.
> Last updated: 2026-03-16

## Current Setup (Local)

- Apple M4, 32GB unified memory, PyTorch MPS
- Baseline: val_bpb=1.763539, 11.5M params, ~17K tok/sec
- ~89 steps in 5 min training, ~13 min total per experiment (incl startup + eval)
- ~110 experiments/day, $0 compute cost (ties up the Mac)

## Recommended Cloud Architecture: Hybrid Local + Modal

### How It Works

```
Mac (orchestration — CPU only)         Modal Cloud (GPU only)
──────────────────────────────         ─────────────────────
1. Scan literature (arXiv API)
2. Council proposes (LLM API)
3. Council critiques (LLM API)
4. Council implements code (LLM API)
5. Send train.py to Modal ──────────►  6. GPU trains for 5 min
                                        7. Returns val_bpb + metrics
8. Evaluate result (local)  ◄──────────
9. Keep/discard (git, local)
10. Log to results.tsv (local)
11. Loop back to step 1
```

- Mac does the thinking (LLM API calls, literature retrieval, git, logging)
- Cloud does only the GPU training (5 min per experiment)
- Results return to Mac for evaluation and next iteration
- GPU spins up only for training, spins down between experiments

### Modal Implementation Pattern

```python
import modal

app = modal.App("rediscover")
vol = modal.Volume.from_name("rediscover-data", create_if_missing=True)

# GPU-accelerated training (runs on Modal cloud)
@app.function(gpu="A10G", timeout=600, volumes={"/data": vol})
def run_experiment(train_code: str, prepare_code: str) -> dict:
    """Run a 5-minute training experiment on cloud GPU."""
    # Write train.py and prepare.py to /tmp
    # Run training subprocess
    # Parse and return val_bpb, metrics
    ...

# Local orchestration calls this remotely:
# result = run_experiment.remote(train_code, prepare_code)
```

### Cost: Modal (RECOMMENDED)

| GPU | Per Hour | 24h (GPU time only) | Notes |
|-----|----------|---------------------|-------|
| T4 | $0.59 | ~$14 | Cheapest, may be slower |
| A10G | $1.10 | ~$26 | Good balance |
| A100 40GB | $2.10 | ~$50 | Fast, probably overkill |
| H100 | $3.95 | ~$95 | Way overkill for 11.5M params |

**Key advantage:** Pay only for actual GPU seconds. The 5-8 min of LLM API calls between experiments costs nothing on Modal (no GPU running).

**Free tier:** $30/month in credits = ~1 full 24h run on A10G at no cost.

**With faster GPU (A10G vs MPS):**
- Training likely completes in 1-2 min instead of 5 min
- Experiments take ~8-10 min total instead of ~13 min
- ~150-180 experiments/day instead of ~110

## Alternative: Vast.ai / RunPod (Cheapest Raw Cost)

### How It Works

SSH into a rented GPU instance, run the entire loop there (both orchestration and training). Use tmux to keep the session alive after disconnecting.

```bash
# From Mac terminal:
ssh -p PORT root@INSTANCE_IP
tmux
python src/app/loop.py --experiments 288
# ctrl+b d to detach, close terminal
# Come back later:
ssh -p PORT root@INSTANCE_IP
tmux attach
```

### Cost

| Provider | GPU | 24h Cost | Notes |
|----------|-----|----------|-------|
| Vast.ai | RTX 4090 | ~$8 | Cheapest, variable reliability |
| RunPod Community | RTX 4090 | ~$8-9 | Better reliability than Vast.ai |
| RunPod Secure | RTX 4090 | ~$14 | Data center grade |

**Downside:** You pay for the full 24h including idle time between experiments (when LLM calls are happening, GPU sits unused).

### RunPod + Claude Code

RunPod has an official guide for running Claude Code on their pods. This means you could:
1. SSH into a RunPod pod with GPU
2. Start Claude Code in tmux
3. Let Claude Code autonomously drive the entire experiment loop
4. Detach, close laptop, come back to check

This is the most "set and forget" option but requires running everything on the cloud instance (no local orchestration).

## NOT Recommended

| Platform | Why Not |
|----------|---------|
| Google Colab Pro/Pro+ | 24h session limit, no background processes, notebook-only |
| Lambda Labs | Expensive ($1.79/hr for A100), chronic availability issues |
| AWS/GCP on-demand | 3-5x more expensive than Vast.ai/RunPod for same GPU |

## Decision Matrix

| Factor | Local (M4 Mac) | Modal Hybrid | Vast.ai/RunPod SSH |
|--------|----------------|-------------|-------------------|
| Cost/day | $0 | $14-26 | $8-9 |
| Setup | Done | Medium | Medium |
| Speed | ~110 exp/day | ~150-180 exp/day | ~200+ exp/day |
| Ties up Mac? | Yes | No (CPU only) | No |
| Reliability | Excellent | Excellent | Variable |
| Overnight? | Mac must stay on | Mac must stay on | Runs independently |
| Free tier | N/A | $30/month | No |

## When to Switch to Cloud

Switch from local to cloud when:
- You've validated the loop works end-to-end locally (current priority)
- You want to run overnight without tying up your Mac
- You want faster experiments (cloud GPU >> MPS)
- You want to run multiple research directions in parallel

## Data Transfer Considerations

The experiment harness uses data cached at `~/.cache/autoresearch/` (~500MB for 4 shards + tokenizer). For cloud deployment:
- Modal: Mount a Volume with the pre-downloaded data, or download on first run and cache in the Volume
- Vast.ai/RunPod: Include data download in the instance setup script, or use a Docker image with pre-baked data

## Implementation Notes for Modal

When ready to implement:
1. Create a `modal_app.py` in the repo root
2. Define `run_experiment()` function with `@app.function(gpu="A10G")`
3. Use Modal Volumes for data persistence (download once, reuse)
4. The main loop in `src/app/loop.py` calls `run_experiment.remote()` instead of subprocess
5. Add `modal` to pyproject.toml dependencies
6. `modal deploy modal_app.py` to register the app, then run locally
