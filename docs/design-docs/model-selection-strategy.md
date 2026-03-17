# Model Selection Strategy

> Defines which models are used for each council role, why, and the multi-cutoff experimental design.
> Status: Active
> Last updated: 2026-03-16

## Core Principle

Every model in the research council must have a training data cutoff BEFORE the validation window. This ensures the agent cannot "cheat" by remembering post-cutoff breakthroughs.

## Current Council: December 2023 Cutoff (Run A — First Experiment)

| Role | Model | Provider | Cutoff | Cost (per 1M tokens) | Why This Model |
|------|-------|----------|--------|---------------------|----------------|
| **Scan** | Mistral 7B Instruct v0.2 | OpenRouter or Groq | Dec 2023 | ~$0.10 | Cheap, fast, just generating search queries |
| **Propose** | Mixtral 8x7B Instruct v0.1 | OpenRouter or Together | Dec 2023 | ~$0.60 | Best open-source reasoning at this cutoff |
| **Critique** | Mistral 7B Instruct v0.2 | OpenRouter or Groq | Dec 2023 | ~$0.10 | Different architecture = independent perspective |
| **Refine** | Mixtral 8x7B Instruct v0.1 | OpenRouter or Together | Dec 2023 | ~$0.60 | Same as propose, addresses critique with synthesis |
| **Implement** | DeepSeek Coder 33B Instruct | OpenRouter or Together | Nov 2023 | ~$0.80 | Best pre-2024 PyTorch coder (73.9% HumanEval, strong DS-1000) |

### Why NOT GPT-4 Turbo?
- Worse at code than DeepSeek Coder 33B (67% vs 73.9% HumanEval)
- OpenAI aggressively retires old snapshots — unreliable long-term access
- Cutoff date documentation is inconsistent across OpenAI sources
- More expensive ($10/$30 per 1M tokens)

### Why NOT Claude 3.5 Sonnet?
- April 2024 cutoff — knows about MLA (May 2024), Mamba-2 (May 2024), and other post-cutoff work
- Even in code-only role, its knowledge could influence implementation choices

### Why NOT GPT-4o?
- Original snapshot (Oct 2023 cutoff) was removed from API in Feb 2026
- Current snapshots have June 2024 cutoff — too late

## LLM Provider Configuration

### OpenRouter (Recommended)
litellm model strings for OpenRouter:
```python
MODEL_MAP = {
    "scan": "openrouter/mistralai/mistral-7b-instruct-v0.2",
    "propose": "openrouter/mistralai/mixtral-8x7b-instruct",
    "critique": "openrouter/mistralai/mistral-7b-instruct-v0.2",
    "refine": "openrouter/mistralai/mixtral-8x7b-instruct",
    "implement": "openrouter/deepseek/deepseek-coder-33b-instruct",
}
```

Requires `OPENROUTER_API_KEY` in .env.

OpenRouter advantages:
- Single API key for all models
- Can specify provider routing for exact versions
- 5.5% platform fee on top of provider costs
- litellm native support: `openrouter/<model-id>`

### Alternative Providers
```python
# Together AI
"together_ai/deepseek-ai/deepseek-coder-33b-instruct"
"together_ai/mistralai/Mixtral-8x7B-Instruct-v0.1"

# Groq (fastest inference)
"groq/mixtral-8x7b-32768"
"groq/mistral-7b-instant"  # Note: check exact model ID availability
```

### Local (Ollama) — For Offline Development
```bash
ollama pull deepseek-coder:33b       # ~22GB, fits M4 32GB
ollama pull mixtral:8x7b             # ~26GB, tight fit
ollama pull mistral:7b-instruct-v0.2 # ~4GB, easy
```

litellm model strings for Ollama:
```python
"ollama/deepseek-coder:33b"
"ollama/mixtral:8x7b"
"ollama/mistral:7b-instruct-v0.2"
```

Note: Cannot run DeepSeek 33B and Mixtral 8x7B simultaneously on 32GB. Must swap between roles.

## Multi-Cutoff Experimental Design (FUTURE)

### The Hypothesis
The amount of prior knowledge available to the agent affects what it can discover. By running the same research direction with models at different cutoff dates, we can measure how knowledge accumulation impacts innovation.

### Planned Runs

| Run | Cutoff | Models | What They Know | What They Don't Know |
|-----|--------|--------|----------------|---------------------|
| **Run A** (current) | Dec 2023 | Mixtral 8x7B + DeepSeek Coder 33B | Standard attention, FA1-2, GQA, MQA, RoPE, Mamba-1 (borderline), early MoE | MLA, FA3, Mamba-2, xLSTM, RWKV v5/v6, DeepSeek-V3 |
| **Run B** | Aug 2023 | Claude 3 Opus (if accessible) + Code Llama 34B | Standard attention, FA1-2, GQA, RoPE, early SSM | Mamba-1, MoE revival, everything 2024+ |
| **Run C** | Apr 2024 | Claude 3.5 Sonnet June + DeepSeek Coder 33B | All of Run A + GLA, early RWKV v5 | MLA (May), FA3 (July), Mamba-2 (May), xLSTM (May) |
| **Run D** | Sep 2022 | Llama 2 70B + Code Llama 34B | Pre-transformer-explosion baseline | Most modern techniques |

### What We Measure Across Runs
1. **Discovery rate**: What % of post-cutoff breakthroughs does each run approach?
2. **Discovery speed**: How many experiments to reach a given insight?
3. **Novelty**: Does an earlier cutoff produce MORE novel ideas (less constrained) or fewer (less foundation)?
4. **Convergence**: Do different cutoffs converge on the same improvements or different ones?
5. **Search queries**: What does each cutoff level look for in the literature? Do they ask different questions?

### Why This Is Valuable
- Tests the relationship between "amount of prior art" and "ability to innovate"
- Creates a controlled experiment across knowledge levels — like testing students at different education levels
- If Run B (Aug 2023) discovers something Run A misses, that suggests too much knowledge can constrain creativity
- If Run A consistently outperforms Run B, that confirms "standing on shoulders of giants" matters
- This methodology itself could be a research contribution

### Execution Order
1. Get Run A (Dec 2023) working first — prove the concept
2. Run B (Aug 2023) as the first comparison — maximum validation window
3. Runs C and D based on what we learn

## Pre-Dec 2023 Model Reference

### Confirmed Cutoff Dates (from research)

| Model | Cutoff | HumanEval | Strength | Source |
|-------|--------|-----------|----------|--------|
| DeepSeek Coder 33B Instruct | Nov 2023 | 73.9% | Best PyTorch coder | DeepSeek paper, GitHub issue #89 |
| WizardCoder 33B V1.1 | Nov 2023 | 79.9% | Highest HumanEval score | WizardLM GitHub |
| DeepSeek Coder 6.7B Instruct | Nov 2023 | 73.2% | Amazing for size | DeepSeek paper |
| Mixtral 8x7B Instruct v0.1 | Dec 2023 | ~45% | Best reasoning (open) | Mistral AI |
| Mistral 7B Instruct v0.2 | Dec 2023 | N/A | Fast, cheap, general | Mistral AI |
| Code Llama 70B | Sep 2022 | 67.8% | Conservative cutoff | Meta, GitHub issue #182 |
| Code Llama 34B Instruct | Sep 2022 | 53.7% | Moderate code ability | Meta |
| Claude 3 Opus | Aug 2023 | N/A | Best reasoning overall | Anthropic help center |
| Llama 2 70B Chat | Sep 2022 | 37.5% | Most conservative cutoff | Meta |
| Phi-2 | Mid 2023 | ~50% | Tiny (2.7B), cheap | Microsoft Research |

### Models That Do NOT Meet Pre-Dec 2023 Cutoff
- GPT-4o (all snapshots): Oct 2023 original removed from API; current = June 2024
- Claude 3.5 Sonnet: April 2024
- DeepSeek Coder V2: 2024
- StarCoder2: 2024
- Gemini (all versions): various 2024+ cutoffs
- GPT-4 Turbo: Dec 2023 claimed but inconsistently documented, deprecated

## Critique Veto Policy

**Always run the experiment regardless of critique.** The critique informs the refinement step but does not have veto power. Let val_bpb decide what works, not opinions. Sometimes "bad" ideas are breakthroughs.
