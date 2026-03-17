# Rediscover — Architecture Map

> This is a **map**, not a manual. It tells you where things are and what doesn't belong.
> For detailed design decisions, see `docs/design-docs/`. For implementation plans, see `docs/exec-plans/`.

## What This Is

Rediscover is an autonomous ML research system that tests whether AI agents can independently discover breakthroughs in machine learning and AI. It uses **retrodiction** as its validation methodology: agents are given only research published before a known knowledge cutoff date, then their discoveries are compared against real breakthroughs published after that date.

**Core question:** If you give AI agents the state of ML research as of December 2023, will they independently converge on ideas like Multi-Head Latent Attention, Mamba-2's SSM-attention equivalence, or FlashAttention-3?

**Domain-agnostic by design.** The system works for any ML/AI research problem where you can define a metric, give the agent code to modify, and evaluate results. Attention mechanisms are the first test case, but the architecture supports any research domain (optimizers, MoE routing, positional encoding, inference efficiency, data efficiency, etc.).

## How It Works

```
┌─────────────────────────────────────────────────┐
│              THE RESEARCH LOOP                   │
│                                                  │
│  1. LITERATURE SCAN (cheap model)                │
│     Query knowledge base of pre-cutoff papers    │
│                                                  │
│  2. PROPOSE (primary model — GPT-4 Turbo)        │
│     Hypothesis + approach based on literature     │
│                                                  │
│  3. CRITIQUE (challenger — Llama 3.1 70B)         │
│     Independent review, concerns, go/no-go        │
│                                                  │
│  4. REFINE (primary model)                        │
│     Address critique, produce implementation plan │
│                                                  │
│  5. IMPLEMENT (code model)                        │
│     Edit train.py with proposed changes           │
│                                                  │
│  6. RUN (local — M4 Mac, 32GB)                    │
│     Fixed 5-min training budget                   │
│     Metrics: perplexity, memory, throughput       │
│                                                  │
│  7. EVALUATE + LOG                                │
│     Better → keep (git commit)                    │
│     Worse  → discard (git reset)                  │
│     Append to results.tsv + experiment_log.md     │
│                                                  │
│  8. LOOP (with guards)                            │
│     Max iterations, budget cap, stuck detection   │
└─────────────────────────────────────────────────┘
```

## Project Structure

```
src/
├── domains/                     Business domains (self-contained)
│   ├── literature/              Paper ingestion and retrieval ✅
│   │   ├── types.py             Paper, SearchResult dataclasses
│   │   ├── config.py            CUTOFF_DATE, CATEGORIES, EMBEDDING_MODEL, CHROMA_PATH
│   │   ├── service.py           LiteratureService: ingest (arXiv→SPECTER→Chroma), search
│   │   └── __init__.py
│   │
│   ├── council/                 Multi-agent deliberation ✅
│   │   ├── types.py             SearchQuery, Proposal, Critique, ExperimentPlan, CouncilResult
│   │   ├── config.py            Prompt templates (5 roles), context helpers, tiered context limits
│   │   ├── service.py           CouncilService: scan → propose → critique → refine → implement
│   │   └── __init__.py
│   │
│   └── validation/              Retrodiction comparison (NOT YET BUILT)
│       ├── types.py             Comparison, SimilarityScore types
│       ├── config.py            Cutoff dates, breakthrough registry
│       ├── service.py           Compare agent output vs post-cutoff research
│       └── __init__.py
│
├── providers/                   Cross-cutting concerns
│   ├── __init__.py              Providers interface
│   ├── llm.py                   ✅ litellm wrapper (role→model routing, cost tracking, budget cap)
│   ├── arxiv.py                 ✅ arXiv API client (date-filtered search, single paper lookup)
│   └── git.py                   ✅ Git operations (commit, reset_last, log, diff, has_changes)
│
├── utils/                       Generic utilities, zero business logic
│   ├── costs.py                 ✅ MODEL_PRICES, estimate_cost, CostTracker, BudgetExceededError
│   └── __init__.py
│
├── types/                       Shared type definitions
│   └── __init__.py
│
└── app/                         App wiring and entry points ✅
    ├── loop.py                  ✅ run_loop() — council + training subprocess + git + guards + logging
    ├── guards.py                ✅ LoopGuards — max_iterations, budget_cap, stuck_detection, error_cascade
    └── __init__.py

experiments/                     Experiment workspace (git-tracked)
├── train.py                     The file agents modify
├── prepare.py                   Fixed setup (data, tokenizer, eval functions)
├── program.md                   Agent instructions for research direction
├── results.tsv                  One row per experiment
└── experiment_log.md            Detailed narrative log

tests/
├── unit/                        Pure function tests
├── integration/                 End-to-end pipeline tests
└── conftest.py
```

### Layer Model (within each domain)

```
Types → Config → Service → Runtime
                   ↑
             Providers (llm, arxiv, runner, git)
```

| Layer | Can Import From | Responsibility |
|-------|----------------|----------------|
| Types | `utils/` only | Pure type definitions |
| Config | Types, `utils/` | Constants, thresholds, model assignments |
| Service | Types, Config, `providers/` | Business logic |

### Providers

Providers is the single interface through which external dependencies enter any domain.

```python
# src/providers/__init__.py
class Providers:
    llm: LLMProvider        # litellm wrapper with model routing + caching
    arxiv: ArxivProvider     # Date-filtered paper retrieval
    runner: RunnerProvider   # Subprocess experiment execution + timing
    git: GitProvider         # Commit, reset, log operations
```

### Cross-Domain Rules

1. Type imports across domains are allowed
2. Service-to-service calls go through Providers or explicit function arguments
3. No circular dependencies between domains
4. Evaluation functions in `experiments/` are pure — no API calls

## Tech Stack

| Component | Choice | Notes |
|-----------|--------|-------|
| Runtime | Python 3.11+ | ML ecosystem requires Python |
| Package Manager | UV | Fast, deterministic |
| LLM API | litellm | Unified interface for all providers |
| Paper Retrieval | arXiv API | Date-filtered, free, comprehensive |
| ML Framework | PyTorch (MPS) | Apple Silicon GPU acceleration |
| Experiment Tracking | File-based (TSV + git) | Zero overhead, full provenance |
| Testing | pytest | Standard Python testing |

## Validation Methodology: Retrodiction

The system uses models with a **December 2023 knowledge cutoff** (GPT-4 Turbo, Llama 3.1 70B). Papers fed to agents are filtered to **before December 31, 2023** via arXiv API.

Agent outputs are compared against breakthroughs published **after** the cutoff:
- Multi-Head Latent Attention (MLA) — May 2024
- FlashAttention-3 — July 2024
- Mamba-2 / SSD theorem — May 2024
- xLSTM — May 2024
- RWKV Eagle/Finch — April 2024
- DeepSeek-V3 — December 2024

See `docs/references/architecture-breakthroughs.md` for the full registry.

## What Does NOT Exist Here

- **No web UI** — CLI-only, file-based interface
- **No database** — Git + TSV for all state
- **No framework** — No LangChain, CrewAI, AutoGen. Plain Python + litellm.
- **No distributed training** — Single machine (M4 Mac, 32GB)
- **No paper writing** — Unlike AI Scientist v2, this system discovers and validates, it doesn't write papers
- **No real-time monitoring dashboard** — Check experiment_log.md and results.tsv
- **No multi-GPU support** — Single MPS device

## Documentation Map

```
ARCHITECTURE.md          ← You are here (code map)
CLAUDE.md                ← Claude Code instructions
AGENTS.md                ← Instructions for Codex/external agents
docs/
├── PLANS.md             ← Index of all execution plans
├── DESIGN.md            ← System design patterns
├── PRODUCT_SENSE.md     ← Product vision, beliefs, north star
├── QUALITY_SCORE.md     ← Quality standards and metrics
├── RELIABILITY.md       ← Reliability standards
├── SECURITY.md          ← Security practices
├── design-docs/         ← Detailed design decisions
├── exec-plans/          ← Active and completed implementation plans
├── generated/           ← Auto-generated docs
├── product-specs/       ← Product specifications
└── references/          ← Research and external reference material
```
