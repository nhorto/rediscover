# Phase 1: Foundation — Get the Loop Running

> Goal: A working autonomous research loop that can run overnight on the M4 Mac.
> Priority: Get something running, then improve it.

## What We're Building

The minimum viable loop:
```
Literature scan → Propose → Critique → Refine → Implement → Train (5 min) → Evaluate → Log → Repeat
```

## Steps

### Step 1: Project Setup
- [ ] Initialize `pyproject.toml` with UV
- [ ] Add dependencies: `torch`, `litellm`, `arxiv`, `chromadb`, `sentence-transformers`, `tiktoken`, `pyarrow`, `rustbpe`
- [ ] Create `.env.example` with required API keys (OPENAI_API_KEY, ANTHROPIC_API_KEY, etc.)
- [ ] Set up `tests/` directory with conftest.py
- [ ] Verify `uv sync` and `uv run pytest` work

**Exit criteria:** `uv run pytest` runs (even with 0 tests)

### Step 2: Experiment Harness (from autoresearch-macos)
- [ ] Adapt `prepare.py` from `miolini/autoresearch-macos` into `experiments/prepare.py`
  - Keep: data download, tokenizer training, dataloader, `evaluate_bpb`
  - Change: Make data path configurable, reduce default shard count for faster setup
- [ ] Adapt `train.py` from `miolini/autoresearch-macos` into `experiments/train.py`
  - Keep: Full MPS-compatible GPT model, MuonAdamW, SDPA attention, time-based loop
  - Keep: Mac-tuned defaults (DEPTH=4, BATCH=16, TOTAL_BATCH=65K)
  - Change: Nothing yet — this is the agent's canvas
- [ ] Create `experiments/program.md` — initial research direction (attention improvements)
- [ ] Create `experiments/results.tsv` with header row
- [ ] Test: `uv run experiments/prepare.py` downloads data and trains tokenizer
- [ ] Test: `uv run experiments/train.py` completes a 5-minute training run on MPS
- [ ] Record baseline val_bpb

**Exit criteria:** `uv run experiments/train.py` runs for 5 minutes on M4 Mac and outputs val_bpb

### Step 3: LLM Provider (litellm wrapper)
- [ ] Create `src/providers/llm.py`
  - litellm wrapper with model routing
  - Prompt caching support (Anthropic)
  - Token counting and cost tracking
  - Budget cap enforcement
  - Model config: which model for which role (scan=cheap, propose=GPT-4-Turbo, critique=Llama-3.1-70B, implement=Claude-3.5-Sonnet)
- [ ] Create `src/utils/costs.py` — cost tracking and budget enforcement
- [ ] Test: Can call each model and get responses
- [ ] Test: Cost tracking correctly accumulates

**Exit criteria:** Can call all council models through litellm with cost tracking

### Step 4: Paper Ingestion Pipeline (arXiv)
- [ ] Create `src/providers/arxiv.py`
  - Query arXiv API with date filtering (before Dec 31, 2023)
  - Download paper metadata (title, abstract, authors, categories, date)
  - Rate limiting (1 req / 3 sec)
  - Save to local JSON cache
- [ ] Create `src/domains/literature/service.py`
  - Embed abstracts with `allenai-specter` (sentence-transformers)
  - Store in Chroma vector DB with year/category metadata
  - Retrieval: query by topic, filtered to pre-cutoff papers
- [ ] Create `src/domains/literature/types.py` — Paper, SearchResult types
- [ ] Create `src/domains/literature/config.py` — categories, cutoff date, embedding model
- [ ] Ingest initial corpus: "attention" + "transformer" papers from cs.LG/cs.CL, 2017-2023
- [ ] Test: Can retrieve top-10 relevant papers for "improving attention efficiency"

**Exit criteria:** ~5,000+ papers indexed in Chroma, retrievable by topic with date filtering

### Step 5: Council Pipeline
- [ ] Create `src/domains/council/types.py` — Proposal, Critique, Decision types
- [ ] Create `src/domains/council/config.py` — model assignments, token budgets per role
- [ ] Create `src/domains/council/service.py`
  - `scan_literature(topic)` → retrieve relevant papers, summarize state of art
  - `propose(context, results_history)` → hypothesis + approach (GPT-4 Turbo)
  - `critique(proposal, context)` → concerns, go/no-go (Llama 3.1 70B)
  - `refine(proposal, critique)` → final implementation plan (GPT-4 Turbo)
  - `implement(plan, train_py)` → code diff to apply (Claude 3.5 Sonnet June 2024)
- [ ] Test: Full pipeline produces a valid code modification

**Exit criteria:** Council can produce a code diff for train.py given a research direction

### Step 6: Git Provider
- [ ] Create `src/providers/git.py`
  - `commit(message)` — commit current state of experiments/
  - `reset_last()` — revert last commit (failed experiment)
  - `log(n)` — get last n commit messages
  - `diff()` — show current changes
- [ ] Test: Can commit, log, and reset in experiments/ directory

**Exit criteria:** Git operations work programmatically

### Step 7: The Loop (Wire It All Together)
- [ ] Create `src/app/loop.py` — main orchestrator
  ```
  1. Scan literature for current research direction
  2. Council: propose → critique → refine → implement
  3. Apply code diff to experiments/train.py
  4. Git commit the change
  5. Run `uv run experiments/train.py` (subprocess, 5-min timeout + startup overhead)
  6. Parse val_bpb from output
  7. If improved: keep commit, log "keep" to results.tsv
  8. If worse: git reset, log "discard" to results.tsv
  9. Append to experiment_log.md (narrative: hypothesis, result, what was learned)
  10. Loop
  ```
- [ ] Create `src/app/guards.py` — loop guards
  - Max iterations (default 500)
  - Budget cap (total LLM spend)
  - Stuck detection (no improvement in last 20 experiments)
  - Hypothesis similarity check (last 3-5 proposals)
  - Error cascade (3 consecutive failures → skip)
- [ ] Create entry point: `uv run src/app/loop.py`
- [ ] Test: Run 3-5 experiments end-to-end

**Exit criteria:** The loop runs autonomously for 3+ experiments, making real changes to train.py and logging results

### Step 8: Validation Framework (Stretch for Phase 1)
- [ ] Create `src/domains/validation/types.py` — Comparison types
- [ ] Create `src/domains/validation/config.py` — post-cutoff breakthrough registry
- [ ] Create `src/domains/validation/service.py`
  - Compare agent proposals against known breakthroughs (embedding similarity)
  - Score: direct hit / adjacent / directionally correct / novel / miss
- [ ] Test: Can score a mock proposal against the MLA breakthrough

**Exit criteria:** Can compare agent output against post-cutoff breakthroughs

## Risks

| Risk | Impact | Mitigation |
|------|--------|------------|
| train.py doesn't run on M4 | Blocks everything | Test Step 2 first, before building anything else |
| LLM-generated code diffs break train.py | Experiments crash | Error cascade guard; git reset on failure |
| Council produces weak hypotheses | No progress | Start with a focused program.md; iterate on prompts |
| arXiv API rate limiting | Slow ingestion | Download corpus once, cache locally |
| Cost overruns | Budget exhausted | Hard budget cap in guards.py |

## Order of Operations

**Do Step 2 FIRST** — if we can't train on the M4, nothing else matters.

Then: 1 → 3 → 6 → 4 → 5 → 7 → 8

Steps 3, 4, and 6 can be parallelized (independent providers).

## Estimated Scope

- Step 1: Small (project setup)
- Step 2: Medium (adapt and test training harness)
- Step 3: Medium (litellm wrapper + cost tracking)
- Step 4: Large (arXiv API + embedding + Chroma)
- Step 5: Large (council pipeline with multiple LLM roles)
- Step 6: Small (git wrapper)
- Step 7: Medium (orchestration loop + guards)
- Step 8: Medium (validation scoring)
