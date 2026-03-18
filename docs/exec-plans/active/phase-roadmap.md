# Rediscover Phase Roadmap

> Full project roadmap from foundation through analysis.
> Status: Active — living document, updated as phases complete.
> Last updated: 2026-03-16

## Phase 1: Foundation (COMPLETE)
**Goal:** Get the autonomous research loop running end-to-end on M4 Mac.

**Delivered:**
- Experiment harness (MPS-compatible, baseline val_bpb=1.763539)
- LLM provider (litellm + OpenRouter, Dec 2023 cutoff council)
- arXiv paper ingestion pipeline (SPECTER embeddings, Chroma vector DB)
- Council pipeline (scan → propose → critique → refine → implement)
- Git provider (commit, reset, log)
- Research loop with guards (budget cap, stuck detection, error cascade)
- 66 tests, ruff clean
- Full documentation system

## Phase 2: Validate the System Works
**Goal:** Prove the loop runs autonomously with real LLM calls and real training.

**Steps:**
- [x] 2a: Ingest papers into knowledge base (3,000+ papers)
- [x] 2b: Run 3-5 experiments with $2-3 budget cap to test end-to-end
- [x] 2c: Fix whatever breaks (prompt issues, parsing failures, timeout handling)
- [ ] 2d: Confirm the loop runs for 10+ experiments without human intervention
- [ ] 2e: Review experiment_log.md — are proposals coherent? Is the council reasoning well?

**Exit criteria:** Loop runs 10+ experiments autonomously, producing coherent proposals and valid training runs.

## Phase 3: Validation Framework
**Goal:** Build retrodiction scoring — measure whether agent discoveries resemble real breakthroughs.

**Steps:**
- [x] 3a: Build src/domains/validation/ — types, config, service
- [x] 3b: Register post-cutoff breakthroughs with descriptions and key mechanisms (MLA, FA3, Mamba-2, etc.)
- [x] 3c: Implement similarity scoring (embedding-based + keyword-based)
- [ ] 3d: Build comparison report generator — for each agent proposal, score against all breakthroughs
- [ ] 3e: Test with synthetic proposals (e.g., feed it a description of MLA and check it scores as "direct hit")

**Exit criteria:** Can score any agent proposal against the breakthrough registry and produce a meaningful similarity rating.

## Phase 4: First Real Experiment
**Goal:** Run the first overnight research session and analyze what the agent discovers.

**Steps:**
- [ ] 4a: Write a focused program.md for the first research direction (attention efficiency)
- [ ] 4b: Run the loop overnight (~100+ experiments, $15-30 budget)
- [ ] 4c: Analyze results:
  - What hypotheses did the agent generate?
  - Which ones improved val_bpb?
  - What search queries did it use?
  - Did any proposals resemble post-cutoff breakthroughs?
  - What patterns emerged in successful vs failed experiments?
- [ ] 4d: Run validation framework on all proposals
- [ ] 4e: Write up findings

**Exit criteria:** Complete analysis of one overnight run with retrodiction scores.

## Phase 5: Multi-Cutoff Comparison
**Goal:** Test how knowledge level affects discovery ability.

**Steps:**
- [ ] 5a: Run the same research direction with Dec 2023 cutoff (Run A — already done in Phase 4)
- [ ] 5b: Set up Aug 2023 cutoff council (Claude 3 Opus if accessible, or Llama 2 70B + Code Llama 34B)
- [ ] 5c: Run same direction with Aug 2023 cutoff (Run B)
- [ ] 5d: Set up Apr 2024 cutoff council (Claude 3.5 Sonnet June)
- [ ] 5e: Run same direction with Apr 2024 cutoff (Run C)
- [ ] 5f: Compare across runs:
  - Discovery rate at each cutoff
  - Discovery speed (experiments to insight)
  - Novelty vs foundation trade-off
  - Search query differences across cutoffs
  - Convergence analysis — do different cutoffs find the same things?

**Exit criteria:** Comparative analysis across 2+ cutoff levels with retrodiction scores.

See docs/design-docs/model-selection-strategy.md for full multi-cutoff experimental design.

## Phase 6: Domain Expansion
**Goal:** Test whether the system generalizes beyond attention mechanisms.

**Steps:**
- [ ] 6a: Write program.md for a second research domain (e.g., optimizer improvements)
- [ ] 6b: Ingest relevant papers for the new domain
- [ ] 6c: Run overnight experiment
- [ ] 6d: Repeat for 2-3 more domains:
  - MoE routing strategies
  - Positional encoding
  - Inference efficiency (quantization, pruning)
  - Training data efficiency
- [ ] 6e: Compare discovery patterns across domains — is the system better at some than others?

**Exit criteria:** Successful runs in 3+ research domains with comparative analysis.

## Phase 7: Scale Up
**Goal:** Move to cloud, run longer experiments, try larger models.

**Steps:**
- [ ] 7a: Set up Modal integration (see docs/references/cloud-gpu-options.md)
- [ ] 7b: Run on A10G GPU — faster training, more experiments per day
- [ ] 7c: Try larger model sizes (60M-125M params) — do discoveries change at scale?
- [ ] 7d: Run parallel research directions simultaneously
- [ ] 7e: Multi-day autonomous runs

**Exit criteria:** System runs on cloud GPU, producing 200+ experiments per day.

## Phase 8: Analysis and Write-Up
**Goal:** Answer the core question — can AI agents independently discover ML breakthroughs?

**Steps:**
- [ ] 8a: Aggregate all results across phases 4-7
- [ ] 8b: Statistical analysis of discovery rates, patterns, and failure modes
- [ ] 8c: Case studies of the most interesting discoveries (hits, near-misses, genuinely novel ideas)
- [ ] 8d: Compare against AI Scientist v2 and other prior art
- [ ] 8e: Write up methodology, results, and conclusions
- [ ] 8f: Determine if results warrant a research paper or blog post

**Exit criteria:** Comprehensive analysis document with clear conclusions about AI research capability.

## Timeline Notes

- Phases are sequential but some overlap is possible
- Phase 2 is a prerequisite for everything after it
- Phases 4-6 can be run in parallel once the system is validated
- Phase 7 can happen anytime after Phase 2 if speed becomes a bottleneck
- Phase 8 is ongoing — start taking notes from Phase 4 onward
