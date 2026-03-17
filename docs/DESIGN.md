# System Design

## Design Principles

1. **File-based everything.** State lives in TSV files and git history, not databases.
2. **Fixed time budgets.** Every training experiment gets exactly 5 minutes. This makes all results directly comparable regardless of architectural changes.
3. **Single metric per experiment.** Validation bits-per-byte (val_bpb) — lower is better. Vocabulary-agnostic, directly comparable.
4. **Reversible by default.** Every code change is a git commit. Failed experiments are `git reset`. Nothing is permanently lost.
5. **Progressive cost escalation.** Cheap models do the scanning and initial assessment. Expensive models only activate for deep reasoning and code generation.

## The Research Loop

```
Literature Scan (cheap) → Propose (primary) → Critique (challenger) → Refine → Implement → Run → Evaluate → Log → Loop
```

Each iteration takes approximately 8-12 minutes:
- Council deliberation: ~2-5 minutes (LLM calls)
- Training experiment: ~5 minutes (fixed budget)
- Evaluation + logging: ~1 minute

**Throughput:** ~5-7 experiments per hour, ~120-170 per day.

## Council Architecture

The council is a **pipeline**, not a roundtable:

| Stage | Model | Role | Cost Tier |
|-------|-------|------|-----------|
| Literature scan | Gemini Flash / DeepSeek | Retrieve relevant papers | Cheap |
| Propose | GPT-4 Turbo | Generate hypothesis + approach | Expensive |
| Critique | Llama 3.1 70B | Challenge proposal, find flaws | Medium |
| Refine | GPT-4 Turbo | Address critique, final plan | Expensive |
| Implement | Claude 3.5 Sonnet (June 2024) | Write the actual code changes | Medium |

All models share a **December 2023 knowledge cutoff** (or are restricted to pre-cutoff context only).

## Loop Guards

| Guard | Trigger | Action |
|-------|---------|--------|
| Max iterations | 500 experiments | Stop, produce summary |
| Budget cap | $X total spend | Stop, produce summary |
| Stuck detection | No improvement in 20 experiments | Force novel direction |
| Similarity check | Last 3 hypotheses cosine sim > 0.9 | Reject, demand different approach |
| Error cascade | 3 consecutive coding failures | Skip hypothesis, move on |
| Timeout | Any single LLM call > 120s | Retry once, then skip |

## Experiment State

```
experiments/
├── results.tsv              # Structured: timestamp, hypothesis, val_bpb, memory_mb, tokens_per_sec, keep/discard
├── experiment_log.md        # Narrative: what was tried, why, what was learned
├── train.py                 # The file agents modify
├── prepare.py               # Fixed: data prep, tokenizer, eval functions
└── program.md               # Standing instructions for research direction
```

Git history provides full provenance. `git log --oneline` = complete experiment timeline.
