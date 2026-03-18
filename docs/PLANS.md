# Execution Plans Index

> All execution plans for Rediscover.

## Active Plans

| Plan | Status | Description |
|------|--------|-------------|
| [phase-roadmap.md](exec-plans/active/phase-roadmap.md) | Active | Full project roadmap: Phases 1-8 |
| [cloud-gpu-modal.md](exec-plans/active/cloud-gpu-modal.md) | Active | Modal cloud GPU integration for experiment training |
| [targeted-code-generation.md](exec-plans/active/targeted-code-generation.md) | Active | Zone-based code generation to eliminate syntax errors |
| [batch-improvements.md](exec-plans/active/batch-improvements.md) | Active | Multiple improvements from user feedback |

## Completed Plans

| Plan | Completed | Description |
|------|-----------|-------------|
| [phase-1-foundation.md](exec-plans/completed/phase-1-foundation.md) | 2026-03-17 | Get the autonomous research loop running end-to-end on M4 Mac |
| [step-1-2-setup-and-harness.md](exec-plans/completed/step-1-2-setup-and-harness.md) | 2026-03-17 | Project setup + experiment harness on MPS |
| [step-3-llm-provider.md](exec-plans/completed/step-3-llm-provider.md) | 2026-03-17 | litellm wrapper with model routing and cost tracking |
| [step-4-arxiv-ingestion.md](exec-plans/completed/step-4-arxiv-ingestion.md) | 2026-03-17 | arXiv paper ingestion into Chroma vector DB |
| [step-5-council-pipeline.md](exec-plans/completed/step-5-council-pipeline.md) | 2026-03-17 | Multi-agent council: scan, propose, critique, refine, implement |
| [step-6-git-provider.md](exec-plans/completed/step-6-git-provider.md) | 2026-03-17 | Programmatic git operations for experiment loop |
| [step-7-research-loop.md](exec-plans/completed/step-7-research-loop.md) | 2026-03-17 | Wire the loop: council + training + git + guards |
| [hypothesis-similarity.md](exec-plans/completed/hypothesis-similarity.md) | 2026-03-17 | Embedding-based hypothesis dedup with retry logic |
| [fix-code-generation.md](exec-plans/completed/fix-code-generation.md) | 2026-03-17 | Fix implement step (superseded by targeted-code-generation) |
| [phase-2-paper-ingestion.md](exec-plans/completed/phase-2-paper-ingestion.md) | 2026-03-17 | Paper ingestion into Chroma knowledge base (3,000+ papers) |

## Tech Debt

See [tech-debt-tracker.md](exec-plans/tech-debt-tracker.md).

## How to Use This

1. **Starting new work?** Check active plans first.
2. **Creating a new plan?** Add to `exec-plans/active/`, update this index.
3. **Finishing a plan?** Move to `exec-plans/completed/`, update this index.
4. **Found tech debt?** Log it in the tech debt tracker.
