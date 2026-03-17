# Execution Plans Index

> All execution plans for Rediscover.

## Active Plans

| Plan | Status | Description |
|------|--------|-------------|
| [phase-1-foundation.md](exec-plans/active/phase-1-foundation.md) | Active | Get the autonomous research loop running end-to-end on M4 Mac |
| [step-1-2-setup-and-harness.md](exec-plans/active/step-1-2-setup-and-harness.md) | Complete | Project setup + experiment harness on MPS |
| [step-3-llm-provider.md](exec-plans/active/step-3-llm-provider.md) | Complete | litellm wrapper with model routing and cost tracking |
| [step-4-arxiv-ingestion.md](exec-plans/active/step-4-arxiv-ingestion.md) | Complete | arXiv paper ingestion into Chroma vector DB |
| [step-6-git-provider.md](exec-plans/active/step-6-git-provider.md) | Complete | Programmatic git operations for experiment loop |
| [step-5-council-pipeline.md](exec-plans/active/step-5-council-pipeline.md) | Complete | Multi-agent council: scan → propose → critique → refine → implement |
| [step-7-research-loop.md](exec-plans/active/step-7-research-loop.md) | Complete | Wire the loop: council + training + git + guards |
| [phase-roadmap.md](exec-plans/active/phase-roadmap.md) | Active | Full project roadmap: Phases 1-8 |
| [phase-2-paper-ingestion.md](exec-plans/active/phase-2-paper-ingestion.md) | Active | Paper ingestion into Chroma knowledge base |

## Completed Plans

| Plan | Completed | Description |
|------|-----------|-------------|

## Tech Debt

See [tech-debt-tracker.md](exec-plans/tech-debt-tracker.md).

## How to Use This

1. **Starting new work?** Check active plans first.
2. **Creating a new plan?** Add to `exec-plans/active/`, update this index.
3. **Finishing a plan?** Move to `exec-plans/completed/`, update this index.
4. **Found tech debt?** Log it in the tech debt tracker.
