# Prior Art — Existing Autonomous ML Research Systems

> What already exists, what we can learn from it, and how Rediscover differs.
> Last updated: 2026-03-16

## Karpathy's autoresearch

- **Repo:** https://github.com/karpathy/autoresearch
- **What it does:** Single agent autonomously modifies training code, runs 5-minute experiments, evaluates val_bpb, keeps improvements, discards failures. Repeat overnight.
- **Architecture:** 3 files — `prepare.py` (fixed), `train.py` (agent modifies), `program.md` (agent instructions)
- **Key design:** Fixed time budget, single metric, git-as-memory, results.tsv log
- **What we borrow:** The experiment loop pattern, fixed time budgets, git-based state, results.tsv format, program.md concept
- **What we add:** Literature context, multi-agent council, retrodiction validation, knowledge cutoff controls

## AI Scientist v2 (Sakana AI)

- **Repo:** https://github.com/SakanaAI/AI-Scientist-v2
- **Paper:** https://arxiv.org/abs/2504.08066
- **What it does:** Full pipeline — literature review → hypothesis → code → run experiments → analyze → write LaTeX paper. Uses agentic tree search.
- **Headline result:** A generated paper passed peer review at an ICLR workshop
- **Failures (per independent evaluation, arxiv 2502.14297):**
  - 42% of experiments failed due to coding errors
  - Hallucinated numerical results
  - Poor novelty detection — "rediscovered" mini-batch SGD as novel
  - Requires human-defined templates that limit autonomy
  - Structural errors in papers (missing figures, placeholder text)
- **What we learn:** The experiment loop works. The paper writing doesn't. Novelty detection is critical. Don't try to do everything at once.
- **How Rediscover differs:** We don't write papers. We validate via retrodiction (ground truth), not LLM-as-judge. We focus on one domain at a time.

## BudgetMLAgent

- **Paper:** https://arxiv.org/abs/2411.07464
- **What it does:** Uses cheap models by default, cascades to expensive models only when needed via "ask-the-expert lifelines"
- **Result:** 94% cost reduction ($0.931 → $0.054/run) with BETTER success rate (22.7% → 33.0%)
- **What we borrow:** The model routing strategy. Cheap models for routine work, expensive for reasoning.

## n-autoresearch

- **Repo:** https://github.com/iii-hq/n-autoresearch
- **What it does:** Multi-GPU parallel extension of autoresearch. Python orchestrator + Rust GPU workers.
- **Architecture:** Workers call REST API to register experiments, get suggestions, report results. Central orchestrator holds all state.
- **What we learn:** The orchestrator/worker split is a good pattern if we ever scale to cloud GPUs.

## MLAgentBench (Stanford SNAP)

- **Repo:** https://github.com/snap-stanford/MLAgentBench
- **Paper:** https://arxiv.org/abs/2310.03302
- **What it does:** Benchmark suite of 13 ML tasks for evaluating research agents (CIFAR-10, BabyLM, etc.)
- **Key finding:** Claude 3 Opus achieved the best success rate across all tested models
- **What we learn:** Defines what "success" looks like for autonomous ML agents. Good reference for evaluation.

## Karpathy's llm-council

- **Repo:** https://github.com/karpathy/llm-council
- **What it does:** Submit a question to multiple LLMs, they respond independently, peer-review each other (anonymized), then a chairman synthesizes
- **Architecture:** FastAPI + React, uses OpenRouter for unified model access
- **What we borrow:** The deliberation pattern (independent response → critique → synthesis). Not the code.
- **How Rediscover differs:** Our council is a pipeline (propose → critique → refine), not a roundtable. Different agents have different roles.

## OpenHands (formerly OpenDevin)

- **Repo:** https://github.com/All-Hands-AI/OpenHands
- **Paper:** https://arxiv.org/abs/2407.16741
- **What it does:** Full Linux sandbox for coding agents (Docker + Jupyter). Event-stream perception-action loop.
- **Why we don't use it:** Adds container orchestration complexity we don't need. Our experiments run directly on the host machine.

## Framework Landscape (Why We Don't Use One)

| Framework | What It Does | Why Not For Us |
|-----------|-------------|----------------|
| LangGraph | Stateful graph with checkpoints | Over-engineered for a while loop |
| CrewAI | Role-based agent teams | Broken logging; can't track experiments reliably |
| AutoGen | Multi-agent conversations | Retry loops can spiral without guards |
| MetaGPT | Software team simulation | Designed for project delivery, not research iteration |
| CAMEL | Role-playing agent pairs | Research prototype, not production-ready |

**Our approach:** `litellm` + plain Python loop + file-based state. A general-purpose agent loop is ~40-70 lines. This is what autoresearch itself uses (no framework), and it works.
