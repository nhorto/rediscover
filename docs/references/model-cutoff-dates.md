# LLM Knowledge Cutoff Dates

> Reference for selecting models with controlled knowledge horizons.
> Last updated: 2026-03-16

## Primary Council Models (December 2023 Cutoff)

| Model | Knowledge Cutoff | API Available (Mar 2026)? | Cost (per 1M tokens) | Capability |
|-------|-----------------|--------------------------|---------------------|------------|
| **GPT-4 Turbo** (`gpt-4-turbo-2024-04-09`) | December 2023 | Deprecated but callable | $10 in / $30 out | Excellent reasoning + code |
| **Llama 3.1 70B** | December 2023 | Yes (Groq, Together, etc.) | ~$0.35-$0.79 | Strong, open source, self-hostable |
| **Llama 3 70B** | December 2023 | Yes (various providers) | ~$0.59/$0.79 | Same cutoff, slightly weaker |

## Alternative Models (Different Cutoffs)

| Model | Knowledge Cutoff | API Available? | Cost | Notes |
|-------|-----------------|----------------|------|-------|
| **Claude 3.5 Sonnet (June 2024)** | April 2024 | Yes | $3/$15 | 4 months later cutoff — smaller validation window |
| **Claude 3.5 Sonnet (Oct 2024)** | April 2024 | Yes | $3/$15 | Same cutoff as June, better capability |
| **Claude 3 Opus** | August 2023 | Researcher access program | $15/$75 | Earliest cutoff — largest validation window |

## Models NOT Suitable

| Model | Why Not |
|-------|---------|
| GPT-4 original | API access removed June 2024 |
| GPT-4o (May 2024 snapshot) | Removed from API Feb 2026 |
| Gemini 1.0/1.5 Pro | Fully retired, returns 404 |
| GPT-3.5 Turbo | Too weak for ML architectural reasoning |
| Claude 3 Haiku | Too weak for novel research |
| Mistral Large | Cutoff dates not publicly documented — can't control knowledge horizon |

## Council Groupings by Cutoff

### Group A: December 2023 (RECOMMENDED)
- GPT-4 Turbo + Llama 3.1 70B
- Massive validation window (Jan 2024 → present)
- Two independent models with identical knowledge horizons

### Group B: April 2024
- Claude 3.5 Sonnet (June) + Claude 3.5 Sonnet (Oct)
- Same cutoff, different capability levels
- Smaller but still substantial validation window (May 2024 → present)

### Group C: August 2023 (richest validation, hardest access)
- Claude 3 Opus (requires researcher access program)
- Largest validation window — includes Mamba-1 itself (Dec 2023)

## Knowledge Contamination Notes

Even with a December 2023 cutoff, models may have absorbed hints from:
- Blog posts and Twitter discussions about upcoming research
- arXiv pre-prints that were available before formal publication
- Conference workshop papers and informal presentations

This doesn't invalidate the approach — it means we're testing "can the agent synthesize and implement promising directions" rather than "can it invent from nothing." This is still how human researchers work.
