# Phase 4 Summary — First Real Experiment

> Date: 2026-03-18
> Status: PARTIAL — blocked by OpenRouter credit limit

## Executive Summary

The Phase 4 experiment run was blocked after 17 experiments ($4.44 spent) when the OpenRouter API key ran out of credits. Analysis of the available data shows the system successfully improved val_bpb by 5.0% (1.763539 → 1.675352) through Performer FAVOR+ and dynamic sparse attention modifications. However, all 10 unique proposals replicated known pre-cutoff methods rather than inventing new mechanisms, and no proposals achieved "direct hit" similarity with any post-cutoff breakthrough. The crash rate of 41% (improved from 80% pre-zone to 33% post-zone) remains high enough to waste significant budget on failed experiments.

## Key Findings

- **5.0% val_bpb improvement** from baseline (1.763539 → 1.675352) — modest but real
- **0 direct hits** against post-cutoff breakthroughs — the system reproduced pre-cutoff methods, not post-cutoff innovations
- **90% "adjacent" similarity** — misleadingly high due to all proposals being in the same semantic neighborhood as breakthroughs (efficient attention)
- **41% crash rate overall** — down from 80% after zone-based generation fixes
- **Low proposal diversity** — all proposals clustered around linear/sparse/kernel attention methods from the same small set of papers
- **All proposals replicated papers** — despite prompt rewrites encouraging originality (the rewrites happened after these experiments ran)

## Discovery Rate

| Level | Count | Rate |
|-------|-------|------|
| Direct hit (same mechanism) | 0 | 0% |
| Adjacent (same problem, different approach) | 9 | 90% |
| Directionally correct (right problem space) | 1 | 10% |
| Novel (not in literature) | 0 | 0% |

**True discovery rate: 0%.** No proposals independently converged on a post-cutoff mechanism. The 90% "adjacent" rate reflects semantic proximity (all about attention), not genuine discovery.

## Most Interesting Proposals

1. **Dynamic sparse attention (val_bpb=1.675790):** "Implementing a sparse attention mechanism could reduce memory complexity while maintaining or potentially improving val_bpb by focusing only on the most relevant parts of the sequence." — This was the most original proposal, adapting k based on sequence length and token importance. Best match: DeepSeek-V3 (0.091 keyword overlap).

2. **Performer FAVOR+ (val_bpb=1.675352, best result):** While this is a pre-cutoff method, the implementation worked well and achieved the best val_bpb. Shows the system can successfully implement known methods.

3. **Linformer-style linear attention:** Closest to MLA (score=0.618) due to low-rank key/value projection — but projected sequence length instead of embedding dimension (the key MLA insight). Crashed during training.

## Failure Modes

1. **Paper replication, not invention** — The biggest issue. Every proposal was "implement method X from paper Y." The prompt rewrites added in Phase 2 (demanding originality) were applied AFTER these experiments.

2. **Crash rate** — 41% of experiments crashed, wasting ~40% of the compute budget on failed code. Main causes:
   - Pre-zone code structure issues (4 crashes — fixed)
   - MuonAdamW 1D parameter errors (2 crashes — zone validation now catches)
   - Shape mismatches (1 crash)

3. **Narrow search** — The council repeatedly found the same ~5 papers and proposed variations on the same themes. Need better diversity in search queries.

4. **OpenRouter credit exhaustion** — The $30 budget was insufficient given the API key's credit limit.

## Recommendations for Phase 5

1. **Add OpenRouter credits** — Need $15-30 of API credits to run 100+ experiments
2. **Run with new prompts** — The originality-focused prompts from the Phase 2 rewrite haven't been tested yet; they should produce more diverse proposals
3. **Consider raising similarity thresholds** — Current "adjacent" threshold (0.55) is too lenient for same-domain comparisons; raise to 0.65 or use domain-specific baselines
4. **Improve search diversity** — Consider injecting different search angles per iteration, or requiring the council to explore at least one unfamiliar topic per cycle
5. **Reduce crash rate further** — Add more pre-flight validation checks (test optimizer compatibility, verify all parameter shapes are 2D+)
6. **Track per-iteration hypothesis diversity** — Measure how different each proposal is from all previous ones, not just the most recent
