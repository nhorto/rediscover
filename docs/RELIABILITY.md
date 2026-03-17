# Reliability Standards

## Long-Running Operation

Rediscover is designed to run autonomously for hours or days. Reliability patterns:

### Error Recovery
- Every LLM API call has exponential backoff (max 3 retries)
- Every experiment is a reversible git commit — failures are `git reset`
- State is checkpointed after each experiment (results.tsv + git)
- On restart, the system reads results.tsv and continues from where it left off

### Loop Guards
- Max iterations hard limit (default: 500)
- Total cost budget cap
- Stuck detection: no improvement in N consecutive experiments → force novel direction
- Hypothesis similarity detection → reject repetitive proposals
- Error cascade: 3 consecutive coding failures → skip, move on
- Per-step timeout: 120s for any single LLM call

### Graceful Degradation
- If expensive model fails → fall back to cheaper model
- If code execution fails 3x on same hypothesis → discard and move on
- If arXiv API is unavailable → use cached papers from knowledge base
- Process supervisor (systemd/supervisord) for auto-restart on crash

### Health Indicators
- `results.tsv` row count = total experiments completed
- `git log --oneline | wc -l` = total commits (successful experiments)
- Last modification time of results.tsv = last activity
- Token spend tracking in utils/costs.py
