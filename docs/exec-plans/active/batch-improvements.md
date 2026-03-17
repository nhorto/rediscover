# Batch Improvements

> Multiple improvements from user feedback session.
> Status: IN PROGRESS
> Started: 2026-03-17

## Progress

- [ ] 1. Update model selection — switch to o1/gpt-4o-2024-05-13 council
- [ ] 2. Implement summarized history — compress old experiments into summaries
- [ ] 3. Fix program.md leakage — remove post-cutoff hints
- [ ] 4. Implement file sandboxing — chmod + diff validation for attention-only changes
- [ ] 5. Plan better tests — audit and improve test rigor

## Details per item

### 1. Model Selection
Switch from Mixtral/DeepSeek to o1/gpt-4o-2024-05-13. Update:
- src/providers/llm.py DEFAULT_MODEL_MAP
- src/utils/costs.py MODEL_PRICES
- docs/design-docs/model-selection-strategy.md
- tests that reference old model names

### 2. Summarized History
Every 20 experiments, compress the batch into a paragraph summary using a cheap model.
The propose step sees: all summaries + last 10 raw results.
This keeps context bounded as experiments grow.

### 3. Program.md Leakage
Remove "Compressing the KV representation into a smaller latent space" and any other
post-cutoff hints. Make it problem-focused ("reduce memory") not solution-focused.

### 4. File Sandboxing
- chmod 444 on prepare.py, results.tsv, program.md before training
- Diff validation: after implement step, verify changes are in attention-related code only
- Restore permissions after training

### 5. Better Tests
- Add adversarial tests for LLM response parsing (malformed, empty, garbage)
- Add tests for the loop with mocked providers
- Add tests for the council when LLM doesn't follow format
- Review existing tests for "made to be passable" patterns
