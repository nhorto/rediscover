# Security Requirements

## API Key Management
- All API keys stored in `.env` file (never committed)
- `.env` is in `.gitignore`
- litellm reads keys from environment variables
- No API keys in code, config files, or documentation

## Agent Sandboxing
- Agents can ONLY modify files in the `experiments/` directory
- Agents cannot modify `prepare.py` (fixed evaluation code)
- Agents cannot access files outside the project directory
- Agents cannot make network calls except through providers

## Code Execution
- Training runs in a subprocess with a hard timeout (5 minutes)
- No arbitrary code execution outside the experiment harness
- train.py changes are git-committed before execution (rollback possible)

## Cost Protection
- Hard budget cap on total LLM spending
- Per-experiment token limits
- Alert when spending exceeds daily threshold

## Data
- No personal data processed
- arXiv papers are publicly available
- Experiment results are not sensitive
- Repository may be made public — no secrets in any file
