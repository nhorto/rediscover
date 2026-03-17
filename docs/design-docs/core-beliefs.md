# Core Beliefs

> Opinionated decisions that guide how Rediscover is built.

1. **Retrodiction is the only honest validation.** Without a ground truth to compare against, you can't know if AI-generated research is actually novel. Using models with known cutoffs and comparing against post-cutoff breakthroughs creates that ground truth.

2. **Small-scale experiments are real science.** FlashAttention, RoPE, and Chinchilla scaling laws were all validated at 60-125M parameters. We don't need H100 clusters to do meaningful ML research.

3. **Simplicity beats sophistication.** AI Scientist v2 tried to do everything and 42% of experiments failed. A working loop that runs overnight beats an elegant system that crashes.

4. **The loop is the product.** The value is in the autonomous iteration cycle: propose → critique → implement → evaluate → learn → repeat. Everything else is supporting infrastructure.

5. **Cost discipline enables longer runs.** A system that costs $200/day runs for one day. A system that costs $15/day runs for two weeks. Longer runs discover more.

6. **Git is the memory.** Every experiment is a commit. Every failure is a revert. The full research history is `git log`. No database needed.

7. **Domain-agnostic by design.** The system should work for any ML/AI research problem where you can define a clear metric and give the agent code to modify — not just attention mechanisms.
