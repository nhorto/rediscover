# Hypothesis Similarity Check

> Goal: Prevent the council from repeating similar hypotheses by embedding-based comparison.
> Status: IN PROGRESS
> Started: 2026-03-17

## Progress

- [x] Add similarity computation to src/app/guards.py — cosine_similarity(), SimilarityResult, check_similarity(), record_hypothesis_embedding()
- [x] Use SPECTER embedder (reused from literature service) to embed hypotheses
- [x] Check new hypothesis against last N hypotheses, reject if cosine sim > threshold
- [x] Update loop.py — retry up to 3x on similar hypothesis, inject "AVOID REPETITION" message, proceed anyway after max retries
- [x] Add tests — 9 new tests (4 cosine sim, 5 hypothesis similarity)
- [x] Code review — offset alignment between hypothesis text and embeddings is fragile but correct; double-embed is minor perf hit; embedder=None handled properly
- [x] Ruff + pytest 75/75 pass
- [x] Update docs

## Status: COMPLETE

## Design

### Where it lives
The similarity check belongs in `LoopGuards` since it's a guard condition, but it needs access to an embedding model. Rather than making guards depend on sentence-transformers directly, the loop will compute the similarity and pass the result to the guard.

### Flow
1. Council produces a proposal with a hypothesis string
2. Loop embeds the hypothesis using SPECTER
3. Loop checks cosine similarity against stored hypothesis embeddings
4. If sim > 0.9 with any of the last 10: reject, append novelty demand to program.md, re-run council (max 3 retries)
5. If all retries exhausted: run the experiment anyway (don't deadlock)

### Implementation
- `src/app/guards.py`: Add `check_similarity(hypothesis_embedding, threshold)` method, store embeddings
- `src/app/loop.py`: Embed hypothesis after council, check before proceeding to training
- Embedding model: reuse LiteratureService.embedder if available, otherwise load standalone

## Exit Criteria
- Repeated hypotheses are detected and rejected
- Max 3 retries before proceeding anyway
- Tests verify detection and retry logic
