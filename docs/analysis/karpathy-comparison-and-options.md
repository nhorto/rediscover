# Karpathy autoresearch vs Rediscover: Comparison & Options

> Date: 2026-03-18
> Purpose: Understand why Karpathy's approach works and ours crashes, then brainstorm fixes.

## How Karpathy's autoresearch Works

### The Simple Truth
Karpathy's approach is radically simple compared to ours:

1. **Single agent** — Claude Opus reads `program.md` and runs autonomously. No council, no multi-step pipeline, no separate propose/critique/refine/implement steps.

2. **Whole-file editing** — The agent edits `train.py` directly. No zone extraction, no code splicing, no frozen sections. The entire file is writable.

3. **No pre-flight validation** — No syntax checks, no quick forward passes, no structure validation. The agent commits the code and runs the full 5-minute training. If it crashes, it reads the stack trace and fixes it (or moves on).

4. **git as the safety net** — If training fails or val_bpb doesn't improve, `git reset --hard` reverts to the last good commit. Simple, reliable, zero-overhead.

5. **`program.md` IS the prompt** — The entire research methodology, loop instructions, constraints, and scoring criteria are in one markdown file. The agent reads it and follows it.

### Key Details

| Aspect | Karpathy | Rediscover |
|--------|----------|------------|
| Agent architecture | Single Claude agent | 5-step council (scan → propose → critique → refine → implement) |
| Code generation | Agent edits full file directly | LLM generates 100-line zone, code splicing puts it back |
| Model for code | Claude Opus 4.6 (same agent doing everything) | Separate model per role (gpt-4o-mini for council, various for implement) |
| Pre-flight checks | None | validate_train_py + validate_zone_structure + quick_validate_code |
| Crash handling | Agent reads stack trace, uses judgment | Automated fix_code loop (up to 3-10 retries via API) |
| Training validation | Full 5-min run is the only gate | Quick forward pass test, then full training |
| Literature search | Built into program.md instructions | Chroma DB with 5,832 papers, SPECTER embeddings |
| Files agent can modify | Only train.py (whole file) | Only the "zone" (~100 lines out of 692) |

## Why Karpathy's Works and Ours Crashes

### Root Cause: The Zone Extraction Pattern

Our #1 problem is the **zone extraction + splicing** approach. Here's what happens:

1. We extract ~100 lines (GPTConfig + helpers + CausalSelfAttention) from train.py
2. Send these 100 lines to an LLM with instructions to modify them
3. The LLM returns modified code
4. We splice it back into the 692-line train.py

**Why this fails:**
- The LLM sees 100 lines of code **without the context** of the other 592 lines. It doesn't know what the frozen code expects.
- We show it "frozen context" (a summary), but the LLM still rewrites from scratch instead of making targeted edits
- The splicing creates subtle bugs — the frozen code calls `norm(x)`, but the LLM deletes `norm()` because it doesn't know the frozen code needs it
- Every invariant (forward signature, ve gating, RoPE, GQA repeat, causal mask) is a potential failure point
- We've added validation checks for each invariant, but there are too many — each check triggers a fix attempt, which burns API budget and often fails

### Root Cause: Too Many LLM Calls

Our council pipeline makes **5+ separate LLM calls** per experiment:
1. Scan (generate search queries)
2. Propose (hypothesis + approach)
3. Critique (review proposal)
4. Refine (address critique, produce plan)
5. Implement (write code)
6. Fix attempts (1-10 more calls if code fails)

Each call is a chance for information loss. The plan from step 4 is translated into code in step 5, but the implement model doesn't have the full context of steps 1-4. It's playing telephone.

**Karpathy uses ONE call** — the same agent that thinks about what to try also writes the code. Zero information loss.

### Root Cause: Context Disconnect

Our implement model receives:
- A plan text (from the refine step)
- The frozen context summary
- Two examples
- The current zone code
- Tensor shape reference
- Interface contract

But it does NOT see:
- The full 692-line train.py
- The actual frozen code (MLP, Block, GPT, optimizer, training loop)
- How the optimizer groups parameters
- How the training loop calls forward()
- What `prepare.py` provides

Karpathy's agent sees the ENTIRE file. It knows every line of context.

## Our Crash Rate by Model

| Model | Experiments | Crashed | Rate |
|-------|------------|---------|------|
| gpt-4o (Phase 2 pre-zone) | 4 | 4 | 100% |
| gpt-4o (Phase 2 post-zone) | 13 | 3 | 23% |
| gpt-4o-mini (Phase 4) | 10 | 10 | 100% |
| gpt-4o-2024-11-20 (Phase 4) | 5 | 4 | 80% |
| Sonnet 4.5 (Phase 4, brief test) | ~3 | ~3 | ~100% |

The pattern is clear: **the zone approach doesn't work reliably with any model.**

## Options for Fixing This

### Option 1: Copy Karpathy's Approach (Recommended)
**What:** Ditch the zone extraction. Give the agent the full train.py and let it edit directly. Use a single Claude agent (via Claude Code or API) that reads program.md and modifies train.py.

**Pros:**
- Proven to work — Karpathy got 100+ experiments overnight
- Zero information loss — agent sees full file
- No splicing bugs
- Simpler code (can delete zone extraction, code splicing, structure validation)

**Cons:**
- Agent might make changes outside the attention mechanism (hyperparameters, optimizer, etc.)
- Harder to constrain to attention-only changes
- Need to use a capable model (Opus-level) for the whole pipeline, not just implement
- Retrodiction validity requires knowledge-limited model for research decisions

**Difficulty:** Medium — need to rewrite loop.py to use a single agent workflow instead of council pipeline. Keep the council for research direction (propose/critique) but let a capable agent do the actual coding.

### Option 2: Send Full train.py to Implement Model
**What:** Keep the council pipeline but send the FULL 692-line train.py to the implement model instead of just the zone. Let it return the full modified file.

**Pros:**
- Full context — model sees everything including optimizer, training loop, frozen code
- Keeps the council research pipeline (retrodiction-valid)
- Minimal code changes (just change what we send to implement)

**Cons:**
- 692 lines is a lot of output — higher cost, more tokens, more chances for errors
- Model might change frozen sections (optimizer, training loop, etc.)
- Need output validation to check frozen sections weren't modified

**Difficulty:** Low — change `_implement()` in service.py to send full file instead of zone.

### Option 3: Diff-Based Code Generation
**What:** Instead of asking the model to output the full zone code, ask it to output a diff/patch. Show it the current code and ask "what lines do you want to change?"

**Pros:**
- Model only outputs the changed lines (much less output)
- Baseline code is preserved by default — model must explicitly change each line
- Less likely to accidentally delete norm(), ve gating, etc.

**Cons:**
- LLMs are notoriously bad at generating well-formed diffs
- Patch application can fail in subtle ways
- More complex parsing logic needed

**Difficulty:** Medium-High — need to build diff parsing and application logic.

### Option 4: Template-Based Code Generation (Option B from earlier)
**What:** Give the model a rigid template with slots to fill. The model only writes the attention computation body (~20 lines), everything else is hardcoded.

**Pros:**
- Guaranteed correct boilerplate
- Very low crash rate (the model can't break what it can't touch)
- Fast, cheap (small output)

**Cons:**
- Very constraining — model can't add new __init__ params, new helper functions, or new config fields
- Limits the kinds of experiments the system can try
- Not how Karpathy does it (his approach has no constraints)

**Difficulty:** Low — build template, fill slots.

### Option 5: Hybrid — Council Research + Single-Agent Implementation
**What:** Keep the council pipeline (scan → propose → critique → refine) for research decisions, but instead of calling an implement API, spawn a separate Claude Code session that reads the plan and modifies train.py directly (like Karpathy's approach).

**Pros:**
- Best of both worlds: retrodiction-valid research (knowledge-limited council) + high-quality code (capable agent edits full file)
- The implementation agent can read the full file, test locally, fix its own bugs
- Matches proven Karpathy pattern for code quality

**Cons:**
- More complex orchestration (council + separate agent session)
- Implementation agent needs to be prompted not to inject post-cutoff knowledge
- Longer per-experiment time (agent might iterate several times before committing)

**Difficulty:** High — need agent orchestration, but would be the most robust solution.

## Recommendation

**Start with Option 2 (send full train.py)** — it's the lowest-effort change that addresses the root cause. If the model can see the full file including the optimizer, training loop, and how parameters are grouped, it should produce dramatically fewer crashes.

If Option 2 still crashes too much, move to **Option 1 (copy Karpathy's approach)** — ditch the zone extraction entirely and let the agent edit the full file.

Save Options 3-5 for later if needed.

## What We've Tried So Far (Complete History)

### Phase 2 — First Experiments (March 17)
- Used gpt-4o-2024-05-13 via OpenRouter for all roles
- Pre-zone approach (full file rewrite): 100% crash rate
- Post-zone approach (zone extraction): 23% crash rate, 4 successful experiments
- Best val_bpb: 1.675352 (5% improvement)
- All proposals replicated pre-cutoff papers (Performer, Linformer, FLuRKA, Reformer)

### Phase 4 — Scaling Attempt (March 18)
- OpenRouter credit limit blocked initial runs (402 errors)
- Switched to gpt-4o-mini: fast/cheap but 88-100% crash rate
- Fixed MuonAdamW None gradient crash
- Added signal-based API timeout (litellm timeout wasn't working)
- Added output line-buffering (Python was buffering all output)
- Added error hard stop (10 consecutive failures → stop loop)
- Fixed loop crash on API timeout during fix_code calls

### Model Experiments (March 18)
- gpt-4o-mini: creative proposals, terrible code (88% crash)
- gpt-4o-2024-11-20: same crash patterns (80% crash)
- Sonnet 4.5: still crashed (same structural issues)
- **Conclusion: the problem is the approach, not the model**

### Prompt Experiments (March 18)
- Added working examples (baseline SDPA + Nyström)
- Added "surgical modifications" prompt
- Added "code translator, not researcher" prompt
- Added causal mask validation
- **None of these significantly reduced crashes** because the root cause is the zone extraction pattern, not the prompt wording

### Infrastructure Improvements (March 18)
- Modal cloud GPU runner (built but not deployed)
- Direct OpenAI API support (bypassing OpenRouter)
- Runner provider abstraction (local vs cloud)
- Error cascade hard stop guard

## Files Changed

All code changes are committed to the `feature/targeted-code-generation` branch. Key files:
- `src/app/loop.py` — Main research loop with validation checks
- `src/providers/llm.py` — Model routing and API timeouts
- `src/domains/council/config.py` — Prompt templates
- `src/domains/council/service.py` — Council pipeline
- `src/utils/code_splicing.py` — Zone extraction (the root cause of crashes)
- `experiments/train.py` — The file being modified
- `experiments/program.md` — Research direction document
