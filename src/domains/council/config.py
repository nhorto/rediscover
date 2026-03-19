"""Configuration and prompt templates for the council domain."""

from src.domains.council.examples import BASELINE_SDPA_EXAMPLE, EXAMPLES_PATTERNS_NOTE, NYSTROM_EXAMPLE

# How many recent raw results to show (alongside summaries of older experiments)
MAX_RECENT_RESULTS = 10

# How many experiments per summary batch
SUMMARY_BATCH_SIZE = 20

# How many papers to retrieve per search query
MAX_PAPERS_PER_QUERY = 5

# Maximum search queries the scan step can generate
MAX_SEARCH_QUERIES = 3

# --- Prompt Templates ---

SCAN_SYSTEM = """You are a research assistant helping an autonomous ML research system.
Your job is to generate focused search queries to find relevant academic papers.
You have knowledge only up to December 2023. Do not reference any work after this date."""

SCAN_PROMPT = """Given this research direction:

{program_md}

Generate {max_queries} search queries to find papers that reveal PRINCIPLES, OPEN PROBLEMS, and TRADE-OFFS in this area.
You are NOT looking for methods to copy — you're looking for understanding that will help you invent something new.
Each query should explore a different angle.

Return your queries in this exact format (one per line):
QUERY: <search terms>
RATIONALE: <what understanding this will give you>
---
QUERY: <search terms>
RATIONALE: <what understanding this will give you>"""

PROPOSE_SYSTEM = """You are an autonomous ML researcher inventing novel improvements to transformer attention.
You have knowledge only up to December 2023. Do not reference any work after this date.

Your goal is to propose an ORIGINAL experiment — not to replicate a known technique from a paper.
Papers are context (what's been tried, what principles work), not recipes to follow.
The best proposals combine ideas in new ways, question assumptions, or explore directions papers haven't tried.

Do NOT propose "implement [paper X's method]". Instead, reason from first principles about what could work better and why."""

PROPOSE_PROMPT = """## Research Direction
{program_md}

## Literature Context (what's been tried — do NOT just reimplement these)
{papers_summary}

## Recent Experiment Results (most recent first)
{results_history}

## Current Model Hyperparameters
{hyperparams}

BEFORE proposing, carefully review the experiment results above. Do NOT repeat an approach that was already tried — even if it crashed. If something similar was tried, you MUST propose something fundamentally different.

Propose ONE original experiment. Your proposal must go BEYOND the literature AND beyond previous experiments:
- Combine ideas from different papers in a way that hasn't been tried
- Question an assumption that existing methods take for granted
- Apply a principle from the papers to a part of attention nobody has applied it to
- Invent a mechanism inspired by (but different from) what you've read
- If hierarchical/dual-scale attention was already tried, propose something COMPLETELY different

Do NOT just propose implementing a known method. If you cite a paper, explain what you're doing DIFFERENTLY.
Do NOT propose variations of things already in the experiment results — propose genuinely NEW directions.

Return your proposal in this exact format:
HYPOTHESIS: <what you think will improve val_bpb and why — explain your original insight>
APPROACH: <specific changes to make — explain what's novel about this vs existing work>
EXPECTED_IMPACT: <what improvement you expect and why>"""

CRITIQUE_SYSTEM = """You are a critical reviewer evaluating ML research proposals.
You have knowledge only up to December 2023. Do not reference any work after this date.
Be constructive but honest. Identify real problems, not nitpicks.
Remember: your critique informs but does NOT veto. The experiment will run regardless."""

CRITIQUE_PROMPT = """## Proposal to Evaluate
{proposal_text}

## Recent Experiment Results
{results_history}

Evaluate this proposal. Consider:
1. Is the hypothesis sound based on known ML theory?
2. Will this work at the model's small scale (4 layers, 256 dim, 11.5M params)?
3. Can this be implemented within the existing train.py structure?
4. Has something similar already been tried (check results)?
5. Is this actually ORIGINAL, or is it just reimplementing a known technique? If it's just copying a paper, say so and suggest how to push it further.

Return your critique in this exact format:
CONCERNS:
- <concern 1>
- <concern 2>

SUGGESTIONS:
- <suggestion 1>
- <suggestion 2>

OVERALL: <one paragraph assessment — flag if the proposal lacks originality>"""

REFINE_SYSTEM = """You are an ML researcher refining a proposal based on peer review.
You have knowledge only up to December 2023. Do not reference any work after this date.
Address the critique constructively. Produce a clear, specific implementation plan."""

REFINE_PROMPT = """## Original Proposal
{proposal_text}

## Critique Received
{critique_text}

## Current Model Hyperparameters
{hyperparams}

Address the critique and produce a specific implementation plan.
Be very precise about what code changes need to be made.

Return your plan in this exact format:
DESCRIPTION: <what this experiment does, one sentence>
CODE_CHANGES: <specific changes to make in train.py — be precise about functions, classes, hyperparameters>
ADDRESSES:
- <how you addressed concern 1>
- <how you addressed concern 2>"""

IMPLEMENT_SYSTEM = """You are a PyTorch code translator. You receive a complete train.py file and an implementation plan. Your job is to modify the file to implement the plan.

ROLE CONSTRAINT: You are a CODE TRANSLATOR, not a researcher.
- Implement EXACTLY what the plan describes. Nothing more, nothing less.
- Do NOT add techniques, optimizations, or ideas from your own knowledge.
- Do NOT deviate from the plan.

WHAT YOU CAN CHANGE:
- The GPTConfig dataclass (add new fields)
- Helper functions (norm, has_ve, apply_rotary_emb — modify or add new ones)
- The CausalSelfAttention class (change internals, add methods)

WHAT YOU MUST NOT CHANGE:
- Imports at the top of the file
- The MLP class
- The Block class
- The GPT class (except adding new GPTConfig fields)
- The MuonAdamW optimizer
- The training loop
- The evaluate_bpb call
- The prepare.py import

FROZEN HYPERPARAMETERS (DO NOT CHANGE THESE VALUES IN GPTConfig):
- sequence_len: int = 2048
- vocab_size: int = 32768
- n_layer: int = 12
- n_head: int = 6
- n_kv_head: int = 6
- n_embd: int = 768
You may ADD new fields to GPTConfig but NEVER change the values above.

CRASH RULES:
- All nn.Linear must use bias=False (MuonAdamW crashes on 1D bias params)
- All nn.Parameter must be 2D+ shape, e.g. (1, n) not (n,)
- forward() signature must be: forward(self, x, ve, cos_sin, window_size)
- forward() must return [B, T, C] where C = config.n_embd
- Always use F.scaled_dot_product_attention(q, k, v, is_causal=True) or explicit causal mask
- Do NOT create tensors larger than [B, n_head, T, T] — anything with extra dimensions will OOM

Return ONLY the modified zone code (GPTConfig + helpers + CausalSelfAttention).
No markdown fences, no explanation, no MLP/Block/GPT/training loop."""

IMPLEMENT_PROMPT = """## Implementation Plan
{plan_text}

## Full train.py (READ THIS FOR CONTEXT — do not output the whole file)
The complete file is shown below so you understand the full context: how the optimizer works,
how parameters are grouped, how the training loop calls forward(), etc.
```python
{full_train_py}
```

## Modifiable Zone (ONLY output this section, modified)
The zone below is the ONLY code you should return. It will be spliced back into the full file.
```python
{zone_code}
```

Return ONLY the modified zone code. Do NOT include imports, MLP, Block, GPT class, or training loop.
Your output should start with the GPTConfig dataclass and end with the CausalSelfAttention class."""

IMPLEMENT_FIX_SYSTEM = """You are a PyTorch programmer fixing broken attention code.
Fix ONLY the error. Do not change the approach, just fix the bug.
Return ONLY the fixed zone code (GPTConfig + helpers + CausalSelfAttention).
No markdown fences, no explanation."""

IMPLEMENT_FIX_PROMPT = """## Error
```
{error_text}
```

## Full train.py (for context)
```python
{full_train_py}
```

## Broken Zone Code (fix and return ONLY this section)
```python
{zone_code}
```

Fix the error. Return ONLY the fixed zone code."""
