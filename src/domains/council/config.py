"""Configuration and prompt templates for the council domain."""

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

Generate {max_queries} focused search queries to find relevant papers on arXiv.
Each query should explore a different angle of this research direction.

Return your queries in this exact format (one per line):
QUERY: <search terms>
RATIONALE: <why this query is useful>
---
QUERY: <search terms>
RATIONALE: <why this query is useful>"""

PROPOSE_SYSTEM = """You are an autonomous ML researcher proposing experiments to improve a transformer model.
You have knowledge only up to December 2023. Do not reference any work after this date.
Your goal is to propose a single, specific, testable hypothesis that could lower validation bits-per-byte (val_bpb).
Be concrete and specific. Vague proposals waste experiment time."""

PROPOSE_PROMPT = """## Research Direction
{program_md}

## Relevant Papers
{papers_summary}

## Recent Experiment Results (most recent first)
{results_history}

## Current Model Hyperparameters
{hyperparams}

Based on the literature and past results, propose ONE specific experiment to try next.

Return your proposal in this exact format:
HYPOTHESIS: <what you think will improve val_bpb and why>
APPROACH: <specific changes to make to the model/training>
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

Return your critique in this exact format:
CONCERNS:
- <concern 1>
- <concern 2>

SUGGESTIONS:
- <suggestion 1>
- <suggestion 2>

OVERALL: <one paragraph assessment>"""

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

IMPLEMENT_SYSTEM = """You are an expert PyTorch programmer. Write clean, correct code.
You will receive a plan describing changes to make to a transformer training script.
Return the COMPLETE modified train.py file. Do not use placeholders or comments like '# ... rest unchanged'.
The entire file must be valid Python that can run with: uv run experiments/train.py"""

IMPLEMENT_PROMPT = """## Implementation Plan
{plan_text}

## Current train.py (COMPLETE FILE — modify and return the full file)
```python
{train_py}
```

Apply the changes described in the plan. Return the COMPLETE modified train.py file.
Do not add any new import dependencies beyond what is already imported.
Do not change the TIME_BUDGET, evaluation logic, or output format.
Return ONLY the Python code, no markdown fences, no explanation."""


def extract_hyperparams(train_py: str) -> str:
    """Extract the hyperparameters section from train.py for context-limited prompts."""
    lines = train_py.split("\n")
    in_section = False
    result = []
    for line in lines:
        if "Hyperparameters" in line and "---" in line:
            in_section = True
            continue
        if in_section and "---" in line:
            break
        if in_section:
            result.append(line)
    if not result:
        # Fallback: look for the ALL_CAPS assignment block
        for line in lines:
            stripped = line.strip()
            if stripped and "=" in stripped and stripped[0].isupper() and stripped.split("=")[0].strip().isupper():
                result.append(line)
    return "\n".join(result).strip()


def format_results_history(results_tsv: str, max_recent: int = MAX_RECENT_RESULTS) -> str:
    """Format results history with summarized older experiments + recent raw results.

    If there are more than max_recent experiments, older ones are summarized into
    batches of SUMMARY_BATCH_SIZE. The most recent max_recent are shown in full.
    """
    lines = results_tsv.strip().split("\n")
    if len(lines) <= 1:
        return "No experiments have been run yet."

    header = lines[0]
    rows = lines[1:]
    total = len(rows)

    if total <= max_recent:
        return header + "\n" + "\n".join(rows)

    # Split into older (to summarize) and recent (show raw)
    older = rows[: total - max_recent]
    recent = rows[total - max_recent :]

    # Summarize older experiments in batches
    summaries = []
    for batch_start in range(0, len(older), SUMMARY_BATCH_SIZE):
        batch = older[batch_start : batch_start + SUMMARY_BATCH_SIZE]
        keeps = []
        discards = []
        crashes = []
        for row in batch:
            parts = row.split("\t")
            if len(parts) >= 5:
                status = parts[3]
                desc = parts[4]
                bpb = parts[1]
                if status == "keep":
                    keeps.append(f"{desc} (val_bpb={bpb})")
                elif status == "discard":
                    discards.append(desc)
                elif status == "crash":
                    crashes.append(desc)

        batch_num = batch_start // SUMMARY_BATCH_SIZE + 1
        summary_parts = [f"[Batch {batch_num}, experiments {batch_start + 1}-{batch_start + len(batch)}]"]
        if keeps:
            summary_parts.append(f"  Kept: {'; '.join(keeps)}")
        if discards:
            summary_parts.append(f"  Discarded ({len(discards)}): {'; '.join(discards[:3])}" + (" ..." if len(discards) > 3 else ""))
        if crashes:
            summary_parts.append(f"  Crashed ({len(crashes)}): {'; '.join(crashes[:2])}" + (" ..." if len(crashes) > 2 else ""))

        summaries.append("\n".join(summary_parts))

    result = "## Experiment History Summary\n"
    result += "\n\n".join(summaries)
    result += f"\n\n## Recent Experiments (last {max_recent})\n"
    result += header + "\n" + "\n".join(recent)
    return result


def format_papers_summary(papers: list) -> str:
    """Format a list of Paper objects into a concise summary for prompts."""
    if not papers:
        return "No relevant papers found."
    summaries = []
    for i, paper in enumerate(papers, 1):
        summaries.append(f"{i}. [{paper.arxiv_id}] {paper.title}\n   {paper.abstract[:300]}...")
    return "\n\n".join(summaries)
