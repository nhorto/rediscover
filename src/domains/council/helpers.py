"""Helper functions for council prompt formatting and context extraction."""

from src.domains.council.config import MAX_RECENT_RESULTS, SUMMARY_BATCH_SIZE


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


def extract_code_structure(train_py: str) -> str:
    """Extract a structural summary of train.py for the implement prompt.

    Gives the model a map of the file without requiring it to parse 500 lines.
    """
    lines = train_py.split("\n")
    imports = []
    classes = []
    functions = []
    constants = []

    for i, line in enumerate(lines):
        stripped = line.strip()
        if stripped.startswith("import ") or stripped.startswith("from "):
            imports.append(stripped)
        elif stripped.startswith("class "):
            classes.append(f"  Line {i + 1}: {stripped.split(':')[0]}")
        elif stripped.startswith("def ") and not line.startswith(" "):
            functions.append(f"  Line {i + 1}: {stripped.split(':')[0]}")
        elif "=" in stripped and stripped[0].isupper() and not stripped.startswith("#"):
            name = stripped.split("=")[0].strip()
            if name.isupper():
                constants.append(f"  {stripped}")

    result = "### Imports\n" + "\n".join(imports[:20])
    result += "\n\n### Classes (do not rename or remove)\n" + "\n".join(classes)
    result += "\n\n### Top-level Functions (do not rename or remove)\n" + "\n".join(functions)
    result += "\n\n### Hyperparameter Constants\n" + "\n".join(constants)
    result += f"\n\n### File length: {len(lines)} lines"
    return result


def format_papers_summary(papers: list) -> str:
    """Format a list of Paper objects into a concise summary for prompts."""
    if not papers:
        return "No relevant papers found."
    summaries = []
    for i, paper in enumerate(papers, 1):
        summaries.append(f"{i}. [{paper.arxiv_id}] {paper.title}\n   {paper.abstract[:300]}...")
    return "\n\n".join(summaries)
