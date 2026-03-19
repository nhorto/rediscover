"""Council service: multi-agent deliberation pipeline for experiment proposals."""

from datetime import datetime

from src.domains.council.config import (
    CRITIQUE_PROMPT,
    CRITIQUE_SYSTEM,
    IMPLEMENT_FIX_PROMPT,
    IMPLEMENT_FIX_SYSTEM,
    IMPLEMENT_PROMPT,
    IMPLEMENT_SYSTEM,
    MAX_PAPERS_PER_QUERY,
    MAX_SEARCH_QUERIES,
    PROPOSE_PROMPT,
    PROPOSE_SYSTEM,
    REFINE_PROMPT,
    REFINE_SYSTEM,
    SCAN_PROMPT,
    SCAN_SYSTEM,
)
from src.domains.council.helpers import (
    extract_hyperparams,
    format_papers_summary,
    format_results_history,
)
from src.domains.council.parsing import (
    clean_code_response,
    extract_field,
    extract_list,
    parse_search_queries,
)
from src.domains.council.types import (
    CouncilResult,
    Critique,
    ExperimentPlan,
    Proposal,
    SearchQuery,
)
from src.domains.literature.service import LiteratureService
from src.providers.llm import LLMProvider
from src.types import Paper
# Zone extraction no longer used — full-file approach (Option 2)
# from src.utils.code_splicing import extract_modifiable_zone, get_frozen_context, replace_modifiable_zone


def _log_step(log: list[dict], step: str, response) -> None:
    """Append a standard log entry for an LLM call."""
    log.append({"step": step, "model": response.model, "input_tokens": response.input_tokens,
                "output_tokens": response.output_tokens, "cost": response.cost,
                "timestamp": datetime.now().isoformat()})


class CouncilService:
    """Orchestrates the research council: scan → propose → critique → refine → implement."""

    def __init__(self, llm: LLMProvider, literature: LiteratureService | None = None):
        self.llm = llm
        self.literature = literature

    def run_council(
        self,
        train_py: str,
        results_tsv: str,
        program_md: str,
    ) -> CouncilResult:
        """Run a full council deliberation cycle. Returns a CouncilResult with new train.py."""
        log: list[dict] = []

        # 1. SCAN — generate search queries and retrieve papers
        search_queries, papers = self._scan(program_md, log)

        # 2. PROPOSE — form hypothesis based on literature and results
        proposal = self._propose(program_md, papers, results_tsv, train_py, search_queries, log)

        # 3. CRITIQUE — independent review of the proposal
        critique = self._critique(proposal, results_tsv, log)

        # 4. REFINE — address critique, produce implementation plan
        plan = self._refine(proposal, critique, train_py, log)

        # 5. IMPLEMENT — write the actual code (gets plan + full train.py, NO research context)
        new_train_py, implement_raw = self._implement(plan, train_py, log)

        return CouncilResult(
            proposal=proposal,
            critique=critique,
            plan=plan,
            new_train_py=new_train_py,
            implement_raw_response=implement_raw,
            log=log,
        )

    def _scan(self, program_md: str, log: list[dict]) -> tuple[list[SearchQuery], list[Paper]]:
        """Generate search queries and retrieve relevant papers."""
        prompt = SCAN_PROMPT.format(program_md=program_md, max_queries=MAX_SEARCH_QUERIES)
        response = self.llm.complete(role="scan", prompt=prompt, system=SCAN_SYSTEM)
        _log_step(log, "scan", response)

        # Parse search queries from response
        queries = parse_search_queries(response.content)

        # Retrieve papers for each query (dedup by arxiv_id)
        all_papers: list[Paper] = []
        seen_ids: set[str] = set()
        if self.literature:
            for sq in queries:
                results = self.literature.search(sq.query, n_results=MAX_PAPERS_PER_QUERY)
                for r in results:
                    if r.paper.arxiv_id not in seen_ids:
                        seen_ids.add(r.paper.arxiv_id)
                        all_papers.append(r.paper)

        log.append({
            "step": "scan_results",
            "queries": [{"query": sq.query, "rationale": sq.rationale} for sq in queries],
            "papers_found": len(all_papers),
        })

        return queries, all_papers

    def _propose(
        self,
        program_md: str,
        papers: list[Paper],
        results_tsv: str,
        train_py: str,
        search_queries: list[SearchQuery],
        log: list[dict],
    ) -> Proposal:
        """Propose a hypothesis based on literature and past results."""
        prompt = PROPOSE_PROMPT.format(
            program_md=program_md,
            papers_summary=format_papers_summary(papers),
            results_history=format_results_history(results_tsv),
            hyperparams=extract_hyperparams(train_py),
        )
        response = self.llm.complete(role="propose", prompt=prompt, system=PROPOSE_SYSTEM)
        _log_step(log, "propose", response)

        # Parse proposal
        hypothesis = extract_field(response.content, "HYPOTHESIS")
        approach = extract_field(response.content, "APPROACH")
        expected_impact = extract_field(response.content, "EXPECTED_IMPACT")

        return Proposal(
            hypothesis=hypothesis,
            approach=approach,
            expected_impact=expected_impact,
            search_queries=search_queries,
            papers_found=papers,
            raw_response=response.content,
        )

    def _critique(self, proposal: Proposal, results_tsv: str, log: list[dict]) -> Critique:
        """Independent critique of the proposal."""
        proposal_text = (
            f"HYPOTHESIS: {proposal.hypothesis}\n"
            f"APPROACH: {proposal.approach}\n"
            f"EXPECTED_IMPACT: {proposal.expected_impact}"
        )
        prompt = CRITIQUE_PROMPT.format(
            proposal_text=proposal_text,
            results_history=format_results_history(results_tsv, max_recent=5),
        )
        response = self.llm.complete(role="critique", prompt=prompt, system=CRITIQUE_SYSTEM)
        _log_step(log, "critique", response)

        concerns = extract_list(response.content, "CONCERNS")
        suggestions = extract_list(response.content, "SUGGESTIONS")
        overall = extract_field(response.content, "OVERALL")

        return Critique(
            concerns=concerns,
            suggestions=suggestions,
            overall_assessment=overall,
            raw_response=response.content,
        )

    def _refine(
        self, proposal: Proposal, critique: Critique, train_py: str, log: list[dict]
    ) -> ExperimentPlan:
        """Refine the proposal into an implementation plan."""
        proposal_text = (
            f"HYPOTHESIS: {proposal.hypothesis}\n"
            f"APPROACH: {proposal.approach}\n"
            f"EXPECTED_IMPACT: {proposal.expected_impact}"
        )
        critique_text = (
            "CONCERNS:\n" + "\n".join(f"- {c}" for c in critique.concerns) + "\n\n"
            "SUGGESTIONS:\n" + "\n".join(f"- {s}" for s in critique.suggestions) + "\n\n"
            f"OVERALL: {critique.overall_assessment}"
        )
        prompt = REFINE_PROMPT.format(
            proposal_text=proposal_text,
            critique_text=critique_text,
            hyperparams=extract_hyperparams(train_py),
        )
        response = self.llm.complete(role="refine", prompt=prompt, system=REFINE_SYSTEM)
        _log_step(log, "refine", response)

        description = extract_field(response.content, "DESCRIPTION")
        code_changes = extract_field(response.content, "CODE_CHANGES")
        addresses = extract_list(response.content, "ADDRESSES")

        return ExperimentPlan(
            description=description,
            code_changes_summary=code_changes,
            addresses_concerns=addresses,
            raw_response=response.content,
        )

    def _implement(self, plan: ExperimentPlan, train_py: str, log: list[dict]) -> tuple[str, str]:
        """Write code changes by sending the FULL train.py to the model.

        The model sees the complete file (including optimizer, training loop, etc.)
        and returns the complete modified file. This gives it full context to avoid
        breaking invariants like MuonAdamW parameter constraints.
        """
        plan_text = (
            f"DESCRIPTION: {plan.description}\n"
            f"CODE_CHANGES: {plan.code_changes_summary}"
        )

        prompt = IMPLEMENT_PROMPT.format(
            plan_text=plan_text,
            full_train_py=train_py,
        )
        response = self.llm.complete(
            role="implement",
            prompt=prompt,
            system=IMPLEMENT_SYSTEM,
            temperature=0.3,
            max_tokens=8192,  # Full file is ~700 lines
        )

        _log_step(log, "implement", response)

        new_code = clean_code_response(response.content)
        return new_code, response.content

    def fix_code(self, broken_code: str, error_text: str, log: list[dict]) -> tuple[str, str]:
        """Fix broken code by sending the error + full file to the model."""
        error_truncated = error_text[:2000]

        prompt = IMPLEMENT_FIX_PROMPT.format(
            error_text=error_truncated, train_py=broken_code,
        )
        response = self.llm.complete(
            role="implement", prompt=prompt, system=IMPLEMENT_FIX_SYSTEM,
            temperature=0.2, max_tokens=8192,
        )
        _log_step(log, "implement_fix", response)

        return clean_code_response(response.content), response.content

