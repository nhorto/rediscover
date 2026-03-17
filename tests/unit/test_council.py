"""Tests for council domain (mocked LLM calls — no real API calls)."""

from unittest.mock import MagicMock, patch

import pytest

from src.domains.council.config import extract_hyperparams, format_papers_summary, format_results_history
from src.domains.council.parsing import clean_code_response, extract_field, extract_list, parse_search_queries
from src.domains.council.service import CouncilService
from src.domains.council.types import CouncilResult, Critique, ExperimentPlan, Proposal, SearchQuery
from src.providers.llm import LLMProvider, LLMResponse
from src.types import Paper


def _make_llm_response(content="test response", role="test", model="test-model"):
    return LLMResponse(content=content, model=model, role=role, input_tokens=100, output_tokens=50, cost=0.001)


def _make_paper(arxiv_id="2301.00001", title="Test Paper"):
    return Paper(
        arxiv_id=arxiv_id, title=title, abstract="Abstract about attention mechanisms.",
        authors=["Author One"], published="2023-01-15T00:00:00+00:00",
        categories=["cs.LG"], primary_category="cs.LG", pdf_url="",
    )


SAMPLE_TRAIN_PY = """import torch
import torch.nn as nn

from prepare import MAX_SEQ_LEN

@dataclass
class GPTConfig:
    n_head: int = 2
    n_embd: int = 256
    n_kv_head: int = 2

def norm(x):
    return x

def has_ve(layer_idx, n_layer):
    return layer_idx % 2 == 0

def apply_rotary_emb(x, cos, sin):
    return x

class CausalSelfAttention(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.c_q = nn.Linear(config.n_embd, config.n_embd)

    def forward(self, x, ve, cos_sin, window_size):
        return self.c_q(x)

class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
    def forward(self, x):
        return x

# ---------------------------------------------------------------------------
# Hyperparameters (agents modify these)
# ---------------------------------------------------------------------------

ASPECT_RATIO = 64
HEAD_DIM = 128
WINDOW_PATTERN = "L"
TOTAL_BATCH_SIZE = 2**16
DEPTH = 4
DEVICE_BATCH_SIZE = 16

model = None
"""

SAMPLE_RESULTS_TSV = """commit\tval_bpb\tmemory_gb\tstatus\tdescription
baseline\t1.763539\t0.0\tkeep\tBaseline: 11.5M params
abc1234\t1.750000\t0.0\tkeep\tIncreased depth to 6
def5678\t1.780000\t0.0\tdiscard\tReduced head dim to 64
"""


@pytest.mark.unit
class TestConfigHelpers:
    def test_extract_hyperparams(self):
        result = extract_hyperparams(SAMPLE_TRAIN_PY)
        assert "ASPECT_RATIO = 64" in result
        assert "DEPTH = 4" in result
        assert "model = build_model()" not in result

    def test_extract_hyperparams_fallback(self):
        """Falls back to ALL_CAPS assignments if no section header found."""
        code = "DEPTH = 4\nBATCH_SIZE = 16\nmodel = GPT()\n"
        result = extract_hyperparams(code)
        assert "DEPTH = 4" in result
        assert "BATCH_SIZE = 16" in result

    def test_format_results_history(self):
        result = format_results_history(SAMPLE_RESULTS_TSV, max_recent=2)
        assert "abc1234" in result or "commit" in result
        assert "def5678" in result

    def test_format_results_history_empty(self):
        result = format_results_history("commit\tval_bpb\n")
        assert "No experiments" in result

    def test_format_papers_summary(self):
        papers = [_make_paper("2301.00001", "Paper One"), _make_paper("2301.00002", "Paper Two")]
        result = format_papers_summary(papers)
        assert "Paper One" in result
        assert "Paper Two" in result
        assert "2301.00001" in result

    def test_format_papers_summary_empty(self):
        assert "No relevant papers" in format_papers_summary([])


@pytest.mark.unit
class TestCouncilTypes:
    def test_search_query(self):
        sq = SearchQuery(query="attention efficiency", rationale="core topic")
        assert sq.query == "attention efficiency"

    def test_proposal(self):
        p = Proposal(
            hypothesis="test", approach="test", expected_impact="test",
            search_queries=[], papers_found=[], raw_response="raw",
        )
        assert p.hypothesis == "test"

    def test_council_result(self):
        cr = CouncilResult(
            proposal=Proposal("h", "a", "e", [], [], "raw"),
            critique=Critique([], [], "ok", "raw"),
            plan=ExperimentPlan("d", "c", [], "raw"),
            new_train_py="print('hello')",
            implement_raw_response="raw",
        )
        assert cr.new_train_py == "print('hello')"


@pytest.mark.unit
class TestCouncilServiceParsing:
    def test_parse_search_queries(self):
        text = """QUERY: attention mechanism efficiency
RATIONALE: core topic
---
QUERY: KV cache compression
RATIONALE: memory optimization"""
        queries = parse_search_queries(text)
        assert len(queries) == 2
        assert queries[0].query == "attention mechanism efficiency"
        assert queries[1].query == "KV cache compression"

    def test_parse_search_queries_fallback(self):
        """Falls back to line-by-line if no structured format."""
        text = "attention mechanism improvements\nKV cache optimization\ntransformer efficiency"
        queries = parse_search_queries(text)
        assert len(queries) == 3

    def test_extract_field(self):
        text = "HYPOTHESIS: Lower learning rate will help\nAPPROACH: Reduce from 0.04 to 0.02"
        result = extract_field(text, "HYPOTHESIS")
        assert result == "Lower learning rate will help"

    def test_extract_list(self):
        text = "CONCERNS:\n- Too aggressive\n- May not converge\n\nOVERALL: Risky"
        result = extract_list(text, "CONCERNS")
        assert len(result) == 2
        assert "Too aggressive" in result[0]

    def test_clean_code_response(self):
        text = "```python\nprint('hello')\n```"
        assert clean_code_response(text) == "print('hello')"

    def test_clean_code_response_no_fences(self):
        text = "print('hello')"
        assert clean_code_response(text) == "print('hello')"


@pytest.mark.unit
class TestCouncilServicePipeline:
    @patch.object(LLMProvider, "complete")
    def test_full_pipeline(self, mock_complete):
        """Test the full council pipeline with mocked LLM responses."""
        scan_response = _make_llm_response(
            "QUERY: attention efficiency\nRATIONALE: core topic\n---\n"
            "QUERY: KV cache\nRATIONALE: memory",
            role="scan",
        )
        propose_response = _make_llm_response(
            "HYPOTHESIS: Reduce head dim from 128 to 64\n"
            "APPROACH: Change HEAD_DIM constant\n"
            "EXPECTED_IMPACT: Faster attention with less memory",
            role="propose",
        )
        critique_response = _make_llm_response(
            "CONCERNS:\n- May lose capacity\n- Needs testing\n\n"
            "SUGGESTIONS:\n- Try 96 first\n\n"
            "OVERALL: Worth trying but be conservative",
            role="critique",
        )
        refine_response = _make_llm_response(
            "DESCRIPTION: Reduce HEAD_DIM to 96 as compromise\n"
            "CODE_CHANGES: Change HEAD_DIM = 128 to HEAD_DIM = 96\n"
            "ADDRESSES:\n- Using 96 instead of 64 to preserve capacity",
            role="refine",
        )
        implement_response = _make_llm_response(
            SAMPLE_TRAIN_PY.replace("HEAD_DIM = 128", "HEAD_DIM = 96"),
            role="implement",
        )

        mock_complete.side_effect = [
            scan_response, propose_response, critique_response, refine_response, implement_response,
        ]

        llm = LLMProvider()
        council = CouncilService(llm=llm, literature=None)
        result = council.run_council(SAMPLE_TRAIN_PY, SAMPLE_RESULTS_TSV, "Improve attention")

        assert isinstance(result, CouncilResult)
        assert result.proposal.hypothesis == "Reduce head dim from 128 to 64"
        assert len(result.critique.concerns) == 2
        assert "HEAD_DIM = 96" in result.plan.code_changes_summary
        assert "HEAD_DIM = 96" in result.new_train_py
        assert len(result.log) == 6  # scan, scan_results, propose, critique, refine, implement
        assert mock_complete.call_count == 5

    @patch.object(LLMProvider, "complete")
    def test_scan_with_literature(self, mock_complete):
        """Test that scan step retrieves papers from literature service."""
        mock_complete.return_value = _make_llm_response(
            "QUERY: attention efficiency\nRATIONALE: core\n", role="scan",
        )

        mock_literature = MagicMock()
        mock_search_result = MagicMock()
        mock_search_result.paper = _make_paper()
        mock_literature.search.return_value = [mock_search_result]

        llm = LLMProvider()
        council = CouncilService(llm=llm, literature=mock_literature)
        queries, papers = council._scan("Improve attention", [])

        assert len(queries) >= 1
        assert len(papers) == 1
        mock_literature.search.assert_called_once()

    @patch.object(LLMProvider, "complete")
    def test_implement_gets_low_temperature(self, mock_complete):
        """Implement step should use low temperature for code generation."""
        mock_complete.return_value = _make_llm_response("print('hello')", role="implement")

        llm = LLMProvider()
        council = CouncilService(llm=llm)
        plan = ExperimentPlan("test", "test changes", [], "raw")
        council._implement(plan, SAMPLE_TRAIN_PY, [])

        call_kwargs = mock_complete.call_args.kwargs
        assert call_kwargs["temperature"] == 0.3
        assert call_kwargs["max_tokens"] == 4096  # Zone output, not full file
