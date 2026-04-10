"""Tests for src.core.critique.

All API calls are mocked. No real network traffic should ever happen.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.core.claim import ExperimentResult
from src.core import critique as critique_mod


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_results() -> list[ExperimentResult]:
    return [
        ExperimentResult(
            claim_id="claim_1",
            model_key="llama_1b",
            paper_id="emotions",
            metrics={"probe_accuracy": 0.73, "per_concept": {"joy": 0.81}},
            success=True,
            metadata={"layer": 12, "seed": 0},
        ),
        ExperimentResult(
            claim_id="claim_2",
            model_key="llama_1b",
            paper_id="emotions",
            metrics={"diagonal_dominance": 0.55},
            success=False,
            metadata={"layer": 12},
        ),
    ]


@pytest.fixture
def sample_sanity() -> list[dict]:
    return [
        {"name": "stimulus_count", "severity": "info", "message": "100 stimuli loaded"},
        {"name": "probe_chance", "severity": "warn", "message": "chance is only 50%"},
    ]


@pytest.fixture
def sample_paper_config() -> dict:
    return {
        "paper": {
            "id": "emotions",
            "title": "Emotions in LMs",
            "authors": "Sofroniew et al.",
        },
        "claims": [
            {"id": "claim_1", "experiment_type": "probe_classification"},
        ],
    }


def _mk_anthropic_mock(text: str) -> MagicMock:
    """Build a mocked anthropic.Anthropic client that returns `text` on messages.create."""
    client = MagicMock()
    fake_block = MagicMock()
    fake_block.text = text
    fake_resp = MagicMock()
    fake_resp.content = [fake_block]
    client.messages.create.return_value = fake_resp
    return client


# ---------------------------------------------------------------------------
# _critique_with_claude
# ---------------------------------------------------------------------------

def test_critique_with_claude_returns_mocked_text(
    sample_results, sample_sanity, sample_paper_config
):
    fake_client = _mk_anthropic_mock("# Mocked critique\n\nTop concern: layer choice.")
    with patch("anthropic.Anthropic", return_value=fake_client) as mock_ctor:
        out = critique_mod._critique_with_claude(
            paper_text="This is the paper text.",
            paper_config=sample_paper_config,
            results=sample_results,
            sanity_reports=sample_sanity,
        )

    assert "Mocked critique" in out
    mock_ctor.assert_called_once()
    # Verify the create call got a system prompt and a user message
    call_kwargs = fake_client.messages.create.call_args.kwargs
    assert "system" in call_kwargs
    assert "Emotions in LMs" in call_kwargs["system"]
    assert call_kwargs["max_tokens"] == critique_mod.MAX_OUTPUT_TOKENS
    # User message includes the results JSON
    user_msg = call_kwargs["messages"][0]["content"]
    assert "probe_accuracy" in user_msg
    assert "sanity_reports" in user_msg.lower() or "Sanity reports" in user_msg


def test_critique_with_claude_returns_placeholder_on_error(
    sample_results, sample_sanity, sample_paper_config
):
    client = MagicMock()
    client.messages.create.side_effect = RuntimeError("boom")
    with patch("anthropic.Anthropic", return_value=client):
        out = critique_mod._critique_with_claude(
            paper_text="",
            paper_config=sample_paper_config,
            results=sample_results,
            sanity_reports=sample_sanity,
        )
    assert "Failed" in out
    assert "boom" in out


# ---------------------------------------------------------------------------
# _critique_with_chatgpt
# ---------------------------------------------------------------------------

def test_critique_with_chatgpt_no_api_key_returns_none(
    monkeypatch, sample_results, sample_sanity, sample_paper_config
):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    out = critique_mod._critique_with_chatgpt(
        paper_text="",
        paper_config=sample_paper_config,
        results=sample_results,
        sanity_reports=sample_sanity,
    )
    assert out is None


def test_critique_with_chatgpt_with_mocked_client(
    monkeypatch, sample_results, sample_sanity, sample_paper_config
):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-fake")
    # Build a fake openai.OpenAI client
    fake_msg = MagicMock()
    fake_msg.content = "# ChatGPT mocked critique"
    fake_choice = MagicMock()
    fake_choice.message = fake_msg
    fake_resp = MagicMock()
    fake_resp.choices = [fake_choice]

    fake_client = MagicMock()
    fake_client.chat.completions.create.return_value = fake_resp

    # The function imports `openai` (module) and then `from openai import OpenAI`.
    # We need to provide both via a mocked openai module.
    fake_openai_mod = MagicMock()
    fake_openai_mod.OpenAI = MagicMock(return_value=fake_client)

    with patch.dict("sys.modules", {"openai": fake_openai_mod}):
        out = critique_mod._critique_with_chatgpt(
            paper_text="",
            paper_config=sample_paper_config,
            results=sample_results,
            sanity_reports=sample_sanity,
        )

    assert out is not None
    assert "ChatGPT mocked critique" in out


# ---------------------------------------------------------------------------
# _evaluator
# ---------------------------------------------------------------------------

def test_evaluator_parses_json(sample_results, sample_sanity):
    fake_json = json.dumps(
        {
            "top_concerns": [
                {
                    "concern": "Probe may be fitting style not content",
                    "evidence_for": "High accuracy on narrow set",
                    "evidence_against": "Generalizes to held-out",
                    "severity": "medium",
                    "next_step": "Run confound stimuli",
                    "estimated_effort": "cheap",
                }
            ],
            "consensus_strengths": ["pipeline runs end-to-end"],
            "consensus_weaknesses": ["small n"],
        }
    )
    fake_client = _mk_anthropic_mock(fake_json)
    with patch("anthropic.Anthropic", return_value=fake_client):
        out = critique_mod._evaluator(
            claude_text="some claude text",
            chatgpt_text="some chatgpt text",
            results=sample_results,
            sanity_reports=sample_sanity,
        )
    assert "top_concerns" in out
    assert len(out["top_concerns"]) == 1
    assert out["top_concerns"][0]["severity"] == "medium"
    assert "pipeline runs end-to-end" in out["consensus_strengths"]


def test_evaluator_handles_fenced_json(sample_results, sample_sanity):
    fake_output = (
        "```json\n"
        + json.dumps(
            {
                "top_concerns": [],
                "consensus_strengths": ["ok"],
                "consensus_weaknesses": [],
            }
        )
        + "\n```"
    )
    fake_client = _mk_anthropic_mock(fake_output)
    with patch("anthropic.Anthropic", return_value=fake_client):
        out = critique_mod._evaluator("c", "g", sample_results, sample_sanity)
    assert out.get("consensus_strengths") == ["ok"]


def test_evaluator_falls_back_on_unparseable(sample_results, sample_sanity):
    fake_client = _mk_anthropic_mock("not json at all")
    with patch("anthropic.Anthropic", return_value=fake_client):
        out = critique_mod._evaluator("c", None, sample_results, sample_sanity)
    assert "raw" in out


# ---------------------------------------------------------------------------
# run_critique_pass end-to-end
# ---------------------------------------------------------------------------

def test_run_critique_pass_writes_all_files(
    tmp_path: Path,
    monkeypatch,
    sample_results,
    sample_sanity,
    sample_paper_config,
):
    output_dir = tmp_path / "critiques"

    # Claude: different text per call. First = critic, second = evaluator.
    eval_json = json.dumps(
        {
            "top_concerns": [
                {
                    "concern": "Layer selection not scanned",
                    "evidence_for": "Only layer 12 reported",
                    "evidence_against": "Paper mentions layer 12 as best",
                    "severity": "high",
                    "next_step": "Scan all layers",
                    "estimated_effort": "cheap",
                }
            ],
            "consensus_strengths": ["cross-model infra works"],
            "consensus_weaknesses": ["single layer only"],
        }
    )

    # Mock Claude such that messages.create returns different content each call
    call_counter = {"i": 0}

    def fake_create(**kwargs):
        call_counter["i"] += 1
        block = MagicMock()
        if call_counter["i"] == 1:
            block.text = "# Claude critique (mocked)\n\nConcern: layer."
        else:
            block.text = eval_json
        resp = MagicMock()
        resp.content = [block]
        return resp

    fake_claude = MagicMock()
    fake_claude.messages.create.side_effect = fake_create

    # ChatGPT: mock the openai module + client
    fake_msg = MagicMock()
    fake_msg.content = "# ChatGPT critique (mocked)\n\nConcern: stimulus length."
    fake_choice = MagicMock()
    fake_choice.message = fake_msg
    fake_resp = MagicMock()
    fake_resp.choices = [fake_choice]
    fake_openai_client = MagicMock()
    fake_openai_client.chat.completions.create.return_value = fake_resp
    fake_openai_mod = MagicMock()
    fake_openai_mod.OpenAI = MagicMock(return_value=fake_openai_client)

    monkeypatch.setenv("OPENAI_API_KEY", "sk-fake")

    with patch("anthropic.Anthropic", return_value=fake_claude), patch.dict(
        "sys.modules", {"openai": fake_openai_mod}
    ):
        out = critique_mod.run_critique_pass(
            paper_id="emotions",
            model_key="llama_1b",
            paper_text="Paper text body.",
            paper_config=sample_paper_config,
            results=sample_results,
            sanity_reports=sample_sanity,
            output_dir=output_dir,
        )

    # All four files present
    assert (output_dir / "claude_critique.md").exists()
    assert (output_dir / "chatgpt_critique.md").exists()
    assert (output_dir / "evaluator.json").exists()
    assert (output_dir / "evaluator.md").exists()

    # Contents look right
    assert "Claude critique (mocked)" in (output_dir / "claude_critique.md").read_text()
    assert "ChatGPT critique (mocked)" in (output_dir / "chatgpt_critique.md").read_text()

    parsed = json.loads((output_dir / "evaluator.json").read_text())
    assert parsed["top_concerns"][0]["severity"] == "high"
    assert parsed["_provenance"]["paper_id"] == "emotions"
    assert parsed["_provenance"]["model_key"] == "llama_1b"
    assert parsed["_provenance"]["chatgpt_available"] is True

    md = (output_dir / "evaluator.md").read_text()
    assert "Layer selection not scanned" in md
    assert "cross-model infra works" in md

    # Return value is the parsed dict
    assert out["top_concerns"][0]["severity"] == "high"


def test_run_critique_pass_no_chatgpt(
    tmp_path: Path,
    monkeypatch,
    sample_results,
    sample_sanity,
    sample_paper_config,
):
    """Without OPENAI_API_KEY, chatgpt_critique.md should not be written."""
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    eval_json = json.dumps(
        {"top_concerns": [], "consensus_strengths": [], "consensus_weaknesses": []}
    )

    call_counter = {"i": 0}

    def fake_create(**kwargs):
        call_counter["i"] += 1
        block = MagicMock()
        block.text = "# critic" if call_counter["i"] == 1 else eval_json
        resp = MagicMock()
        resp.content = [block]
        return resp

    fake_claude = MagicMock()
    fake_claude.messages.create.side_effect = fake_create

    output_dir = tmp_path / "critiques"
    with patch("anthropic.Anthropic", return_value=fake_claude):
        out = critique_mod.run_critique_pass(
            paper_id="emotions",
            model_key="llama_1b",
            paper_text="",
            paper_config=sample_paper_config,
            results=sample_results,
            sanity_reports=sample_sanity,
            output_dir=output_dir,
        )

    assert (output_dir / "claude_critique.md").exists()
    assert not (output_dir / "chatgpt_critique.md").exists()
    assert (output_dir / "evaluator.json").exists()
    parsed = json.loads((output_dir / "evaluator.json").read_text())
    assert parsed["_provenance"]["chatgpt_available"] is False
    assert out["_provenance"]["chatgpt_available"] is False
