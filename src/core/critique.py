"""Critique agents for automated review of replication results.

At the end of each model's claim sweep, this module spawns critique agents
(Claude + optionally ChatGPT) to review the results against the paper, then
runs an evaluator pass to prioritize the concerns into an actionable list.

The user previously performed this loop by hand -- pasting results into
Claude.ai and ChatGPT separately, then synthesizing the critiques. This
module automates that flow.

Outputs (written to ``output_dir``):
    - ``claude_critique.md``     Raw Claude markdown critique
    - ``chatgpt_critique.md``    Raw ChatGPT markdown critique (if available)
    - ``evaluator.json``         Structured top-concerns dict
    - ``evaluator.md``           Human-readable rendering of evaluator.json

All file writes are atomic (.tmp then rename) so an interruption mid-write
cannot corrupt a previous critique.

Usage:
    from src.core.critique import run_critique_pass

    run_critique_pass(
        paper_id="emotions",
        model_key="llama_1b",
        paper_text=paper_md,
        paper_config=cfg_dict,
        results=list_of_experiment_results,
        sanity_reports=list_of_sanity_dicts,
        output_dir=Path("results/emotions/llama_1b/critiques"),
    )
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

import yaml

from src.core.claim import ExperimentResult

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

CLAUDE_MODELS_TO_TRY = [
    "claude-sonnet-4-5-20250929",
    "claude-opus-4-1-20250805",
]

CHATGPT_MODELS_TO_TRY = [
    "gpt-5",
    "gpt-4o-2024-11-20",
]

MAX_PAPER_CHARS = 30_000
MAX_OUTPUT_TOKENS = 1500
API_TIMEOUT_SECONDS = 60.0


# ---------------------------------------------------------------------------
# Serialization helpers
# ---------------------------------------------------------------------------

def _serialize_result(result: Any) -> dict[str, Any]:
    """Convert an ExperimentResult (or dict) to a JSON-safe dict.

    Drops any fields that look like tensors or activations -- we only want
    metrics + metadata in the critique payload.
    """
    if isinstance(result, ExperimentResult):
        return {
            "claim_id": result.claim_id,
            "model_key": result.model_key,
            "paper_id": result.paper_id,
            "metrics": result.metrics,
            "success": result.success,
            "metadata": result.metadata,
        }
    if is_dataclass(result):
        return asdict(result)
    if isinstance(result, dict):
        return result
    return {"repr": repr(result)}


def _build_payload(
    paper_text: str,
    paper_config: dict,
    results: list[ExperimentResult],
    sanity_reports: list[dict],
) -> dict[str, Any]:
    """Assemble a JSON-safe payload of everything a critic needs to see."""
    truncated_paper = paper_text[:MAX_PAPER_CHARS]
    if len(paper_text) > MAX_PAPER_CHARS:
        truncated_paper += (
            f"\n\n[... paper truncated at {MAX_PAPER_CHARS} chars; "
            f"original was {len(paper_text)} chars ...]"
        )
    return {
        "paper_text": truncated_paper,
        "paper_config_yaml": yaml.safe_dump(paper_config, sort_keys=False),
        "results": [_serialize_result(r) for r in results],
        "sanity_reports": sanity_reports or [],
    }


def _atomic_write_text(path: Path, content: str) -> None:
    """Atomically write text to a file (.tmp, then rename)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(content)
    tmp.rename(path)


def _atomic_write_json(path: Path, data: Any) -> None:
    """Atomically write JSON to a file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w") as f:
        json.dump(data, f, indent=2, default=str)
    tmp.rename(path)


# ---------------------------------------------------------------------------
# Prompt builders
# ---------------------------------------------------------------------------

def _critic_system_prompt(paper_title: str) -> str:
    return (
        f"You are a senior interpretability researcher reviewing a replication "
        f"of {paper_title}. You have the full paper text, the experiment configs, "
        f"and the raw metric outputs. Identify the top 5 reasons these results "
        f"might be misleading or wrong. Reference specific paper sections. "
        f"Reference any relevant gotcha from GOTCHAS.md if you can. Be specific "
        f"about what to do next. Output markdown with numbered sections for each "
        f"concern."
    )


def _critic_user_prompt(payload: dict[str, Any]) -> str:
    return (
        "Here is the replication context. Produce a markdown critique.\n\n"
        "## Paper text (possibly truncated)\n"
        f"{payload['paper_text']}\n\n"
        "## Paper config (YAML)\n"
        f"```yaml\n{payload['paper_config_yaml']}\n```\n\n"
        "## Experiment results (JSON)\n"
        f"```json\n{json.dumps(payload['results'], indent=2, default=str)}\n```\n\n"
        "## Sanity reports (JSON)\n"
        f"```json\n{json.dumps(payload['sanity_reports'], indent=2, default=str)}\n```\n"
    )


_EVALUATOR_SYSTEM = (
    "You are an evaluator. Read the two critiques below. Identify which concerns "
    "are (a) valid AND (b) actionable. Output a JSON object with exactly this "
    "schema and no extra keys: "
    "{ \"top_concerns\": [ {\"concern\": str, \"evidence_for\": str, "
    "\"evidence_against\": str, \"severity\": \"low\"|\"medium\"|\"high\", "
    "\"next_step\": str, \"estimated_effort\": \"cheap\"|\"medium\"|\"expensive\"} ], "
    "\"consensus_strengths\": [str, ...], \"consensus_weaknesses\": [str, ...] }. "
    "Cap top_concerns at 5. Respond with valid JSON only -- no markdown fences, "
    "no prose before or after."
)


def _evaluator_user_prompt(
    claude_text: str,
    chatgpt_text: str | None,
    results: list[ExperimentResult],
    sanity_reports: list[dict],
) -> str:
    results_json = json.dumps(
        [_serialize_result(r) for r in results], indent=2, default=str
    )
    sanity_json = json.dumps(sanity_reports or [], indent=2, default=str)
    parts = [
        "## Claude critique\n",
        claude_text or "(none)",
        "\n\n## ChatGPT critique\n",
        chatgpt_text or "(ChatGPT critic was not run)",
        "\n\n## Original results\n```json\n",
        results_json,
        "\n```\n\n## Sanity reports\n```json\n",
        sanity_json,
        "\n```\n\nProduce the evaluator JSON now.",
    ]
    return "".join(parts)


# ---------------------------------------------------------------------------
# Individual critics
# ---------------------------------------------------------------------------

def _call_claude(
    system_prompt: str,
    user_prompt: str,
    max_tokens: int = MAX_OUTPUT_TOKENS,
) -> str:
    """Call Claude via the anthropic SDK, trying models in priority order.

    Returns the text content of the first successful response.
    Raises the last exception if all model attempts fail.
    """
    import anthropic  # late import -- keeps tests fast when mocked

    client = anthropic.Anthropic(timeout=API_TIMEOUT_SECONDS)
    last_exc: Exception | None = None
    for model_name in CLAUDE_MODELS_TO_TRY:
        try:
            resp = client.messages.create(
                model=model_name,
                max_tokens=max_tokens,
                system=system_prompt,
                messages=[{"role": "user", "content": user_prompt}],
            )
            # Concatenate text blocks from the response
            chunks: list[str] = []
            for block in resp.content:
                text = getattr(block, "text", None)
                if text:
                    chunks.append(text)
            return "".join(chunks)
        except Exception as exc:
            last_exc = exc
            logger.debug("Claude model %s failed: %s", model_name, exc)
            continue
    raise last_exc if last_exc else RuntimeError("No Claude models attempted")


def _critique_with_claude(
    paper_text: str,
    paper_config: dict,
    results: list[ExperimentResult],
    sanity_reports: list[dict],
) -> str:
    """Produce a Claude critique markdown. Never raises -- returns placeholder on error."""
    paper_title = paper_config.get("paper", {}).get("title", "an interpretability paper")
    system_prompt = _critic_system_prompt(paper_title)
    payload = _build_payload(paper_text, paper_config, results, sanity_reports)
    user_prompt = _critic_user_prompt(payload)
    try:
        return _call_claude(system_prompt, user_prompt)
    except Exception as exc:
        logger.warning("Claude critique failed: %s", exc)
        return f"# Claude critique\n\n(Failed: {exc})\n"


def _critique_with_chatgpt(
    paper_text: str,
    paper_config: dict,
    results: list[ExperimentResult],
    sanity_reports: list[dict],
) -> str | None:
    """Produce a ChatGPT critique markdown, or None if openai is unavailable.

    Silently skips (returns None) if the openai SDK isn't installed or
    OPENAI_API_KEY is not set. On API error, returns a placeholder string.
    """
    if not os.environ.get("OPENAI_API_KEY"):
        logger.info("ChatGPT critic skipped (no API key)")
        return None
    try:
        import openai  # noqa: F401
    except ImportError:
        logger.info("ChatGPT critic skipped (no API key)")
        return None

    try:
        from openai import OpenAI
    except Exception as exc:
        logger.warning("ChatGPT critic import failed: %s", exc)
        return None

    paper_title = paper_config.get("paper", {}).get("title", "an interpretability paper")
    system_prompt = _critic_system_prompt(paper_title)
    payload = _build_payload(paper_text, paper_config, results, sanity_reports)
    user_prompt = _critic_user_prompt(payload)

    client = OpenAI(timeout=API_TIMEOUT_SECONDS)
    last_exc: Exception | None = None
    for model_name in CHATGPT_MODELS_TO_TRY:
        try:
            resp = client.chat.completions.create(
                model=model_name,
                max_tokens=MAX_OUTPUT_TOKENS,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
            )
            return resp.choices[0].message.content or ""
        except Exception as exc:
            last_exc = exc
            logger.debug("ChatGPT model %s failed: %s", model_name, exc)
            continue
    logger.warning("ChatGPT critique failed: %s", last_exc)
    return f"# ChatGPT critique\n\n(Failed: {last_exc})\n"


# ---------------------------------------------------------------------------
# Evaluator
# ---------------------------------------------------------------------------

def _parse_evaluator_json(text: str) -> dict[str, Any]:
    """Parse evaluator output, tolerant of code fences / surrounding prose."""
    if not text:
        return {"raw": ""}
    stripped = text.strip()
    # Strip markdown fences if present
    if stripped.startswith("```"):
        lines = stripped.splitlines()
        # drop first fence line
        lines = lines[1:]
        # drop trailing fence
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        stripped = "\n".join(lines).strip()
    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        # Try to find the outermost {...}
        start = stripped.find("{")
        end = stripped.rfind("}")
        if start != -1 and end > start:
            try:
                return json.loads(stripped[start : end + 1])
            except json.JSONDecodeError:
                pass
    return {"raw": text}


def _evaluator(
    claude_text: str,
    chatgpt_text: str | None,
    results: list[ExperimentResult],
    sanity_reports: list[dict],
) -> dict[str, Any]:
    """Run a single Claude call to merge critiques into a prioritized dict."""
    user_prompt = _evaluator_user_prompt(claude_text, chatgpt_text, results, sanity_reports)
    try:
        raw = _call_claude(_EVALUATOR_SYSTEM, user_prompt)
    except Exception as exc:
        logger.warning("Evaluator call failed: %s", exc)
        return {
            "top_concerns": [],
            "consensus_strengths": [],
            "consensus_weaknesses": [],
            "error": str(exc),
        }
    return _parse_evaluator_json(raw)


def _render_evaluator_markdown(evaluator_dict: dict[str, Any]) -> str:
    """Render the structured evaluator dict as a human-readable markdown table."""
    if "raw" in evaluator_dict and "top_concerns" not in evaluator_dict:
        return f"# Evaluator output (unparsed)\n\n{evaluator_dict.get('raw', '')}\n"

    lines: list[str] = ["# Evaluator summary\n"]

    top = evaluator_dict.get("top_concerns") or []
    if top:
        lines.append("## Top concerns (prioritized)\n")
        lines.append(
            "| # | Concern | Severity | Effort | Next step |\n"
            "|---|---------|----------|--------|-----------|"
        )
        for i, c in enumerate(top, 1):
            concern = str(c.get("concern", "")).replace("|", "\\|")
            sev = str(c.get("severity", "")).replace("|", "\\|")
            eff = str(c.get("estimated_effort", "")).replace("|", "\\|")
            nxt = str(c.get("next_step", "")).replace("|", "\\|")
            lines.append(f"| {i} | {concern} | {sev} | {eff} | {nxt} |")
        lines.append("")

        lines.append("## Evidence per concern\n")
        for i, c in enumerate(top, 1):
            lines.append(f"### {i}. {c.get('concern', '')}")
            lines.append(f"- Evidence for: {c.get('evidence_for', '')}")
            lines.append(f"- Evidence against: {c.get('evidence_against', '')}")
            lines.append("")

    strengths = evaluator_dict.get("consensus_strengths") or []
    if strengths:
        lines.append("## Consensus strengths\n")
        for s in strengths:
            lines.append(f"- {s}")
        lines.append("")

    weaknesses = evaluator_dict.get("consensus_weaknesses") or []
    if weaknesses:
        lines.append("## Consensus weaknesses\n")
        for w in weaknesses:
            lines.append(f"- {w}")
        lines.append("")

    if "error" in evaluator_dict:
        lines.append(f"\n> Evaluator error: {evaluator_dict['error']}\n")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def run_critique_pass(
    paper_id: str,
    model_key: str,
    paper_text: str,
    paper_config: dict,
    results: list[ExperimentResult],
    sanity_reports: list[dict],
    output_dir: Path,
    use_claude: bool = True,
    use_chatgpt: bool = True,
) -> dict[str, Any]:
    """Spawn critique agents and an evaluator. Returns the prioritized dict.

    Writes ``claude_critique.md``, ``chatgpt_critique.md`` (if available),
    ``evaluator.json``, and ``evaluator.md`` into ``output_dir``.

    Args:
        paper_id: Paper identifier (used only for logging/metadata).
        model_key: Model key (used only for logging/metadata).
        paper_text: Full paper.md content (may be empty). Truncated to
            ~30k chars before being sent to any critic.
        paper_config: Parsed paper_config.yaml as a dict.
        results: List of ExperimentResult objects from this (paper, model).
        sanity_reports: Parallel list of sanity-check dicts.
        output_dir: Directory for all output files.
        use_claude: If False, skip the Claude critic entirely.
        use_chatgpt: If False, skip the ChatGPT critic entirely.

    Returns:
        The parsed evaluator dict. Even on catastrophic failure, returns a
        dict with an ``error`` key rather than raising.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    claude_text = ""
    if use_claude:
        claude_text = _critique_with_claude(
            paper_text, paper_config, results, sanity_reports
        )
        _atomic_write_text(output_dir / "claude_critique.md", claude_text)
    else:
        claude_text = "# Claude critique\n\n(skipped)\n"
        _atomic_write_text(output_dir / "claude_critique.md", claude_text)

    chatgpt_text: str | None = None
    if use_chatgpt:
        chatgpt_text = _critique_with_chatgpt(
            paper_text, paper_config, results, sanity_reports
        )
    if chatgpt_text is not None:
        _atomic_write_text(output_dir / "chatgpt_critique.md", chatgpt_text)

    evaluator_dict = _evaluator(claude_text, chatgpt_text, results, sanity_reports)
    # Add provenance so the committed artifact is self-describing.
    evaluator_dict.setdefault("_provenance", {})
    evaluator_dict["_provenance"].update(
        {
            "paper_id": paper_id,
            "model_key": model_key,
            "n_results": len(results),
            "n_sanity_reports": len(sanity_reports or []),
            "chatgpt_available": chatgpt_text is not None,
        }
    )

    _atomic_write_json(output_dir / "evaluator.json", evaluator_dict)
    _atomic_write_text(
        output_dir / "evaluator.md",
        _render_evaluator_markdown(evaluator_dict),
    )

    logger.info(
        "Critique pass complete for %s/%s -> %s",
        paper_id,
        model_key,
        output_dir,
    )
    return evaluator_dict
