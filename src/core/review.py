"""Post-replication review prompts — the single source of truth.

This module builds the standardized review prompts documented in
``CONTRIBUTING.md#ai-review-policy``. Three entry points:

  - :func:`build_pr_prompt` — maintainer reviewing an incoming PR.
    Emits the "Paper: / Replication: {PR URL}" prompt.

  - :func:`build_self_review` — contributor checking their own
    replication before opening a PR. Writes an artifact bundle to a
    file and emits a prompt that says "upload the attached bundle",
    explicitly nudging the user to use an AI *different* from the one
    that helped them run the replication so the feedback is neutral.

  - :func:`build_bundle` — raw concatenated markdown of every text
    artifact (paper, writeup, configs, result/sanity JSONs, harness
    critiques). Used by both prompt entry points as well as
    ``scripts/review_prompt.py --bundle``.

The prompt text is kept in constants at the top of this file so there
is exactly one place to edit the protocol. When these change, the
historical protocol note in the emotions reviews README should be
updated too for archival honesty.

See ``scripts/review_prompt.py`` for the CLI wrapper.
"""

from __future__ import annotations

import json
from io import StringIO
from pathlib import Path
from typing import Any

import yaml

from src.utils.env import get_data_root, get_project_root


REPO_URL = "https://github.com/zachgoldfine44/mechinterp-replication"


# The two-prompt review protocol. Preserved verbatim from the examples in
# writeup/emotions/.../reviews/README.md so scores stay comparable across
# submissions over time.
PROMPT_1 = (
    "Please read the following paper, and then the following replication of "
    "the paper, and rate the replication on a scale of 1-10 "
    "(1 being terrible experimental design, no meaningful insights due to "
    "serious methodological flaws, etc; 10 being perfect experimental "
    "design, insights are useful due to rigorous methodology, meaningfully "
    "contributes to scientific body of knowledge), along with justifications "
    "for the 1-10 score you assign:"
)

PROMPT_2 = (
    "Can you turn this into a harsher peer-review-style referee report "
    "with major concerns / minor concerns / accept-reject recommendation?"
)

SELF_REVIEW_INTRO = (
    "Before opening your PR, get a second opinion on your replication.\n"
    "Paste the prompt below into an AI assistant *other than* the one that "
    "helped you run this replication — if you used Claude, try ChatGPT or "
    "Gemini; if you used ChatGPT, try Claude or Gemini; and so on. An "
    "independent read catches blind spots in methodology and framing that "
    "the AI you co-wrote with is more likely to miss."
)


# ---------------------------------------------------------------------------
# Path helpers (shared by all three entry points)
# ---------------------------------------------------------------------------

def _load_yaml(path: Path) -> dict[str, Any]:
    with open(path) as f:
        return yaml.safe_load(f) or {}


def _list_replications(paper_id: str) -> list[str]:
    reps_dir = get_project_root() / "config" / "papers" / paper_id / "replications"
    if not reps_dir.is_dir():
        return []
    return sorted(p.name for p in reps_dir.iterdir() if p.is_dir())


def _paths(paper_id: str, replication_id: str) -> dict[str, Path]:
    base = get_project_root()
    return {
        "paper_md": base / f"config/papers/{paper_id}/paper.md",
        "paper_config": base / f"config/papers/{paper_id}/replications/{replication_id}/paper_config.yaml",
        "stimuli_config": base / f"config/papers/{paper_id}/replications/{replication_id}/stimuli_config.yaml",
        "metadata": base / f"config/papers/{paper_id}/replications/{replication_id}/metadata.yaml",
        "results_dir": base / f"results/{paper_id}/{replication_id}",
        "writeup": base / f"writeup/{paper_id}/{replication_id}/draft.md",
        "figures_dir": base / f"figures/{paper_id}/{replication_id}",
    }


def _rel(p: Path) -> str:
    return p.relative_to(get_project_root()).as_posix()


def _paper_url(paper_id: str) -> str:
    """Read the paper URL from any replication's paper_config.yaml."""
    candidates: list[Path] = []
    reps_dir = get_project_root() / "config" / "papers" / paper_id / "replications"
    if reps_dir.is_dir():
        for rep in sorted(reps_dir.iterdir()):
            if rep.is_dir():
                candidates.append(rep / "paper_config.yaml")
    candidates.append(
        get_project_root() / "config" / "papers" / paper_id / "paper_config.yaml"
    )
    for cfg in candidates:
        if cfg.exists():
            url = _load_yaml(cfg).get("paper", {}).get("url", "")
            if url:
                return url
    raise FileNotFoundError(
        f"Could not find a paper URL for {paper_id!r}. Check that at least "
        f"one paper_config.yaml exists with a paper.url field."
    )


def list_replications(paper_id: str) -> list[str]:
    """Public re-export so CLI can error-list without importing private helpers."""
    return _list_replications(paper_id)


# ---------------------------------------------------------------------------
# Bundle mode (concatenated markdown of all text artifacts)
# ---------------------------------------------------------------------------

def _read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except OSError as e:
        return f"(could not read {path}: {e})"


def _fenced(content: str, lang: str = "") -> str:
    return f"```{lang}\n{content.rstrip()}\n```\n"


def _collect_result_files(results_dir: Path) -> list[tuple[str, Path, Path | None]]:
    out: list[tuple[str, Path, Path | None]] = []
    if not results_dir.is_dir():
        return out
    for model_dir in sorted(results_dir.iterdir()):
        if not model_dir.is_dir():
            continue
        for claim_dir in sorted(model_dir.iterdir()):
            if not claim_dir.is_dir() or claim_dir.name == "critiques":
                continue
            result = claim_dir / "result.json"
            sanity = claim_dir / "sanity.json"
            if result.exists():
                out.append((
                    f"{model_dir.name}/{claim_dir.name}",
                    result,
                    sanity if sanity.exists() else None,
                ))
    return out


def _collect_critiques(results_dir: Path) -> list[Path]:
    out: list[Path] = []
    if not results_dir.is_dir():
        return out
    for model_dir in sorted(results_dir.iterdir()):
        crit = model_dir / "critiques"
        if crit.is_dir():
            for f in sorted(crit.iterdir()):
                if f.suffix in (".md", ".json"):
                    out.append(f)
    return out


def build_bundle(paper_id: str, replication_id: str) -> str:
    """Concatenate every text artifact into a single markdown document.

    Skips binaries (`*.pt`) and per-layer probe dumps. The emitted doc
    has a predictable section layout so an AI reviewer (or human) can
    scan it top-to-bottom.
    """
    paths = _paths(paper_id, replication_id)
    if not paths["paper_config"].exists():
        raise FileNotFoundError(f"No paper_config.yaml at {paths['paper_config']}")
    cfg = _load_yaml(paths["paper_config"])
    meta = _load_yaml(paths["metadata"]) if paths["metadata"].exists() else {}
    paper = cfg.get("paper", {})

    out = StringIO()
    out.write(f"# Review bundle: `{replication_id}`\n\n")
    out.write(
        f"**Paper:** {paper.get('title', paper_id)}"
        + (f" — {paper.get('url', '')}" if paper.get("url") else "")
        + "\n"
    )
    out.write(
        f"**Replicator:** {meta.get('replicator', '?')} "
        f"(@{meta.get('github_handle', '')})\n"
        f"**Models tested:** {', '.join(meta.get('models_tested', [])) or '?'}\n"
        f"**Status:** {meta.get('status', '?')}\n\n"
    )
    out.write(
        "This bundle concatenates every text artifact that an AI referee "
        "needs to score the replication. Sections: the paper (oracle), the "
        "replicator's writeup + config, per-claim result and sanity JSONs, "
        "and any harness critique outputs. Binaries (`*.pt`) and per-layer "
        "probe dumps are omitted.\n\n---\n\n"
    )

    if paths["paper_md"].exists():
        out.write("## 1. Original paper (oracle)\n\n")
        out.write(f"_Source: `{_rel(paths['paper_md'])}`_\n\n")
        out.write(_read_text(paths["paper_md"]))
        out.write("\n\n---\n\n")

    if paths["writeup"].exists():
        out.write("## 2. Replicator's writeup\n\n")
        out.write(f"_Source: `{_rel(paths['writeup'])}`_\n\n")
        out.write(_read_text(paths["writeup"]))
        out.write("\n\n---\n\n")

    out.write("## 3. Replicator's `paper_config.yaml`\n\n")
    out.write(f"_Source: `{_rel(paths['paper_config'])}`_\n\n")
    out.write(_fenced(_read_text(paths["paper_config"]), "yaml"))
    out.write("\n---\n\n")

    if paths["stimuli_config"].exists():
        out.write("## 4. Replicator's `stimuli_config.yaml`\n\n")
        out.write(f"_Source: `{_rel(paths['stimuli_config'])}`_\n\n")
        out.write(_fenced(_read_text(paths["stimuli_config"]), "yaml"))
        out.write("\n---\n\n")

    if paths["metadata"].exists():
        out.write("## 5. Replicator's `metadata.yaml`\n\n")
        out.write(f"_Source: `{_rel(paths['metadata'])}`_\n\n")
        out.write(_fenced(_read_text(paths["metadata"]), "yaml"))
        out.write("\n---\n\n")

    result_files = _collect_result_files(paths["results_dir"])
    if result_files:
        out.write("## 6. Results\n\n")
        for i, (claim_id, result_path, sanity_path) in enumerate(result_files, 1):
            out.write(f"### 6.{i} `{claim_id}`\n\n")
            out.write(f"_Source: `{_rel(result_path)}`_\n\n")
            out.write(_fenced(_read_text(result_path), "json"))
            if sanity_path is not None:
                out.write(f"\n_Sanity check: `{_rel(sanity_path)}`_\n\n")
                out.write(_fenced(_read_text(sanity_path), "json"))
            out.write("\n")
        out.write("---\n\n")

    critiques = _collect_critiques(paths["results_dir"])
    if critiques:
        out.write("## 7. Harness critique outputs\n\n")
        out.write(
            "_These are the outputs of the pipeline's own Claude / ChatGPT "
            "critique step (not the post-submission peer review this bundle "
            "is for). Included for reference._\n\n"
        )
        for crit in critiques:
            out.write(f"### `{_rel(crit)}`\n\n")
            text = _read_text(crit)
            if crit.suffix == ".json":
                try:
                    text = json.dumps(json.loads(text), indent=2)
                except json.JSONDecodeError:
                    pass
                out.write(_fenced(text, "json"))
            else:
                out.write(text)
                out.write("\n")
            out.write("\n")
        out.write("---\n\n")

    out.write("## Review instructions\n\n")
    out.write("**Prompt 1:** " + PROMPT_1 + "\n\n")
    out.write("**Prompt 2 (send after Prompt 1):** " + PROMPT_2 + "\n")
    return out.getvalue()


# ---------------------------------------------------------------------------
# PR-mode prompt (maintainer reviewing an incoming PR)
# ---------------------------------------------------------------------------

def build_pr_prompt(
    paper_id: str,
    pr_number: int,
    replication_id: str | None = None,
) -> str:
    """Emit the paste-ready prompt for a PR review.

    Inline format with Paper: and Replication: lines, matching the
    exact wording in CONTRIBUTING.md#ai-review-policy. The PR URL is
    what the AI reviewer reads to see the full submission.
    """
    paper_url = _paper_url(paper_id)
    pr_url = f"{REPO_URL}/pull/{pr_number}"

    out = StringIO()
    header = (
        f"# Review request: `{replication_id}` (PR #{pr_number})"
        if replication_id
        else f"# Review request: {paper_id} (PR #{pr_number})"
    )
    out.write(header + "\n\n")
    out.write("Paste the following into claude.ai, gemini.google.com, "
              "chatgpt.com. Prompt 2 is a follow-up — send it after the "
              "model responds to Prompt 1.\n\n---\n\n")
    out.write("## Prompt 1\n\n")
    out.write(PROMPT_1 + "\n\n")
    out.write(f"Paper: {paper_url}\n")
    out.write(f"Replication: {pr_url}\n\n")
    out.write("---\n\n")
    out.write("## Prompt 2 (send after Prompt 1)\n\n")
    out.write(PROMPT_2 + "\n\n---\n\n")
    if replication_id:
        save_hint = f"writeup/{paper_id}/{replication_id}/reviews/{{reviewer}}.md"
    else:
        save_hint = f"writeup/{paper_id}/{{replication_id}}/reviews/{{reviewer}}.md"
    out.write(
        f"Save responses under `{save_hint}` and regenerate the README "
        "table with `python scripts/generate_replications_table.py`.\n"
    )
    return out.getvalue()


# ---------------------------------------------------------------------------
# Self-review (contributor checking their own replication before the PR)
# ---------------------------------------------------------------------------

def default_self_review_bundle_path(paper_id: str, replication_id: str) -> Path:
    """Where build_self_review writes the bundle by default.

    Under ``local_data/`` so it's gitignored — the self-review bundle is
    a derived artifact; what matters long-term is the reviews you get
    back (those get committed under
    ``writeup/{paper}/{replication}/reviews/``).
    """
    return (
        get_data_root() / "reviews" / paper_id / replication_id
        / "self_review_bundle.md"
    )


def build_self_review(
    paper_id: str,
    replication_id: str,
    bundle_path: Path | None = None,
    used_ai: str | None = None,
) -> tuple[str, Path]:
    """Write an artifact bundle and return a (prompt_text, bundle_path) pair.

    Args:
        paper_id, replication_id: the replication to self-review.
        bundle_path: where to write the bundle. Defaults to
            ``local_data/reviews/{paper}/{replication}/self_review_bundle.md``.
        used_ai: optional string naming the AI the user ran the
            replication with (e.g., "Claude", "ChatGPT", "Codex").
            When provided, the prompt message explicitly suggests the
            other two.

    Returns:
        (prompt_text, bundle_path). Caller prints prompt_text to stdout
        and/or a log; bundle_path is where to find the file to upload.
    """
    if bundle_path is None:
        bundle_path = default_self_review_bundle_path(paper_id, replication_id)

    bundle_text = build_bundle(paper_id, replication_id)
    bundle_path.parent.mkdir(parents=True, exist_ok=True)
    bundle_path.write_text(bundle_text, encoding="utf-8")

    paper_url = _paper_url(paper_id)

    # Compose the suggestion of which other AI to use. Conservative: if
    # we don't know which they used, suggest all three.
    all_ais = {"Claude": "claude.ai", "ChatGPT": "chatgpt.com", "Gemini": "gemini.google.com"}
    if used_ai and used_ai in all_ais:
        others = {k: v for k, v in all_ais.items() if k != used_ai}
        suggest = (
            f"You ran this replication with {used_ai}. For an independent read, "
            f"try one of: {', '.join(f'{k} ({v})' for k, v in others.items())}."
        )
    else:
        suggest = (
            "Try it in an AI *other than* the one you used to run this "
            f"replication — {', '.join(f'{k} ({v})' for k, v in all_ais.items())}."
        )

    bundle_rel = bundle_path.resolve()
    try:
        bundle_rel = bundle_rel.relative_to(get_project_root())
    except ValueError:
        pass  # bundle is outside the repo; use absolute path

    out = StringIO()
    out.write(f"# Self-review: `{replication_id}`\n\n")
    out.write(SELF_REVIEW_INTRO + "\n\n")
    out.write(suggest + "\n\n")
    out.write(
        f"A bundle of every text artifact from your replication has been "
        f"written to:\n\n    {bundle_rel}\n\n"
        f"Upload that file (most web AIs accept a markdown attachment) or "
        f"paste its contents, then send this prompt:\n\n"
    )
    out.write("---\n\n")
    out.write("## Prompt 1\n\n")
    out.write(PROMPT_1 + "\n\n")
    out.write(f"Paper: {paper_url}\n")
    out.write(f"Replication: [attached bundle — see {bundle_rel}]\n\n")
    out.write("---\n\n")
    out.write("## Prompt 2 (send after the model responds to Prompt 1)\n\n")
    out.write(PROMPT_2 + "\n\n")
    out.write("---\n\n")
    out.write(
        "If the feedback surfaces anything you want to address before "
        "opening the PR, fix it, re-run the pipeline, and regenerate this "
        "bundle:\n\n"
        f"    python scripts/review_prompt.py --paper {paper_id} "
        f"--replication {replication_id} --self-review\n\n"
        "Save any reviews you decide to keep under "
        f"`writeup/{paper_id}/{replication_id}/reviews/` before committing.\n"
    )
    return out.getvalue(), bundle_path
