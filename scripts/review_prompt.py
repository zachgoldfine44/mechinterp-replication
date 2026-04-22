"""Emit an AI-review prompt (or artifact bundle) for a replication PR.

Two modes:

  1. **Prompt mode** (default) — prints the standardized review prompt
     with just two links: the paper URL and the PR URL. Ready to paste
     into claude.ai, gemini.google.com, chatgpt.com, etc. The PR page
     itself is the canonical view of everything the replicator
     submitted, which is what the AI reviewer should read.

  2. **Bundle mode** (``--bundle``) — concatenates every text artifact
     (paper.md, writeup/draft.md, paper_config.yaml, stimuli_config.yaml,
     metadata.yaml, each claim's result.json + sanity.json, and any
     harness critiques) into a single markdown document with section
     headers. Binaries (*.pt) and per-layer probe dumps are skipped.
     Fallback for reviewers that can't fetch URLs, or for upload as a
     single file.

Usage:
    # Paste-ready prompt for a PR (the maintainer's common case)
    python scripts/review_prompt.py --paper geometry_of_truth --pr 2

    # Same, with an explicit replication_id in the header
    python scripts/review_prompt.py \\
        --paper geometry_of_truth --pr 2 \\
        --replication geometry_of_truth-tulaneadam-qwen_1_5b

    # Offline bundle (requires --replication; no auto-select)
    python scripts/review_prompt.py \\
        --paper geometry_of_truth \\
        --replication geometry_of_truth-tulaneadam-qwen_1_5b \\
        --bundle -o /tmp/review.md

Replication IDs are never auto-selected — when bundle mode needs one
and you haven't provided it, the script lists the available options
and exits. The maintainer is expected to know which replication they're
reviewing; guessing would risk bundling the wrong one.
"""

from __future__ import annotations

import argparse
import json
import sys
from io import StringIO
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
REPO_URL = "https://github.com/zachgoldfine44/mechinterp-replication"


# The review protocol — kept in sync with CONTRIBUTING.md#ai-review-policy
# and the prompt text used historically (preserved in
# writeup/emotions/.../reviews/README.md). If these prompts change,
# update both places.
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


def _load_yaml(path: Path) -> dict[str, Any]:
    with open(path) as f:
        return yaml.safe_load(f) or {}


def _list_replications(paper_id: str) -> list[str]:
    reps_dir = REPO_ROOT / "config" / "papers" / paper_id / "replications"
    if not reps_dir.is_dir():
        return []
    return sorted(p.name for p in reps_dir.iterdir() if p.is_dir())


def _require_replication(paper_id: str, replication_id: str | None) -> str:
    """Return the replication_id, or exit with a helpful error.

    No auto-selection: even if a paper has exactly one replication, the
    maintainer must pass --replication explicitly. Reviews are high-stakes
    enough (they get pinned to the submitter's PR) that silent defaulting
    could quietly target the wrong replication.
    """
    if replication_id:
        return replication_id
    reps = _list_replications(paper_id)
    if not reps:
        raise SystemExit(
            f"Paper {paper_id!r} has no replications under "
            f"config/papers/{paper_id}/replications/."
        )
    opts = ", ".join(reps)
    raise SystemExit(
        f"--replication is required for this mode. "
        f"Paper {paper_id!r} has {len(reps)} replication(s): {opts}"
    )


def _paper_url_only(paper_id: str) -> str:
    """Read the paper URL from any replication's paper_config.yaml, or
    the paper-level config.yaml if present.

    All replications of a paper share the same paper URL, so any
    paper_config.yaml is authoritative.
    """
    candidates: list[Path] = []
    reps_dir = REPO_ROOT / "config" / "papers" / paper_id / "replications"
    if reps_dir.is_dir():
        for rep in sorted(reps_dir.iterdir()):
            if rep.is_dir():
                candidates.append(rep / "paper_config.yaml")
    candidates.append(
        REPO_ROOT / "config" / "papers" / paper_id / "paper_config.yaml"
    )
    for cfg in candidates:
        if cfg.exists():
            url = _load_yaml(cfg).get("paper", {}).get("url", "")
            if url:
                return url
    raise SystemExit(
        f"Could not find a paper URL for {paper_id!r}. Check that at least "
        f"one paper_config.yaml exists with a paper.url field."
    )


def _gh_blob(path_from_root: str, ref: str) -> str:
    return f"{REPO_URL}/blob/{ref}/{path_from_root}"


def _gh_tree(path_from_root: str, ref: str) -> str:
    return f"{REPO_URL}/tree/{ref}/{path_from_root}"


def _paths(paper_id: str, replication_id: str) -> dict[str, Path]:
    """Return the canonical paths for a replication (may not all exist)."""
    base = REPO_ROOT
    return {
        "paper_md": base / f"config/papers/{paper_id}/paper.md",
        "paper_config": base / f"config/papers/{paper_id}/replications/{replication_id}/paper_config.yaml",
        "stimuli_config": base / f"config/papers/{paper_id}/replications/{replication_id}/stimuli_config.yaml",
        "metadata": base / f"config/papers/{paper_id}/replications/{replication_id}/metadata.yaml",
        "stimuli_dir": base / f"config/papers/{paper_id}/replications/{replication_id}/stimuli",
        "results_dir": base / f"results/{paper_id}/{replication_id}",
        "writeup": base / f"writeup/{paper_id}/{replication_id}/draft.md",
        "writeup_dir": base / f"writeup/{paper_id}/{replication_id}",
        "figures_dir": base / f"figures/{paper_id}/{replication_id}",
    }


def _rel(p: Path) -> str:
    return p.relative_to(REPO_ROOT).as_posix()


# ---------------------------------------------------------------------------
# Prompt mode
# ---------------------------------------------------------------------------

def build_prompt(
    paper_id: str,
    pr_number: int,
    replication_id: str | None,
) -> str:
    """Emit the standardized review prompt for a PR.

    Output is plain text, matching the exact format in CONTRIBUTING.md#ai-review-policy —
    one paragraph + two labeled URL lines. Copy-paste into claude.ai,
    gemini.google.com, chatgpt.com. Prompt 2 (the follow-up referee
    request) is appended below with a short header so the maintainer
    can send it after the model's first response.

    replication_id is optional here: the prompt body only contains the
    paper URL + PR URL (the PR is the canonical view of the submission).
    When given, it's used for a discreet "Review request: {id}" header
    so the maintainer can tell instances apart if running several.
    """
    paper_url = _paper_url_only(paper_id)
    pr_url = f"{REPO_URL}/pull/{pr_number}"

    out = StringIO()
    if replication_id:
        out.write(f"# Review request: `{replication_id}` (PR #{pr_number})\n\n")
    else:
        out.write(f"# Review request: {paper_id} (PR #{pr_number})\n\n")
    out.write("Paste the following into claude.ai, gemini.google.com, "
              "chatgpt.com. Prompt 2 is a follow-up — send it after the "
              "model responds to Prompt 1.\n\n")
    out.write("---\n\n")
    out.write("## Prompt 1\n\n")
    out.write(PROMPT_1 + "\n\n")
    out.write(f"Paper: {paper_url}\n")
    out.write(f"Replication: {pr_url}\n\n")
    out.write("---\n\n")
    out.write("## Prompt 2 (send after Prompt 1)\n\n")
    out.write(PROMPT_2 + "\n\n")
    out.write("---\n\n")
    if replication_id:
        out.write(
            "Save responses under "
            f"`writeup/{paper_id}/{replication_id}/reviews/{{reviewer}}.md` "
            "and regenerate the README table with "
            "`python scripts/generate_replications_table.py`.\n"
        )
    else:
        out.write(
            "Save responses under "
            f"`writeup/{paper_id}/{{replication_id}}/reviews/{{reviewer}}.md` "
            "and regenerate the README table with "
            "`python scripts/generate_replications_table.py`.\n"
        )
    return out.getvalue()


# ---------------------------------------------------------------------------
# Bundle mode
# ---------------------------------------------------------------------------

def _read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except OSError as e:
        return f"(could not read {path}: {e})"


def _fenced(content: str, lang: str = "") -> str:
    return f"```{lang}\n{content.rstrip()}\n```\n"


def _collect_result_files(results_dir: Path) -> list[tuple[str, Path, Path | None]]:
    """Return (claim_id, result.json, sanity.json | None) for each claim.

    Walks results_dir/{model}/{claim}/ — model key is carried into the
    claim_id string as "{model}/{claim}" so model-level structure is
    visible in section headers. Per-layer probe JSONs are excluded to
    keep bundles compact; the summary numbers live in result.json.
    """
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
    """Collect harness critique outputs from results/.../critiques/ dirs."""
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
    paths = _paths(paper_id, replication_id)
    if not paths["paper_config"].exists():
        raise SystemExit(f"No paper_config.yaml at {paths['paper_config']}")
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
        "probe dumps are omitted.\n\n"
    )
    out.write("---\n\n")

    # 1. Paper (the oracle)
    if paths["paper_md"].exists():
        out.write("## 1. Original paper (oracle)\n\n")
        out.write(f"_Source: `{_rel(paths['paper_md'])}`_\n\n")
        out.write(_read_text(paths["paper_md"]))
        out.write("\n\n---\n\n")

    # 2. Writeup
    if paths["writeup"].exists():
        out.write("## 2. Replicator's writeup\n\n")
        out.write(f"_Source: `{_rel(paths['writeup'])}`_\n\n")
        out.write(_read_text(paths["writeup"]))
        out.write("\n\n---\n\n")

    # 3. Paper config
    out.write("## 3. Replicator's `paper_config.yaml`\n\n")
    out.write(f"_Source: `{_rel(paths['paper_config'])}`_\n\n")
    out.write(_fenced(_read_text(paths["paper_config"]), "yaml"))
    out.write("\n---\n\n")

    # 4. Stimuli config
    if paths["stimuli_config"].exists():
        out.write("## 4. Replicator's `stimuli_config.yaml`\n\n")
        out.write(f"_Source: `{_rel(paths['stimuli_config'])}`_\n\n")
        out.write(_fenced(_read_text(paths["stimuli_config"]), "yaml"))
        out.write("\n---\n\n")

    # 5. Metadata
    if paths["metadata"].exists():
        out.write("## 5. Replicator's `metadata.yaml`\n\n")
        out.write(f"_Source: `{_rel(paths['metadata'])}`_\n\n")
        out.write(_fenced(_read_text(paths["metadata"]), "yaml"))
        out.write("\n---\n\n")

    # 6. Per-claim results + sanity
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

    # 7. Harness critiques (if any were produced by the pipeline's own
    # critique step when the replication ran).
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
                # Pretty-print JSON for readability
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

    # Footer with the review prompts, so the AI reviewer has everything
    # it needs in one document.
    out.write("## Review instructions\n\n")
    out.write("**Prompt 1:** " + PROMPT_1 + "\n\n")
    out.write("**Prompt 2 (send after Prompt 1):** " + PROMPT_2 + "\n")

    return out.getvalue()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--paper", required=True,
        help="Paper ID (folder under config/papers/).",
    )
    parser.add_argument(
        "--pr", type=int, default=None,
        help=(
            "PR number. Required for prompt mode — the PR URL goes into "
            "the prompt as 'Replication: https://github.com/.../pull/N', "
            "which is what the AI reviewer reads to see the full "
            "submission (files changed, diff, discussion). Not used in "
            "bundle mode."
        ),
    )
    parser.add_argument(
        "--replication", default=None,
        help=(
            "Replication ID (folder under "
            "config/papers/{paper}/replications/). Optional in prompt "
            "mode (used only for the header label). Required in "
            "--bundle mode. Never auto-selected, even when a paper has "
            "only one replication — the maintainer should know which "
            "one they're reviewing."
        ),
    )
    parser.add_argument(
        "--bundle", action="store_true",
        help="Emit a concatenated markdown bundle of all text artifacts "
             "instead of the paste-ready prompt. Requires --replication.",
    )
    parser.add_argument(
        "-o", "--output", default=None,
        help="Write to file instead of stdout.",
    )
    args = parser.parse_args()

    if args.bundle:
        replication_id = _require_replication(args.paper, args.replication)
        text = build_bundle(args.paper, replication_id)
    else:
        if args.pr is None:
            raise SystemExit(
                "--pr is required for prompt mode "
                "(e.g. --pr 2). Use --bundle for the offline "
                "concatenated-markdown mode, which doesn't need --pr."
            )
        text = build_prompt(args.paper, args.pr, args.replication)

    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(text)
        sys.stderr.write(
            f"Wrote {len(text):,} chars to {out_path}\n"
        )
    else:
        sys.stdout.write(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
