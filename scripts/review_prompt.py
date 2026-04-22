"""Emit an AI-review prompt (or artifact bundle) for a replication.

Two modes, both keyed off the replication_id:

  1. **Prompt mode** (default) — prints a ready-to-paste prompt containing
     the standardized review protocol from CONTRIBUTING.md#ai-review-policy
     plus the paper URL and the GitHub blob/tree URLs for this specific
     replication's writeup, config, results, and (if present) figures.
     Good for any AI reviewer that can fetch URLs.

  2. **Bundle mode** (``--bundle``) — concatenates every text artifact
     (paper.md, writeup/draft.md, paper_config.yaml, stimuli_config.yaml,
     metadata.yaml, each claim's result.json + sanity.json, and any
     harness critiques) into a single markdown document with section
     headers. Binaries (*.pt) and per-layer probe dumps are skipped.
     Good for pasting into a web chat that can't fetch URLs, or for
     uploading as a single file.

Usage:
    python scripts/review_prompt.py --paper emotions
    python scripts/review_prompt.py \\
        --paper geometry_of_truth \\
        --replication geometry_of_truth-tulaneadam-qwen_1_5b
    python scripts/review_prompt.py \\
        --paper geometry_of_truth --bundle -o /tmp/review.md
    python scripts/review_prompt.py \\
        --paper geometry_of_truth --ref refactor/per-replication-namespacing

When --replication is omitted and the paper has exactly one replication,
it's auto-selected (same rule as the pipeline).
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


def _resolve_replication(paper_id: str, replication_id: str | None) -> str:
    if replication_id:
        return replication_id
    reps = _list_replications(paper_id)
    if len(reps) == 1:
        return reps[0]
    if not reps:
        raise SystemExit(
            f"Paper {paper_id!r} has no replications under "
            f"config/papers/{paper_id}/replications/."
        )
    raise SystemExit(
        f"Paper {paper_id!r} has {len(reps)} replications; pass --replication "
        f"to pick one.\nOptions: {reps}"
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

def build_prompt(paper_id: str, replication_id: str, ref: str) -> str:
    paths = _paths(paper_id, replication_id)
    if not paths["paper_config"].exists():
        raise SystemExit(f"No paper_config.yaml at {paths['paper_config']}")
    cfg = _load_yaml(paths["paper_config"])
    meta = (
        _load_yaml(paths["metadata"]) if paths["metadata"].exists() else {}
    )
    paper = cfg.get("paper", {})
    paper_url = paper.get("url", "")
    paper_title = paper.get("title", paper_id)

    # Assemble GitHub URLs for everything that exists.
    links: list[str] = []
    if paths["writeup"].exists():
        links.append(f"- Writeup: {_gh_blob(_rel(paths['writeup']), ref)}")
    links.append(
        f"- Replication config: {_gh_blob(_rel(paths['paper_config']), ref)}"
    )
    if paths["metadata"].exists():
        links.append(
            f"- Metadata: {_gh_blob(_rel(paths['metadata']), ref)}"
        )
    if paths["stimuli_config"].exists():
        links.append(
            f"- Stimuli config: {_gh_blob(_rel(paths['stimuli_config']), ref)}"
        )
    if paths["results_dir"].is_dir():
        links.append(
            f"- Results (all claims): {_gh_tree(_rel(paths['results_dir']), ref)}"
        )
    if paths["figures_dir"].is_dir():
        links.append(
            f"- Figures: {_gh_tree(_rel(paths['figures_dir']), ref)}"
        )
    if paths["paper_md"].exists():
        links.append(
            f"- Full paper text (oracle): {_gh_blob(_rel(paths['paper_md']), ref)}"
        )

    replicator = meta.get("replicator") or cfg.get("replication", {}).get("replicator", "Unknown")
    handle = meta.get("github_handle") or cfg.get("replication", {}).get("github_handle", "")
    models = meta.get("models_tested") or cfg.get("replication", {}).get("models_tested", [])
    status = meta.get("status") or cfg.get("replication", {}).get("status", "")

    handle_str = f"@{handle}" if handle else ""
    models_str = ", ".join(models) if models else "(unspecified)"

    # Two-prompt protocol, matching CONTRIBUTING.md#ai-review-policy.
    out = StringIO()
    out.write(f"# Review request: `{replication_id}`\n\n")
    out.write(
        f"**Paper:** {paper_title}"
        + (f" — <{paper_url}>" if paper_url else "")
        + "\n"
    )
    out.write(
        f"**Replicator:** {replicator} {handle_str}\n"
        f"**Models tested:** {models_str}\n"
        f"**Status:** {status or '—'}\n\n"
    )
    out.write("---\n\n")
    out.write("## Prompt 1 — 1-to-10 scoring\n\n")
    out.write("> " + PROMPT_1.replace("\n", "\n> ") + "\n")
    if paper_url:
        out.write(f"> - Paper: {paper_url}\n")
    out.write("> - Replication artifacts:\n")
    for ln in links:
        # indent under the "Replication artifacts:" bullet
        out.write("> " + ln.replace("- ", "  - ", 1) + "\n")
    out.write("\n")
    out.write("## Prompt 2 — harsher referee report (send after Prompt 1)\n\n")
    out.write("> " + PROMPT_2 + "\n\n")
    out.write("---\n\n")
    out.write(
        f"Save responses under "
        f"`writeup/{paper_id}/{replication_id}/reviews/{{reviewer}}.md` "
        f"and regenerate the README table with "
        f"`python scripts/generate_replications_table.py`.\n"
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
        "--replication", default=None,
        help=(
            "Replication ID (folder under config/papers/{paper}/replications/)."
            " Auto-detected when the paper has exactly one."
        ),
    )
    parser.add_argument(
        "--bundle", action="store_true",
        help="Emit a concatenated markdown bundle of all text artifacts "
             "instead of the clickable prompt.",
    )
    parser.add_argument(
        "--ref", default="main",
        help="Git ref for GitHub URLs in prompt mode (branch, tag, or SHA). "
             "Default: main.",
    )
    parser.add_argument(
        "-o", "--output", default=None,
        help="Write to file instead of stdout.",
    )
    args = parser.parse_args()

    replication_id = _resolve_replication(args.paper, args.replication)

    if args.bundle:
        text = build_bundle(args.paper, replication_id)
    else:
        text = build_prompt(args.paper, replication_id, args.ref)

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
