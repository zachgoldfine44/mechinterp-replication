"""Generate the Completed Replications table in README.md.

Scans ``config/papers/*/replications/*/metadata.yaml`` and rewrites the
section of README.md between two HTML-comment sentinels:

    <!-- replications-table:start -->
    ... generated table ...
    <!-- replications-table:end -->

Run this script after merging a new replication PR. CI could also run it
on every PR to warn if the table is stale, but the source of truth is
always the metadata.yaml files.

Why this exists: the README table is the single most collision-prone
thing in the repo once multiple people start contributing. A shared
table row edited by 10 contributors is a merge-conflict engine. By
contrast, each replication's metadata.yaml lives in its own directory
and the table is a derived artifact. Contributors write metadata; the
table is regenerated.

Usage:
    python scripts/generate_replications_table.py
    python scripts/generate_replications_table.py --check    # fail if stale
    python scripts/generate_replications_table.py --print    # stdout only
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


REPO_ROOT = Path(__file__).resolve().parent.parent
PAPERS_DIR = REPO_ROOT / "config" / "papers"
README_PATH = REPO_ROOT / "README.md"

SENTINEL_START = "<!-- replications-table:start -->"
SENTINEL_END = "<!-- replications-table:end -->"


@dataclass
class ReplicationRow:
    """One row in the generated table (one replication attempt)."""

    paper_id: str
    paper_title: str
    paper_url: str
    replication_id: str
    replicator: str
    github_handle: str
    status: str
    models_tested: list[str]
    summary: str
    writeup_path: str
    config_path: str
    ai_reviews: list[dict[str, str]]

    def model_span(self) -> str:
        """Compact render of models tested, e.g. '6 (1B–9B)' or '1 (1.5B)'."""
        n = len(self.models_tested)
        if n == 0:
            return "—"
        # Try to find min/max params from model keys by pattern match.
        # We don't want to hard-code the registry here; the metadata.yaml
        # owner can override by adding a models_span field.
        return f"{n}" if n == 1 else f"{n}"


def _read_paper_metadata(paper_dir: Path) -> tuple[str, str]:
    """Return (title, url) by reading paper-level paper_config.yaml or any
    replication's paper_config.yaml. All replications of a paper share the
    same paper metadata so we can fall back between them.
    """
    candidate_configs: list[Path] = []
    reps_dir = paper_dir / "replications"
    if reps_dir.is_dir():
        for rep in sorted(reps_dir.iterdir()):
            if rep.is_dir():
                candidate_configs.append(rep / "paper_config.yaml")
    candidate_configs.append(paper_dir / "paper_config.yaml")

    for cfg in candidate_configs:
        if cfg.exists():
            with open(cfg) as f:
                raw = yaml.safe_load(f) or {}
            p = raw.get("paper", {}) or {}
            return (
                p.get("title", paper_dir.name),
                p.get("url", ""),
            )
    return (paper_dir.name, "")


def _discover_ai_reviews(writeup_dir: Path) -> list[dict[str, str]]:
    """Find AI reviews in writeup/{paper}/{replication}/reviews/.

    Returns a list of dicts with keys {name, score, path}. Score parsing
    is heuristic — looks at the first ~30 lines of each review for a
    pattern like 'Score: 7/10' or '**Score:** 8.5/10'. Reviewers who
    don't give a score get score='—'.
    """
    reviews_dir = writeup_dir / "reviews"
    if not reviews_dir.is_dir():
        return []
    out: list[dict[str, str]] = []
    for md in sorted(reviews_dir.glob("*.md")):
        if md.name.lower() == "readme.md":
            continue
        # Parse a display name from filename
        stem = md.stem
        if "chatgpt" in stem.lower():
            display = "ChatGPT"
        elif "claude" in stem.lower():
            display = "Claude"
        elif "gemini" in stem.lower():
            display = "Gemini"
        else:
            display = stem
        # Try to pull a score. Tolerates "7/10", "7 / 10", "**7 / 10**".
        score = "—"
        try:
            import re
            head = md.read_text(errors="replace").splitlines()[:30]
            pat = re.compile(r"(\d+(?:\.\d+)?)\s*/\s*10")
            for line in head:
                if "score" not in line.lower():
                    continue
                m = pat.search(line)
                if m:
                    score = f"{m.group(1)}/10"
                    break
        except OSError:
            pass
        path_rel = md.relative_to(REPO_ROOT).as_posix()
        out.append({"name": display, "score": score, "path": path_rel})
    return out


def collect_replications() -> list[ReplicationRow]:
    """Walk config/papers/*/replications/*/metadata.yaml."""
    rows: list[ReplicationRow] = []
    if not PAPERS_DIR.is_dir():
        return rows
    for paper_dir in sorted(PAPERS_DIR.iterdir()):
        if not paper_dir.is_dir():
            continue
        paper_id = paper_dir.name
        paper_title, paper_url = _read_paper_metadata(paper_dir)
        reps_dir = paper_dir / "replications"
        if not reps_dir.is_dir():
            continue
        for rep_dir in sorted(reps_dir.iterdir()):
            if not rep_dir.is_dir():
                continue
            meta_path = rep_dir / "metadata.yaml"
            if not meta_path.exists():
                continue
            with open(meta_path) as f:
                meta: dict[str, Any] = yaml.safe_load(f) or {}
            writeup_path = meta.get(
                "writeup_path",
                f"writeup/{paper_id}/{rep_dir.name}/draft.md",
            )
            writeup_dir = REPO_ROOT / writeup_path
            if writeup_dir.suffix == ".md":
                writeup_dir = writeup_dir.parent
            rows.append(ReplicationRow(
                paper_id=paper_id,
                paper_title=paper_title,
                paper_url=paper_url,
                replication_id=meta.get("id", rep_dir.name),
                replicator=meta.get("replicator", "Unknown"),
                github_handle=meta.get("github_handle", ""),
                status=meta.get("status", ""),
                models_tested=meta.get("models_tested", []) or [],
                summary=(meta.get("summary") or "").strip().replace("\n", " "),
                writeup_path=writeup_path,
                config_path=meta.get(
                    "config_path",
                    f"config/papers/{paper_id}/replications/{rep_dir.name}/",
                ),
                ai_reviews=_discover_ai_reviews(writeup_dir),
            ))
    return rows


def _handle_link(handle: str) -> str:
    if not handle:
        return "—"
    return f"[@{handle}](https://github.com/{handle})"


def _reviews_cell(reviews: list[dict[str, str]]) -> str:
    if not reviews:
        return ""
    return " · ".join(f"[{r['name']} {r['score']}]({r['path']})" for r in reviews)


def render_table(rows: list[ReplicationRow]) -> str:
    """Render the rows as a markdown table, matching README conventions."""
    header = (
        "| Paper | Replicator | Replication ID | Models | Status | Key findings | Writeup | Config | AI peer reviews |\n"
        "|-------|:---:|---|:---:|:---:|---|:---:|:---:|:---|"
    )
    lines = [header]
    for r in sorted(rows, key=lambda x: (x.paper_id, x.replication_id)):
        title_cell = (
            f"[{r.paper_title}]({r.paper_url})" if r.paper_url else r.paper_title
        )
        n_models = len(r.models_tested)
        models_cell = f"{n_models}" if n_models else "—"
        lines.append(
            "| " + " | ".join([
                title_cell,
                _handle_link(r.github_handle),
                f"`{r.replication_id}`",
                models_cell,
                r.status or "—",
                r.summary or "",
                f"[draft.md]({r.writeup_path})" if r.writeup_path else "—",
                f"[config/]({r.config_path})" if r.config_path else "—",
                _reviews_cell(r.ai_reviews),
            ]) + " |"
        )
    # Blank-row marker so the shape of the table is obvious to humans reading
    # README.md even before any replications exist.
    lines.append(
        "| *Your replication here* | *(you)* | "
        "`{paper}-{handle}-{model_scope}` | | | | | | |"
    )
    return "\n".join(lines)


def splice_into_readme(table_md: str) -> tuple[str, bool]:
    """Replace content between sentinels. Returns (new_text, changed).

    If the sentinels are absent, they're inserted around the existing
    replications table (first markdown table under the "Completed
    replications" header).
    """
    text = README_PATH.read_text()
    if SENTINEL_START in text and SENTINEL_END in text:
        pre, _, rest = text.partition(SENTINEL_START)
        _, _, post = rest.partition(SENTINEL_END)
        new = f"{pre}{SENTINEL_START}\n{table_md}\n{SENTINEL_END}{post}"
        return new, new != text
    # First-time install: insert sentinels around the existing table.
    # Find the table block under "### Completed replications".
    anchor = "### Completed replications"
    if anchor not in text:
        raise RuntimeError(
            "README.md is missing the 'Completed replications' section. "
            "Add the section before running this script."
        )
    pre, _, rest = text.partition(anchor)
    # Everything until the next top-level heading (---) is the block.
    block, sep, tail = rest.partition("\n---")
    new_block = (
        f"{anchor}{_preamble_from(block)}\n\n"
        f"{SENTINEL_START}\n{table_md}\n{SENTINEL_END}\n"
        f"{_postamble_from(block)}"
    )
    new = f"{pre}{new_block}{sep}{tail}"
    return new, new != text


def _preamble_from(block: str) -> str:
    """Everything before the first table row starting with '|'."""
    lines = block.splitlines()
    out: list[str] = []
    for ln in lines:
        if ln.strip().startswith("|"):
            break
        out.append(ln)
    return "\n".join(out).rstrip()


def _postamble_from(block: str) -> str:
    """Everything after the last table row starting with '|'."""
    lines = block.splitlines()
    last_table_idx = -1
    for i, ln in enumerate(lines):
        if ln.strip().startswith("|"):
            last_table_idx = i
    if last_table_idx == -1:
        return ""
    return "\n".join(lines[last_table_idx + 1:]).lstrip("\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check", action="store_true",
        help="Exit non-zero if README would change. Use in CI.",
    )
    parser.add_argument(
        "--print", dest="print_only", action="store_true",
        help="Print the generated table to stdout without touching README.",
    )
    args = parser.parse_args()

    rows = collect_replications()
    table = render_table(rows)

    if args.print_only:
        print(table)
        return 0

    new_text, changed = splice_into_readme(table)
    if args.check:
        if changed:
            sys.stderr.write(
                "README.md replications table is stale — "
                "re-run scripts/generate_replications_table.py\n"
            )
            return 1
        return 0

    if changed:
        README_PATH.write_text(new_text)
        sys.stderr.write(
            f"Updated README with {len(rows)} replication(s).\n"
        )
    else:
        sys.stderr.write("README replications table already up to date.\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
