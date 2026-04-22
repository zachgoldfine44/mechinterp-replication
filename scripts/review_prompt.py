"""Emit an AI-review prompt for a replication.

Four modes:

  1. **PR review** (default, for maintainers) —
         python scripts/review_prompt.py --paper X --pr N [--replication ID]
     Prints the paste-ready prompt with Paper: + Replication: {PR URL}.
     The reviewer's AI must be able to fetch URLs; use --reviewer-ready
     if that's a problem.

  2. **Self-review** (for contributors, pre-PR) —
         python scripts/review_prompt.py --paper X --replication ID --self-review
     Writes a bundle of all text artifacts to
     ``local_data/reviews/{paper}/{id}/self_review_bundle.md`` and
     prints a prompt that nudges the contributor to ask an AI *other
     than* the one that helped them run the replication.

  3. **Reviewer-ready** (one-shot upload-and-paste) —
         python scripts/review_prompt.py --paper X --replication ID --reviewer-ready
     Writes a bundle to
     ``local_data/reviews/{paper}/{id}/reviewer_bundle.md`` and prints
     step-by-step instructions plus the two prompts, framed neutrally
     (no "different AI" nudge). Use this when the user just wants to
     kick off a review in claude.ai / gemini.google.com / chatgpt.com
     without dealing with URL-fetch restrictions.

  4. **Bundle only** —
         python scripts/review_prompt.py --paper X --replication ID --bundle
     Emits the concatenated-markdown bundle to stdout (or -o FILE).
     Useful for ad-hoc offline review; no prompt text attached.

Replication IDs are never auto-selected; if a mode needs one and you
haven't provided it, the script lists the available options and exits.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Support running from the repo root without `pip install -e .`.
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.core.review import (  # noqa: E402
    build_bundle,
    build_pr_prompt,
    build_reviewer_ready,
    build_self_review,
    list_replications,
)


def _require_replication(paper_id: str, replication_id: str | None) -> str:
    if replication_id:
        return replication_id
    reps = list_replications(paper_id)
    if not reps:
        raise SystemExit(
            f"Paper {paper_id!r} has no replications under "
            f"config/papers/{paper_id}/replications/."
        )
    raise SystemExit(
        "--replication is required for this mode. "
        f"Paper {paper_id!r} has {len(reps)} replication(s): {', '.join(reps)}"
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--paper", required=True,
        help="Paper ID (folder under config/papers/).")
    parser.add_argument("--pr", type=int, default=None,
        help="PR number for maintainer PR-review mode.")
    parser.add_argument("--replication", default=None,
        help="Replication ID (never auto-selected).")

    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--self-review", action="store_true",
        help="Write a bundle and emit a contributor-oriented self-review "
             "prompt nudging toward a different AI than the one used to "
             "run the replication. Requires --replication.")
    mode.add_argument("--reviewer-ready", action="store_true",
        help="Write a bundle and emit step-by-step upload-and-paste "
             "instructions (neutral framing, no 'different AI' nudge). "
             "The go-to mode when you just want a review prompt you can "
             "drop into claude.ai / gemini.google.com / chatgpt.com. "
             "Requires --replication.")
    mode.add_argument("--bundle", action="store_true",
        help="Emit the concatenated-markdown bundle only (no prompt text). "
             "Requires --replication.")

    parser.add_argument("--used-ai", default=None,
        help="Optional: which AI helped you run the replication "
             "(Claude, ChatGPT, Gemini, Codex…). Used in --self-review "
             "mode to suggest the other two specifically.")
    parser.add_argument("-o", "--output", default=None,
        help="Write to file instead of stdout.")
    args = parser.parse_args()

    if args.self_review:
        replication_id = _require_replication(args.paper, args.replication)
        bundle_path = Path(args.output).resolve() if args.output else None
        text, written_to = build_self_review(
            args.paper, replication_id,
            bundle_path=bundle_path,
            used_ai=args.used_ai,
        )
        sys.stderr.write(f"Wrote bundle to {written_to}\n")
        sys.stdout.write(text)
        return 0

    if args.reviewer_ready:
        replication_id = _require_replication(args.paper, args.replication)
        bundle_path = Path(args.output).resolve() if args.output else None
        text, written_to = build_reviewer_ready(
            args.paper, replication_id,
            bundle_path=bundle_path,
        )
        sys.stderr.write(f"Wrote bundle to {written_to}\n")
        sys.stdout.write(text)
        return 0

    if args.bundle:
        replication_id = _require_replication(args.paper, args.replication)
        text = build_bundle(args.paper, replication_id)
    else:
        if args.pr is None:
            raise SystemExit(
                "--pr is required for PR-review mode (e.g. --pr 2), or "
                "use --self-review (for contributors) / --bundle "
                "(for an offline artifact dump)."
            )
        text = build_pr_prompt(args.paper, args.pr, args.replication)

    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(text)
        sys.stderr.write(f"Wrote {len(text):,} chars to {out_path}\n")
    else:
        sys.stdout.write(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
