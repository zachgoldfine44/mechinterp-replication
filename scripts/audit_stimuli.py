"""Audit training stimuli for forbidden-word leaks and length issues.

The training stimuli for the emotions paper are LLM- or hand-generated
stories that are supposed to *evoke* an emotion without *naming* it. If
the word "happy" appears in a story labeled "happy", a probe trained on
those stories may be picking up on the word, not the internal emotion
representation.

This script audits every training stimulus file and reports:

  - leaks: stimuli that contain their own concept word as a whole word
    (case-insensitive). Listed per-concept with the offending excerpts.
  - close-form leaks: -ed/-ing/-ly/-ness/-y/-ier/-iest derivations of
    the concept word (e.g. "afraid" stimulus containing "fearful" is
    NOT flagged here, but "fearfully" -> we flag "fear*" if any
    derivation appears).
  - very short stimuli: < 30 words (flag for being thin)
  - very long stimuli: > 200 words (flag for going beyond the
    intended length)

Usage:
    python scripts/audit_stimuli.py --paper emotions

Exits with status 0 if no leaks found, 1 if any leaks present.
This is a fast static check — no model loading.
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# Common stems that count as a "leak" of the concept word.
# We strip the trailing vowel for some emotions before stem-matching to
# catch -ed/-ing forms (e.g., "love" -> "lov" catches "loving").
def concept_stems(concept: str) -> list[str]:
    """Build a list of word stems whose presence flags a leak."""
    c = concept.lower().strip()
    stems = {c}

    # Drop trailing 'e' for -ing/-ed (love -> lov)
    if c.endswith("e"):
        stems.add(c[:-1])
    # Drop trailing 'y' (happy -> happ)
    if c.endswith("y"):
        stems.add(c[:-1])
    # Drop trailing 'ed' for past tense (loved -> lov; not common but cheap)
    if c.endswith("ed"):
        stems.add(c[:-2])

    # Hand-mapped close synonyms / family roots that critique-2-#1 specifically
    # warned about. We only flag the EXACT concept word + its inflections, not
    # arbitrary synonyms — flagging synonyms would create too many false
    # positives. The stronger nuisance-control test is the lexical baseline.
    return sorted(stems)


# Whitelist of common-English words that share a prefix with an emotion
# stem but mean something different. The aggressive prefix matcher would
# otherwise produce false positives ("happened" matching "happ" stem from
# "happy" — but "happen" has nothing to do with "happy").
FALSE_POSITIVE_WHITELIST = {
    "happen", "happens", "happening", "happened",
    "love",  # "love" appears in non-loving contexts; only flag "loving"/"loved"
    "calmly",  # "calmly" *is* a leak; left here as illustration of the trade-off
    # NB: we keep "calmly" out of the whitelist — it IS a leak.
}
# Remove the entries we still want to flag (illustration above):
FALSE_POSITIVE_WHITELIST.discard("calmly")
# Keep "love" out: a "love" stem appearing in a "loving" stimulus probably IS a leak.
FALSE_POSITIVE_WHITELIST.discard("love")


def find_leaks(
    text: str, concept: str
) -> list[tuple[str, str]]:
    """Return list of (stem, excerpt) for each leak in `text`.

    Stem-prefix matching with a whitelist for common-English false positives
    (e.g., "happened" matches the stem "happ" from "happy" but is not a leak).
    """
    leaks = []
    text_lower = text.lower()
    for stem in concept_stems(concept):
        # Whole-word or stem-prefix match. We use \b<stem> (NOT \b<stem>\b)
        # so that "lov" matches "loved", "loving", "lovely" — i.e. any
        # word that starts with the stem. This is intentionally aggressive.
        pattern = re.compile(r"\b(" + re.escape(stem) + r"\w*)", re.IGNORECASE)
        for match in pattern.finditer(text_lower):
            matched_word = match.group(1).lower()
            if matched_word in FALSE_POSITIVE_WHITELIST:
                continue
            start = max(0, match.start() - 30)
            end = min(len(text), match.end() + 30)
            excerpt = text[start:end].replace("\n", " ")
            leaks.append((stem, f"...{excerpt}..."))
            break  # one match per stem is enough
    return leaks


def audit_concept_file(
    path: Path, concept: str, min_words: int = 30, max_words: int = 250
) -> dict:
    """Audit a single concept's JSON file."""
    with open(path) as f:
        items = json.load(f)

    n = len(items)
    leak_items = []
    short_items = []
    long_items = []
    word_counts = []

    for item in items:
        text = item.get("text", "")
        n_words = len(text.split())
        word_counts.append(n_words)
        if n_words < min_words:
            short_items.append((item.get("id", "?"), n_words))
        if n_words > max_words:
            long_items.append((item.get("id", "?"), n_words))
        leaks = find_leaks(text, concept)
        if leaks:
            leak_items.append((item.get("id", "?"), leaks))

    return {
        "n_items": n,
        "leaks": leak_items,
        "short": short_items,
        "long": long_items,
        "min_words": min(word_counts) if word_counts else 0,
        "max_words": max(word_counts) if word_counts else 0,
        "mean_words": sum(word_counts) / len(word_counts) if word_counts else 0.0,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--paper", default="emotions")
    parser.add_argument("--data-root", default="drive_data")
    parser.add_argument("--min-words", type=int, default=30)
    parser.add_argument("--max-words", type=int, default=250)
    parser.add_argument("--strict", action="store_true",
                        help="Exit non-zero on length issues too, not just leaks")
    args = parser.parse_args()

    training_dir = Path(args.data_root) / "data" / args.paper / "training"
    if not training_dir.exists():
        logger.error("Training directory not found: %s", training_dir)
        return 2

    json_files = sorted(
        f for f in training_dir.glob("*.json") if f.stem != "all_stimuli"
    )
    if not json_files:
        logger.error("No concept files found in %s", training_dir)
        return 2

    logger.info("Auditing %d concept files in %s", len(json_files), training_dir)

    total_leaks = 0
    total_short = 0
    total_long = 0
    per_concept_summary: dict[str, dict] = {}

    for f in json_files:
        concept = f.stem
        report = audit_concept_file(f, concept, args.min_words, args.max_words)
        per_concept_summary[concept] = report
        total_leaks += len(report["leaks"])
        total_short += len(report["short"])
        total_long += len(report["long"])

    # Print summary
    print()
    print("=" * 70)
    print(f"STIMULUS AUDIT — paper={args.paper}, dir={training_dir}")
    print("=" * 70)
    print(f"  {'concept':14s}  {'n':>3s}  {'leaks':>5s}  {'short':>5s}  {'long':>4s}  {'words(min/mean/max)'}")
    print("-" * 70)
    for concept, report in per_concept_summary.items():
        print(
            f"  {concept:14s}  {report['n_items']:>3d}  "
            f"{len(report['leaks']):>5d}  "
            f"{len(report['short']):>5d}  "
            f"{len(report['long']):>4d}  "
            f"{report['min_words']:>3d} / {report['mean_words']:>5.1f} / {report['max_words']:>3d}"
        )
    print("-" * 70)
    print(
        f"  TOTAL leaks={total_leaks}, short(<{args.min_words}w)={total_short}, "
        f"long(>{args.max_words}w)={total_long}"
    )
    print()

    # Print leak details
    if total_leaks > 0:
        print("LEAK DETAILS:")
        for concept, report in per_concept_summary.items():
            if not report["leaks"]:
                continue
            print(f"\n  [{concept}]")
            for item_id, leaks in report["leaks"][:5]:  # cap at 5 per concept
                for stem, excerpt in leaks:
                    print(f"    {item_id} (stem '{stem}'): {excerpt}")
            if len(report["leaks"]) > 5:
                print(f"    ... and {len(report['leaks']) - 5} more in this concept")
        print()

    # Save full audit
    out_path = Path(args.data_root) / "results" / args.paper / "stimulus_audit.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path.with_suffix(".tmp"), "w") as f:
        json.dump(per_concept_summary, f, indent=2, default=str)
    out_path.with_suffix(".tmp").rename(out_path)
    print(f"Full audit saved to {out_path}")

    # Exit code
    has_problems = total_leaks > 0 or (args.strict and (total_short > 0 or total_long > 0))
    return 1 if has_problems else 0


if __name__ == "__main__":
    sys.exit(main())
