"""Generate stimuli/training/{true,false}.json from the paper's cities CSV."""

from __future__ import annotations

import csv
import io
import json
import random
import urllib.request
from pathlib import Path

CSV_URL = (
    "https://raw.githubusercontent.com/saprmarks/geometry-of-truth/main/"
    "datasets/cities.csv"
)
SEED = 42
PER_CONCEPT = 50

HERE = Path(__file__).resolve().parent
OUT_DIR = HERE / "stimuli" / "training"


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(CSV_URL) as r:
        text = r.read().decode("utf-8")
    rows = list(csv.DictReader(io.StringIO(text)))
    true_rows = [r for r in rows if r["label"] == "1"]
    false_rows = [r for r in rows if r["label"] == "0"]
    rng = random.Random(SEED)
    rng.shuffle(true_rows)
    rng.shuffle(false_rows)
    for concept, subset in [("true", true_rows[:PER_CONCEPT]),
                             ("false", false_rows[:PER_CONCEPT])]:
        items = [{"concept": concept, "text": r["statement"]} for r in subset]
        (OUT_DIR / f"{concept}.json").write_text(
            json.dumps(items, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        print(f"wrote {len(items)} items -> {OUT_DIR / f'{concept}.json'}")


if __name__ == "__main__":
    main()
