"""Convert Azaria-Mitchell / Marks-Tegmark CSV datasets to harness stimulus JSON format.

Input:  /tmp/got_datasets/{dataset}.csv  (statement, label, ...)
Output: config/papers/geometry_of_truth/stimuli/{training/true.json, training/false.json,
         sp_en_trans.json, neg_cities.json, larger_than.json}

Each stimulus: {"id": str, "concept": "true"|"false", "text": str, "source": dataset_name}
"""
from __future__ import annotations

import csv
import json
import random
from pathlib import Path

DATA_SRC = Path("/tmp/got_datasets")
STIMULI_OUT = Path(
    "/Users/zacharygoldfine/Claude/repos/mechinterp-replication/config/papers/geometry_of_truth/stimuli"
)

SEED = 42
random.seed(SEED)


def load_csv(name: str) -> list[dict]:
    """Load a GoT-style CSV into a list of {statement, label} dicts."""
    rows: list[dict] = []
    with open(DATA_SRC / f"{name}.csv", newline="") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(r)
    return rows


def to_stimuli(rows: list[dict], source: str, n_per_concept: int | None = None) -> dict[str, list[dict]]:
    """Convert rows to per-concept stimulus lists.

    Returns {"true": [...], "false": [...]} where each item has id/concept/text/source.
    If n_per_concept is set, sample that many per concept (after a class-balanced shuffle).
    """
    by_concept: dict[str, list[dict]] = {"true": [], "false": []}
    for i, r in enumerate(rows):
        lbl = str(r.get("label", "")).strip()
        stmt = r.get("statement", "").strip()
        if not stmt or lbl not in ("0", "1"):
            continue
        concept = "true" if lbl == "1" else "false"
        by_concept[concept].append({
            "id": f"{source}_{concept}_{i:05d}",
            "concept": concept,
            "text": stmt,
            "source": source,
        })

    if n_per_concept is not None:
        for c in by_concept:
            random.shuffle(by_concept[c])
            by_concept[c] = by_concept[c][:n_per_concept]

    return by_concept


def write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)
    tmp.rename(path)
    print(f"wrote {path}  ({len(obj) if isinstance(obj, list) else 'dict'} items)")


def main() -> None:
    # ─── Training set: cities ────────────────────────────────────────────────
    cities = load_csv("cities")
    training = to_stimuli(cities, "cities", n_per_concept=100)
    print(f"cities totals: true={len(training['true'])}  false={len(training['false'])}")
    write_json(STIMULI_OUT / "training" / "true.json", training["true"])
    write_json(STIMULI_OUT / "training" / "false.json", training["false"])

    # ─── Generalization test sets (combined JSON files) ─────────────────────
    # Each test set = 50 true + 50 false, pooled into one list.
    for ds in ["sp_en_trans", "neg_cities", "larger_than"]:
        rows = load_csv(ds)
        by_concept = to_stimuli(rows, ds, n_per_concept=50)
        combined = by_concept["true"] + by_concept["false"]
        random.shuffle(combined)
        print(f"{ds} totals: true={len(by_concept['true'])}  false={len(by_concept['false'])}")
        write_json(STIMULI_OUT / f"{ds}.json", combined)


if __name__ == "__main__":
    main()
