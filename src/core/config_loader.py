"""Load and validate paper configurations from YAML files.

This module is the single entry point for reading all YAML configs:
  - Paper configs:   config/papers/{paper_id}/paper_config.yaml
                     (or config/papers/{paper_id}/replications/{replication_id}/paper_config.yaml
                      for papers that have per-replication layouts)
  - Stimuli configs: config/papers/{paper_id}/stimuli_config.yaml
                     (or .../replications/{replication_id}/stimuli_config.yaml)
  - Model matrix:    config/models.yaml
  - Active paper:    config/active_paper.txt

Per-replication layout (v4+): a paper may host multiple independent
replication attempts under ``config/papers/{paper_id}/replications/``.
Each replication owns its own paper_config.yaml, stimuli, results, and
writeup. See CONTRIBUTING.md for the naming convention.

Usage:
    from src.core.config_loader import load_paper_config, get_active_paper

    paper = load_paper_config("emotions")
    paper = load_paper_config("emotions", "emotions-zachgoldfine44-6models")
    print(paper.title, len(paper.claims), "claims")

    models = get_models_for_tier("small", model_variant="instruct")
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from src.core.claim import ClaimConfig
from src.utils.env import get_project_root

logger = logging.getLogger(__name__)


@dataclass
class PaperConfig:
    """Parsed representation of a paper's replication configuration.

    Attributes:
        id: Short identifier (e.g., 'emotions', 'ioi').
        title: Full paper title.
        authors: Author string (e.g., 'Sofroniew et al.').
        url: Link to the paper.
        original_model: What model the paper originally studied.
        model_variant: 'instruct' or 'base' -- determines which models
            from the matrix to use.
        techniques_required: List of technique module names needed
            (e.g., ['probes', 'contrastive', 'steering']).
        claims: List of ClaimConfig objects to replicate.
        paper_text_path: Relative path under config/papers/{id}/ to a
            markdown file containing the full paper text. Defaults to
            'paper.md'. The pipeline reads this once and exposes it via
            ``paper_text`` so experiments and critique agents can compare
            their behavior to the paper as ground truth.
        paper_text: Full text of the paper, loaded from
            ``paper_text_path`` if that file exists. Empty string if no
            paper.md is present (with a logged warning).
        replication_id: Identifier for this specific replication attempt
            (e.g., 'geometry_of_truth-tulaneadam-qwen_1_5b'). When set,
            downstream paths (results, stimuli, writeup) are namespaced
            under it. None means legacy un-namespaced layout.
        replication_metadata: Raw ``replication:`` section from the yaml
            (replicator, github_handle, date, models_tested, status,
            notes). Used by the README table generator. Empty dict if
            no ``replication:`` section is present.
    """

    id: str
    title: str
    authors: str
    url: str
    original_model: str
    model_variant: str
    techniques_required: list[str]
    claims: list[ClaimConfig]
    paper_text_path: str = "paper.md"
    paper_text: str = ""
    replication_id: str | None = None
    replication_metadata: dict[str, Any] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.replication_metadata is None:
            object.__setattr__(self, "replication_metadata", {})


def _resolve_paper_config_path(paper_id: str, replication_id: str | None) -> Path:
    """Return the path to paper_config.yaml for a given paper/replication.

    If ``replication_id`` is given and a per-replication config exists at
    ``config/papers/{paper_id}/replications/{replication_id}/paper_config.yaml``,
    that wins. Otherwise fall back to the paper-level
    ``config/papers/{paper_id}/paper_config.yaml``.

    This fallback is deliberate: it lets the plumbing land before any files
    move, and it means users can keep running legacy un-namespaced configs
    while contributors adopt the per-replication layout.
    """
    project_root = get_project_root()
    if replication_id:
        per_rep = (
            project_root
            / "config" / "papers" / paper_id
            / "replications" / replication_id / "paper_config.yaml"
        )
        if per_rep.exists():
            return per_rep
    return project_root / "config" / "papers" / paper_id / "paper_config.yaml"


def list_replications(paper_id: str) -> list[str]:
    """List replication_ids present under config/papers/{paper_id}/replications/.

    Returns an empty list if no per-replication layout is in use.
    """
    project_root = get_project_root()
    reps_dir = project_root / "config" / "papers" / paper_id / "replications"
    if not reps_dir.is_dir():
        return []
    return sorted(p.name for p in reps_dir.iterdir() if p.is_dir())


def load_paper_config(
    paper_id: str, replication_id: str | None = None,
) -> PaperConfig:
    """Load and validate a paper config from YAML.

    Args:
        paper_id: Identifier matching a folder under config/papers/.
        replication_id: Optional replication identifier. When given, the
            loader reads ``config/papers/{paper_id}/replications/{replication_id}/paper_config.yaml``
            (falling back to the paper-level config if that file doesn't
            exist yet). Also used to populate ``PaperConfig.replication_id``,
            which downstream code uses for namespacing result/writeup paths.

            If ``replication_id`` is None, the loader will read a
            ``replication:`` section from the yaml itself (if present) and
            use the ``id`` field there. Explicit argument wins over yaml.

    Returns:
        Parsed PaperConfig with all claims.

    Raises:
        FileNotFoundError: If the config file doesn't exist.
        KeyError: If required fields are missing from the YAML.
    """
    config_path = _resolve_paper_config_path(paper_id, replication_id)

    if not config_path.exists():
        raise FileNotFoundError(f"Paper config not found: {config_path}")

    with open(config_path) as f:
        raw = yaml.safe_load(f)

    paper = raw["paper"]

    # Resolve replication_id: explicit CLI arg wins, then yaml's
    # replication.id, then None (legacy).
    replication_meta = raw.get("replication", {}) or {}
    resolved_replication_id = (
        replication_id or replication_meta.get("id") or None
    )

    claims: list[ClaimConfig] = []
    for c in raw["claims"]:
        claims.append(
            ClaimConfig(
                paper_id=paper["id"],
                claim_id=c["id"],
                description=c["description"],
                experiment_type=c["experiment_type"],
                params=c.get("params", {}),
                success_metric=c["success_metric"],
                success_threshold=float(c["success_threshold"]),
                depends_on=c.get("depends_on"),
                notes=c.get("notes", ""),
                paper_section=c.get("paper_section", ""),
                replication_id=resolved_replication_id,
            )
        )

    # Paper text (paper.md) lives at the paper level, not the replication
    # level, because it is the shared oracle all replications check against.
    project_root = get_project_root()
    paper_text_path = paper.get("paper_text_path", "paper.md")
    paper_text_full_path = (
        project_root / "config" / "papers" / paper_id / paper_text_path
    )
    paper_text = ""
    if paper_text_full_path.exists():
        try:
            paper_text = paper_text_full_path.read_text(encoding="utf-8")
            logger.info(
                "Loaded paper text from %s (%d chars)",
                paper_text_full_path, len(paper_text),
            )
        except OSError as e:
            logger.warning(
                "Could not read paper text at %s: %s", paper_text_full_path, e
            )
    else:
        logger.warning(
            "No paper text found at %s. Critique agents and sanity checks "
            "will run without ground-truth context. Save the paper as "
            "markdown there to enable paper-as-oracle behavior.",
            paper_text_full_path,
        )

    return PaperConfig(
        id=paper["id"],
        title=paper["title"],
        authors=paper["authors"],
        url=paper["url"],
        original_model=paper["original_model"],
        model_variant=paper["model_variant"],
        techniques_required=raw["techniques_required"],
        claims=claims,
        paper_text_path=paper_text_path,
        paper_text=paper_text,
        replication_id=resolved_replication_id,
        replication_metadata=replication_meta,
    )


def load_model_config() -> dict[str, Any]:
    """Load the full model matrix from config/models.yaml.

    Returns:
        Raw parsed YAML dict. Top-level keys typically include 'models'
        with per-model entries specifying hf_id, size_tier, family, etc.

    Raises:
        FileNotFoundError: If models.yaml doesn't exist.
    """
    project_root = get_project_root()
    config_path = project_root / "config" / "models.yaml"

    if not config_path.exists():
        raise FileNotFoundError(f"Model config not found: {config_path}")

    with open(config_path) as f:
        return yaml.safe_load(f)


def load_stimuli_config(
    paper_id: str, replication_id: str | None = None,
) -> dict[str, Any]:
    """Load stimuli config for a paper (optionally per-replication).

    Args:
        paper_id: Identifier matching a folder under config/papers/.
        replication_id: Optional replication identifier. When given, the
            loader checks for
            ``config/papers/{paper_id}/replications/{replication_id}/stimuli_config.yaml``
            first, falling back to the paper-level file if absent.

    Returns:
        Raw parsed YAML dict defining stimulus sets.

    Raises:
        FileNotFoundError: If neither the per-replication nor the
            paper-level stimuli config exists.
    """
    project_root = get_project_root()
    paper_dir = project_root / "config" / "papers" / paper_id

    candidates: list[Path] = []
    if replication_id:
        candidates.append(
            paper_dir / "replications" / replication_id / "stimuli_config.yaml"
        )
    candidates.append(paper_dir / "stimuli_config.yaml")

    for path in candidates:
        if path.exists():
            with open(path) as f:
                return yaml.safe_load(f)

    raise FileNotFoundError(
        f"Stimuli config not found. Tried: {', '.join(str(p) for p in candidates)}"
    )


def get_models_for_tier(tier: str, model_variant: str = "instruct") -> list[dict[str, Any]]:
    """Get all models matching a size tier from the model matrix.

    Args:
        tier: One of 'small', 'medium', 'large'.
        model_variant: 'instruct' or 'base'. Currently returns all models
            in the tier -- the caller should filter by variant if the
            model matrix includes both.

    Returns:
        List of dicts, each containing 'key' plus all fields from the
        model's YAML entry (hf_id, size_tier, family, etc.).
    """
    config = load_model_config()
    models: list[dict[str, Any]] = []

    for key, model in config["models"].items():
        if model["size_tier"] == tier:
            models.append({"key": key, **model})

    return models


def get_active_paper() -> str:
    """Read the active paper ID from config/active_paper.txt.

    Returns:
        Stripped paper_id string.

    Raises:
        FileNotFoundError: If active_paper.txt doesn't exist.
    """
    project_root = get_project_root()
    path = project_root / "config" / "active_paper.txt"

    if not path.exists():
        raise FileNotFoundError(
            f"Active paper file not found: {path}. "
            f"Create it with: echo 'paper_id' > {path}"
        )

    return path.read_text().strip()
