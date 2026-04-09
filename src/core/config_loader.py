"""Load and validate paper configurations from YAML files.

This module is the single entry point for reading all YAML configs:
  - Paper configs:   config/papers/{paper_id}/paper_config.yaml
  - Stimuli configs: config/papers/{paper_id}/stimuli_config.yaml
  - Model matrix:    config/models.yaml
  - Active paper:    config/active_paper.txt

Usage:
    from src.core.config_loader import load_paper_config, get_active_paper

    paper = load_paper_config("emotions")
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
    """

    id: str
    title: str
    authors: str
    url: str
    original_model: str
    model_variant: str
    techniques_required: list[str]
    claims: list[ClaimConfig]


def load_paper_config(paper_id: str) -> PaperConfig:
    """Load and validate a paper config from YAML.

    Args:
        paper_id: Identifier matching a folder under config/papers/.

    Returns:
        Parsed PaperConfig with all claims.

    Raises:
        FileNotFoundError: If the config file doesn't exist.
        KeyError: If required fields are missing from the YAML.
    """
    project_root = get_project_root()
    config_path = project_root / "config" / "papers" / paper_id / "paper_config.yaml"

    if not config_path.exists():
        raise FileNotFoundError(f"Paper config not found: {config_path}")

    with open(config_path) as f:
        raw = yaml.safe_load(f)

    paper = raw["paper"]
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
            )
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


def load_stimuli_config(paper_id: str) -> dict[str, Any]:
    """Load stimuli config for a paper.

    Args:
        paper_id: Identifier matching a folder under config/papers/.

    Returns:
        Raw parsed YAML dict defining stimulus sets.

    Raises:
        FileNotFoundError: If the stimuli config doesn't exist.
    """
    project_root = get_project_root()
    config_path = project_root / "config" / "papers" / paper_id / "stimuli_config.yaml"

    if not config_path.exists():
        raise FileNotFoundError(f"Stimuli config not found: {config_path}")

    with open(config_path) as f:
        return yaml.safe_load(f)


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
