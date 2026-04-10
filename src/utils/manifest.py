"""Run manifest: capture environment, code version, and library state.

A run manifest is a dict that gets attached to every ExperimentResult's
metadata field. It captures the state of the world when the experiment
ran, so that:

  - You can tell if a cached result is stale because the code changed
    (compare git SHAs).
  - You can tell if numerical differences across runs are explained by
    library version drift (different torch / sklearn / TL versions).
  - You can tell what device / dtype / precision the result was computed
    in (debugging fp16 vs fp32 mismatches, MPS vs CUDA, etc.).
  - You can pin a paper-replication report to a specific commit and
    library set so external readers can reproduce.

This was added in response to external critique #1-B which flagged the
caching system as fragile because there was no manifest keyed on these
properties.

Usage:
    from src.utils.manifest import build_run_manifest

    manifest = build_run_manifest(extra={"layer_subsampling": [0, 4, 8]})
    # ... store in ExperimentResult.metadata['run_manifest'] = manifest
"""

from __future__ import annotations

import logging
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def _get_git_sha(short: bool = True) -> str | None:
    """Return the current git commit SHA, or None if not in a git repo."""
    try:
        from src.utils.env import get_project_root
        repo_root = get_project_root()
        cmd = ["git", "rev-parse", "HEAD"]
        if short:
            cmd.insert(2, "--short")
        out = subprocess.check_output(
            cmd, cwd=str(repo_root), stderr=subprocess.DEVNULL,
            timeout=2,
        )
        return out.decode().strip()
    except Exception:
        return None


def _get_git_dirty() -> bool | None:
    """Return True if working tree has uncommitted changes, None on error."""
    try:
        from src.utils.env import get_project_root
        repo_root = get_project_root()
        out = subprocess.check_output(
            ["git", "status", "--porcelain"],
            cwd=str(repo_root), stderr=subprocess.DEVNULL,
            timeout=2,
        )
        return bool(out.decode().strip())
    except Exception:
        return None


def _get_lib_version(name: str) -> str | None:
    """Return the version string of an installed package, or None."""
    try:
        import importlib.metadata
        return importlib.metadata.version(name)
    except Exception:
        return None


def _get_torch_info() -> dict[str, Any]:
    """Return torch + CUDA / MPS environment info."""
    info: dict[str, Any] = {}
    try:
        import torch
        info["torch_version"] = torch.__version__
        info["cuda_available"] = bool(torch.cuda.is_available())
        if torch.cuda.is_available():
            info["cuda_device_name"] = torch.cuda.get_device_name(0)
            info["cuda_version"] = torch.version.cuda
        info["mps_available"] = (
            hasattr(torch.backends, "mps")
            and torch.backends.mps.is_available()
        )
    except Exception as e:
        info["torch_import_error"] = str(e)
    return info


def build_run_manifest(extra: dict[str, Any] | None = None) -> dict[str, Any]:
    """Build a manifest dict capturing the state of the world for a run.

    Captures:
      - timestamp (UTC ISO 8601)
      - git: short_sha, full_sha, dirty (uncommitted changes flag)
      - python: version, executable
      - platform: system, machine, processor
      - torch: version, cuda, mps, device names
      - libs: versions of transformer-lens, transformers, sklearn, scipy, numpy
      - extra: caller-supplied dict (e.g., layer subsampling, dtype choice,
               aggregation strategy, seed)

    The manifest is intentionally LIBERAL — it captures everything cheap
    and lets downstream consumers decide what they care about.
    """
    manifest: dict[str, Any] = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "git": {
            "short_sha": _get_git_sha(short=True),
            "full_sha": _get_git_sha(short=False),
            "dirty": _get_git_dirty(),
        },
        "python": {
            "version": sys.version.split()[0],
            "executable": sys.executable,
        },
        "platform": {
            "system": platform.system(),
            "machine": platform.machine(),
            "processor": platform.processor(),
        },
        "torch": _get_torch_info(),
        "libs": {
            "transformer_lens": _get_lib_version("transformer-lens"),
            "transformers": _get_lib_version("transformers"),
            "scikit-learn": _get_lib_version("scikit-learn"),
            "scipy": _get_lib_version("scipy"),
            "numpy": _get_lib_version("numpy"),
            "huggingface_hub": _get_lib_version("huggingface-hub"),
        },
    }
    if extra:
        manifest["extra"] = extra
    return manifest


def manifest_summary(manifest: dict[str, Any]) -> str:
    """Compact human-readable summary of a manifest, for log output."""
    git = manifest.get("git", {})
    sha = git.get("short_sha", "?")
    dirty = git.get("dirty")
    dirty_marker = "+dirty" if dirty else ""
    torch_info = manifest.get("torch", {})
    torch_v = torch_info.get("torch_version", "?")
    device = (
        "cuda" if torch_info.get("cuda_available")
        else "mps" if torch_info.get("mps_available")
        else "cpu"
    )
    lib_vs = manifest.get("libs", {})
    tl_v = lib_vs.get("transformer_lens", "?")
    return (
        f"git={sha}{dirty_marker} torch={torch_v}/{device} "
        f"transformer_lens={tl_v}"
    )
