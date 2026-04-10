"""Environment detection and path resolution.

The harness writes to TWO roots, intentionally:

  1. ``get_data_root()`` — local-only cache + heavy artifacts. Activation
     caches, probe weights, per-stimulus tensors. **Never committed to git.**
     Defaults to ``<repo>/local_data/``. Override with the
     ``MECHINTERP_DATA_ROOT`` env var if you want to point at an external
     mount (e.g., a Colab Drive mount you want to keep across sessions).

  2. ``get_committed_artifacts_root()`` — small artifacts that document
     what happened. ``result.json``, ``sanity.json``, critique reports,
     ``*.png`` figures. **These ARE committed to git** (whitelisted in
     ``.gitignore``) so the user can see results land in real time and a
     fresh clone can rebuild the writeup without re-running experiments.
     Always ``<repo>/results/``. Not overridable.

Past versions of this module had a Google Drive symlink concept
(``./drive_data``) and a Colab ``/content/drive`` default. Both were
removed in the 2026-04-09 storage-simplification pass. The user couldn't
see where results were going, and nothing flowed to git during runs. Now
everything important lives in the repo.

Usage::

    from src.utils.env import (
        get_data_root,
        get_committed_artifacts_root,
        get_device,
        get_project_root,
    )

    cache_dir = get_data_root() / "cache" / "activations"
    result_path = get_committed_artifacts_root() / paper_id / model_key / claim_id / "result.json"
"""

from __future__ import annotations

import os
import platform
from pathlib import Path


def detect_environment() -> str:
    """Detect the current execution environment.

    Returns:
        'colab' if running inside Google Colab,
        'macbook' if running on an Apple Silicon Mac,
        'other' otherwise.
    """
    try:
        shell = get_ipython()  # type: ignore[name-defined]
        if "google.colab" in str(type(shell)):
            return "colab"
    except NameError:
        pass

    if platform.system() == "Darwin" and platform.processor() == "arm":
        return "macbook"

    return "other"


def get_project_root() -> Path:
    """Return the project root directory (where CLAUDE.md lives).

    Computed relative to this file's location:
        src/utils/env.py  ->  ../../  =  project root
    """
    return Path(__file__).resolve().parent.parent.parent


def get_data_root() -> Path:
    """Return the root for **local-only** caches and heavy artifacts.

    Resolution order:
      1. ``MECHINTERP_DATA_ROOT`` environment variable (explicit override).
      2. ``<repo>/local_data/``.

    Heavy artifacts that go here: per-stimulus activation tensors, probe
    weights, concept-vector ``.pt`` files, anything large enough that
    committing it to git would bloat the repo.

    The returned path is NOT guaranteed to exist — callers should mkdir
    as needed.
    """
    env_override = os.environ.get("MECHINTERP_DATA_ROOT")
    if env_override:
        return Path(env_override)
    return get_project_root() / "local_data"


def get_committed_artifacts_root() -> Path:
    """Return the root for **git-tracked** result artifacts.

    Always ``<repo>/results/``. Not overridable, on purpose: this is the
    canonical location for everything that should flow through git history
    during a run (small JSONs, sanity reports, critique reports, figures).

    The ``.gitignore`` is configured to ignore ``results/`` by default
    while whitelisting the specific small files that document what
    happened. New file types added under ``results/`` need a corresponding
    ``!results/...`` line in ``.gitignore`` to be picked up.
    """
    return get_project_root() / "results"


def get_device() -> str:
    """Return the best available torch device string for the current environment.

    Returns:
        'cuda' if NVIDIA GPU is available,
        'mps' if Apple Metal is available,
        'cpu' otherwise.

    Note: Some TransformerLens ops fail on MPS. Callers doing probe
    training locally may want to force 'cpu' even when 'mps' is available.
    """
    import torch

    if torch.cuda.is_available():
        return "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"
