"""Environment detection and path resolution.

Detects whether we're running on MacBook (local dev), Colab (GPU), or other.
All data paths flow through get_data_root() -- never hardcode paths.

Usage:
    from src.utils.env import get_data_root, get_device, detect_environment

    data_root = get_data_root()          # Path to data/results on Google Drive
    device = get_device()                # "cuda", "mps", or "cpu"
    env = detect_environment()           # "colab", "macbook", or "other"
    project = get_project_root()         # Repo root (where CLAUDE.md lives)
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


def get_data_root() -> Path:
    """Return the root directory for all persistent data (stimuli, results, activations).

    Resolution order:
      1. MECHINTERP_DATA_ROOT environment variable (explicit override).
      2. Colab default: /content/drive/MyDrive/mechinterp-replication
      3. MacBook default: ./drive_data  (symlink to Google Drive)
      4. Fallback: ./local_data

    The returned path is NOT guaranteed to exist -- callers should mkdir as needed.
    """
    env_override = os.environ.get("MECHINTERP_DATA_ROOT")
    if env_override:
        return Path(env_override)

    env = detect_environment()
    if env == "colab":
        return Path("/content/drive/MyDrive/mechinterp-replication")
    elif env == "macbook":
        return Path("./drive_data")
    else:
        return Path("./local_data")


def get_device() -> str:
    """Return the best available torch device string for the current environment.

    Returns:
        'cuda' if NVIDIA GPU is available,
        'mps' if Apple Metal is available,
        'cpu' otherwise.

    Note: Some TransformerLens ops fail on MPS. Callers doing probe training
    locally may want to force 'cpu' even when 'mps' is available.
    """
    import torch

    if torch.cuda.is_available():
        return "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def get_project_root() -> Path:
    """Return the project root directory (where CLAUDE.md lives).

    Computed relative to this file's location:
        src/utils/env.py  ->  ../../  =  project root
    """
    return Path(__file__).resolve().parent.parent.parent
