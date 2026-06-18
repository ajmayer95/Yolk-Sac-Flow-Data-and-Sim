"""General utility helpers for the modular GNN edge-flow workflow.

This module contains small functions that are used across the package:
random seed control, device selection, safe numeric conversion, pickle
compatibility patches, and lightweight CSV/JSON writing.
"""

from __future__ import annotations

import csv
import importlib
import json
import math
import random
import sys
from pathlib import Path
from typing import Sequence

import numpy as np
import torch

def install_numpy_pickle_compat() -> None:
    """Install compatibility aliases for NetworkX pickles saved with NumPy 2.x names."""
    aliases = {
        "numpy._core": "numpy.core",
        "numpy._core.numeric": "numpy.core.numeric",
        "numpy._core.multiarray": "numpy.core.multiarray",
        "numpy._core._multiarray_umath": "numpy.core._multiarray_umath",
        "numpy._core.umath": "numpy.core.umath",
        "numpy._core.fromnumeric": "numpy.core.fromnumeric",
    }
    for new_name, old_name in aliases.items():
        if new_name not in sys.modules:
            try:
                sys.modules[new_name] = importlib.import_module(old_name)
            except Exception:
                pass


def set_seed(seed: int) -> None:
    """Set Python, NumPy, and PyTorch random seeds."""
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.backends.mps.is_available():
        try:
            torch.mps.manual_seed(int(seed))
        except Exception:
            pass


def resolve_device(name: str) -> torch.device:
    """Resolve a user device string into a PyTorch device."""
    if name == "cpu":
        return torch.device("cpu")
    if name == "mps":
        if not torch.backends.mps.is_available():
            raise SystemExit("Requested --device mps, but MPS is unavailable.")
        return torch.device("mps")
    return torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")


def safe_float(value, default: float = float("nan")) -> float:
    """Convert a value to a finite float, returning default on failure."""
    try:
        x = float(value)
    except (TypeError, ValueError):
        return default
    return x if math.isfinite(x) else default


def write_csv(path: Path, rows: Sequence[dict]) -> None:
    """Write a list of dictionaries to CSV."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, obj: object) -> None:
    """Write a JSON file with parent-directory creation."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, allow_nan=True)