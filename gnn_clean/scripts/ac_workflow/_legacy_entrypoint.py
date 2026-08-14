#!/usr/bin/env python
"""Run an existing AC workflow script through its legacy entrypoint."""

from __future__ import annotations

import runpy
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def run_legacy(relative_path: str) -> None:
    target = PROJECT_ROOT / relative_path
    if not target.exists():
        raise FileNotFoundError(target)
    target_parent = str(target.parent)
    if target_parent not in sys.path:
        sys.path.insert(0, target_parent)
    project_root = str(PROJECT_ROOT)
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
    sys.argv[0] = str(target)
    runpy.run_path(str(target), run_name="__main__")
