#!/usr/bin/env python
"""Step 01 DC solver: boundary-parameter calibration sweep."""

from __future__ import annotations

import sys
from pathlib import Path


WORKFLOW_ROOT = Path(__file__).resolve().parents[1]
if str(WORKFLOW_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKFLOW_ROOT))

from _legacy_entrypoint import run_legacy


if __name__ == "__main__":
    run_legacy("scripts/python/run_boundary_weight_sweep.py")
