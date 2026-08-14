#!/usr/bin/env python
"""Step 01 AC analysis: boundary-parameter calibration summaries."""

from __future__ import annotations

import sys
from pathlib import Path


WORKFLOW_ROOT = Path(__file__).resolve().parents[1]
if str(WORKFLOW_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKFLOW_ROOT))

from _legacy_entrypoint import run_legacy


if __name__ == "__main__":
    run_legacy("scripts/python/analyze_ac_boundary_parameter_calibration_legacy.py")
