#!/usr/bin/env python
"""Compatibility wrapper for the organized AC Step 02 solver."""

from __future__ import annotations

from _workflow_entrypoint import run_workflow


if __name__ == "__main__":
    run_workflow("scripts/ac_workflow/solver/02_physics_weight_sweep.py")
