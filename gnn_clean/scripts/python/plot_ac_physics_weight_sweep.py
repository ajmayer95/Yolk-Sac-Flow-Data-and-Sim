#!/usr/bin/env python
"""Compatibility wrapper for the organized AC Step 02 plotting entrypoint."""

from __future__ import annotations

from _workflow_entrypoint import run_workflow


if __name__ == "__main__":
    run_workflow("scripts/ac_workflow/plotting/02_physics_weight_sweep_plots.py")
