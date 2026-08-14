#!/usr/bin/env python
"""Compatibility wrapper for the organized AC Step 00 plotting entrypoint."""

from __future__ import annotations

from _workflow_entrypoint import run_workflow


if __name__ == "__main__":
    run_workflow("scripts/ac_workflow/plotting/00_ideal_models_plots.py")
