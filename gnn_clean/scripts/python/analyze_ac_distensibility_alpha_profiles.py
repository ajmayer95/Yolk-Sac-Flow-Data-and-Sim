#!/usr/bin/env python
"""Compatibility wrapper for the organized AC Step 03 analysis entrypoint."""

from __future__ import annotations

from _workflow_entrypoint import run_workflow


if __name__ == "__main__":
    run_workflow("scripts/ac_workflow/analysis/03_distensibility_alpha_profiles_analysis.py")
