#!/usr/bin/env python
"""Compatibility wrapper for the organized AC Step 03 solver."""

from __future__ import annotations

from _workflow_entrypoint import run_workflow


if __name__ == "__main__":
    run_workflow("scripts/ac_workflow/solver/03_distensibility_alpha_profiles.py")
