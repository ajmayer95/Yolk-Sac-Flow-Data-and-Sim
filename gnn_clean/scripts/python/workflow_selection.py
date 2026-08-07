"""Helpers for resolving default artifacts from prior workflow steps."""

from __future__ import annotations

import csv
import math
from pathlib import Path


def _safe_float(value: object) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return parsed if math.isfinite(parsed) else float("nan")


def resolve_balanced_dc_run_dir(step2_root: Path, explicit_run_dir: Path | None = None) -> Path:
    """Resolve the balanced DC Step 2 representative run directory."""

    if explicit_run_dir is not None:
        return explicit_run_dir.expanduser().resolve()

    step2_root = step2_root.expanduser().resolve()
    rep_csv = step2_root / "representative_configurations.csv"
    if not rep_csv.exists():
        raise FileNotFoundError(
            f"Missing {rep_csv}. Run DC Step 2 analysis first or pass --b1-run-dir."
        )

    with rep_csv.open("r", newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    balanced = [
        row
        for row in rows
        if str(row.get("selection_category", "")).strip() == "balanced"
    ]
    if not balanced:
        raise ValueError(
            f"No balanced representative found in {rep_csv}. Pass --b1-run-dir explicitly."
        )

    balanced.sort(
        key=lambda row: (
            _safe_float(row.get("selection_rank_within_regime")),
            _safe_float(row.get("selection_score")),
            str(row.get("run_name", "")),
        )
    )
    chosen = balanced[0]

    run_dir = str(chosen.get("run_dir", "")).strip()
    if run_dir:
        return Path(run_dir).expanduser().resolve()

    run_name = str(chosen.get("run_name", "")).strip()
    if not run_name:
        raise ValueError(
            f"Balanced representative in {rep_csv} is missing both run_dir and run_name."
        )
    return (step2_root / "gnn" / run_name).resolve()
