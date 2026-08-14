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


def _float_matches(value: object, target: float, atol: float = 1.0e-12) -> bool:
    parsed = _safe_float(value)
    return math.isfinite(parsed) and math.isclose(parsed, float(target), rel_tol=0.0, abs_tol=atol)


def _euclidean_error(row: dict[str, object]) -> float:
    flow = _safe_float(row.get("flow_rmse_nl_s"))
    kirchhoff = _safe_float(row.get("kirchhoff_rms_per_internal_node_nl_s"))
    if not (math.isfinite(flow) and math.isfinite(kirchhoff)):
        return float("inf")
    return math.hypot(flow, kirchhoff)


def load_dc_representative_rows(rep_csv: Path) -> list[dict[str, object]]:
    rep_csv = rep_csv.expanduser().resolve()
    if not rep_csv.exists():
        raise FileNotFoundError(f"Missing representative CSV: {rep_csv}")
    with rep_csv.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def load_dc_all_run_rows(step2_root: Path) -> list[dict[str, object]]:
    step2_root = step2_root.expanduser().resolve()
    all_runs_csv = step2_root / "physics_weight_all_runs.csv"
    if not all_runs_csv.exists():
        raise FileNotFoundError(f"Missing DC Step 2 all-runs CSV: {all_runs_csv}")
    with all_runs_csv.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def resolve_dc_representative_row(
    rep_csv: Path,
    lambda_q: float | None = None,
    lambda_k: float | None = None,
    lambda_delta: float | None = None,
) -> dict[str, object]:
    """Resolve a single Step 2 DC representative row.

    Default behavior selects the best balanced representative.
    If any lambda is explicitly provided, all three must be provided and must
    match a row in ``representative_configurations.csv``.
    """

    rows = load_dc_representative_rows(rep_csv)
    explicit = any(value is not None for value in (lambda_q, lambda_k, lambda_delta))
    if explicit:
        if None in {lambda_q, lambda_k, lambda_delta}:
            raise ValueError(
                "To select an explicit Step 2 configuration, provide all of "
                "--lambda-q, --lambda-k, and --lambda-delta."
            )
        matches = [
            row
            for row in rows
            if _float_matches(row.get("lambda_q"), float(lambda_q))
            and _float_matches(row.get("lambda_k"), float(lambda_k))
            and _float_matches(row.get("lambda_delta"), float(lambda_delta))
        ]
        if not matches:
            raise ValueError(
                "No Step 2 representative matches "
                f"(lambda_q={lambda_q}, lambda_k={lambda_k}, lambda_delta={lambda_delta}) "
                f"in {Path(rep_csv).expanduser().resolve()}."
            )
        matches.sort(
            key=lambda row: (
                _safe_float(row.get("selection_rank_within_regime")),
                _euclidean_error(row),
                str(row.get("run_name", "")),
            )
        )
        return matches[0]

    balanced = [
        row
        for row in rows
        if str(row.get("selection_category", "")).strip() == "balanced"
    ]
    if not balanced:
        raise ValueError(
            f"No balanced representative found in {Path(rep_csv).expanduser().resolve()}."
        )
    balanced.sort(
        key=lambda row: (
            _safe_float(row.get("selection_rank_within_regime")),
            _euclidean_error(row),
            str(row.get("run_name", "")),
        )
    )
    return balanced[0]


def resolve_dc_step2_row(
    step2_root: Path,
    lambda_q: float | None = None,
    lambda_k: float | None = None,
    lambda_delta: float | None = None,
) -> dict[str, object]:
    """Resolve a single Step 2 DC row.

    Default behavior selects the best balanced representative from
    ``representative_configurations.csv``.
    If explicit lambdas are provided, the match may come from the full
    ``physics_weight_all_runs.csv`` table so callers can target any completed
    Step 2 configuration, not only the saved representative subset.
    """

    step2_root = step2_root.expanduser().resolve()
    rep_csv = step2_root / "representative_configurations.csv"
    explicit = any(value is not None for value in (lambda_q, lambda_k, lambda_delta))
    if not explicit:
        return resolve_dc_representative_row(rep_csv)
    if None in {lambda_q, lambda_k, lambda_delta}:
        raise ValueError(
            "To select an explicit Step 2 configuration, provide all of "
            "--lambda-q, --lambda-k, and --lambda-delta."
        )
    rows = load_dc_all_run_rows(step2_root)
    matches = [
        row
        for row in rows
        if str(row.get("model_family", "")).strip() == "gnn"
        and _float_matches(row.get("lambda_q"), float(lambda_q))
        and _float_matches(row.get("lambda_k"), float(lambda_k))
        and _float_matches(row.get("lambda_delta"), float(lambda_delta))
    ]
    if not matches:
        raise ValueError(
            "No DC Step 2 GNN run matches "
            f"(lambda_q={lambda_q}, lambda_k={lambda_k}, lambda_delta={lambda_delta}) "
            f"in {(step2_root / 'physics_weight_all_runs.csv').resolve()}."
        )
    matches.sort(
        key=lambda row: (
            _safe_float(row.get("pareto_rank")),
            _euclidean_error(row),
            str(row.get("run_name", "")),
        )
    )
    return matches[0]


def resolve_balanced_dc_run_dir(step2_root: Path, explicit_run_dir: Path | None = None) -> Path:
    """Resolve the balanced DC Step 2 representative run directory."""

    if explicit_run_dir is not None:
        return explicit_run_dir.expanduser().resolve()

    step2_root = step2_root.expanduser().resolve()
    rep_csv = step2_root / "representative_configurations.csv"
    chosen = resolve_dc_representative_row(rep_csv)

    run_dir = str(chosen.get("run_dir", "")).strip()
    if run_dir:
        return Path(run_dir).expanduser().resolve()

    run_name = str(chosen.get("run_name", "")).strip()
    if not run_name:
        raise ValueError(
            f"Balanced representative in {rep_csv} is missing both run_dir and run_name."
        )
    return (step2_root / "gnn" / run_name).resolve()


def resolve_dc_run_dir(
    step2_root: Path,
    explicit_run_dir: Path | None = None,
    lambda_q: float | None = None,
    lambda_k: float | None = None,
    lambda_delta: float | None = None,
) -> Path:
    """Resolve a DC Step 2 representative run directory.

    Precedence:
    1. ``explicit_run_dir`` when provided
    2. explicit ``lambda_q/lambda_k/lambda_delta`` representative selection
    3. default balanced representative
    """

    if explicit_run_dir is not None:
        return explicit_run_dir.expanduser().resolve()

    step2_root = step2_root.expanduser().resolve()
    chosen = resolve_dc_step2_row(
        step2_root,
        lambda_q=lambda_q,
        lambda_k=lambda_k,
        lambda_delta=lambda_delta,
    )

    run_dir = str(chosen.get("run_dir", "")).strip()
    if run_dir:
        return Path(run_dir).expanduser().resolve()

    run_name = str(chosen.get("run_name", "")).strip()
    if not run_name:
        raise ValueError(f"Resolved Step 2 row in {step2_root} is missing both run_dir and run_name.")
    return (step2_root / "gnn" / run_name).resolve()
