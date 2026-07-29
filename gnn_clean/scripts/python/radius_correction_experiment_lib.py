"""Shared helpers for the Step 5 radius-refinement experiment."""

from __future__ import annotations

import csv
import math
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_GRAPH = PROJECT_ROOT / "datasets" / "harmonized_scaled_dataset.gpickle"
DEFAULT_STEP2_ROOT = PROJECT_ROOT / "outputs" / "dc" / "02_physics_weight_sweep"
DEFAULT_REPRESENTATIVE_CSV = DEFAULT_STEP2_ROOT / "representative_configurations.csv"
DEFAULT_REPRESENTATIVE_LABELS_CSV = (
    DEFAULT_STEP2_ROOT / "figures" / "representative_plot_labels.csv"
)
DEFAULT_TARGETED_EDGE_CSV = PROJECT_ROOT / "datasets" / "fix_vessels.csv"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "outputs" / "dc" / "05_radius_corrections"

SHARED_CONDITIONS = ("p_original", "g_original")
STRATEGY_ORDER = ("targeted_166", "low_snr_20pct")
STRATEGY_DISPLAY = {
    "targeted_166": "Targeted 166-vessel set",
    "low_snr_20pct": "Lowest-SNR 20%",
}
CONDITION_ORDER = (
    "p_original",
    "p_corrected",
    "g_original",
    "g_fixed",
    "g_retrained",
)
CONDITION_DISPLAY = {
    "p_original": "P-original",
    "p_corrected": "P-corrected",
    "g_original": "G-original",
    "g_fixed": "G-fixed",
    "g_retrained": "G-retrained",
}
REGIME_PREFIX = {
    "flow_prioritized": "F",
    "balanced": "B",
    "conservation_prioritized": "K",
    "correction_regularized": "C",
}


def representative_label(selection_category: str, rank: object) -> str:
    prefix = REGIME_PREFIX.get(str(selection_category), "R")
    try:
        rank_int = int(float(rank))
    except (TypeError, ValueError):
        rank_int = 0
    return f"{prefix}{rank_int}" if rank_int > 0 else prefix


def safe_float(value) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return result if math.isfinite(result) else float("nan")


def normalize_bool(value, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return default
    text = str(value).strip().lower()
    if text in {"1", "true", "t", "yes", "y"}:
        return True
    if text in {"0", "false", "f", "no", "n", ""}:
        return False
    return default


def expected_condition_files() -> tuple[str, ...]:
    return ("summary.csv", "summary.yaml", "edge_predictions.csv", "node_predictions.csv")


def write_rows(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    fieldnames: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def read_summary_csv(path: Path) -> dict[str, str]:
    rows = read_rows(path)
    if not rows:
        raise RuntimeError(f"Summary CSV is empty: {path}")
    return rows[0]


def read_representatives(
    representative_csv: Path,
    representative_labels_csv: Path | None = None,
) -> pd.DataFrame:
    reps = pd.read_csv(representative_csv)
    required = {
        "run_name",
        "lambda_q",
        "lambda_k",
        "lambda_delta",
        "selection_category",
        "selection_rank_within_regime",
    }
    missing = sorted(required - set(reps.columns))
    if missing:
        raise ValueError(f"Representative CSV missing required columns: {missing}")
    reps = reps.copy()
    reps["generated_label"] = [
        representative_label(row["selection_category"], row["selection_rank_within_regime"])
        for _, row in reps.iterrows()
    ]
    if representative_labels_csv is not None and representative_labels_csv.exists():
        labels = pd.read_csv(representative_labels_csv)
        if {"run_name", "plot_label"} <= set(labels.columns):
            reps = reps.merge(labels[["run_name", "plot_label"]], on="run_name", how="left")
    if "plot_label" not in reps.columns:
        reps["plot_label"] = reps["generated_label"]
    reps["plot_label"] = reps["plot_label"].fillna(reps["generated_label"])
    return reps


def select_step2_run(
    representative_csv: Path,
    representative_labels_csv: Path | None,
    selected_run_name: str | None,
    requested_representative_label: str | None,
) -> dict[str, object]:
    reps = read_representatives(representative_csv, representative_labels_csv)
    if selected_run_name:
        match = reps[reps["run_name"].astype(str) == str(selected_run_name)].copy()
        if match.empty:
            raise ValueError(f"Could not find selected run name {selected_run_name!r}")
        row = match.iloc[0]
    elif requested_representative_label:
        match = reps[reps["plot_label"].astype(str) == str(requested_representative_label)].copy()
        if match.empty:
            raise ValueError(
                f"Could not find representative label {requested_representative_label!r}"
            )
        row = match.iloc[0]
    else:
        balanced = reps[reps["selection_category"].astype(str) == "balanced"].copy()
        if balanced.empty:
            raise ValueError("No balanced representative found.")
        balanced["selection_rank_within_regime"] = pd.to_numeric(
            balanced["selection_rank_within_regime"], errors="coerce"
        )
        balanced = balanced.sort_values(
            ["selection_rank_within_regime", "run_name"], na_position="last"
        )
        row = balanced.iloc[0]
    run_name = str(row["run_name"])
    run_dir = step2_run_dir(DEFAULT_STEP2_ROOT, run_name)
    return {
        "run_name": run_name,
        "plot_label": str(row.get("plot_label", row.get("generated_label", ""))),
        "selection_category": str(row["selection_category"]),
        "selection_rank_within_regime": int(float(row["selection_rank_within_regime"])),
        "lambda_q": float(row["lambda_q"]),
        "lambda_k": float(row["lambda_k"]),
        "lambda_delta": float(row["lambda_delta"]),
    }


def step2_run_dir(step2_root: Path, run_name: str) -> Path:
    return step2_root / "gnn" / run_name


def shared_condition_dir(output_root: Path, condition_name: str) -> Path:
    return output_root / "shared" / condition_name


def strategy_dir(output_root: Path, strategy_name: str) -> Path:
    return output_root / strategy_name


def condition_dir(output_root: Path, strategy_name: str, condition_name: str) -> Path:
    if strategy_name == "shared":
        return shared_condition_dir(output_root, condition_name)
    return strategy_dir(output_root, strategy_name) / condition_name
