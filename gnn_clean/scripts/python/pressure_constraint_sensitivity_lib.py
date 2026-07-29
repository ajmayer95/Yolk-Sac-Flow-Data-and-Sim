"""Shared helpers for Step 3 pressure-constraint sensitivity."""

from __future__ import annotations

import math
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_GRAPH = PROJECT_ROOT / "datasets" / "harmonized_scaled_dataset.gpickle"
DEFAULT_STEP2_ROOT = PROJECT_ROOT / "outputs" / "dc" / "02_physics_weight_sweep"
DEFAULT_REPRESENTATIVE_CSV = DEFAULT_STEP2_ROOT / "representative_configurations.csv"
DEFAULT_REPRESENTATIVE_LABELS_CSV = (
    DEFAULT_STEP2_ROOT / "figures" / "representative_plot_labels.csv"
)
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "outputs" / "dc" / "03_pressure_constraint_sensitivity"

CONSTRAINT_ORDER = (
    "gauge_only",
    "equal_av",
    "equal_drop",
    "fixed_drop_10pa",
)
CONSTRAINT_DISPLAY = {
    "gauge_only": "Gauge only",
    "equal_av": "Equal A/V",
    "equal_drop": "Equal drop",
    "fixed_drop_10pa": "Fixed drop, 10 Pa",
}
CONSTRAINT_SPECS = {
    "gauge_only": {
        "pressure_constraints": ["gauge-only"],
        "alpha_pa": None,
        "description": "Gauge only: P[V_reference] = 0.",
    },
    "equal_av": {
        "pressure_constraints": ["equal-a-equal-v"],
        "alpha_pa": None,
        "description": "Equal arterial and equal venous pressures with gauge.",
    },
    "equal_drop": {
        "pressure_constraints": ["equal-av-pressure-drop"],
        "alpha_pa": None,
        "description": "Equal arterial-venous drops with gauge.",
    },
    "fixed_drop_10pa": {
        "pressure_constraints": ["mean-a-minus-v-alpha-equal-v"],
        "alpha_pa": 10.0,
        "description": "Mean arterial minus venous pressure equals 10 Pa with equal venous pressure.",
    },
}
REGIME_PREFIX = {
    "flow_prioritized": "F",
    "balanced": "B",
    "conservation_prioritized": "K",
    "correction_regularized": "C",
}


def representative_label(selection_category: str, selection_rank_within_regime: object) -> str:
    prefix = REGIME_PREFIX.get(str(selection_category), "R")
    try:
        rank = int(float(selection_rank_within_regime))
    except (TypeError, ValueError):
        rank = 0
    return f"{prefix}{rank}" if rank > 0 else prefix


def format_lambda_token(value: float) -> str:
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError(f"Lambda value must be positive and finite, got {value!r}")
    return f"{value:g}".replace(".", "p")


def poiseuille_pair_run_name(lambda_q: float, lambda_k: float) -> str:
    return f"poiseuille__q_{format_lambda_token(lambda_q)}__k_{format_lambda_token(lambda_k)}"


def gnn_constraint_run_name(step2_run_name: str, constraint_type: str) -> str:
    return f"{step2_run_name}__constraint_{constraint_type}"


def poiseuille_constraint_run_name(lambda_q: float, lambda_k: float, constraint_type: str) -> str:
    return f"{poiseuille_pair_run_name(lambda_q, lambda_k)}__constraint_{constraint_type}"


def expected_run_files() -> tuple[str, ...]:
    return ("summary.csv", "summary.yaml", "edge_predictions.csv", "node_predictions.csv")


def launcher_metadata_path(run_dir: Path) -> Path:
    return run_dir / "launcher_run_config.yaml"
