#!/usr/bin/env python
"""Analyze completed Step 2 physics-weight sweep runs."""

from __future__ import annotations

import argparse
import csv
import math
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from physics_weight_sweep_lib import (
    WEIGHTING_REGIME_ORDER,
    category_scores,
    classify_weighting_regime,
    dominates,
    launcher_metadata_path,
    nondominated_sort,
    parse_scalar,
    percentile_scale,
    representative_sort_key,
)
from utils import load_yaml, write_yaml


NL_PER_M3 = 1.0e12


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument(
        "--representatives-only",
        action="store_true",
        help="Re-rank and rewrite representative outputs from existing summary CSVs without touching sweep summaries.",
    )
    return parser.parse_args()


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


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


def safe_float(value) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return result if math.isfinite(result) else float("nan")


REGIME_PREFIX = {
    "flow_prioritized": "F",
    "balanced": "B",
    "conservation_prioritized": "K",
    "correction_regularized": "C",
}

REGIME_SCORE_FORMULAS = {
    "flow_prioritized": "0.75*flow_rmse_nl_s + 0.25*kirchhoff_rms_per_internal_node_nl_s",
    "balanced": "0.5*flow_rmse_nl_s + 0.5*kirchhoff_rms_per_internal_node_nl_s",
    "conservation_prioritized": "0.25*flow_rmse_nl_s + 0.75*kirchhoff_rms_per_internal_node_nl_s",
    "correction_regularized": "0.5*flow_rmse_nl_s + 0.5*kirchhoff_rms_per_internal_node_nl_s",
}


def physical_selection_score(regime: str, flow_rmse_nl_s: float, kirchhoff_rms_nl_s: float) -> float:
    if not (math.isfinite(flow_rmse_nl_s) and math.isfinite(kirchhoff_rms_nl_s)):
        return float("nan")
    if regime == "flow_prioritized":
        return 0.75 * flow_rmse_nl_s + 0.25 * kirchhoff_rms_nl_s
    if regime == "conservation_prioritized":
        return 0.25 * flow_rmse_nl_s + 0.75 * kirchhoff_rms_nl_s
    if regime in {"balanced", "correction_regularized"}:
        return 0.5 * flow_rmse_nl_s + 0.5 * kirchhoff_rms_nl_s
    return float("nan")


def label_for_regime_rank(regime: str, rank: int) -> str:
    return f"{REGIME_PREFIX[regime]}{rank}"


def representative_label_rows(representatives: list[dict[str, object]]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for row in representatives:
        rows.append(
            {
                "plot_label": row["plot_label"],
                "run_name": row["run_name"],
                "selection_category": row["selection_category"],
                "selection_rank_within_regime": row["selection_rank_within_regime"],
                "selection_score": row["selection_score"],
                "selection_score_formula": row["selection_score_formula"],
                "selection_score_regime": row["selection_score_regime"],
                "lambda_q": row["lambda_q"],
                "lambda_k": row["lambda_k"],
                "lambda_delta": row["lambda_delta"],
                "flow_rmse_nl_s": row["flow_rmse_nl_s"],
                "kirchhoff_rms_per_internal_node_nl_s": row["kirchhoff_rms_per_internal_node_nl_s"],
                "delta_rms": row["delta_rms"],
            }
        )
    return rows


def validate_and_print_representatives(representatives: list[dict[str, object]]) -> None:
    for regime in WEIGHTING_REGIME_ORDER:
        subset = [row for row in representatives if str(row["selection_category"]) == regime]
        if not subset:
            continue
        subset.sort(key=lambda row: safe_float(row["selection_rank_within_regime"]))
        previous_score = -float("inf")
        for expected_rank, row in enumerate(subset, start=1):
            actual_rank = int(safe_float(row["selection_rank_within_regime"]))
            if actual_rank != expected_rank:
                raise AssertionError(
                    f"{regime}: expected consecutive ranks 1..N, got rank {actual_rank} at position {expected_rank}"
                )
            score = safe_float(row["selection_score"])
            if math.isfinite(previous_score) and score < previous_score - 1.0e-12:
                raise AssertionError(
                    f"{regime}: selection scores are not monotonic with rank ({score} < {previous_score})"
                )
            previous_score = score
        print(f"{regime}:")
        for row in subset:
            print(
                "  "
                f"{row['plot_label']}  {row['run_name']}  "
                f"score={safe_float(row['selection_score']):.12g}  "
                f"flow_rmse_nl_s={safe_float(row['flow_rmse_nl_s']):.12g}  "
                f"kirchhoff_rms_per_internal_node_nl_s={safe_float(row['kirchhoff_rms_per_internal_node_nl_s']):.12g}  "
                f"delta_rms={safe_float(row['delta_rms']):.12g}"
            )


def load_metadata(path: Path) -> dict[str, object]:
    payload = load_yaml(path)
    if not isinstance(payload, dict):
        raise RuntimeError(f"Unexpected launcher metadata format: {path}")
    return payload


def discover_run_dirs(input_root: Path) -> list[Path]:
    return sorted(
        path.parent
        for path in input_root.rglob("launcher_run_config.yaml")
        if path.is_file()
    )


def mean_or_nan(values: list[float]) -> float:
    finite = [value for value in values if math.isfinite(value)]
    if not finite:
        return float("nan")
    return float(sum(finite) / len(finite))


def std_or_nan(values: list[float]) -> float:
    finite = [value for value in values if math.isfinite(value)]
    if not finite:
        return float("nan")
    mu = mean_or_nan(finite)
    return float(math.sqrt(sum((value - mu) ** 2 for value in finite) / len(finite)))


def percentile_or_nan(values: list[float], percentile: float) -> float:
    finite = [value for value in values if math.isfinite(value)]
    if not finite:
        return float("nan")
    return float(np.percentile(np.asarray(finite, dtype=np.float64), percentile))


def compute_from_gnn_run(run_dir: Path, metadata: dict[str, object]) -> dict[str, object]:
    summary = {key: parse_scalar(value) for key, value in read_csv_rows(run_dir / "summary.csv")[0].items()}
    edge_rows = [{key: parse_scalar(value) for key, value in row.items()} for row in read_csv_rows(run_dir / "edge_predictions.csv")]
    node_rows = [{key: parse_scalar(value) for key, value in row.items()} for row in read_csv_rows(run_dir / "node_predictions.csv")]

    q_obs = []
    q_pred = []
    delta = []
    ratio = []
    for row in edge_rows:
        if bool(row.get("valid_observed_flow", False)):
            q_obs.append(safe_float(row.get("q_obs_m3_s")))
            q_pred.append(safe_float(row.get("q_pred_m3_s")))
        delta.append(safe_float(row.get("delta_e")))
        ratio.append(safe_float(row.get("Gcorr_over_G0")))

    q_obs_arr = np.asarray([value for value in q_obs if math.isfinite(value)], dtype=np.float64)
    q_pred_arr = np.asarray([value for value in q_pred if math.isfinite(value)], dtype=np.float64)
    residual_arr = q_pred_arr - q_obs_arr if q_obs_arr.size == q_pred_arr.size else np.asarray([], dtype=np.float64)
    flow_scale_nl_s = safe_float(summary.get("flow_scale_m3_s")) * NL_PER_M3

    internal_residual_nl_s: list[float] = []
    pressure_values: list[float] = []
    arterial_pressures: list[float] = []
    venous_pressures: list[float] = []
    for row in node_rows:
        pressure_pa = safe_float(row.get("pressure_pa"))
        if math.isfinite(pressure_pa):
            pressure_values.append(pressure_pa)
        role = str(row.get("boundary_role", ""))
        if role == "arterial":
            arterial_pressures.append(pressure_pa)
        elif role == "venous":
            venous_pressures.append(pressure_pa)
        else:
            internal_residual_nl_s.append(safe_float(row.get("kirchhoff_residual_m3_s")) * NL_PER_M3)

    finite_delta = [value for value in delta if math.isfinite(value)]
    finite_ratio = [value for value in ratio if math.isfinite(value)]
    delta_rms = math.sqrt(sum(value * value for value in finite_delta) / len(finite_delta)) if finite_delta else float("nan")
    delta_mean_abs = mean_or_nan([abs(value) for value in finite_delta])
    delta_saturation_fraction = safe_float(summary.get("exploration_delta_saturation_fraction"))
    if not math.isfinite(delta_saturation_fraction):
        sat_tol = 5.0e-3
        delta_saturation_fraction = mean_or_nan(
            [
                1.0 if (value <= -0.5 + sat_tol or value >= 0.5 - sat_tol) else 0.0
                for value in finite_delta
            ]
        )

    boundary_count = max(int(safe_float(summary.get("reduced_constraint_count"))), 0)
    boundary_l2 = safe_float(summary.get("pressure_solver_constraint_residual_l2"))
    boundary_residual_rms_pa = (
        boundary_l2 / math.sqrt(boundary_count)
        if boundary_count > 0 and math.isfinite(boundary_l2)
        else float("nan")
    )

    row = {
        "run_name": str(metadata["run_name"]),
        "model_family": "gnn",
        "lambda_q": float(metadata["lambda_q"]),
        "lambda_k": float(metadata["lambda_k"]),
        "lambda_b": float(metadata["lambda_b"]),
        "lambda_delta": float(metadata["lambda_delta"]),
        "message_passing_layers": int(metadata["message_passing_layers"]),
        "weighting_regime": classify_weighting_regime(
            float(metadata["lambda_q"]),
            float(metadata["lambda_k"]),
            float(metadata["lambda_delta"]),
        ),
        "flow_rmse_nl_s": float(np.sqrt(np.mean((residual_arr * NL_PER_M3) ** 2))) if residual_arr.size else float("nan"),
        "flow_mae_nl_s": float(np.mean(np.abs(residual_arr * NL_PER_M3))) if residual_arr.size else float("nan"),
        "flow_nrmse_median": (
            float(np.sqrt(np.mean((residual_arr * NL_PER_M3) ** 2))) / flow_scale_nl_s
            if residual_arr.size and math.isfinite(flow_scale_nl_s) and flow_scale_nl_s > 0.0
            else float("nan")
        ),
        "kirchhoff_rms_per_internal_node_nl_s": mean_or_nan([]),  # overwritten below
        "kirchhoff_mae_per_internal_node_nl_s": mean_or_nan([abs(value) for value in internal_residual_nl_s]),
        "kirchhoff_p95_abs_nl_s": percentile_or_nan([abs(value) for value in internal_residual_nl_s], 95.0),
        "kirchhoff_max_abs_nl_s": max((abs(value) for value in internal_residual_nl_s if math.isfinite(value)), default=float("nan")),
        "kirchhoff_rms_normalized_median": float("nan"),
        "pressure_min_pa": min((value for value in pressure_values if math.isfinite(value)), default=float("nan")),
        "pressure_max_pa": max((value for value in pressure_values if math.isfinite(value)), default=float("nan")),
        "pressure_mean_pa": mean_or_nan(pressure_values),
        "arterial_pressure_mean_pa": mean_or_nan(arterial_pressures),
        "venous_pressure_mean_pa": mean_or_nan(venous_pressures),
        "arterial_pressure_spread_pa": (
            max(arterial_pressures) - min(arterial_pressures)
            if len([v for v in arterial_pressures if math.isfinite(v)]) >= 2
            else float("nan")
        ),
        "venous_pressure_spread_pa": (
            max(venous_pressures) - min(venous_pressures)
            if len([v for v in venous_pressures if math.isfinite(v)]) >= 2
            else float("nan")
        ),
        "arterial_equality_residual_pa": safe_float(summary.get("arterial_equality_residual_pa")),
        "venous_equality_residual_pa": safe_float(summary.get("venous_equality_residual_pa")),
        "boundary_residual_rms_pa": boundary_residual_rms_pa,
        "boundary_residual_max_pa": safe_float(summary.get("pressure_solver_constraint_residual_max")),
        "delta_mean": mean_or_nan(finite_delta),
        "delta_std": std_or_nan(finite_delta),
        "delta_mean_abs": delta_mean_abs,
        "delta_rms": delta_rms,
        "delta_min_observed": min(finite_delta) if finite_delta else float("nan"),
        "delta_max_observed": max(finite_delta) if finite_delta else float("nan"),
        "delta_saturation_fraction": delta_saturation_fraction,
        "conductance_ratio_mean": mean_or_nan(finite_ratio),
        "conductance_ratio_median": percentile_or_nan(finite_ratio, 50.0),
        "conductance_ratio_min": min(finite_ratio) if finite_ratio else float("nan"),
        "conductance_ratio_max": max(finite_ratio) if finite_ratio else float("nan"),
        "conductance_ratio_p05": percentile_or_nan(finite_ratio, 5.0),
        "conductance_ratio_p95": percentile_or_nan(finite_ratio, 95.0),
        "convergence_status": True,
        "solver_success": True,
        "runtime_seconds": safe_float(summary.get("runtime_seconds")),
        "graph_path": summary.get("graph_path", ""),
        "output_dir": str(run_dir),
        "boundary_role_constraint_count": boundary_count,
    }
    if internal_residual_nl_s:
        row["kirchhoff_rms_per_internal_node_nl_s"] = math.sqrt(
            sum(value * value for value in internal_residual_nl_s) / len(internal_residual_nl_s)
        )
    if math.isfinite(row["kirchhoff_rms_per_internal_node_nl_s"]) and math.isfinite(flow_scale_nl_s) and flow_scale_nl_s > 0.0:
        row["kirchhoff_rms_normalized_median"] = row["kirchhoff_rms_per_internal_node_nl_s"] / flow_scale_nl_s
    if math.isfinite(row["pressure_min_pa"]) and math.isfinite(row["pressure_max_pa"]):
        row["pressure_range_pa"] = row["pressure_max_pa"] - row["pressure_min_pa"]
    else:
        row["pressure_range_pa"] = float("nan")
    return row


def compute_from_poiseuille_run(run_dir: Path, metadata: dict[str, object]) -> dict[str, object]:
    summary = {key: parse_scalar(value) for key, value in read_csv_rows(run_dir / "summary.csv")[0].items()}
    row = {
        "run_name": str(metadata["run_name"]),
        "model_family": "poiseuille_baseline",
        "lambda_q": float(metadata["lambda_q"]),
        "lambda_k": float(metadata["lambda_k"]),
        "lambda_b": float(metadata["lambda_b"]),
        "lambda_delta": float("nan"),
        "message_passing_layers": 0,
        "weighting_regime": "",
        "flow_rmse_nl_s": safe_float(summary.get("flow_rmse_nl_s")),
        "flow_mae_nl_s": safe_float(summary.get("flow_mae_nl_s")),
        "flow_nrmse_median": safe_float(summary.get("flow_nrmse_median")),
        "kirchhoff_rms_per_internal_node_nl_s": safe_float(summary.get("kirchhoff_rms_per_internal_node_nl_s")),
        "kirchhoff_mae_per_internal_node_nl_s": safe_float(summary.get("kirchhoff_mae_per_internal_node_nl_s")),
        "kirchhoff_p95_abs_nl_s": safe_float(summary.get("kirchhoff_p95_abs_nl_s")),
        "kirchhoff_max_abs_nl_s": safe_float(summary.get("kirchhoff_max_abs_nl_s")),
        "kirchhoff_rms_normalized_median": safe_float(summary.get("kirchhoff_rms_normalized_median")),
        "pressure_min_pa": safe_float(summary.get("pressure_min_pa")),
        "pressure_max_pa": safe_float(summary.get("pressure_max_pa")),
        "pressure_range_pa": safe_float(summary.get("pressure_range_pa")),
        "pressure_mean_pa": safe_float(summary.get("pressure_mean_pa")),
        "arterial_pressure_mean_pa": safe_float(summary.get("arterial_pressure_mean_pa")),
        "venous_pressure_mean_pa": safe_float(summary.get("venous_pressure_mean_pa")),
        "arterial_pressure_spread_pa": safe_float(summary.get("arterial_pressure_spread_pa")),
        "venous_pressure_spread_pa": safe_float(summary.get("venous_pressure_spread_pa")),
        "arterial_equality_residual_pa": safe_float(summary.get("arterial_equality_residual_pa")),
        "venous_equality_residual_pa": safe_float(summary.get("venous_equality_residual_pa")),
        "boundary_residual_rms_pa": safe_float(summary.get("boundary_residual_rms_pa")),
        "boundary_residual_max_pa": safe_float(summary.get("boundary_residual_max_pa")),
        "delta_mean": float("nan"),
        "delta_std": float("nan"),
        "delta_mean_abs": float("nan"),
        "delta_rms": float("nan"),
        "delta_min_observed": float("nan"),
        "delta_max_observed": float("nan"),
        "delta_saturation_fraction": float("nan"),
        "conductance_ratio_mean": float("nan"),
        "conductance_ratio_median": float("nan"),
        "conductance_ratio_min": float("nan"),
        "conductance_ratio_max": float("nan"),
        "conductance_ratio_p05": float("nan"),
        "conductance_ratio_p95": float("nan"),
        "convergence_status": bool(summary.get("solver_success", True)),
        "solver_success": bool(summary.get("solver_success", True)),
        "runtime_seconds": safe_float(summary.get("runtime_seconds")),
        "graph_path": summary.get("graph_path", ""),
        "output_dir": str(run_dir),
    }
    return row


def analyze_rows(rows: list[dict[str, object]]) -> tuple[list[dict[str, object]], list[dict[str, object]], list[dict[str, object]], list[dict[str, object]]]:
    gnn_rows = [row for row in rows if row["model_family"] == "gnn"]
    poiseuille_rows = [row for row in rows if row["model_family"] == "poiseuille_baseline"]

    valid_gnn = [
        row for row in gnn_rows
        if bool(row.get("solver_success"))
        and math.isfinite(safe_float(row.get("flow_rmse_nl_s")))
        and math.isfinite(safe_float(row.get("kirchhoff_rms_per_internal_node_nl_s")))
    ]
    flow_values = [safe_float(row["flow_rmse_nl_s"]) for row in valid_gnn]
    kirch_values = [safe_float(row["kirchhoff_rms_per_internal_node_nl_s"]) for row in valid_gnn]
    flow_p05 = percentile_or_nan(flow_values, 5.0)
    flow_p95 = percentile_or_nan(flow_values, 95.0)
    kirch_p05 = percentile_or_nan(kirch_values, 5.0)
    kirch_p95 = percentile_or_nan(kirch_values, 95.0)

    for row in gnn_rows:
        flow_scaled = percentile_scale(safe_float(row.get("flow_rmse_nl_s")), flow_p05, flow_p95)
        kirch_scaled = percentile_scale(
            safe_float(row.get("kirchhoff_rms_per_internal_node_nl_s")),
            kirch_p05,
            kirch_p95,
        )
        row["flow_rmse_p05"] = flow_p05
        row["flow_rmse_p95"] = flow_p95
        row["kirchhoff_rms_p05"] = kirch_p05
        row["kirchhoff_rms_p95"] = kirch_p95
        row["flow_rmse_rank_scaled"] = flow_scaled
        row["kirchhoff_rms_rank_scaled"] = kirch_scaled
        scores = category_scores(flow_scaled, kirch_scaled)
        row.update(scores)
        row["selection_score"] = float("nan")
        row["selection_score_formula"] = ""
        row["selection_score_regime"] = ""

    ranks = nondominated_sort(valid_gnn, ("flow_rmse_nl_s", "kirchhoff_rms_per_internal_node_nl_s"))
    for row in gnn_rows:
        row["pareto_rank"] = float("nan")
        row["is_pareto_front"] = False
        row["dominated"] = False
        row["n_dominating_runs"] = float("nan")
        row["n_dominated_runs"] = float("nan")

    for row, rank in zip(valid_gnn, ranks):
        row["pareto_rank"] = rank if rank is not None else float("nan")
        row["is_pareto_front"] = rank == 1
        row["dominated"] = rank is not None and rank > 1
        n_dominating = 0
        n_dominated = 0
        for other in valid_gnn:
            if other is row:
                continue
            if dominates(other, row, ("flow_rmse_nl_s", "kirchhoff_rms_per_internal_node_nl_s")):
                n_dominating += 1
            if dominates(row, other, ("flow_rmse_nl_s", "kirchhoff_rms_per_internal_node_nl_s")):
                n_dominated += 1
        row["n_dominating_runs"] = n_dominating
        row["n_dominated_runs"] = n_dominated
    representatives: list[dict[str, object]] = []
    for row in gnn_rows:
        row["selected_representative"] = False
        row["selection_rank_within_regime"] = float("nan")
        row["selection_category"] = row["weighting_regime"]
        row["plot_label"] = ""

    for regime in WEIGHTING_REGIME_ORDER:
        candidates = [row for row in valid_gnn if row["weighting_regime"] == regime]
        for row in candidates:
            row["selection_score"] = physical_selection_score(
                regime,
                safe_float(row.get("flow_rmse_nl_s")),
                safe_float(row.get("kirchhoff_rms_per_internal_node_nl_s")),
            )
            row["selection_score_formula"] = REGIME_SCORE_FORMULAS[regime]
            row["selection_score_regime"] = regime
        if regime == "correction_regularized":
            candidates.sort(
                key=lambda row: (
                    safe_float(row.get("selection_score")),
                    safe_float(row.get("delta_rms")),
                    str(row.get("run_name", "")),
                )
            )
        else:
            candidates.sort(
                key=lambda row: (
                    safe_float(row.get("selection_score")),
                    str(row.get("run_name", "")),
                )
            )
        for rank_index, row in enumerate(candidates[:4], start=1):
            row["selected_representative"] = True
            row["selection_rank_within_regime"] = rank_index
            row["selection_category"] = regime
            row["plot_label"] = label_for_regime_rank(regime, rank_index)
            representatives.append(row)
    for row in gnn_rows:
        if not row.get("selection_score_formula"):
            regime = str(row.get("weighting_regime", ""))
            row["selection_score"] = physical_selection_score(
                regime,
                safe_float(row.get("flow_rmse_nl_s")),
                safe_float(row.get("kirchhoff_rms_per_internal_node_nl_s")),
            )
            row["selection_score_formula"] = REGIME_SCORE_FORMULAS.get(regime, "")
            row["selection_score_regime"] = regime
    for row in poiseuille_rows:
        row["flow_rmse_rank_scaled"] = float("nan")
        row["kirchhoff_rms_rank_scaled"] = float("nan")
        row["flow_rmse_p05"] = float("nan")
        row["flow_rmse_p95"] = float("nan")
        row["kirchhoff_rms_p05"] = float("nan")
        row["kirchhoff_rms_p95"] = float("nan")
        row["pareto_rank"] = float("nan")
        row["is_pareto_front"] = False
        row["dominated"] = False
        row["n_dominating_runs"] = float("nan")
        row["n_dominated_runs"] = float("nan")
        row["score_flow_prioritized"] = float("nan")
        row["score_conservation_prioritized"] = float("nan")
        row["score_balanced"] = float("nan")
        row["score_correction_regularized"] = float("nan")
        row["selection_score"] = float("nan")
        row["selection_score_formula"] = ""
        row["selection_score_regime"] = ""
        row["selected_representative"] = False
        row["selection_rank_within_regime"] = float("nan")
        row["selection_category"] = ""
        row["plot_label"] = ""

    representatives.sort(
        key=lambda row: (
            WEIGHTING_REGIME_ORDER.index(str(row["weighting_regime"])),
            safe_float(row["selection_rank_within_regime"]),
        )
    )
    validate_and_print_representatives(representatives)
    return rows, gnn_rows, poiseuille_rows, representatives


def main() -> None:
    args = parse_args()
    input_root = args.input_root.expanduser().resolve()
    if args.representatives_only:
        gnn_rows = [
            {key: parse_scalar(value) for key, value in row.items()}
            for row in read_csv_rows(input_root / "physics_weight_gnn_summary.csv")
        ]
        all_rows = [
            {key: parse_scalar(value) for key, value in row.items()}
            for row in read_csv_rows(input_root / "physics_weight_all_runs.csv")
        ]
        poiseuille_rows = [
            {key: parse_scalar(value) for key, value in row.items()}
            for row in read_csv_rows(input_root / "physics_weight_poiseuille_summary.csv")
        ]
        _, _, _, representatives = analyze_rows(all_rows)
        write_rows(input_root / "representative_configurations.csv", representatives)
        figures_dir = input_root / "figures"
        write_rows(figures_dir / "representative_plot_labels.csv", representative_label_rows(representatives))
        write_yaml(
            input_root / "physics_weight_analysis.yaml",
            {
                "n_all_runs": len(all_rows),
                "n_gnn_runs": len(gnn_rows),
                "n_poiseuille_runs": len(poiseuille_rows),
                "n_representatives": len(representatives),
                "representative_csv": str(input_root / "representative_configurations.csv"),
                "representative_plot_labels_csv": str(figures_dir / "representative_plot_labels.csv"),
                "representatives_only": True,
            },
        )
        return
    run_dirs = discover_run_dirs(input_root)
    rows: list[dict[str, object]] = []
    for run_dir in run_dirs:
        metadata = load_metadata(launcher_metadata_path(run_dir))
        model_family = str(metadata["model_family"])
        summary_csv = run_dir / "summary.csv"
        if not summary_csv.exists():
            rows.append(
                {
                    "run_name": str(metadata["run_name"]),
                    "model_family": model_family,
                    "lambda_q": float(metadata["lambda_q"]),
                    "lambda_k": float(metadata["lambda_k"]),
                    "lambda_b": float(metadata["lambda_b"]),
                    "lambda_delta": float(metadata.get("lambda_delta", float("nan"))),
                    "message_passing_layers": int(metadata["message_passing_layers"]),
                    "convergence_status": False,
                    "solver_success": False,
                    "output_dir": str(run_dir),
                }
            )
            continue
        if model_family == "gnn":
            rows.append(compute_from_gnn_run(run_dir, metadata))
        else:
            rows.append(compute_from_poiseuille_run(run_dir, metadata))

    all_rows, gnn_rows, poiseuille_rows, representatives = analyze_rows(rows)
    write_rows(input_root / "physics_weight_all_runs.csv", all_rows)
    write_rows(input_root / "physics_weight_gnn_summary.csv", gnn_rows)
    write_rows(input_root / "physics_weight_poiseuille_summary.csv", poiseuille_rows)
    write_rows(input_root / "representative_configurations.csv", representatives)
    write_rows(input_root / "figures" / "representative_plot_labels.csv", representative_label_rows(representatives))
    write_yaml(
        input_root / "physics_weight_analysis.yaml",
        {
            "n_all_runs": len(all_rows),
            "n_gnn_runs": len(gnn_rows),
            "n_poiseuille_runs": len(poiseuille_rows),
            "n_representatives": len(representatives),
            "representative_csv": str(input_root / "representative_configurations.csv"),
            "representative_plot_labels_csv": str(input_root / "figures" / "representative_plot_labels.csv"),
        },
    )


if __name__ == "__main__":
    main()
