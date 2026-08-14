#!/usr/bin/env python
"""Stage 1: fixed-admittance harmonic model comparison on a compatible graph."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from harmonic_utils import (  # noqa: E402
    DEG_PER_RAD,
    NL_PER_M3,
    best_f0_hz,
    build_harmonic_measurements,
    complex_rmse,
    edge_distensibility_values,
    edge_geometry_m,
    fixed_transmission_line_admittance,
    float_rmse,
    principal_phase_residual_deg,
    solve_complex_pressure_direct,
    solve_complex_pressure_with_nodal_injections,
    solve_complex_pressure_with_nodal_injections_and_flow,
    taylor_transmission_line_admittance,
    validate_edge_frequencies,
    wrap_phase_rad,
)
from real_data import MU, build_real_gnn_data, load_graph  # noqa: E402
from utils import resolve_device, set_random_seed, write_yaml  # noqa: E402
from workflow_selection import resolve_balanced_dc_run_dir  # noqa: E402


DEFAULT_DC_STEP2_ROOT = PROJECT_ROOT / "outputs" / "dc" / "02_physics_weight_sweep"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "ac" / "00_ideal_models" / "harmonic_stage1_admittance_model_comparison"

MODEL_SPECS = (
    ("full_ideal", "Full ideal admittance"),
    ("taylor_ideal", "Taylor-expanded ideal admittance"),
    ("taylor_dc_transferred", "Taylor-expanded + transferred DC conductance"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stage 1: fixed-admittance harmonic model comparison.")
    parser.add_argument("--graph-path", type=Path, required=True)
    parser.add_argument("--dc-step2-root", type=Path, default=DEFAULT_DC_STEP2_ROOT)
    parser.add_argument("--b1-run-dir", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--harmonic-number", type=int, default=1)
    parser.add_argument("--harmonic-numbers", type=int, nargs="+", default=None)
    parser.add_argument("--D0", type=float, default=1.0e-3)
    parser.add_argument("--alpha", type=float, default=0.0)
    parser.add_argument("--viscosity-pa-s", type=float, default=float(MU))
    parser.add_argument("--f0-hz", type=float, default=None)
    parser.add_argument("--boundary-amplitude-scale", type=float, default=1.0)
    parser.add_argument(
        "--arterial-boundary-mode",
        choices=("all", "per_tip_highest_snr"),
        default="all",
        help=(
            "How to choose arterial boundary injections for the hard source term. "
            "'all' uses every arterial boundary node. "
            "'per_tip_highest_snr' keeps only one arterial node per arterial tip, "
            "choosing the modeled node with the highest adjacent-edge SNR."
        ),
    )
    parser.add_argument(
        "--venous-boundary-mode",
        choices=("observed", "rebalance_to_sources"),
        default="observed",
        help=(
            "How to handle venous/sink harmonic injections. "
            "'observed' uses the measured sink phasors directly. "
            "'rebalance_to_sources' rescales the sink phasors together so the "
            "net harmonic boundary injection is exactly zero while preserving "
            "the selected arterial source phasors."
        ),
    )
    parser.add_argument("--phase-threshold-nl-s", type=float, default=None)
    parser.add_argument(
        "--pressure-solver-mode",
        choices=("pure_direct", "constrained_least_squares"),
        default="pure_direct",
    )
    parser.add_argument("--lambda-q", type=float, default=0.0)
    parser.add_argument("--lambda-k", type=float, default=1.0)
    parser.add_argument("--lambda-b", type=float, default=100.0)
    parser.add_argument(
        "--phase-offset",
        type=float,
        default=0.0,
        help="Deprecated compatibility option; must remain 0 for the common-phase constraint.",
    )
    parser.add_argument("--max-phase-iterations", type=int, default=20)
    parser.add_argument("--phase-tol", type=float, default=2.0e-3)
    parser.add_argument("--lstsq-backend", choices=("numpy", "torch"), default="numpy")
    parser.add_argument("--no-observed-flow-snr-weighting", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--require-cuda", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def requested_harmonic_numbers(args: argparse.Namespace) -> list[int]:
    values = args.harmonic_numbers if args.harmonic_numbers is not None else [int(args.harmonic_number)]
    harmonics: list[int] = []
    for value in values:
        harmonic = int(value)
        if harmonic < 1:
            raise ValueError(f"Harmonic numbers must be positive; got {harmonic}.")
        if harmonic not in harmonics:
            harmonics.append(harmonic)
    return harmonics


def minimal_real_data_config(seed: int) -> dict[str, object]:
    return {
        "training": {"seed": int(seed)},
        "data": {
            "include_boundary_nodes_in_pressure_solve": True,
            "split_fractions": {"train": 0.70, "validation": 0.15, "test": 0.15},
            "flow_normalization_reference_flux_nL_per_s": 1.0,
            "use_tilewise_flow_normalization": False,
        },
        "physics": {"use_observed_flow_snr_weighting": True},
    }


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


def read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)


def load_matplotlib_pyplot():
    try:
        import matplotlib.pyplot as plt
    except ModuleNotFoundError:
        return None
    return plt


def graph_positions(graph) -> dict[str, tuple[float, float]]:
    positions: dict[str, tuple[float, float]] = {}
    for node_id, node_data in graph.nodes(data=True):
        try:
            x = float(node_data.get("x", node_data.get("graph_x")))
            y = float(node_data.get("y", node_data.get("graph_y")))
        except (TypeError, ValueError):
            continue
        if math.isfinite(x) and math.isfinite(y):
            positions[str(node_id)] = (x, y)
    return positions


def phase_eval_threshold(valid_mask: np.ndarray, q_obs_nl_s: np.ndarray, explicit: float | None) -> float:
    if explicit is not None and math.isfinite(float(explicit)) and float(explicit) > 0.0:
        return float(explicit)
    amp_values = np.abs(q_obs_nl_s[valid_mask & np.isfinite(np.abs(q_obs_nl_s))])
    return max(1.0e-6, 1.0e-3 * float(np.median(amp_values))) if amp_values.size else 1.0e-6


def boundary_phase_difference_deg(arterial_pressures: np.ndarray) -> float:
    if arterial_pressures.size < 2:
        return float("nan")
    return float(abs(wrap_phase_rad(float(np.angle(arterial_pressures[0]) - np.angle(arterial_pressures[1]))) * DEG_PER_RAD))


def boundary_amplitude_difference_pa(arterial_pressures: np.ndarray) -> float:
    if arterial_pressures.size < 2:
        return float("nan")
    return float(abs(np.abs(arterial_pressures[0]) - np.abs(arterial_pressures[1])))


def phase_min_max_range_deg(values: np.ndarray) -> tuple[float, float, float]:
    phases = np.angle(values)
    finite = phases[np.isfinite(phases)]
    if finite.size == 0:
        return float("nan"), float("nan"), float("nan")
    anchor = np.angle(np.mean(np.exp(1j * finite)))
    wrapped = np.asarray([wrap_phase_rad(float(v - anchor)) for v in finite], dtype=np.float64)
    return (
        float(np.min(wrapped) * DEG_PER_RAD),
        float(np.max(wrapped) * DEG_PER_RAD),
        float((np.max(wrapped) - np.min(wrapped)) * DEG_PER_RAD),
    )


def rel_complex_error(numerator: np.ndarray, denominator: np.ndarray) -> np.ndarray:
    denom = np.abs(denominator)
    out = np.full(numerator.shape, np.nan, dtype=np.float64)
    mask = np.isfinite(np.abs(numerator)) & np.isfinite(denom) & (denom > 0.0)
    out[mask] = np.abs(numerator[mask]) / denom[mask]
    return out


def summarize_distribution(values: np.ndarray) -> dict[str, float]:
    finite = np.asarray(values[np.isfinite(values)], dtype=np.float64)
    if finite.size == 0:
        return {
            "mean": float("nan"),
            "median": float("nan"),
            "max": float("nan"),
            "p05": float("nan"),
            "p25": float("nan"),
            "p75": float("nan"),
            "p95": float("nan"),
            "fraction_lt_0p01": float("nan"),
            "fraction_lt_0p1": float("nan"),
            "fraction_lt_1": float("nan"),
        }
    return {
        "mean": float(np.mean(finite)),
        "median": float(np.median(finite)),
        "max": float(np.max(finite)),
        "p05": float(np.quantile(finite, 0.05)),
        "p25": float(np.quantile(finite, 0.25)),
        "p75": float(np.quantile(finite, 0.75)),
        "p95": float(np.quantile(finite, 0.95)),
        "fraction_lt_0p01": float(np.mean(finite < 0.01)),
        "fraction_lt_0p1": float(np.mean(finite < 0.1)),
        "fraction_lt_1": float(np.mean(finite < 1.0)),
    }


def safe_json_value(value):
    if isinstance(value, dict):
        return {str(k): safe_json_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [safe_json_value(v) for v in value]
    if isinstance(value, np.ndarray):
        return safe_json_value(value.tolist())
    if isinstance(value, (np.floating, float)):
        return float(value)
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, complex):
        return {"real": float(np.real(value)), "imag": float(np.imag(value))}
    return value


def magnitude_triplet(values: np.ndarray, *, nonzero_only: bool = False) -> dict[str, float]:
    mags = np.abs(np.asarray(values))
    finite = mags[np.isfinite(mags)]
    if nonzero_only:
        finite = finite[finite > 0.0]
    if finite.size == 0:
        return {"min": float("nan"), "median": float("nan"), "max": float("nan")}
    return {
        "min": float(np.min(finite)),
        "median": float(np.median(finite)),
        "max": float(np.max(finite)),
    }


def complex_vector_rows(node_ids: np.ndarray, node_indices: np.ndarray, values: np.ndarray) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for idx in np.asarray(node_indices, dtype=np.int64):
        z = complex(values[int(idx)])
        rows.append(
            {
                "node_index": int(idx),
                "node_id": str(node_ids[int(idx)]),
                "abs_pa": float(np.abs(z)),
                "phase_deg": float(np.angle(z) * DEG_PER_RAD),
                "real_pa": float(np.real(z)),
                "imag_pa": float(np.imag(z)),
            }
        )
    return rows


def write_pressure_diagnostics(
    *,
    path: Path,
    data,
    model_name: str,
    harmonic_number: int,
    q_obs_nl_s: np.ndarray,
    valid_mask: np.ndarray,
    pressure: np.ndarray,
    solve: dict[str, object],
    source_vector_nl_s: np.ndarray,
    admittance_diag: np.ndarray,
    admittance_off: np.ndarray,
    arterial_idx: np.ndarray,
    venous_idx: np.ndarray,
    lambda_q: float,
    lambda_k: float,
    lambda_b: float,
) -> None:
    node_ids = np.asarray(data.node_id.tolist(), dtype=object)
    laplacian = np.asarray(solve["laplacian_m3_s_per_pa"], dtype=np.complex128)
    flow_matrix = np.asarray(solve["flow_matrix_m3_s_per_pa"], dtype=np.complex128)
    source_vector_m3_s = np.asarray(solve["source_vector_m3_s"], dtype=np.complex128)
    q_pred_m3_s = np.asarray(solve["flow_pred_m3_s"], dtype=np.complex128)
    q_obs_m3_s = np.asarray(q_obs_nl_s, dtype=np.complex128) / NL_PER_M3
    nodal_balance_m3_s = np.asarray(solve["nodal_balance_m3_s"], dtype=np.complex128)
    nodal_residual_m3_s = np.asarray(solve["nodal_residual_m3_s"], dtype=np.complex128)
    ls_info = dict(solve.get("lsqr_info", {}))
    block_diag = dict(ls_info.get("block_diagnostics", {}))
    laplacian_scale = float(ls_info.get("laplacian_scale", 1.0))
    flow_scale = float(ls_info.get("flow_scale", 1.0))

    valid_mask = np.asarray(valid_mask, dtype=bool)
    flow_rows = np.flatnonzero(valid_mask)
    q_resid_m3_s = q_pred_m3_s[flow_rows] - q_obs_m3_s[flow_rows]
    r_k_raw = nodal_balance_m3_s - source_vector_m3_s
    r_k_scaled = math.sqrt(lambda_k) * (r_k_raw / max(laplacian_scale, 1.0e-30))
    r_q_scaled = math.sqrt(lambda_q) * (q_resid_m3_s / max(flow_scale, 1.0e-30))

    phase_rows = np.zeros((0, 2 * len(pressure)), dtype=np.float64)
    phase_rhs = np.zeros((0,), dtype=np.float64)
    r_b_raw = np.zeros((0,), dtype=np.float64)
    r_b_scaled = np.zeros((0,), dtype=np.float64)
    if lambda_b > 0.0 and arterial_idx.size > 0:
        from harmonic_utils import phase_only_constraint_rows  # local import for exact solve-time rows

        phase_rows_t, phase_rhs_t, phase_phi, amp_floor = phase_only_constraint_rows(
            pressures=pressure,
            arterial_idx=arterial_idx,
            num_nodes=len(pressure),
            device=resolve_device("cpu"),
        )
        phase_rows = phase_rows_t.detach().cpu().numpy().astype(np.float64, copy=False)
        phase_rhs = phase_rhs_t.detach().cpu().numpy().astype(np.float64, copy=False)
        pressure_real = np.concatenate([np.real(pressure), np.imag(pressure)])
        r_b_raw = phase_rows @ pressure_real - phase_rhs
        r_b_scaled = math.sqrt(lambda_b) * r_b_raw
    else:
        phase_phi = float("nan")
        amp_floor = float("nan")

    top_idx = np.argsort(np.abs(pressure))[-10:][::-1]
    optimal_offset = complex(np.mean(pressure))
    shifted_pressure = pressure - optimal_offset
    shifted_q_pred_m3_s = flow_matrix @ shifted_pressure
    shifted_r_k_raw = laplacian @ shifted_pressure - source_vector_m3_s

    diag = {
        "model_name": model_name,
        "harmonic_number": int(harmonic_number),
        "admittance_scale": {
            "abs_Ys": magnitude_triplet(admittance_diag),
            "abs_Yt": magnitude_triplet(admittance_off),
            "laplacian_nonzero_abs": magnitude_triplet(laplacian, nonzero_only=True),
            "observed_H1_flow_amplitude_nl_s": magnitude_triplet(q_obs_nl_s[valid_mask]),
            "pressure_scale_from_median_Q_over_median_Y_pa": (
                float(np.median(np.abs(q_obs_m3_s[flow_rows]))) / max(magnitude_triplet(admittance_diag)["median"], 1.0e-30)
                if flow_rows.size
                else float("nan")
            ),
            "pressure_scale_from_max_Q_over_median_Y_pa": (
                float(np.max(np.abs(q_obs_m3_s[flow_rows]))) / max(magnitude_triplet(admittance_diag)["median"], 1.0e-30)
                if flow_rows.size
                else float("nan")
            ),
            "pressure_scale_from_median_Q_over_min_laplacian_pa": (
                float(np.median(np.abs(q_obs_m3_s[flow_rows]))) / max(magnitude_triplet(laplacian, nonzero_only=True)["min"], 1.0e-30)
                if flow_rows.size
                else float("nan")
            ),
            "pressure_scale_from_max_Q_over_min_laplacian_pa": (
                float(np.max(np.abs(q_obs_m3_s[flow_rows]))) / max(magnitude_triplet(laplacian, nonzero_only=True)["min"], 1.0e-30)
                if flow_rows.size
                else float("nan")
            ),
        },
        "objective_blocks": block_diag,
        "final_real_matrix": {
            "shape": ls_info.get("final_real_matrix_diagnostics", {}).get("shape", []),
            "rank": ls_info.get("rank"),
            "largest_singular_value": (
                ls_info.get("largest_singular_values", [float("nan")])[0]
                if ls_info.get("largest_singular_values")
                else float("nan")
            ),
            "smallest_nonzero_singular_value": ls_info.get("smallest_nonzero_singular_value", float("nan")),
            "condition_number": ls_info.get("acond", float("nan")),
            "smallest_singular_values": ls_info.get("smallest_singular_values", []),
            "largest_singular_values": ls_info.get("largest_singular_values", []),
        },
        "pressure_solution": {
            "abs_pressure_pa": magnitude_triplet(pressure),
            "top10_nodes": complex_vector_rows(node_ids, top_idx, pressure),
            "arterial_anchor_pressures": complex_vector_rows(node_ids, arterial_idx, pressure),
            "venous_boundary_pressures": complex_vector_rows(node_ids, venous_idx, pressure),
        },
        "residual_decomposition": {
            "kirchhoff_raw_norm_m3_s": float(np.linalg.norm(r_k_raw)),
            "kirchhoff_raw_rms_m3_s": float(complex_rmse(r_k_raw)),
            "kirchhoff_scaled_norm": float(np.linalg.norm(r_k_scaled)),
            "kirchhoff_scaled_rms": float(complex_rmse(r_k_scaled)),
            "kirchhoff_max_abs_nl_s": float(np.max(np.abs(r_k_raw)) * NL_PER_M3) if r_k_raw.size else 0.0,
            "edge_flow_raw_norm_m3_s": float(np.linalg.norm(q_resid_m3_s)),
            "edge_flow_raw_rms_m3_s": float(complex_rmse(q_resid_m3_s)),
            "edge_flow_scaled_norm": float(np.linalg.norm(r_q_scaled)),
            "edge_flow_scaled_rms": float(complex_rmse(r_q_scaled)),
            "edge_flow_max_abs_nl_s": float(np.max(np.abs(q_resid_m3_s)) * NL_PER_M3) if q_resid_m3_s.size else 0.0,
            "arterial_phase_raw_norm": float(np.linalg.norm(r_b_raw)),
            "arterial_phase_raw_rms": float(np.sqrt(np.mean(np.square(r_b_raw)))) if r_b_raw.size else 0.0,
            "arterial_phase_scaled_norm": float(np.linalg.norm(r_b_scaled)),
            "arterial_phase_scaled_rms": float(np.sqrt(np.mean(np.square(r_b_scaled)))) if r_b_scaled.size else 0.0,
            "arterial_phase_common_phase_deg": float(phase_phi * DEG_PER_RAD) if np.isfinite(phase_phi) else float("nan"),
            "arterial_phase_amplitude_floor_pa": float(amp_floor),
        },
        "nullspace_gauge": {
            **dict(ls_info.get("nullspace_diagnostics", {})),
            "optimal_constant_offset_real_pa": float(np.real(optimal_offset)),
            "optimal_constant_offset_imag_pa": float(np.imag(optimal_offset)),
            "shifted_abs_pressure_pa": magnitude_triplet(shifted_pressure),
            "shifted_flow_change_norm_m3_s": float(np.linalg.norm(shifted_q_pred_m3_s - q_pred_m3_s)),
            "shifted_kirchhoff_change_norm_m3_s": float(np.linalg.norm(shifted_r_k_raw - r_k_raw)),
        },
        "source_vector_abs_nl_s": magnitude_triplet(source_vector_nl_s),
        "final_real_matrix_extra": dict(ls_info.get("final_real_matrix_diagnostics", {})),
    }
    path.write_text(json.dumps(safe_json_value(diag), indent=2), encoding="utf-8")


def resolve_boundary_mappings(graph, node_ids: np.ndarray, node_index: dict[object, int]) -> list[dict[str, object]]:
    mappings: list[dict[str, object]] = []
    for boundary_node, node_data in graph.nodes(data=True):
        boundary_type = str(node_data.get("boundary_type", ""))
        if boundary_type not in {"source", "sink"}:
            continue
        if boundary_node in node_index:
            neighbors = [neighbor for neighbor in graph.neighbors(boundary_node)]
            if len(neighbors) != 1:
                raise ValueError(f"Boundary node {boundary_node!r} does not have a unique adjacent edge.")
            modeled_node = boundary_node
            neighbor = neighbors[0]
        else:
            neighbors = [neighbor for neighbor in graph.neighbors(boundary_node) if neighbor in node_index]
            if len(neighbors) != 1:
                raise ValueError(
                    f"Boundary node {boundary_node!r} does not map unambiguously to a single modeled node."
                )
            neighbor = neighbors[0]
            modeled_node = neighbor
        mappings.append(
            {
                "boundary_node": boundary_node,
                "modeled_node": modeled_node,
                "modeled_node_index": int(node_index[modeled_node]),
                "adjacent_graph_neighbor": neighbor,
                "boundary_type": boundary_type,
                "edge_data": graph.edges[boundary_node, neighbor],
                "boundary_node_data": node_data,
            }
        )
    mappings.sort(key=lambda row: (0 if row["boundary_type"] == "source" else 1, str(row["boundary_node"])))
    return mappings


def _finite_boundary_snr_values(edge_data: dict[str, object]) -> list[float]:
    values: list[float] = []
    for key in (
        "snr_f0_piv",
        "snr_harm_fit_db_piv",
        "snr_ac_fit_db_piv",
        "_h_total_snr",
        "snr_pulse",
        "snr_db",
        "Q_DC_snr_db",
        "mean_Q_snr_db",
    ):
        try:
            value = float(edge_data.get(key))
        except (TypeError, ValueError):
            continue
        if math.isfinite(value):
            values.append(value)
    return values


def arterial_boundary_snr_score(graph, mapping: dict[str, object]) -> float:
    modeled_node = mapping["modeled_node"]
    best = float("-inf")
    for neighbor in graph.neighbors(modeled_node):
        if neighbor == mapping["boundary_node"]:
            continue
        edge_data = graph.edges[modeled_node, neighbor]
        if edge_data.get("synthetic_boundary_edge"):
            continue
        for value in _finite_boundary_snr_values(edge_data):
            best = max(best, float(value))
    if math.isfinite(best):
        return float(best)
    boundary_edge = graph.edges[mapping["boundary_node"], mapping["adjacent_graph_neighbor"]]
    boundary_values = _finite_boundary_snr_values(boundary_edge)
    if boundary_values:
        return float(max(boundary_values))
    return float("-inf")


def select_boundary_mappings(graph, mappings: list[dict[str, object]], arterial_boundary_mode: str) -> list[dict[str, object]]:
    if str(arterial_boundary_mode) == "all":
        return mappings

    selected_sources: list[dict[str, object]] = []
    sinks = [mapping for mapping in mappings if mapping["boundary_type"] == "sink"]
    source_groups: dict[object, list[dict[str, object]]] = {}
    ungrouped_sources: list[dict[str, object]] = []
    for mapping in mappings:
        if mapping["boundary_type"] != "source":
            continue
        boundary_node_data = mapping.get("boundary_node_data", {})
        tip = boundary_node_data.get("cut_boundary_origin_tip")
        if tip is None:
            ungrouped_sources.append(mapping)
            continue
        source_groups.setdefault(tip, []).append(mapping)

    for tip, group in sorted(source_groups.items(), key=lambda item: item[0]):
        ranked = sorted(
            group,
            key=lambda mapping: (
                arterial_boundary_snr_score(graph, mapping),
                float(mapping.get("boundary_node_data", {}).get("cut_boundary_weight", float("-inf"))),
                str(mapping["boundary_node"]),
            ),
            reverse=True,
        )
        selected_sources.append(ranked[0])

    selected_sources.extend(ungrouped_sources)
    selected = selected_sources + sinks
    selected.sort(key=lambda row: (0 if row["boundary_type"] == "source" else 1, str(row["boundary_node"])))
    return selected


def phase_constraint_indices(
    data,
    active_mappings: list[dict[str, object]],
    arterial_boundary_mode: str,
) -> np.ndarray:
    if str(arterial_boundary_mode) != "per_tip_highest_snr":
        return data.arterial_node_indices.detach().cpu().numpy().astype(np.int64).flatten()
    selected = sorted(
        {
            int(mapping["modeled_node_index"])
            for mapping in active_mappings
            if mapping["boundary_type"] == "source"
        }
    )
    return np.asarray(selected, dtype=np.int64)


def validate_boundary_mapping_edges(data, mappings: list[dict[str, object]]) -> None:
    for mapping in mappings:
        boundary_node = mapping["boundary_node"]
        adjacent_node = mapping["adjacent_graph_neighbor"]
        if boundary_node == adjacent_node:
            raise ValueError(f"Boundary mapping for node {boundary_node!r} uses itself as its adjacent graph neighbor.")
        match_count = sum(
            1
            for u, v in data.edge_ids
            if (u == boundary_node and v == adjacent_node) or (u == adjacent_node and v == boundary_node)
        )
        if match_count != 1:
            raise ValueError(
                f"Boundary edge ({boundary_node!r}, {adjacent_node!r}) appears {match_count} times in data.edge_ids; expected exactly once."
            )


def validate_b1_edge_compatibility(data, edge_df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    edge_df = edge_df.copy()
    for column in ("edge_id", "source_index", "target_index", "delta_e", "Gcorr_over_G0", "q_pred_m3_s"):
        if column in edge_df.columns:
            edge_df[column] = pd.to_numeric(edge_df[column], errors="coerce")
    edge_df["source"] = edge_df["source"].astype(str)
    edge_df["target"] = edge_df["target"].astype(str)
    src_idx = data.edge_index[0].detach().cpu().numpy().astype(np.int64)
    dst_idx = data.edge_index[1].detach().cpu().numpy().astype(np.int64)
    has_delta = "delta_e" in edge_df.columns
    has_ratio = "Gcorr_over_G0" in edge_df.columns
    delta = np.zeros(len(data.edge_ids), dtype=np.float64)
    ratio = np.ones(len(data.edge_ids), dtype=np.float64)
    expected_edges: list[tuple[int, tuple[object, object]]] = list(enumerate(data.edge_ids))
    strict_index_check = True
    if len(edge_df) != len(expected_edges):
        boundary_mask = np.asarray(
            data.source_node_mask.detach().cpu().numpy() | data.sink_node_mask.detach().cpu().numpy(),
            dtype=bool,
        )
        expected_edges = [
            (edge_id, (u, v))
            for edge_id, (u, v) in enumerate(data.edge_ids)
            if not (bool(boundary_mask[int(src_idx[edge_id])]) or bool(boundary_mask[int(dst_idx[edge_id])]))
        ]
        strict_index_check = False
    if len(edge_df) != len(expected_edges):
        raise ValueError(
            f"B1 edge_predictions.csv has {len(edge_df)} rows, expected either {len(data.edge_ids)} full-graph rows "
            f"or {len(expected_edges)} non-boundary rows."
        )
    for local_edge_id, (edge_id, (u, v)) in enumerate(expected_edges):
        row = edge_df.iloc[local_edge_id]
        if strict_index_check and int(row["edge_id"]) != edge_id:
            raise ValueError(f"B1 edge row {edge_id} has mismatched edge_id={row['edge_id']}.")
        if (not strict_index_check) and int(row["edge_id"]) != local_edge_id:
            raise ValueError(f"B1 edge row {local_edge_id} has mismatched edge_id={row['edge_id']}.")
        if row["source"] != str(u) or row["target"] != str(v):
            raise ValueError(f"B1 edge row {edge_id} does not match graph edge {(u, v)}.")
        if strict_index_check and (
            int(row["source_index"]) != int(src_idx[edge_id]) or int(row["target_index"]) != int(dst_idx[edge_id])
        ):
            raise ValueError(f"B1 edge row {edge_id} has mismatched source/target indices.")
        if has_delta:
            delta[edge_id] = float(row["delta_e"])
        if has_ratio:
            ratio[edge_id] = float(row["Gcorr_over_G0"])
    if not np.all(np.isfinite(delta)):
        raise ValueError("B1 edge delta_e contains non-finite values.")
    if not np.all(np.isfinite(ratio)):
        raise ValueError("B1 edge Gcorr_over_G0 contains non-finite values.")
    return delta, ratio


def validate_b1_node_compatibility(data, node_df: pd.DataFrame) -> pd.DataFrame:
    node_df = node_df.copy()
    for column in (
        "node_index",
        "predicted_net_flow_nl_s",
        "pressure_pa",
        "x_px",
        "y_px",
        "kirchhoff_residual_m3_s",
        "boundary_injection_m3_s",
    ):
        if column in node_df.columns:
            node_df[column] = pd.to_numeric(node_df[column], errors="coerce")
    node_df["node_id"] = node_df["node_id"].astype(str)
    if "predicted_net_flow_nl_s" not in node_df.columns:
        if {"kirchhoff_residual_m3_s", "boundary_injection_m3_s"} <= set(node_df.columns):
            node_df["predicted_net_flow_nl_s"] = (
                pd.to_numeric(node_df["kirchhoff_residual_m3_s"], errors="coerce")
                + pd.to_numeric(node_df["boundary_injection_m3_s"], errors="coerce")
            ) * NL_PER_M3
        else:
            raise ValueError(
                "B1 node_predictions.csv must contain predicted_net_flow_nl_s or the pair "
                "{kirchhoff_residual_m3_s, boundary_injection_m3_s}."
            )
    expected_node_ids = [str(node_id) for node_id in data.node_id]
    if len(node_df) != len(expected_node_ids):
        boundary_mask = np.asarray(
            data.source_node_mask.detach().cpu().numpy() | data.sink_node_mask.detach().cpu().numpy(),
            dtype=bool,
        )
        expected_node_ids = [
            str(node_id) for node_id, is_boundary in zip(data.node_id, boundary_mask) if not bool(is_boundary)
        ]
    if len(node_df) != len(expected_node_ids):
        raise ValueError(
            f"B1 node_predictions.csv has {len(node_df)} rows, expected either {len(data.node_id)} full-graph rows "
            f"or {len(expected_node_ids)} non-boundary rows."
        )
    for node_index, node_id in enumerate(expected_node_ids):
        row = node_df.iloc[node_index]
        if int(row["node_index"]) != node_index or row["node_id"] != node_id:
            raise ValueError(f"B1 node row {node_index} does not match graph node {node_id}.")
    return node_df


def load_b1_artifacts(run_dir: Path | None, data) -> tuple[np.ndarray, np.ndarray, pd.DataFrame, pd.DataFrame, str]:
    neutral_edge_df = pd.DataFrame(
        {
            "edge_id": np.arange(len(data.edge_ids), dtype=np.int64),
            "source": [str(u) for u, _ in data.edge_ids],
            "target": [str(v) for _, v in data.edge_ids],
            "source_index": data.edge_index[0].detach().cpu().numpy().astype(np.int64),
            "target_index": data.edge_index[1].detach().cpu().numpy().astype(np.int64),
            "delta_e": np.zeros(len(data.edge_ids), dtype=np.float64),
            "Gcorr_over_G0": np.ones(len(data.edge_ids), dtype=np.float64),
        }
    )
    neutral_summary_df = pd.DataFrame([{"graph_path": "", "output_dir": "", "artifact_mode": "neutral_unity_transfer"}])

    if run_dir is None:
        return (
            np.zeros(len(data.edge_ids), dtype=np.float64),
            np.ones(len(data.edge_ids), dtype=np.float64),
            neutral_edge_df,
            neutral_summary_df,
            "neutral_unity_transfer",
        )

    edge_path = run_dir / "edge_predictions.csv"
    summary_path = run_dir / "summary.csv"
    if not edge_path.exists() or not summary_path.exists():
        warnings.warn(
            f"B1 run directory {run_dir} is missing edge_predictions.csv or summary.csv; "
            "using neutral transferred conductance (delta_e=0, Gcorr_over_G0=1).",
            stacklevel=2,
        )
        return (
            np.zeros(len(data.edge_ids), dtype=np.float64),
            np.ones(len(data.edge_ids), dtype=np.float64),
            neutral_edge_df,
            neutral_summary_df,
            "neutral_unity_transfer_missing_artifacts",
        )

    edge_df = read_csv(edge_path)
    summary_df = read_csv(summary_path)
    delta_e_dc, conductance_ratio = validate_b1_edge_compatibility(data, edge_df)
    return delta_e_dc, conductance_ratio, edge_df, summary_df, "loaded_from_b1_run_dir"


def build_boundary_injection_vector(
    mappings: list[dict[str, object]],
    num_nodes: int,
    harmonic_number: int,
    global_f0_hz: float,
    boundary_amplitude_scale: float,
    venous_boundary_mode: str,
) -> tuple[np.ndarray, list[dict[str, object]]]:
    source_vector = np.zeros(int(num_nodes), dtype=np.complex128)
    rows: list[dict[str, object]] = []
    raw_entries: list[dict[str, object]] = []
    for mapping in mappings:
        boundary_node = mapping["boundary_node"]
        modeled_node = mapping["modeled_node"]
        modeled_idx = int(mapping["modeled_node_index"])
        adjacent_node = mapping["adjacent_graph_neighbor"]
        edge_data = mapping["edge_data"]
        observed_phasor_nl_s, ok, _ = build_boundary_edge_phasor(
            edge_data,
            boundary_node,
            adjacent_node,
            harmonic_number,
            global_f0_hz,
        )
        if not ok:
            raise ValueError(
                f"Boundary edge {(boundary_node, adjacent_node)} does not have a valid H{harmonic_number} phasor."
            )
        raw_entries.append(
            {
                "mapping": mapping,
                "boundary_node": boundary_node,
                "modeled_node": modeled_node,
                "modeled_node_index": modeled_idx,
                "adjacent_node": adjacent_node,
                "boundary_type": str(mapping["boundary_type"]),
                "observed_phasor_nl_s": complex(float(boundary_amplitude_scale) * observed_phasor_nl_s),
                "raw_observed_phasor_nl_s": complex(observed_phasor_nl_s),
            }
        )

    source_total = sum(
        entry["observed_phasor_nl_s"]
        for entry in raw_entries
        if entry["boundary_type"] == "source"
    )
    sink_total = sum(
        entry["observed_phasor_nl_s"]
        for entry in raw_entries
        if entry["boundary_type"] == "sink"
    )
    sink_scale = 1.0 + 0.0j
    if str(venous_boundary_mode) == "rebalance_to_sources" and raw_entries:
        if abs(sink_total) > 0.0:
            sink_scale = complex(-source_total / sink_total)
        elif abs(source_total) > 0.0:
            raise ValueError(
                "Cannot rebalance venous boundary injections because the selected sink phasors sum to zero."
            )

    for entry in raw_entries:
        used_phasor = complex(entry["observed_phasor_nl_s"])
        if entry["boundary_type"] == "sink":
            used_phasor *= sink_scale
        mapping = entry["mapping"]
        boundary_node = entry["boundary_node"]
        modeled_node = entry["modeled_node"]
        modeled_idx = int(entry["modeled_node_index"])
        adjacent_node = entry["adjacent_node"]
        source_vector[modeled_idx] += used_phasor
        rows.append(
            {
                "harmonic_number": int(harmonic_number),
                "boundary_node": str(boundary_node),
                "adjacent_graph_neighbor": str(adjacent_node),
                "modeled_node": str(modeled_node),
                "modeled_node_index": modeled_idx,
                "boundary_type": str(mapping["boundary_type"]),
                "boundary_amplitude_scale": float(boundary_amplitude_scale),
                "venous_boundary_mode": str(venous_boundary_mode),
                "venous_sink_scale_real": float(np.real(sink_scale)),
                "venous_sink_scale_imag": float(np.imag(sink_scale)),
                "observed_boundary_real_nl_s": float(np.real(entry["observed_phasor_nl_s"])),
                "observed_boundary_imag_nl_s": float(np.imag(entry["observed_phasor_nl_s"])),
                "observed_boundary_amplitude_nl_s": float(abs(entry["observed_phasor_nl_s"])),
                "observed_boundary_phase_deg": float(np.angle(entry["observed_phasor_nl_s"]) * DEG_PER_RAD),
                "used_boundary_real_nl_s": float(np.real(used_phasor)),
                "used_boundary_imag_nl_s": float(np.imag(used_phasor)),
                "used_boundary_amplitude_nl_s": float(abs(used_phasor)),
                "used_boundary_phase_deg": float(np.angle(used_phasor) * DEG_PER_RAD),
                "modeled_node_net_used_boundary_real_nl_s": float(np.real(source_vector[modeled_idx])),
                "modeled_node_net_used_boundary_imag_nl_s": float(np.imag(source_vector[modeled_idx])),
                "modeled_node_net_used_boundary_amplitude_nl_s": float(np.abs(source_vector[modeled_idx])),
            }
        )
        harmonic_key = f"h{int(harmonic_number)}"
        rows[-1][f"observed_boundary_{harmonic_key}_real_nl_s"] = rows[-1]["observed_boundary_real_nl_s"]
        rows[-1][f"observed_boundary_{harmonic_key}_imag_nl_s"] = rows[-1]["observed_boundary_imag_nl_s"]
        rows[-1][f"observed_boundary_{harmonic_key}_amplitude_nl_s"] = rows[-1]["observed_boundary_amplitude_nl_s"]
        rows[-1][f"observed_boundary_{harmonic_key}_phase_deg"] = rows[-1]["observed_boundary_phase_deg"]
        rows[-1][f"used_boundary_{harmonic_key}_real_nl_s"] = rows[-1]["used_boundary_real_nl_s"]
        rows[-1][f"used_boundary_{harmonic_key}_imag_nl_s"] = rows[-1]["used_boundary_imag_nl_s"]
        rows[-1][f"used_boundary_{harmonic_key}_amplitude_nl_s"] = rows[-1]["used_boundary_amplitude_nl_s"]
        rows[-1][f"used_boundary_{harmonic_key}_phase_deg"] = rows[-1]["used_boundary_phase_deg"]
        rows[-1][f"modeled_node_net_used_boundary_{harmonic_key}_real_nl_s"] = rows[-1]["modeled_node_net_used_boundary_real_nl_s"]
        rows[-1][f"modeled_node_net_used_boundary_{harmonic_key}_imag_nl_s"] = rows[-1]["modeled_node_net_used_boundary_imag_nl_s"]
        rows[-1][f"modeled_node_net_used_boundary_{harmonic_key}_amplitude_nl_s"] = rows[-1]["modeled_node_net_used_boundary_amplitude_nl_s"]
    total_source_phasor = sum(
        complex(row["used_boundary_real_nl_s"], row["used_boundary_imag_nl_s"])
        for row in rows
        if row["boundary_type"] == "source"
    )
    total_sink_phasor = sum(
        complex(row["used_boundary_real_nl_s"], row["used_boundary_imag_nl_s"])
        for row in rows
        if row["boundary_type"] == "sink"
    )
    net_boundary_phasor = total_source_phasor + total_sink_phasor
    for row in rows:
        row["total_source_phasor_nl_s"] = str(total_source_phasor)
        row["total_source_phasor_real_nl_s"] = float(np.real(total_source_phasor))
        row["total_source_phasor_imag_nl_s"] = float(np.imag(total_source_phasor))
        row["total_sink_phasor_nl_s"] = str(total_sink_phasor)
        row["total_sink_phasor_real_nl_s"] = float(np.real(total_sink_phasor))
        row["total_sink_phasor_imag_nl_s"] = float(np.imag(total_sink_phasor))
        row["net_boundary_phasor_nl_s"] = str(net_boundary_phasor)
        row["net_boundary_phasor_real_nl_s"] = float(np.real(net_boundary_phasor))
        row["net_boundary_phasor_imag_nl_s"] = float(np.imag(net_boundary_phasor))
        row["net_boundary_amplitude_nl_s"] = float(abs(net_boundary_phasor))
    return source_vector, rows


def build_boundary_edge_phasor(edge_data: dict, boundary_node: object, modeled_node: object, harmonic_number: int, global_f0_hz: float) -> tuple[complex, bool, float]:
    from harmonic_utils import signed_measurement_phasor_nl_s  # local import avoids circular confusion in type checkers

    return signed_measurement_phasor_nl_s(edge_data, boundary_node, modeled_node, harmonic_number, global_f0_hz)


def build_model_admittances(
    graph,
    edge_ids: list[tuple[object, object]],
    radii_m: np.ndarray,
    lengths_m: np.ndarray,
    edge_distensibility: np.ndarray,
    omega_n: float,
    viscosity_pa_s: float,
    conductance_ratio: np.ndarray,
) -> tuple[dict[str, dict[str, np.ndarray]], np.ndarray, np.ndarray]:
    model_arrays: dict[str, dict[str, np.ndarray]] = {}
    g_e_base = np.full(len(edge_ids), np.nan, dtype=np.float64)
    c_e = np.full(len(edge_ids), np.nan, dtype=np.float64)
    for model_name, _ in MODEL_SPECS:
        model_arrays[model_name] = {
            "admittance_diag": np.zeros(len(edge_ids), dtype=np.complex128),
            "admittance_off": np.zeros(len(edge_ids), dtype=np.complex128),
            "kL": np.zeros(len(edge_ids), dtype=np.complex128),
        }
    for edge_idx, (u, v) in enumerate(edge_ids):
        edge_data = graph.edges[u, v]
        radius_m = float(radii_m[edge_idx])
        length_m = float(lengths_m[edge_idx])
        d_edge = float(edge_distensibility[edge_idx])
        y_s_full, y_t_full, kL_full = fixed_transmission_line_admittance(
            radius_m=radius_m,
            length_m=length_m,
            omega_n=omega_n,
            viscosity_pa_s=viscosity_pa_s,
            distensibility_d=d_edge,
        )
        y_s_taylor, y_t_taylor, kL_taylor, g_edge, c_edge = taylor_transmission_line_admittance(
            radius_m=radius_m,
            length_m=length_m,
            omega_n=omega_n,
            viscosity_pa_s=viscosity_pa_s,
            distensibility_d=d_edge,
            conductance_scale=1.0,
        )
        y_s_transfer, y_t_transfer, kL_transfer, _, _ = taylor_transmission_line_admittance(
            radius_m=radius_m,
            length_m=length_m,
            omega_n=omega_n,
            viscosity_pa_s=viscosity_pa_s,
            distensibility_d=d_edge,
            conductance_scale=float(conductance_ratio[edge_idx]),
        )
        model_arrays["full_ideal"]["admittance_diag"][edge_idx] = y_s_full
        model_arrays["full_ideal"]["admittance_off"][edge_idx] = y_t_full
        model_arrays["full_ideal"]["kL"][edge_idx] = kL_full
        model_arrays["taylor_ideal"]["admittance_diag"][edge_idx] = y_s_taylor
        model_arrays["taylor_ideal"]["admittance_off"][edge_idx] = y_t_taylor
        model_arrays["taylor_ideal"]["kL"][edge_idx] = kL_taylor
        model_arrays["taylor_dc_transferred"]["admittance_diag"][edge_idx] = y_s_transfer
        model_arrays["taylor_dc_transferred"]["admittance_off"][edge_idx] = y_t_transfer
        model_arrays["taylor_dc_transferred"]["kL"][edge_idx] = kL_transfer
        g_e_base[edge_idx] = g_edge
        c_e[edge_idx] = c_edge
    return model_arrays, g_e_base, c_e


def node_rows_for_model(
    data,
    positions: dict[str, tuple[float, float]],
    pressure: np.ndarray,
    source_vector_nl_s: np.ndarray,
    nodal_residual_nl_s: np.ndarray,
) -> list[dict[str, object]]:
    arterial = set(data.arterial_node_indices.detach().cpu().numpy().astype(np.int64).tolist())
    venous = set(data.venous_node_indices.detach().cpu().numpy().astype(np.int64).tolist())
    rows: list[dict[str, object]] = []
    for idx, node_id in enumerate(data.node_id):
        role = "arterial" if idx in arterial else "venous" if idx in venous else "internal"
        coords = positions.get(str(node_id))
        x = float(coords[0]) if coords is not None else float("nan")
        y = float(coords[1]) if coords is not None else float("nan")
        rows.append(
            {
                "node_index": int(idx),
                "node_id": str(node_id),
                "node_type": role,
                "boundary_role": role,
                "pressure_real_pa": float(np.real(pressure[idx])),
                "pressure_imag_pa": float(np.imag(pressure[idx])),
                "pressure_amplitude_pa": float(np.abs(pressure[idx])),
                "pressure_phase_deg": float(np.angle(pressure[idx]) * DEG_PER_RAD),
                "source_injection_real_nl_s": float(np.real(source_vector_nl_s[idx])),
                "source_injection_imag_nl_s": float(np.imag(source_vector_nl_s[idx])),
                "source_injection_amplitude_nl_s": float(np.abs(source_vector_nl_s[idx])),
                "kirchhoff_residual_real_nl_s": float(np.real(nodal_residual_nl_s[idx])),
                "kirchhoff_residual_imag_nl_s": float(np.imag(nodal_residual_nl_s[idx])),
                "kirchhoff_residual_abs_nl_s": float(np.abs(nodal_residual_nl_s[idx])),
                "x_px": x,
                "y_px": y,
            }
        )
    return rows


def edge_rows_for_model(
    data,
    q_obs_nl_s: np.ndarray,
    valid_mask: np.ndarray,
    q_pred_nl_s: np.ndarray,
    admittance_diag: np.ndarray,
    admittance_off: np.ndarray,
    g_e_base: np.ndarray,
    c_e: np.ndarray,
    kL: np.ndarray,
    delta_e_dc: np.ndarray,
    conductance_ratio: np.ndarray,
    model_name: str,
) -> list[dict[str, object]]:
    src_idx = data.edge_index[0].detach().cpu().numpy().astype(np.int64)
    dst_idx = data.edge_index[1].detach().cpu().numpy().astype(np.int64)
    rows: list[dict[str, object]] = []
    for edge_idx, (u, v) in enumerate(data.edge_ids):
        q_obs = q_obs_nl_s[edge_idx]
        q_pred = q_pred_nl_s[edge_idx]
        rows.append(
            {
                "edge_id": int(edge_idx),
                "source": str(u),
                "target": str(v),
                "source_index": int(src_idx[edge_idx]),
                "target_index": int(dst_idx[edge_idx]),
                "valid_harmonic_observation": bool(valid_mask[edge_idx]),
                "observed_flow_real_nl_s": float(np.real(q_obs)),
                "observed_flow_imag_nl_s": float(np.imag(q_obs)),
                "observed_flow_amplitude_nl_s": float(np.abs(q_obs)),
                "observed_flow_phase_deg": float(np.angle(q_obs) * DEG_PER_RAD),
                "predicted_flow_real_nl_s": float(np.real(q_pred)),
                "predicted_flow_imag_nl_s": float(np.imag(q_pred)),
                "predicted_flow_amplitude_nl_s": float(np.abs(q_pred)),
                "predicted_flow_phase_deg": float(np.angle(q_pred) * DEG_PER_RAD),
                "flow_residual_abs_nl_s": float(np.abs(q_pred - q_obs)),
                "Ys_real_m3_s_pa": float(np.real(admittance_diag[edge_idx])),
                "Ys_imag_m3_s_pa": float(np.imag(admittance_diag[edge_idx])),
                "Yt_real_m3_s_pa": float(np.real(admittance_off[edge_idx])),
                "Yt_imag_m3_s_pa": float(np.imag(admittance_off[edge_idx])),
                "g_e_m3_s_pa": float(g_e_base[edge_idx]),
                "c_e_m3_pa": float(c_e[edge_idx]),
                "kL_real": float(np.real(kL[edge_idx])),
                "kL_imag": float(np.imag(kL[edge_idx])),
                "abs_kL": float(np.abs(kL[edge_idx])),
                "delta_e_dc": float(delta_e_dc[edge_idx]) if model_name == "taylor_dc_transferred" else float("nan"),
                "g_e_star_over_g_e": float(conductance_ratio[edge_idx]) if model_name == "taylor_dc_transferred" else float("nan"),
            }
        )
    return rows


def solve_one_model(
    model_name: str,
    model_label: str,
    data,
    positions: dict[str, tuple[float, float]],
    q_obs_nl_s: np.ndarray,
    valid_mask: np.ndarray,
    source_vector_nl_s: np.ndarray,
    arterial_idx: np.ndarray,
    venous_idx: np.ndarray,
    admittance_diag: np.ndarray,
    admittance_off: np.ndarray,
    g_e_base: np.ndarray,
    c_e: np.ndarray,
    kL: np.ndarray,
    delta_e_dc: np.ndarray,
    conductance_ratio: np.ndarray,
    args: argparse.Namespace,
    phase_threshold_nl_s: float,
    output_root: Path,
    harmonic_number: int,
    multi_harmonic: bool,
) -> dict[str, object]:
    model_dir = output_root / "models" / (f"H{harmonic_number}" if multi_harmonic else "") / model_name
    model_dir.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    source_vector_m3_s = np.asarray(source_vector_nl_s, dtype=np.complex128) / NL_PER_M3
    if str(args.pressure_solver_mode) == "pure_direct":
        pressure, solve = solve_complex_pressure_direct(
            admittance_diag=admittance_diag,
            admittance_off=admittance_off,
            edge_index=data.edge_index.detach().cpu().numpy().astype(np.int64),
            source_vector_m3_s=source_vector_m3_s,
            reference_node=int(data.reference_node),
            device=resolve_device(args.device, require_cuda=bool(args.require_cuda)),
        )
    else:
        if float(args.lambda_q) > 0.0:
            pressure, solve = solve_complex_pressure_with_nodal_injections_and_flow(
                admittance_diag=admittance_diag,
                admittance_off=admittance_off,
                edge_index=data.edge_index.detach().cpu().numpy().astype(np.int64),
                q_obs_m3_s=np.asarray(q_obs_nl_s, dtype=np.complex128) / NL_PER_M3,
                valid_edge_mask=np.asarray(valid_mask, dtype=bool),
                flow_row_weights=(
                    None
                    if bool(args.no_observed_flow_snr_weighting)
                    else data.observed_flow_weight.detach().cpu().numpy().astype(np.float64)
                ),
                source_vector_m3_s=source_vector_m3_s,
                arterial_idx=arterial_idx,
                lambda_q=float(args.lambda_q),
                lambda_k=float(args.lambda_k),
                lambda_b=float(args.lambda_b),
                pressure_constraint_type="equal_phase",
                phase_offset=float(args.phase_offset),
                max_iterations=int(args.max_phase_iterations),
                tol=float(args.phase_tol),
                device=resolve_device(args.device, require_cuda=bool(args.require_cuda)),
                lstsq_backend=str(args.lstsq_backend),
            )
        else:
            pressure, solve = solve_complex_pressure_with_nodal_injections(
                admittance_diag=admittance_diag,
                admittance_off=admittance_off,
                edge_index=data.edge_index.detach().cpu().numpy().astype(np.int64),
                source_vector_m3_s=source_vector_m3_s,
                arterial_idx=arterial_idx,
                lambda_k=float(args.lambda_k),
                lambda_b=float(args.lambda_b),
                pressure_constraint_type="equal_phase",
                phase_offset=float(args.phase_offset),
                max_iterations=int(args.max_phase_iterations),
                tol=float(args.phase_tol),
                device=resolve_device(args.device, require_cuda=bool(args.require_cuda)),
                lstsq_backend=str(args.lstsq_backend),
            )
    runtime = time.perf_counter() - start
    matrix_diag = dict(solve["lsqr_info"].get("matrix_diagnostics", {}))

    q_pred_m3_s = np.asarray(solve["flow_pred_m3_s"], dtype=np.complex128)
    nodal_residual_m3_s = np.asarray(solve["nodal_residual_m3_s"], dtype=np.complex128)
    q_pred_nl_s = q_pred_m3_s * NL_PER_M3
    nodal_residual_nl_s = nodal_residual_m3_s * NL_PER_M3
    diagnostics_path = model_dir / "pressure_diagnostics.json"
    write_pressure_diagnostics(
        path=diagnostics_path,
        data=data,
        model_name=model_name,
        harmonic_number=harmonic_number,
        q_obs_nl_s=q_obs_nl_s,
        valid_mask=valid_mask,
        pressure=pressure,
        solve=solve,
        source_vector_nl_s=source_vector_nl_s,
        admittance_diag=admittance_diag,
        admittance_off=admittance_off,
        arterial_idx=arterial_idx,
        venous_idx=venous_idx,
        lambda_q=float(args.lambda_q),
        lambda_k=float(args.lambda_k),
        lambda_b=float(args.lambda_b),
    )
    if not np.allclose(q_pred_nl_s / NL_PER_M3, q_pred_m3_s, rtol=1.0e-12, atol=1.0e-18):
        raise RuntimeError(f"{model_name} flow conversion check failed.")
    if not np.all(np.isfinite(np.abs(pressure))):
        raise RuntimeError(f"{model_name} returned non-finite pressure amplitudes.")
    if float(np.nanmax(np.abs(pressure))) > 1.0e10:
        print(
            f"[diag] wrote pressure diagnostics to {diagnostics_path} "
            f"(max |P|={float(np.nanmax(np.abs(pressure))):.6g} Pa)",
            flush=True,
        )
        raise RuntimeError(f"{model_name} pressure amplitudes remain implausibly large; check unit conversion.")
    arterial_pressures = pressure[arterial_idx]
    boundary_phase_rows = np.asarray(solve["boundary_phase_target_residual_deg"], dtype=np.float64)
    internal_mask = np.ones(len(data.node_id), dtype=bool)
    internal_mask[arterial_idx] = False
    venous_idx = data.venous_node_indices.detach().cpu().numpy().astype(np.int64).flatten()
    internal_mask[venous_idx] = False
    phase_mask = valid_mask & np.isfinite(np.abs(q_obs_nl_s)) & (np.abs(q_obs_nl_s) >= phase_threshold_nl_s)
    phase_residual_deg = principal_phase_residual_deg(np.angle(q_pred_nl_s[phase_mask]), np.angle(q_obs_nl_s[phase_mask]))
    pressure_phase_min_deg, pressure_phase_max_deg, pressure_phase_range_deg = phase_min_max_range_deg(pressure)
    arterial_pressure_amplitudes = np.abs(arterial_pressures)

    node_rows = node_rows_for_model(
        data=data,
        positions=positions,
        pressure=pressure,
        source_vector_nl_s=source_vector_nl_s,
        nodal_residual_nl_s=nodal_residual_nl_s,
    )
    edge_rows = edge_rows_for_model(
        data=data,
        q_obs_nl_s=q_obs_nl_s,
        valid_mask=valid_mask,
        q_pred_nl_s=q_pred_nl_s,
        admittance_diag=admittance_diag,
        admittance_off=admittance_off,
        g_e_base=g_e_base,
        c_e=c_e,
        kL=kL,
        delta_e_dc=delta_e_dc,
        conductance_ratio=conductance_ratio,
        model_name=model_name,
    )
    write_rows(model_dir / "node_predictions.csv", node_rows)
    write_rows(model_dir / "edge_predictions.csv", edge_rows)
    np.savez(
        model_dir / "arrays.npz",
        pressure=pressure,
        q_pred_m3_s=q_pred_m3_s,
        q_pred_nl_s=q_pred_nl_s,
        q_obs_nl_s=q_obs_nl_s,
        nodal_residual_m3_s=nodal_residual_m3_s,
        nodal_residual_nl_s=nodal_residual_nl_s,
        source_vector_m3_s=source_vector_m3_s,
        source_vector_nl_s=source_vector_nl_s,
        admittance_diag=admittance_diag,
        admittance_off=admittance_off,
        g_e=g_e_base,
        c_e=c_e,
        kL=kL,
        delta_e_dc=delta_e_dc,
        conductance_ratio=conductance_ratio,
    )

    summary_row = {
        "model_name": model_name,
        "model_label": model_label,
        "harmonic_number": int(harmonic_number),
        "complex_flow_rmse_nl_s": float(complex_rmse(q_pred_nl_s[valid_mask] - q_obs_nl_s[valid_mask])),
        "flow_amplitude_rmse_nl_s": float(float_rmse(np.abs(q_pred_nl_s[valid_mask]) - np.abs(q_obs_nl_s[valid_mask]))),
        "flow_phase_rmse_deg": float(float_rmse(phase_residual_deg)),
        "phase_eval_edge_count": int(np.count_nonzero(phase_mask)),
        "phase_eval_amplitude_threshold_nl_s": float(phase_threshold_nl_s),
        "kirchhoff_rms_per_internal_node_nl_s": float(complex_rmse(nodal_residual_nl_s[internal_mask])),
        "common_arterial_pressure_phase_residual_deg": float(float_rmse(boundary_phase_rows)),
        "arterial_pressure_amplitude_difference_pa": float(boundary_amplitude_difference_pa(arterial_pressures)),
        "arterial_pressure_phase_difference_deg": float(boundary_phase_difference_deg(arterial_pressures)),
        "arterial_pressure_amplitude_min_pa": float(np.min(arterial_pressure_amplitudes)) if arterial_pressure_amplitudes.size else float("nan"),
        "arterial_pressure_amplitude_max_pa": float(np.max(arterial_pressure_amplitudes)) if arterial_pressure_amplitudes.size else float("nan"),
        "arterial_pressure_amplitude_mean_pa": float(np.mean(arterial_pressure_amplitudes)) if arterial_pressure_amplitudes.size else float("nan"),
        "global_pressure_amplitude_max_pa": float(np.max(np.abs(pressure))),
        "pressure_amplitude_min_pa": float(np.min(np.abs(pressure))),
        "pressure_amplitude_max_pa": float(np.max(np.abs(pressure))),
        "pressure_amplitude_range_pa": float(np.max(np.abs(pressure)) - np.min(np.abs(pressure))),
        "pressure_phase_min_deg": float(pressure_phase_min_deg),
        "pressure_phase_max_deg": float(pressure_phase_max_deg),
        "pressure_phase_range_deg": float(pressure_phase_range_deg),
        "harmonic_matrix_full_rank": bool(solve["lsqr_info"].get("harmonic_matrix_full_rank", matrix_diag.get("is_full_column_rank", False))),
        "harmonic_matrix_rank": int(solve["lsqr_info"].get("harmonic_matrix_rank", matrix_diag.get("matrix_rank", 0))),
        "harmonic_matrix_condition_number": float(solve["lsqr_info"].get("harmonic_matrix_condition_number", matrix_diag.get("condition_number", float("nan")))),
        "harmonic_matrix_ones_residual_norm": float(solve["lsqr_info"].get("harmonic_matrix_ones_residual_norm", float("nan"))),
        "gauge_applied": bool(solve["lsqr_info"].get("gauge_applied", False)),
        "gauge_reason": str(solve["lsqr_info"].get("gauge_reason", "")),
        "direct_system_residual_norm": float(solve["lsqr_info"].get("direct_system_residual_norm", float("nan"))),
        "max_abs_kirchhoff_residual_m3_s": float(solve["lsqr_info"].get("max_abs_kirchhoff_residual_m3_s", float("nan"))),
        "rms_kirchhoff_residual_m3_s": float(solve["lsqr_info"].get("rms_kirchhoff_residual_m3_s", float("nan"))),
        "relative_direct_residual": float(solve["lsqr_info"].get("relative_direct_residual", float("nan"))),
        "relative_pressure_difference": float(solve["lsqr_info"].get("relative_pressure_difference", float("nan"))),
        "matrix_rows": int(matrix_diag.get("matrix_rows", 0)),
        "matrix_cols": int(matrix_diag.get("matrix_cols", 0)),
        "min_singular_value": float(matrix_diag.get("min_singular_value", float("nan"))),
        "max_singular_value": float(matrix_diag.get("max_singular_value", float("nan"))),
        "net_boundary_injection_abs_nl_s": float(np.abs(np.sum(source_vector_nl_s))),
        "net_boundary_injection_abs_m3_s": float(np.abs(np.sum(source_vector_m3_s))),
        "runtime_seconds": float(runtime),
        "solver_success": bool(solve["lsqr_info"].get("success", False)),
        "pressure_solver_backend": str(solve["lsqr_info"].get("backend", "")),
        "phase_iterations_used": float(solve["lsqr_info"].get("phase_iterations_used", float("nan"))),
        "phase_iteration_relative_change": float(solve["lsqr_info"].get("phase_iteration_relative_change", float("nan"))),
        "phase_constraint_kind": str(solve["lsqr_info"].get("phase_constraint_kind", "")),
        "arterial_antiphase_count": int(solve["lsqr_info"].get("arterial_antiphase_count", 0)),
        "common_arterial_phase_deg": float(solve["lsqr_info"].get("common_arterial_phase_deg", float("nan"))),
        "phase_constraint_amplitude_floor_pa": float(solve["lsqr_info"].get("phase_constraint_amplitude_floor_pa", float("nan"))),
        "pressure_solver_mode": str(args.pressure_solver_mode),
        "lambda_q": float(args.lambda_q),
        "lstsq_backend": str(args.lstsq_backend) if str(args.pressure_solver_mode) == "constrained_least_squares" else "",
        "run_dir": str(model_dir),
    }
    with (model_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary_row, handle, indent=2, sort_keys=True)
    summary_row["pressure"] = pressure
    summary_row["q_pred_nl_s"] = q_pred_nl_s
    summary_row["node_rows"] = node_rows
    summary_row["edge_rows"] = edge_rows
    return summary_row


def plot_metric_bars(summary_df: pd.DataFrame, output_dir: Path) -> None:
    plt = load_matplotlib_pyplot()
    if plt is None:
        return
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5), constrained_layout=True)
    axes[0].bar(summary_df["model_label"], summary_df["complex_flow_rmse_nl_s"], color=["#1b9e77", "#7570b3", "#d95f02"])
    axes[0].set_ylabel("Complex Flow RMSE (nL/s)")
    axes[0].tick_params(axis="x", rotation=20)
    axes[1].bar(summary_df["model_label"], summary_df["kirchhoff_rms_per_internal_node_nl_s"], color=["#1b9e77", "#7570b3", "#d95f02"])
    axes[1].set_ylabel("Kirchhoff RMS (nL/s per internal node)")
    axes[1].tick_params(axis="x", rotation=20)
    save_figure(fig, output_dir / "model_metrics.png")


def save_figure(fig, path: Path) -> None:
    plt = load_matplotlib_pyplot()
    if plt is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_pressure_maps(results: list[dict[str, object]], value_key: str, title: str, cbar_label: str, output_path: Path) -> None:
    plt = load_matplotlib_pyplot()
    if plt is None:
        return
    fig, axes = plt.subplots(1, len(results), figsize=(5 * len(results), 4.5), constrained_layout=True)
    if len(results) == 1:
        axes = [axes]
    vmin = min(np.nanmin([row[value_key] for row in result["node_rows"]]) for result in results)
    vmax = max(np.nanmax([row[value_key] for row in result["node_rows"]]) for result in results)
    for ax, result in zip(axes, results, strict=True):
        node_df = pd.DataFrame(result["node_rows"])
        sc = ax.scatter(
            node_df["x_px"],
            node_df["y_px"],
            c=node_df[value_key],
            cmap="viridis",
            s=10,
            vmin=vmin,
            vmax=vmax,
        )
        arterial = node_df["boundary_role"] == "arterial"
        venous = node_df["boundary_role"] == "venous"
        ax.scatter(node_df.loc[arterial, "x_px"], node_df.loc[arterial, "y_px"], marker="^", color="black", s=20)
        ax.scatter(node_df.loc[venous, "x_px"], node_df.loc[venous, "y_px"], marker="s", color="black", s=18)
        ax.set_title(result["model_label"])
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_aspect("equal")
    fig.suptitle(title)
    cbar = fig.colorbar(sc, ax=axes, shrink=0.9)
    cbar.set_label(cbar_label)
    save_figure(fig, output_path)


def plot_flow_amplitude_maps(
    results: list[dict[str, object]],
    positions: dict[str, tuple[float, float]],
    output_path: Path,
) -> None:
    plt = load_matplotlib_pyplot()
    if plt is None:
        return
    from matplotlib.collections import LineCollection
    from matplotlib.colors import LogNorm

    fig, axes = plt.subplots(1, len(results), figsize=(5 * len(results), 4.5), constrained_layout=True)
    if len(results) == 1:
        axes = [axes]

    amplitude_arrays = []
    for result in results:
        edge_df = pd.DataFrame(result["edge_rows"]).copy()
        edge_df["predicted_flow_amplitude_nl_s"] = pd.to_numeric(
            edge_df["predicted_flow_amplitude_nl_s"], errors="coerce"
        )
        amplitude_arrays.append(edge_df["predicted_flow_amplitude_nl_s"].to_numpy(dtype=float))
    finite_amp = np.concatenate(
        [values[np.isfinite(values) & (values > 0.0)] for values in amplitude_arrays if values.size],
        axis=0,
    )
    if finite_amp.size == 0:
        return
    norm = LogNorm(
        vmin=max(float(np.nanpercentile(finite_amp, 1.0)), 1.0e-6),
        vmax=max(float(np.nanpercentile(finite_amp, 99.5)), 1.0e-6),
    )

    colored = None
    for ax, result in zip(axes, results, strict=True):
        edge_df = pd.DataFrame(result["edge_rows"]).copy()
        edge_df["predicted_flow_amplitude_nl_s"] = pd.to_numeric(
            edge_df["predicted_flow_amplitude_nl_s"], errors="coerce"
        )
        node_df = pd.DataFrame(result["node_rows"]).copy()
        segments: list[np.ndarray] = []
        amplitudes: list[float] = []
        for _, row in edge_df.iterrows():
            source = str(row["source"])
            target = str(row["target"])
            if source not in positions or target not in positions:
                continue
            x1, y1 = positions[source]
            x2, y2 = positions[target]
            segments.append(np.asarray([[x1, y1], [x2, y2]], dtype=np.float64))
            amplitudes.append(float(row["predicted_flow_amplitude_nl_s"]))
        if not segments:
            continue
        widths = 0.5 + 2.0 * np.clip(np.log10(np.clip(np.abs(amplitudes), 1.0e-6, None)) + 3.0, 0.0, 3.0) / 3.0
        background = LineCollection(segments, colors="#d0cbc4", linewidths=0.5, alpha=0.35, zorder=1)
        ax.add_collection(background)
        colored = LineCollection(segments, cmap="coolwarm", norm=norm, linewidths=widths, zorder=2)
        colored.set_array(np.clip(np.abs(np.asarray(amplitudes, dtype=float)), 1.0e-12, None))
        ax.add_collection(colored)
        ax.scatter(node_df["x_px"], node_df["y_px"], s=5, c="#5f5f5f", linewidths=0.0, zorder=3)
        ax.set_title(result["model_label"])
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_aspect("equal")
    if colored is not None:
        fig.suptitle("Predicted Flow Amplitude")
        cbar = fig.colorbar(colored, ax=axes, shrink=0.9)
        cbar.set_label("Predicted Flow Amplitude (nL/s)")
        save_figure(fig, output_path)


def plot_flow_scatter(results: list[dict[str, object]], q_obs_nl_s: np.ndarray, valid_mask: np.ndarray, amplitude: bool, output_path: Path) -> None:
    plt = load_matplotlib_pyplot()
    if plt is None:
        return
    fig, axes = plt.subplots(1, len(results), figsize=(5 * len(results), 4.5), constrained_layout=True)
    if len(results) == 1:
        axes = [axes]
    for ax, result in zip(axes, results, strict=True):
        q_pred = np.asarray(result["q_pred_nl_s"], dtype=np.complex128)
        if amplitude:
            x = np.abs(q_obs_nl_s[valid_mask])
            y = np.abs(q_pred[valid_mask])
            ax.set_xlabel("Observed Flow Amplitude (nL/s)")
            ax.set_ylabel("Predicted Flow Amplitude (nL/s)")
        else:
            mask = valid_mask & np.isfinite(np.abs(q_obs_nl_s)) & (np.abs(q_obs_nl_s) > 0.0)
            x = np.angle(q_obs_nl_s[mask]) * DEG_PER_RAD
            y = np.angle(q_pred[mask]) * DEG_PER_RAD
            ax.set_xlabel("Observed Flow Phase (deg)")
            ax.set_ylabel("Predicted Flow Phase (deg)")
        ax.scatter(x, y, s=10, alpha=0.6)
        if x.size:
            bounds = [min(np.min(x), np.min(y)), max(np.max(x), np.max(y))]
            ax.plot(bounds, bounds, color="black", linewidth=1.0)
        ax.set_title(result["model_label"])
    save_figure(fig, output_path)


def plot_abs_kL_distribution(abs_kL: np.ndarray, output_path: Path) -> None:
    plt = load_matplotlib_pyplot()
    if plt is None:
        return
    finite = abs_kL[np.isfinite(abs_kL) & (abs_kL >= 0.0)]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5), constrained_layout=True)
    axes[0].hist(finite, bins=40, color="#4c78a8", alpha=0.8)
    axes[0].set_xlabel("|kL|")
    axes[0].set_ylabel("Edge count")
    axes[1].hist(np.log10(np.clip(finite, 1.0e-12, None)), bins=40, color="#f58518", alpha=0.8)
    axes[1].set_xlabel("log10(|kL|)")
    axes[1].set_ylabel("Edge count")
    save_figure(fig, output_path)


def plot_taylor_relative_errors(edge_df: pd.DataFrame, output_path: Path) -> None:
    plt = load_matplotlib_pyplot()
    if plt is None:
        return
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5), constrained_layout=True)
    axes[0].scatter(edge_df["abs_kL"], edge_df["rel_error_Ys_taylor_vs_full"], s=8, alpha=0.5)
    axes[0].set_xlabel("|kL|")
    axes[0].set_ylabel(r"|Ys_taylor - Ys_full| / |Ys_full|")
    axes[1].scatter(edge_df["abs_kL"], edge_df["rel_error_Yt_taylor_vs_full"], s=8, alpha=0.5)
    axes[1].set_xlabel("|kL|")
    axes[1].set_ylabel(r"|Yt_taylor - Yt_full| / |Yt_full|")
    save_figure(fig, output_path)


def plot_model_difference_maps(
    left: dict[str, object],
    right: dict[str, object],
    node_value: np.ndarray,
    edge_value: np.ndarray,
    node_label: str,
    edge_label: str,
    output_path: Path,
) -> None:
    plt = load_matplotlib_pyplot()
    if plt is None:
        return
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5), constrained_layout=True)
    node_df = pd.DataFrame(left["node_rows"]).copy()
    node_df["diff"] = node_value
    sc = axes[0].scatter(node_df["x_px"], node_df["y_px"], c=node_df["diff"], cmap="magma", s=10)
    axes[0].set_title(f"{left['model_label']} vs {right['model_label']} pressure")
    axes[0].set_xticks([])
    axes[0].set_yticks([])
    axes[0].set_aspect("equal")
    cbar = fig.colorbar(sc, ax=axes[0], shrink=0.9)
    cbar.set_label(node_label)
    axes[1].hist(edge_value[np.isfinite(edge_value)], bins=40, color="#4c78a8", alpha=0.8)
    axes[1].set_title(f"{left['model_label']} vs {right['model_label']} flow")
    axes[1].set_xlabel(edge_label)
    axes[1].set_ylabel("Edge count")
    save_figure(fig, output_path)


def build_edge_diagnostics_df(
    data,
    radii_m: np.ndarray,
    lengths_m: np.ndarray,
    g_e_base: np.ndarray,
    c_e: np.ndarray,
    kL_full: np.ndarray,
    models: dict[str, dict[str, np.ndarray]],
    delta_e_dc: np.ndarray,
    conductance_ratio: np.ndarray,
    results: list[dict[str, object]],
) -> pd.DataFrame:
    q_pred_lookup = {row["model_name"]: np.asarray(row["q_pred_nl_s"], dtype=np.complex128) for row in results}
    rows: list[dict[str, object]] = []
    ys_full = models["full_ideal"]["admittance_diag"]
    yt_full = models["full_ideal"]["admittance_off"]
    ys_taylor = models["taylor_ideal"]["admittance_diag"]
    yt_taylor = models["taylor_ideal"]["admittance_off"]
    for edge_idx, (u, v) in enumerate(data.edge_ids):
        rows.append(
            {
                "edge_id": int(edge_idx),
                "source": str(u),
                "target": str(v),
                "source_index": int(data.edge_index[0, edge_idx]),
                "target_index": int(data.edge_index[1, edge_idx]),
                "radius_m": float(radii_m[edge_idx]),
                "length_m": float(lengths_m[edge_idx]),
                "g_e_m3_s_pa": float(g_e_base[edge_idx]),
                "c_e_m3_pa": float(c_e[edge_idx]),
                "kL_real": float(np.real(kL_full[edge_idx])),
                "kL_imag": float(np.imag(kL_full[edge_idx])),
                "abs_kL": float(np.abs(kL_full[edge_idx])),
                "Ys_full_real_m3_s_pa": float(np.real(ys_full[edge_idx])),
                "Ys_full_imag_m3_s_pa": float(np.imag(ys_full[edge_idx])),
                "Yt_full_real_m3_s_pa": float(np.real(yt_full[edge_idx])),
                "Yt_full_imag_m3_s_pa": float(np.imag(yt_full[edge_idx])),
                "Ys_taylor_real_m3_s_pa": float(np.real(ys_taylor[edge_idx])),
                "Ys_taylor_imag_m3_s_pa": float(np.imag(ys_taylor[edge_idx])),
                "Yt_taylor_real_m3_s_pa": float(np.real(yt_taylor[edge_idx])),
                "Yt_taylor_imag_m3_s_pa": float(np.imag(yt_taylor[edge_idx])),
                "rel_error_Ys_taylor_vs_full": float(rel_complex_error(np.asarray([ys_taylor[edge_idx] - ys_full[edge_idx]]), np.asarray([ys_full[edge_idx]]))[0]),
                "rel_error_Yt_taylor_vs_full": float(rel_complex_error(np.asarray([yt_taylor[edge_idx] - yt_full[edge_idx]]), np.asarray([yt_full[edge_idx]]))[0]),
                "delta_e_dc": float(delta_e_dc[edge_idx]),
                "g_e_star_over_g_e": float(conductance_ratio[edge_idx]),
                "full_flow_real_nl_s": float(np.real(q_pred_lookup["full_ideal"][edge_idx])),
                "full_flow_imag_nl_s": float(np.imag(q_pred_lookup["full_ideal"][edge_idx])),
                "taylor_flow_real_nl_s": float(np.real(q_pred_lookup["taylor_ideal"][edge_idx])),
                "taylor_flow_imag_nl_s": float(np.imag(q_pred_lookup["taylor_ideal"][edge_idx])),
                "transferred_flow_real_nl_s": float(np.real(q_pred_lookup["taylor_dc_transferred"][edge_idx])),
                "transferred_flow_imag_nl_s": float(np.imag(q_pred_lookup["taylor_dc_transferred"][edge_idx])),
            }
        )
    return pd.DataFrame(rows)


def build_arterial_diagnostics_df(results: list[dict[str, object]], boundary_rows: list[dict[str, object]]) -> pd.DataFrame:
    boundary_df = pd.DataFrame(boundary_rows)
    arterial_sources = boundary_df[boundary_df["boundary_type"] == "source"].copy()
    injection_lookup = {
        (int(row["harmonic_number"]), str(row["boundary_node"])): float(row["used_boundary_amplitude_nl_s"])
        for _, row in arterial_sources.iterrows()
    }
    rows: list[dict[str, object]] = []
    for result in results:
        node_df = pd.DataFrame(result["node_rows"])
        arterial_df = node_df[node_df["boundary_role"] == "arterial"].copy()
        for _, row in arterial_df.iterrows():
            rows.append(
                {
                    "model_name": result["model_name"],
                    "model_label": result["model_label"],
                    "harmonic_number": int(result["harmonic_number"]),
                    "arterial_node_id": str(row["node_id"]),
                    "arterial_node_index": int(row["node_index"]),
                    "arterial_pressure_real_pa": float(row["pressure_real_pa"]),
                    "arterial_pressure_imag_pa": float(row["pressure_imag_pa"]),
                    "arterial_pressure_amplitude_pa": float(row["pressure_amplitude_pa"]),
                    "arterial_pressure_phase_deg": float(row["pressure_phase_deg"]),
                    "arterial_source_injection_amplitude_nl_s": float(
                        injection_lookup.get((int(result["harmonic_number"]), str(row["node_id"])), float("nan"))
                    ),
                }
            )
    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    if abs(float(args.phase_offset)) > 1.0e-15:
        raise ValueError(
            "--phase-offset must be 0. The equal-phase boundary condition "
            "does not prescribe an absolute harmonic pressure phase."
        )
    set_random_seed(int(args.seed))
    graph_path = args.graph_path.expanduser().resolve()
    b1_run_dir = resolve_balanced_dc_run_dir(args.dc_step2_root, args.b1_run_dir)
    output_root = args.output_dir.expanduser().resolve()
    if output_root.exists() and not args.overwrite:
        existing = output_root / "summary.csv"
        if existing.exists():
            raise FileExistsError(f"{output_root} already contains outputs. Re-run with --overwrite to replace them.")
    output_root.mkdir(parents=True, exist_ok=True)
    figures_dir = output_root / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    graph = load_graph(graph_path)
    data_config = minimal_real_data_config(int(args.seed))
    data_config["physics"]["use_observed_flow_snr_weighting"] = not bool(args.no_observed_flow_snr_weighting)
    data = build_real_gnn_data(graph_path, data_config)
    node_ids = np.asarray(data.node_id.tolist(), dtype=object)
    node_index = {node: idx for idx, node in enumerate(node_ids)}
    positions = graph_positions(graph)

    delta_e_dc, conductance_ratio, b1_edge_df, b1_summary_df, b1_artifact_mode = load_b1_artifacts(b1_run_dir, data)
    if b1_run_dir is not None and "graph_path" in b1_summary_df.columns and str(b1_summary_df.iloc[0].get("graph_path", "")).strip():
        b1_graph = Path(str(b1_summary_df.iloc[0]["graph_path"])).expanduser().resolve()
        if b1_graph != graph_path:
            raise ValueError(f"B1 run graph {b1_graph} does not match requested graph {graph_path}.")

    radii_m = np.zeros(len(data.edge_ids), dtype=np.float64)
    lengths_m = np.zeros(len(data.edge_ids), dtype=np.float64)
    for edge_idx, (u, v) in enumerate(data.edge_ids):
        radii_m[edge_idx], lengths_m[edge_idx] = edge_geometry_m(graph.edges[u, v])
    edge_distensibility, reference_radius_m = edge_distensibility_values(
        radii_m=radii_m,
        d0=float(args.D0),
        alpha=float(args.alpha),
    )

    mappings = resolve_boundary_mappings(graph, node_ids, node_index)
    validate_boundary_mapping_edges(data, mappings)
    active_mappings = select_boundary_mappings(
        graph,
        mappings,
        arterial_boundary_mode=str(args.arterial_boundary_mode),
    )
    arterial_idx = phase_constraint_indices(
        data,
        active_mappings,
        arterial_boundary_mode=str(args.arterial_boundary_mode),
    )
    venous_idx = data.venous_node_indices.detach().cpu().numpy().astype(np.int64).flatten()
    arterial_boundary_count_total = sum(1 for mapping in mappings if mapping["boundary_type"] == "source")
    arterial_boundary_count_active = sum(1 for mapping in active_mappings if mapping["boundary_type"] == "source")

    harmonic_numbers = requested_harmonic_numbers(args)
    multi_harmonic = len(harmonic_numbers) > 1
    global_f0_hz = float(args.f0_hz) if args.f0_hz is not None else best_f0_hz(graph, data)
    omega0 = 2.0 * math.pi * global_f0_hz

    all_results: list[dict[str, object]] = []
    all_summary_rows: list[dict[str, object]] = []
    all_edge_diag_frames: list[pd.DataFrame] = []
    all_boundary_rows: list[dict[str, object]] = []

    for harmonic_number in harmonic_numbers:
        q_obs_nl_s, valid_mask, edge_f0_hz = build_harmonic_measurements(
            graph,
            data,
            harmonic_number=int(harmonic_number),
            global_f0_hz=global_f0_hz,
        )
        validate_edge_frequencies(edge_f0_hz, valid_mask)
        omega_n = omega0 * int(harmonic_number)
        source_vector_nl_s, boundary_rows = build_boundary_injection_vector(
            mappings=active_mappings,
            num_nodes=len(data.node_id),
            harmonic_number=int(harmonic_number),
            global_f0_hz=global_f0_hz,
            boundary_amplitude_scale=float(args.boundary_amplitude_scale),
            venous_boundary_mode=str(args.venous_boundary_mode),
        )
        all_boundary_rows.extend(boundary_rows)
        phase_threshold_nl_s = phase_eval_threshold(valid_mask, q_obs_nl_s, args.phase_threshold_nl_s)
        model_arrays, g_e_base, c_e = build_model_admittances(
            graph=graph,
            edge_ids=data.edge_ids,
            radii_m=radii_m,
            lengths_m=lengths_m,
            edge_distensibility=edge_distensibility,
            omega_n=omega_n,
            viscosity_pa_s=float(args.viscosity_pa_s),
            conductance_ratio=conductance_ratio,
        )

        results: list[dict[str, object]] = []
        for model_name, model_label in MODEL_SPECS:
            result = solve_one_model(
                model_name=model_name,
                model_label=model_label,
                data=data,
                positions=positions,
                q_obs_nl_s=q_obs_nl_s,
                valid_mask=valid_mask,
                source_vector_nl_s=source_vector_nl_s,
                arterial_idx=arterial_idx,
                venous_idx=venous_idx,
                admittance_diag=model_arrays[model_name]["admittance_diag"],
                admittance_off=model_arrays[model_name]["admittance_off"],
                g_e_base=g_e_base,
                c_e=c_e,
                kL=model_arrays[model_name]["kL"],
                delta_e_dc=delta_e_dc,
                conductance_ratio=conductance_ratio,
                args=args,
                phase_threshold_nl_s=phase_threshold_nl_s,
                output_root=output_root,
                harmonic_number=int(harmonic_number),
                multi_harmonic=multi_harmonic,
            )
            results.append(result)
            all_results.append(result)
            all_summary_rows.append(
                {
                    "graph_path": str(graph_path),
                    "b1_run_dir": str(b1_run_dir) if b1_run_dir is not None else "",
                    "b1_artifact_mode": b1_artifact_mode,
                    "model_name": result["model_name"],
                    "model_label": result["model_label"],
                    "harmonic_number": int(harmonic_number),
                    "f0_hz": float(global_f0_hz),
                    "omega0_rad_s": float(omega0),
                    "omega_n_rad_s": float(omega_n),
                    "D0": float(args.D0),
                    "alpha": float(args.alpha),
                    "viscosity_pa_s": float(args.viscosity_pa_s),
                    "boundary_amplitude_scale": float(args.boundary_amplitude_scale),
                    "arterial_boundary_mode": str(args.arterial_boundary_mode),
                    "venous_boundary_mode": str(args.venous_boundary_mode),
                    "pressure_solver_mode": str(args.pressure_solver_mode),
                    "lambda_q": float(args.lambda_q),
                    "lambda_k": float(args.lambda_k),
                    "lambda_b": float(args.lambda_b),
                    "use_observed_flow_snr_weighting": not bool(args.no_observed_flow_snr_weighting),
                    "lstsq_backend": str(args.lstsq_backend) if str(args.pressure_solver_mode) == "constrained_least_squares" else "",
                    "n_nodes": int(len(data.node_id)),
                    "n_edges": int(len(data.edge_ids)),
                    "boundary_mapping_count_total": int(len(mappings)),
                    "boundary_mapping_count_active": int(len(active_mappings)),
                    "arterial_boundary_count_total": int(arterial_boundary_count_total),
                    "arterial_boundary_count_active": int(arterial_boundary_count_active),
                    "phase_constraint_node_count": int(len(arterial_idx)),
                    **{
                        key: value
                        for key, value in data.observed_flow_weight_stats.items()
                    },
                    **{
                        k: v
                        for k, v in result.items()
                        if k not in {"pressure", "q_pred_nl_s", "node_rows", "edge_rows", "model_name", "model_label"}
                    },
                }
            )

        edge_diag_df = build_edge_diagnostics_df(
            data=data,
            radii_m=radii_m,
            lengths_m=lengths_m,
            g_e_base=g_e_base,
            c_e=c_e,
            kL_full=model_arrays["full_ideal"]["kL"],
            models=model_arrays,
            delta_e_dc=delta_e_dc,
            conductance_ratio=conductance_ratio,
            results=results,
        )
        edge_diag_df.insert(0, "harmonic_number", int(harmonic_number))
        all_edge_diag_frames.append(edge_diag_df)

        kL_stats = summarize_distribution(np.abs(model_arrays["full_ideal"]["kL"]))
        approx_stats = {f"abs_kL_{key}": value for key, value in kL_stats.items()}
        approx_stats.update(
            {
                "rel_error_Ys_mean": float(np.nanmean(edge_diag_df["rel_error_Ys_taylor_vs_full"])),
                "rel_error_Ys_median": float(np.nanmedian(edge_diag_df["rel_error_Ys_taylor_vs_full"])),
                "rel_error_Ys_max": float(np.nanmax(edge_diag_df["rel_error_Ys_taylor_vs_full"])),
                "rel_error_Yt_mean": float(np.nanmean(edge_diag_df["rel_error_Yt_taylor_vs_full"])),
                "rel_error_Yt_median": float(np.nanmedian(edge_diag_df["rel_error_Yt_taylor_vs_full"])),
                "rel_error_Yt_max": float(np.nanmax(edge_diag_df["rel_error_Yt_taylor_vs_full"])),
            }
        )
        for row in all_summary_rows[-len(MODEL_SPECS):]:
            row.update(approx_stats)

        harmonic_figures_dir = figures_dir / f"H{harmonic_number}" if multi_harmonic else figures_dir
        harmonic_figures_dir.mkdir(parents=True, exist_ok=True)
        plot_metric_bars(pd.DataFrame(all_summary_rows[-len(MODEL_SPECS):]), harmonic_figures_dir)
        plot_pressure_maps(results, "pressure_amplitude_pa", f"H{harmonic_number} pressure amplitude", "Pressure amplitude (Pa)", harmonic_figures_dir / "pressure_amplitude_maps.png")
        plot_pressure_maps(results, "pressure_phase_deg", f"H{harmonic_number} pressure phase", "Pressure phase (deg)", harmonic_figures_dir / "pressure_phase_maps.png")
        plot_flow_amplitude_maps(results, positions, harmonic_figures_dir / "flow_amplitude_maps.png")
        plot_flow_scatter(results, q_obs_nl_s, valid_mask, True, harmonic_figures_dir / "predicted_vs_observed_flow_amplitude.png")
        plot_flow_scatter(results, q_obs_nl_s, valid_mask, False, harmonic_figures_dir / "predicted_vs_observed_flow_phase.png")
        plot_abs_kL_distribution(np.abs(model_arrays["full_ideal"]["kL"]), harmonic_figures_dir / "abs_kL_distribution.png")
        plot_taylor_relative_errors(edge_diag_df, harmonic_figures_dir / "taylor_relative_errors.png")

        full = next(row for row in results if row["model_name"] == "full_ideal")
        taylor = next(row for row in results if row["model_name"] == "taylor_ideal")
        transferred = next(row for row in results if row["model_name"] == "taylor_dc_transferred")
        plot_model_difference_maps(
            left=full,
            right=taylor,
            node_value=np.abs(np.asarray(full["pressure"]) - np.asarray(taylor["pressure"])),
            edge_value=np.abs(np.asarray(full["q_pred_nl_s"]) - np.asarray(taylor["q_pred_nl_s"])),
            node_label="|ΔP| (Pa)",
            edge_label="|ΔQ| (nL/s)",
            output_path=harmonic_figures_dir / "model1_vs_model2_differences.png",
        )
        plot_model_difference_maps(
            left=taylor,
            right=transferred,
            node_value=np.abs(np.asarray(taylor["pressure"]) - np.asarray(transferred["pressure"])),
            edge_value=np.abs(np.asarray(taylor["q_pred_nl_s"]) - np.asarray(transferred["q_pred_nl_s"])),
            node_label="|ΔP| (Pa)",
            edge_label="|ΔQ| (nL/s)",
            output_path=harmonic_figures_dir / "model2_vs_model3_differences.png",
        )

    summary_df = pd.DataFrame(all_summary_rows)
    edge_diag_df = pd.concat(all_edge_diag_frames, ignore_index=True) if all_edge_diag_frames else pd.DataFrame()
    summary_df.to_csv(output_root / "summary.csv", index=False)
    edge_diag_df.to_csv(output_root / "edge_diagnostics.csv", index=False)
    pd.DataFrame(all_boundary_rows).to_csv(output_root / "boundary_injections.csv", index=False)
    build_arterial_diagnostics_df(all_results, all_boundary_rows).to_csv(output_root / "arterial_pressures.csv", index=False)
    write_yaml(
        output_root / "config.yaml",
        {
            "graph_path": str(graph_path),
            "b1_run_dir": str(b1_run_dir) if b1_run_dir is not None else "",
            "b1_artifact_mode": b1_artifact_mode,
            "output_dir": str(output_root),
            "harmonic_number": int(harmonic_numbers[0]) if len(harmonic_numbers) == 1 else None,
            "harmonic_numbers": [int(value) for value in harmonic_numbers],
            "D0": float(args.D0),
            "alpha": float(args.alpha),
            "viscosity_pa_s": float(args.viscosity_pa_s),
            "f0_hz": float(global_f0_hz),
            "omega0_rad_s": float(omega0),
            "boundary_amplitude_scale": float(args.boundary_amplitude_scale),
            "pressure_solver_mode": str(args.pressure_solver_mode),
            "lambda_q": float(args.lambda_q),
            "lambda_k": float(args.lambda_k),
            "lambda_b": float(args.lambda_b),
            "use_observed_flow_snr_weighting": not bool(args.no_observed_flow_snr_weighting),
            "lstsq_backend": str(args.lstsq_backend) if str(args.pressure_solver_mode) == "constrained_least_squares" else "",
            "reference_radius_m": float(reference_radius_m),
            **{
                key: value
                for key, value in data.observed_flow_weight_stats.items()
            },
        },
    )

    print(f"[ok] Wrote Stage 1 admittance comparison outputs to {output_root}")


if __name__ == "__main__":
    main()
