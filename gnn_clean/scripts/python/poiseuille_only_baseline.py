#!/usr/bin/env python
"""Poiseuille-only baseline pressure/flow prediction for the real-data GNN graph."""

from __future__ import annotations

import argparse
import csv
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from physics_layer import (
    build_weighted_laplacian,
    constrained_dc_solve_equal_A_equal_V_torch,
)
from real_data import MU, build_real_gnn_data, load_graph
from utils import load_yaml, write_yaml


DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "dc" / "00_ideal_models" / "poiseuille_only_baseline"
NL_PER_M3 = 1.0e12


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("graph", type=Path)
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--viscosity-pa-s", type=float, default=float(MU))
    parser.add_argument("--run-name", default="")
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--dc-solve-mode",
        choices=(
            "partitioned-flow-gauge",
            "reduced-hard-constrained-lstsq",
            "reduced-soft-constrained-lstsq",
        ),
        default="partitioned-flow-gauge",
    )
    parser.add_argument(
        "--arterial-flow-mode",
        choices=("dataset", "none"),
        default="dataset",
    )
    parser.add_argument(
        "--pressure-constraint",
        action="append",
        choices=(
            "gauge_only",
            "equal-a-equal-v",
            "equal-av-pressure-drop",
            "mean-a-minus-v-alpha-equal-v",
        ),
        default=None,
    )
    parser.add_argument("--alpha-pa", type=float, default=None)
    parser.add_argument("--lambda-kirchhoff", type=float, default=1.0)
    parser.add_argument("--lambda-pressure-constraints", type=float, default=1.0)
    parser.add_argument("--lambda-flow-residual", type=float, default=1.0)
    parser.add_argument("--no-observed-flow-snr-weighting", action="store_true")
    parser.add_argument(
        "--flip-observed-flow-sign",
        action="store_true",
        help="Flip the sign of observed DC edge flows before fitting. Intended for sign-convention diagnostics.",
    )
    parser.add_argument(
        "--sign-eps-relative",
        type=float,
        default=1.0e-6,
        help=(
            "Relative threshold for sign diagnostics. Sign is only evaluated "
            "when both |q_obs| and |q_pred| exceed this fraction of the "
            "observed-flow scale."
        ),
    )
    return parser.parse_args()


def _default_config() -> dict:
    return {
        "training": {"seed": 0},
        "physics": {"solver_kind": "constrained_dc_equal_A_equal_V"},
        "data": {
            "include_boundary_nodes_in_pressure_solve": False,
            "split_fractions": {"train": 0.70, "validation": 0.15, "test": 0.15},
            "flow_normalization_reference_flux_nL_per_s": 1.0,
            "use_tilewise_flow_normalization": False,
        },
    }


def _deep_update(base: dict, update: dict) -> dict:
    merged = dict(base)
    for key, value in update.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_update(merged[key], value)
        else:
            merged[key] = value
    return merged


def _load_config(path: Path | None) -> tuple[dict, Path | None]:
    config = _default_config()
    resolved = None
    if path is not None:
        resolved = path.expanduser().resolve()
        config = _deep_update(config, load_yaml(resolved))
    return config, resolved


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
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


def _observed_flow_m3_s(data) -> np.ndarray:
    velocity = data.velocity_observed_m_s[:, 0, 0].detach().cpu().numpy()
    area = data.area_m2.detach().cpu().numpy()
    return velocity * area


def _valid_observed_mask(data, q_obs: np.ndarray) -> np.ndarray:
    split_mask = (data.train_mask | data.val_mask | data.test_mask).detach().cpu().numpy()
    return split_mask.astype(bool) & np.isfinite(q_obs)


def _rmse(pred: np.ndarray, obs: np.ndarray, mask: np.ndarray) -> float:
    if not np.any(mask):
        return float("nan")
    return float(np.sqrt(np.mean((pred[mask] - obs[mask]) ** 2)))


def _mae(pred: np.ndarray, obs: np.ndarray, mask: np.ndarray) -> float:
    if not np.any(mask):
        return float("nan")
    return float(np.mean(np.abs(pred[mask] - obs[mask])))


def _normalized_rmse(pred: np.ndarray, obs: np.ndarray, mask: np.ndarray) -> float:
    if not np.any(mask):
        return float("nan")
    denom = float(np.sum(obs[mask] ** 2))
    if denom <= 1.0e-30:
        return float("nan")
    return float(np.sqrt(np.sum((pred[mask] - obs[mask]) ** 2) / denom))


def _relative_rmse(pred: np.ndarray, obs: np.ndarray, mask: np.ndarray) -> float:
    if not np.any(mask):
        return float("nan")
    denom = float(np.sqrt(np.mean(obs[mask] ** 2)))
    if denom <= 1.0e-30:
        return float("nan")
    return _rmse(pred, obs, mask) / denom


def _flow_scale(q_obs: np.ndarray, mask: np.ndarray) -> float:
    values = np.abs(q_obs[mask & np.isfinite(q_obs)])
    values = values[values > 0.0]
    if values.size == 0:
        return float("nan")
    return float(np.median(values))


def _sign_flip_mask(
    q_pred: np.ndarray,
    q_obs: np.ndarray,
    mask: np.ndarray,
    eps_abs: float,
) -> np.ndarray:
    use = mask & (np.abs(q_obs) > eps_abs) & (np.abs(q_pred) > eps_abs)
    flips = np.zeros_like(mask, dtype=bool)
    flips[use] = np.sign(q_pred[use]) != np.sign(q_obs[use])
    return flips


def _sign_count_fraction(
    q_pred: np.ndarray,
    q_obs: np.ndarray,
    mask: np.ndarray,
    eps_abs: float,
) -> tuple[int, float, int]:
    eligible = mask & (np.abs(q_obs) > eps_abs) & (np.abs(q_pred) > eps_abs)
    if not np.any(eligible):
        return 0, float("nan"), 0
    flips = np.sign(q_pred[eligible]) != np.sign(q_obs[eligible])
    return int(np.sum(flips)), float(np.mean(flips)), int(np.sum(eligible))


def _safe_max_abs(x: np.ndarray, mask: np.ndarray | None = None) -> float:
    values = x if mask is None else x[mask]
    values = values[np.isfinite(values)]
    if values.size == 0:
        return float("nan")
    return float(np.max(np.abs(values)))


def _rms(values: np.ndarray) -> float:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return float("nan")
    return float(np.sqrt(np.mean(finite**2)))


def _mean_abs(values: np.ndarray) -> float:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return float("nan")
    return float(np.mean(np.abs(finite)))


def _percentile_abs(values: np.ndarray, percentile: float) -> float:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return float("nan")
    return float(np.percentile(np.abs(finite), percentile))


def _mean_or_nan(values: np.ndarray) -> float:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return float("nan")
    return float(np.mean(finite))


def _constraint_residuals_pa(
    data,
    pressure_pa: np.ndarray,
    pressure_constraints: list[str],
    alpha_pa: float | None,
) -> list[tuple[str, float]]:
    pressure = torch.tensor(pressure_pa, dtype=torch.float64)
    rows, rhs, labels = _pressure_constraint_rows(
        data=data,
        num_nodes=int(len(data.node_id)),
        pressure_constraints=pressure_constraints,
        alpha_pa=alpha_pa,
        dtype=torch.float64,
        device=torch.device("cpu"),
    )
    residuals: list[tuple[str, float]] = []
    for row, rhs_value, label in zip(rows, rhs, labels):
        if label.startswith("gauge:"):
            continue
        residual = float(torch.dot(row.cpu(), pressure) - rhs_value.cpu())
        residuals.append((label, residual))
    return residuals


def _node_coordinates_px(data, index: int) -> tuple[float, float]:
    coords = getattr(data, "node_xy_px", None)
    if coords is None:
        return float("nan"), float("nan")
    try:
        row = np.asarray(coords[index], dtype=np.float64).reshape(-1)
    except Exception:
        return float("nan"), float("nan")
    if row.size < 2:
        return float("nan"), float("nan")
    return float(row[0]), float(row[1])


def _selected_pressure_constraints(args: argparse.Namespace) -> list[str]:
    if args.pressure_constraint:
        return [
            "gauge-only" if value == "gauge_only" else value
            for value in list(dict.fromkeys(args.pressure_constraint))
        ]
    return ["equal-a-equal-v"]


def _boundary_role_indices(data) -> tuple[np.ndarray, np.ndarray]:
    arterial = np.sort(data.arterial_node_indices.detach().cpu().numpy().astype(np.int64))
    venous = np.sort(data.venous_node_indices.detach().cpu().numpy().astype(np.int64))
    return arterial, venous


def _node_label(data, index: int) -> str:
    return str(data.node_id[int(index)])


def _pressure_constraint_rows(
    data,
    num_nodes: int,
    pressure_constraints: list[str],
    alpha_pa: float | None,
    dtype: torch.dtype,
    device: torch.device,
) -> tuple[list[torch.Tensor], list[torch.Tensor], list[str]]:
    arterial, venous = _boundary_role_indices(data)
    rows: list[torch.Tensor] = []
    rhs: list[torch.Tensor] = []
    labels: list[str] = []

    if venous.size:
        row = torch.zeros(num_nodes, dtype=dtype, device=device)
        row[int(venous[0])] = 1.0
        rows.append(row)
        rhs.append(torch.zeros((), dtype=dtype, device=device))
        labels.append(f"gauge:{_node_label(data, int(venous[0]))}=0")
    else:
        row = torch.zeros(num_nodes, dtype=dtype, device=device)
        row[int(data.reference_node)] = 1.0
        rows.append(row)
        rhs.append(torch.zeros((), dtype=dtype, device=device))
        labels.append(f"gauge:{_node_label(data, int(data.reference_node))}=0")

    for constraint in pressure_constraints:
        if constraint in {"gauge-only", "gauge_only"}:
            continue
        if constraint == "equal-a-equal-v":
            if arterial.size >= 2:
                row = torch.zeros(num_nodes, dtype=dtype, device=device)
                row[int(arterial[0])] = 1.0
                row[int(arterial[1])] = -1.0
                rows.append(row)
                rhs.append(torch.zeros((), dtype=dtype, device=device))
                labels.append(
                    f"equal_a:{_node_label(data, int(arterial[0]))}="
                    f"{_node_label(data, int(arterial[1]))}"
                )
            if venous.size >= 2:
                row = torch.zeros(num_nodes, dtype=dtype, device=device)
                row[int(venous[0])] = 1.0
                row[int(venous[1])] = -1.0
                rows.append(row)
                rhs.append(torch.zeros((), dtype=dtype, device=device))
                labels.append(
                    f"equal_v:{_node_label(data, int(venous[0]))}="
                    f"{_node_label(data, int(venous[1]))}"
                )

        elif constraint == "equal-av-pressure-drop":
            if arterial.size < 2 or venous.size < 2:
                raise ValueError(
                    "equal-av-pressure-drop requires two arterial and two venous nodes."
                )
            row = torch.zeros(num_nodes, dtype=dtype, device=device)
            row[int(arterial[0])] = 1.0
            row[int(venous[0])] = -1.0
            row[int(arterial[1])] = -1.0
            row[int(venous[1])] = 1.0
            rows.append(row)
            rhs.append(torch.zeros((), dtype=dtype, device=device))
            labels.append(
                "equal_av_pressure_drop:"
                f"{_node_label(data, int(arterial[0]))}-{_node_label(data, int(venous[0]))}"
                "="
                f"{_node_label(data, int(arterial[1]))}-{_node_label(data, int(venous[1]))}"
            )

        elif constraint == "mean-a-minus-v-alpha-equal-v":
            if arterial.size < 2 or venous.size < 1:
                raise ValueError(
                    "mean-a-minus-v-alpha-equal-v requires at least two arterial nodes "
                    "and one venous node."
                )
            if alpha_pa is None or not math.isfinite(float(alpha_pa)):
                raise ValueError(
                    "--alpha-pa must be provided for mean-a-minus-v-alpha-equal-v."
                )

            if venous.size >= 2:
                row = torch.zeros(num_nodes, dtype=dtype, device=device)
                row[int(venous[0])] = 1.0
                row[int(venous[1])] = -1.0
                rows.append(row)
                rhs.append(torch.zeros((), dtype=dtype, device=device))
                labels.append(
                    f"equal_v:{_node_label(data, int(venous[0]))}="
                    f"{_node_label(data, int(venous[1]))}"
                )

            row = torch.zeros(num_nodes, dtype=dtype, device=device)
            row[int(arterial[0])] = 0.5
            row[int(arterial[1])] = 0.5
            row[int(venous[0])] = -1.0
            rows.append(row)
            rhs.append(torch.tensor(float(alpha_pa), dtype=dtype, device=device))
            labels.append(
                "mean_a_minus_v_alpha:"
                f"0.5*({_node_label(data, int(arterial[0]))}+"
                f"{_node_label(data, int(arterial[1]))})"
                f"-{_node_label(data, int(venous[0]))}={float(alpha_pa):.6g}"
            )

        else:
            raise ValueError(f"Unsupported pressure constraint: {constraint}")

    return rows, rhs, labels


def _dataset_source_vector(
    data,
    arterial_flow_mode: str,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    num_nodes = int(len(data.node_id))
    if arterial_flow_mode == "dataset":
        return data.boundary_injection_m3_s.to(device=device, dtype=dtype).flatten()
    if arterial_flow_mode == "none":
        return torch.zeros(num_nodes, dtype=dtype, device=device)
    raise ValueError(f"Unsupported arterial-flow mode: {arterial_flow_mode}")


def _gauge_node_index(data) -> int:
    venous_nodes = data.venous_node_indices.detach().cpu().numpy().astype(np.int64)
    if venous_nodes.size:
        return int(venous_nodes[0])
    return int(data.reference_node)


def _laplacian_scale_value(matrix: torch.Tensor) -> torch.Tensor:
    nonzero = torch.abs(matrix[torch.abs(matrix) > 0.0])
    if nonzero.numel() == 0:
        return matrix.new_tensor(1.0)
    return torch.median(nonzero).clamp_min(1.0e-30)


def _observed_flow_block(
    data,
    conductance: torch.Tensor,
    unknown_nodes: torch.Tensor,
    edge_index: torch.Tensor,
    device: torch.device,
    dtype: torch.dtype,
    *,
    flip_observed_flow_sign: bool = False,
    flow_row_weights: torch.Tensor | None = None,
) -> dict[str, object]:
    q_obs = (
        data.velocity_observed_m_s[:, 0, 0].to(device=device, dtype=dtype)
        * data.area_m2.to(device=device, dtype=dtype)
    )
    if flip_observed_flow_sign:
        q_obs = -q_obs
    valid_mask = (
        (data.train_mask | data.val_mask | data.test_mask).to(device=device)
        & torch.isfinite(q_obs)
    )
    valid_edges = torch.nonzero(valid_mask, as_tuple=False).flatten()
    n_rows = int(valid_edges.numel())
    n_unknowns = int(unknown_nodes.numel())
    if n_rows == 0:
        return {
            "matrix": torch.zeros((0, n_unknowns), dtype=dtype, device=device),
            "rhs": torch.zeros((0,), dtype=dtype, device=device),
            "edge_count": 0,
        }
    node_to_col = torch.full(
        (int(len(data.node_id)),),
        -1,
        dtype=torch.long,
        device=device,
    )
    node_to_col[unknown_nodes] = torch.arange(n_unknowns, dtype=torch.long, device=device)
    src = edge_index[0].index_select(0, valid_edges)
    dst = edge_index[1].index_select(0, valid_edges)
    cols_src = node_to_col[src]
    cols_dst = node_to_col[dst]
    row_idx = torch.arange(n_rows, dtype=torch.long, device=device)
    matrix = torch.zeros((n_rows, n_unknowns), dtype=dtype, device=device)
    edge_g = conductance.index_select(0, valid_edges)
    src_keep = cols_src >= 0
    dst_keep = cols_dst >= 0
    if bool(torch.any(src_keep)):
        matrix[row_idx[src_keep], cols_src[src_keep]] += edge_g[src_keep]
    if bool(torch.any(dst_keep)):
        matrix[row_idx[dst_keep], cols_dst[dst_keep]] -= edge_g[dst_keep]
    rhs = q_obs.index_select(0, valid_edges)
    if flow_row_weights is not None:
        row_weights = flow_row_weights.to(device=device, dtype=dtype).index_select(0, valid_edges)
        row_sqrt = torch.sqrt(row_weights.clamp_min(1.0e-12)).unsqueeze(1)
        matrix = matrix * row_sqrt
        rhs = rhs * row_sqrt.squeeze(1)
    return {
        "matrix": matrix,
        "rhs": rhs,
        "edge_count": n_rows,
    }


def _partitioned_system(
    data,
    conductance: torch.Tensor,
    arterial_flow_mode: str,
    device: torch.device,
) -> dict[str, object]:
    dtype = torch.float64
    edge_index = data.edge_index.to(device=device)
    conductance = conductance.to(device=device, dtype=dtype)
    num_nodes = int(len(data.node_id))
    laplacian = build_weighted_laplacian(
        edge_index=edge_index,
        conductance=conductance,
        n_nodes=num_nodes,
    ).to(dtype=dtype, device=device)
    source_vector = _dataset_source_vector(
        data=data,
        arterial_flow_mode=arterial_flow_mode,
        dtype=dtype,
        device=device,
    )
    gauge_node = _gauge_node_index(data)
    unknown_nodes = torch.nonzero(
        torch.arange(num_nodes, device=device) != gauge_node,
        as_tuple=False,
    ).flatten()
    reduced_matrix = laplacian.index_select(0, unknown_nodes).index_select(1, unknown_nodes)
    reduced_rhs = source_vector.index_select(0, unknown_nodes)
    net_injection = float(source_vector.sum().detach().cpu())
    formulation_warning = ""
    if abs(net_injection) > 1.0e-10:
        formulation_warning = (
            "WARNING: net injection is not balanced; pure flow-driven Laplacian solve may be inconsistent."
        )
    return {
        "dtype": dtype,
        "edge_index": edge_index,
        "conductance": conductance,
        "num_nodes": num_nodes,
        "laplacian": laplacian,
        "source_vector": source_vector,
        "gauge_node": gauge_node,
        "unknown_nodes": unknown_nodes,
        "reduced_matrix": reduced_matrix,
        "reduced_rhs": reduced_rhs,
        "formulation_warning": formulation_warning,
    }


def _reduced_constraint_system(
    data,
    num_nodes: int,
    unknown_nodes: torch.Tensor,
    gauge_node: int,
    pressure_constraints: list[str],
    alpha_pa: float | None,
    device: torch.device,
) -> dict[str, object]:
    dtype = torch.float64
    constraint_rows, constraint_rhs, constraint_labels = _pressure_constraint_rows(
        data=data,
        num_nodes=num_nodes,
        pressure_constraints=pressure_constraints,
        alpha_pa=alpha_pa,
        dtype=dtype,
        device=device,
    )
    filtered_rows: list[torch.Tensor] = []
    filtered_rhs: list[torch.Tensor] = []
    filtered_labels: list[str] = []
    for row, rhs, label in zip(constraint_rows, constraint_rhs, constraint_labels):
        is_gauge = bool(
            torch.count_nonzero(row).item() == 1
            and abs(float(row[gauge_node].detach().cpu()) - 1.0) < 1.0e-12
            and abs(float(rhs.detach().cpu())) < 1.0e-12
        )
        if is_gauge:
            continue
        filtered_rows.append(row)
        filtered_rhs.append(rhs)
        filtered_labels.append(label)
    if not filtered_rows:
        return {
            "constraint_matrix_reduced": torch.zeros(
                (0, int(unknown_nodes.numel())), dtype=dtype, device=device
            ),
            "constraint_rhs_reduced": torch.zeros((0,), dtype=dtype, device=device),
            "constraint_labels": [],
        }
    constraint_matrix_full = torch.stack(filtered_rows, dim=0)
    constraint_rhs_full = torch.stack(filtered_rhs, dim=0)
    return {
        "constraint_matrix_reduced": constraint_matrix_full.index_select(1, unknown_nodes),
        "constraint_rhs_reduced": constraint_rhs_full,
        "constraint_labels": filtered_labels,
    }


def _solve_partitioned_flow_gauge_pressure(
    data,
    conductance: torch.Tensor,
    arterial_flow_mode: str,
    device: torch.device,
) -> dict[str, object]:
    dtype = torch.float64
    edge_index = data.edge_index.to(device=device)
    conductance = conductance.to(device=device, dtype=dtype)
    num_nodes = int(len(data.node_id))

    laplacian = build_weighted_laplacian(
        edge_index=edge_index,
        conductance=conductance,
        n_nodes=num_nodes,
    ).to(dtype=dtype, device=device)

    source_vector = _dataset_source_vector(
        data=data,
        arterial_flow_mode=arterial_flow_mode,
        dtype=dtype,
        device=device,
    )

    venous_nodes = data.venous_node_indices.to(device=device, dtype=torch.long).flatten()
    arterial_nodes = data.arterial_node_indices.to(device=device, dtype=torch.long).flatten()
    gauge_node = _gauge_node_index(data)

    known_mask = torch.zeros(num_nodes, dtype=torch.bool, device=device)
    known_mask[gauge_node] = True
    unknown_nodes = torch.nonzero(~known_mask, as_tuple=False).flatten()
    known_nodes = torch.nonzero(known_mask, as_tuple=False).flatten()

    reduced_matrix = laplacian.index_select(0, unknown_nodes).index_select(1, unknown_nodes)
    reduced_rhs = source_vector.index_select(0, unknown_nodes)

    pressure = torch.zeros(num_nodes, dtype=dtype, device=device)
    used_lstsq = False
    formulation_warning = ""
    net_injection = float(source_vector.sum().detach().cpu())
    if abs(net_injection) > 1.0e-10:
        formulation_warning = (
            "WARNING: net injection is not balanced; pure flow-driven Laplacian solve may be inconsistent."
        )

    try:
        reduced_pressure = torch.linalg.solve(reduced_matrix, reduced_rhs)
    except RuntimeError:
        used_lstsq = True
        reduced_pressure = torch.linalg.lstsq(reduced_matrix, reduced_rhs).solution

    pressure.index_copy_(0, unknown_nodes, reduced_pressure)
    pressure.index_fill_(0, known_nodes, 0.0)

    edge_pressure_drop = pressure[edge_index[0]] - pressure[edge_index[1]]
    edge_flow = conductance * edge_pressure_drop
    nodal_residual = laplacian @ pressure - source_vector
    partitioned_residual = reduced_matrix @ reduced_pressure - reduced_rhs

    diagnostics = {
        "pressure_solver_iterations_used": pressure.new_tensor(1.0),
        "pressure_solver_final_relative_residual": (
            torch.linalg.vector_norm(partitioned_residual)
            / torch.linalg.vector_norm(reduced_rhs).clamp_min(1.0e-30)
        ),
        "pressure_solver_reached_max_iterations": pressure.new_tensor(0.0),
        "hard_boundary_constraint_residual_l2": pressure.new_tensor(0.0),
        "hard_boundary_constraint_residual_max": pressure.new_tensor(0.0),
        "constrained_equation_residual_l2": torch.linalg.vector_norm(partitioned_residual),
        "constrained_equation_residual_max": (
            torch.max(torch.abs(partitioned_residual))
            if partitioned_residual.numel()
            else pressure.new_tensor(0.0)
        ),
        "arterial_boundary_mismatch_l2": (
            torch.linalg.vector_norm(nodal_residual[arterial_nodes])
            if arterial_nodes.numel()
            else pressure.new_tensor(0.0)
        ),
        "arterial_boundary_mismatch_max": (
            torch.max(torch.abs(nodal_residual[arterial_nodes]))
            if arterial_nodes.numel()
            else pressure.new_tensor(0.0)
        ),
        "venous_boundary_mismatch_l2": (
            torch.linalg.vector_norm(nodal_residual[venous_nodes])
            if venous_nodes.numel()
            else pressure.new_tensor(0.0)
        ),
        "venous_boundary_mismatch_max": (
            torch.max(torch.abs(nodal_residual[venous_nodes]))
            if venous_nodes.numel()
            else pressure.new_tensor(0.0)
        ),
        "arterial_dataset_flow_total_m3_s": (
            source_vector[arterial_nodes].sum()
            if arterial_nodes.numel()
            else pressure.new_tensor(0.0)
        ),
        "venous_dataset_flow_total_m3_s": (
            source_vector[venous_nodes].sum()
            if venous_nodes.numel()
            else pressure.new_tensor(0.0)
        ),
        "arterial_predicted_flow_total_m3_s": (
            (laplacian @ pressure)[arterial_nodes].sum()
            if arterial_nodes.numel()
            else pressure.new_tensor(0.0)
        ),
        "venous_predicted_flow_total_m3_s": (
            (laplacian @ pressure)[venous_nodes].sum()
            if venous_nodes.numel()
            else pressure.new_tensor(0.0)
        ),
        "gauge_node_index": pressure.new_tensor(float(gauge_node)),
        "unknown_pressure_node_count": pressure.new_tensor(float(unknown_nodes.numel())),
        "total_injection_m3_s": source_vector.sum(),
        "net_injection_m3_s": source_vector.sum(),
        "partitioned_system_residual_l2": torch.linalg.vector_norm(partitioned_residual),
        "partitioned_system_residual_max": (
            torch.max(torch.abs(partitioned_residual))
            if partitioned_residual.numel()
            else pressure.new_tensor(0.0)
        ),
        "partitioned_solver_used_lstsq": pressure.new_tensor(1.0 if used_lstsq else 0.0),
    }

    return {
        "pressure_pa": pressure.to(dtype=torch.float32),
        "edge_flow_m3_s": edge_flow.to(dtype=torch.float32),
        "nodal_residual_m3_s": nodal_residual.to(dtype=torch.float32),
        "source_vector_m3_s": source_vector.to(dtype=torch.float32),
        "dc_solve_mode": "partitioned-flow-gauge",
        "solver_kind_used": "partitioned_flow_gauge",
        "constraint_labels": ["partitioned_flow_gauge:gauge_only"],
        "gauge_node_index": gauge_node,
        "gauge_node_id": _node_label(data, gauge_node),
        "pressure_prescribed_node_ids": _node_label(data, gauge_node),
        "partitioned_solver_used_lstsq": used_lstsq,
        "formulation_warning": formulation_warning,
        "diagnostics": {
            key: value.to(dtype=torch.float32)
            for key, value in diagnostics.items()
        },
    }


def _solve_reduced_constrained_pressure(
    data,
    conductance: torch.Tensor,
    arterial_flow_mode: str,
    pressure_constraints: list[str],
    alpha_pa: float | None,
    device: torch.device,
    dc_solve_mode: str,
    lambda_kirchhoff: float,
    lambda_pressure_constraints: float,
    lambda_flow_residual: float,
    flip_observed_flow_sign: bool = False,
    use_observed_flow_snr_weighting: bool = True,
) -> dict[str, object]:
    system = _partitioned_system(
        data=data,
        conductance=conductance,
        arterial_flow_mode=arterial_flow_mode,
        device=device,
    )
    dtype = system["dtype"]
    edge_index = system["edge_index"]
    conductance = system["conductance"]
    num_nodes = system["num_nodes"]
    laplacian = system["laplacian"]
    source_vector = system["source_vector"]
    gauge_node = system["gauge_node"]
    unknown_nodes = system["unknown_nodes"]
    reduced_matrix = system["reduced_matrix"]
    reduced_rhs = system["reduced_rhs"]
    formulation_warning = system["formulation_warning"]

    venous_nodes = data.venous_node_indices.to(device=device, dtype=torch.long).flatten()
    arterial_nodes = data.arterial_node_indices.to(device=device, dtype=torch.long).flatten()

    constraint_payload = _reduced_constraint_system(
        data=data,
        num_nodes=num_nodes,
        unknown_nodes=unknown_nodes,
        gauge_node=gauge_node,
        pressure_constraints=pressure_constraints,
        alpha_pa=alpha_pa,
        device=device,
    )
    reduced_constraints = constraint_payload["constraint_matrix_reduced"]
    reduced_constraint_rhs = constraint_payload["constraint_rhs_reduced"]
    constraint_labels = constraint_payload["constraint_labels"]
    flow_payload = _observed_flow_block(
        data=data,
        conductance=conductance,
        unknown_nodes=unknown_nodes,
        edge_index=edge_index,
        device=device,
        dtype=dtype,
        flip_observed_flow_sign=flip_observed_flow_sign,
        flow_row_weights=(
            data.observed_flow_weight.to(device=device, dtype=dtype)
            if bool(use_observed_flow_snr_weighting)
            else None
        ),
    )
    reduced_flow_matrix = flow_payload["matrix"]
    reduced_flow_rhs = flow_payload["rhs"]

    laplacian_scale = _laplacian_scale_value(reduced_matrix)
    reduced_matrix_scaled = reduced_matrix / laplacian_scale
    reduced_rhs_scaled = reduced_rhs / laplacian_scale
    flow_row_scale = _laplacian_scale_value(reduced_flow_matrix)
    reduced_flow_matrix_scaled = reduced_flow_matrix / flow_row_scale
    reduced_flow_rhs_scaled = reduced_flow_rhs / flow_row_scale

    if not math.isfinite(lambda_kirchhoff) or lambda_kirchhoff < 0.0:
        raise ValueError("--lambda-kirchhoff must be finite and non-negative.")
    if not math.isfinite(lambda_pressure_constraints) or lambda_pressure_constraints < 0.0:
        raise ValueError("--lambda-pressure-constraints must be finite and non-negative.")
    if not math.isfinite(lambda_flow_residual) or lambda_flow_residual < 0.0:
        raise ValueError("--lambda-flow-residual must be finite and non-negative.")

    if dc_solve_mode == "reduced-hard-constrained-lstsq":
        n_unknowns = int(unknown_nodes.numel())
        n_constraints = int(reduced_constraints.shape[0])
        normal_matrix = float(lambda_kirchhoff) * (
            reduced_matrix_scaled.transpose(0, 1) @ reduced_matrix_scaled
        )
        kkt = torch.zeros(
            (n_unknowns + n_constraints, n_unknowns + n_constraints),
            dtype=dtype,
            device=device,
        )
        kkt[:n_unknowns, :n_unknowns] = normal_matrix
        if n_constraints:
            kkt[:n_unknowns, n_unknowns:] = reduced_constraints.transpose(0, 1)
            kkt[n_unknowns:, :n_unknowns] = reduced_constraints
        rhs = torch.zeros(n_unknowns + n_constraints, dtype=dtype, device=device)
        rhs[:n_unknowns] = float(lambda_kirchhoff) * (
            reduced_matrix_scaled.transpose(0, 1) @ reduced_rhs_scaled
        )
        if n_constraints:
            rhs[n_unknowns:] = reduced_constraint_rhs
        reduced_pressure = torch.linalg.lstsq(kkt, rhs).solution[:n_unknowns]
        constraints_status = "reduced_hard"
        solver_kind_used = "reduced_hard_constrained_lstsq"
    elif dc_solve_mode == "reduced-soft-constrained-lstsq":
        block_matrices = []
        block_rhs = []
        if float(lambda_kirchhoff) > 0.0:
            block_matrices.append(math.sqrt(float(lambda_kirchhoff)) * reduced_matrix_scaled)
            block_rhs.append(math.sqrt(float(lambda_kirchhoff)) * reduced_rhs_scaled)
        if float(lambda_pressure_constraints) > 0.0 and int(reduced_constraints.shape[0]) > 0:
            constraint_scale = math.sqrt(float(lambda_pressure_constraints))
            block_matrices.append(constraint_scale * reduced_constraints)
            block_rhs.append(constraint_scale * reduced_constraint_rhs)
        if float(lambda_flow_residual) > 0.0 and int(reduced_flow_matrix.shape[0]) > 0:
            flow_scale = math.sqrt(float(lambda_flow_residual))
            block_matrices.append(flow_scale * reduced_flow_matrix_scaled)
            block_rhs.append(flow_scale * reduced_flow_rhs_scaled)
        if not block_matrices:
            raise ValueError(
                "At least one of --lambda-kirchhoff, --lambda-pressure-constraints, "
                "or --lambda-flow-residual must be positive for reduced-soft-constrained-lstsq."
            )
        stacked_matrix = torch.cat(block_matrices, dim=0)
        stacked_rhs = torch.cat(block_rhs, dim=0)
        reduced_pressure = torch.linalg.lstsq(stacked_matrix, stacked_rhs).solution
        constraints_status = "reduced_soft"
        solver_kind_used = "reduced_soft_constrained_lstsq"
    else:
        raise ValueError(f"Unsupported reduced constrained mode: {dc_solve_mode}")

    pressure = torch.zeros(num_nodes, dtype=dtype, device=device)
    pressure.index_copy_(0, unknown_nodes, reduced_pressure)
    pressure[gauge_node] = 0.0

    edge_pressure_drop = pressure[edge_index[0]] - pressure[edge_index[1]]
    edge_flow = conductance * edge_pressure_drop
    nodal_residual = laplacian @ pressure - source_vector
    constrained_residual = reduced_matrix @ reduced_pressure - reduced_rhs
    constraint_residual = (
        reduced_constraints @ reduced_pressure - reduced_constraint_rhs
        if reduced_constraints.numel()
        else pressure.new_zeros((0,))
    )

    partitioned_result = _solve_partitioned_flow_gauge_pressure(
        data=data,
        conductance=conductance,
        arterial_flow_mode=arterial_flow_mode,
        device=device,
    )
    partitioned_pressure = partitioned_result["pressure_pa"].to(device=device, dtype=dtype)
    partitioned_flow = partitioned_result["edge_flow_m3_s"].to(device=device, dtype=dtype)
    partitioned_constraint_residual = (
        reduced_constraints @ partitioned_pressure.index_select(0, unknown_nodes)
        - reduced_constraint_rhs
        if reduced_constraints.numel()
        else pressure.new_zeros((0,))
    )
    flow_residual = (
        reduced_flow_matrix @ reduced_pressure - reduced_flow_rhs
        if reduced_flow_matrix.numel()
        else pressure.new_zeros((0,))
    )

    diagnostics = {
        "pressure_solver_iterations_used": pressure.new_tensor(1.0),
        "pressure_solver_final_relative_residual": (
            torch.linalg.vector_norm(constrained_residual)
            / torch.linalg.vector_norm(reduced_rhs).clamp_min(1.0e-30)
        ),
        "pressure_solver_reached_max_iterations": pressure.new_tensor(0.0),
        "hard_boundary_constraint_residual_l2": (
            torch.linalg.vector_norm(constraint_residual)
            if constraint_residual.numel()
            else pressure.new_tensor(0.0)
        ),
        "hard_boundary_constraint_residual_max": (
            torch.max(torch.abs(constraint_residual))
            if constraint_residual.numel()
            else pressure.new_tensor(0.0)
        ),
        "constrained_equation_residual_l2": torch.linalg.vector_norm(constrained_residual),
        "constrained_equation_residual_max": (
            torch.max(torch.abs(constrained_residual))
            if constrained_residual.numel()
            else pressure.new_tensor(0.0)
        ),
        "arterial_boundary_mismatch_l2": (
            torch.linalg.vector_norm(nodal_residual[arterial_nodes])
            if arterial_nodes.numel()
            else pressure.new_tensor(0.0)
        ),
        "arterial_boundary_mismatch_max": (
            torch.max(torch.abs(nodal_residual[arterial_nodes]))
            if arterial_nodes.numel()
            else pressure.new_tensor(0.0)
        ),
        "venous_boundary_mismatch_l2": (
            torch.linalg.vector_norm(nodal_residual[venous_nodes])
            if venous_nodes.numel()
            else pressure.new_tensor(0.0)
        ),
        "venous_boundary_mismatch_max": (
            torch.max(torch.abs(nodal_residual[venous_nodes]))
            if venous_nodes.numel()
            else pressure.new_tensor(0.0)
        ),
        "arterial_dataset_flow_total_m3_s": (
            source_vector[arterial_nodes].sum()
            if arterial_nodes.numel()
            else pressure.new_tensor(0.0)
        ),
        "venous_dataset_flow_total_m3_s": (
            source_vector[venous_nodes].sum()
            if venous_nodes.numel()
            else pressure.new_tensor(0.0)
        ),
        "arterial_predicted_flow_total_m3_s": (
            (laplacian @ pressure)[arterial_nodes].sum()
            if arterial_nodes.numel()
            else pressure.new_tensor(0.0)
        ),
        "venous_predicted_flow_total_m3_s": (
            (laplacian @ pressure)[venous_nodes].sum()
            if venous_nodes.numel()
            else pressure.new_tensor(0.0)
        ),
        "gauge_node_index": pressure.new_tensor(float(gauge_node)),
        "unknown_pressure_node_count": pressure.new_tensor(float(unknown_nodes.numel())),
        "active_equation_row_count": pressure.new_tensor(float(unknown_nodes.numel())),
        "total_injection_m3_s": source_vector.sum(),
        "net_injection_m3_s": source_vector.sum(),
        "partitioned_system_residual_l2": pressure.new_tensor(float("nan")),
        "partitioned_system_residual_max": pressure.new_tensor(float("nan")),
        "partitioned_solver_used_lstsq": pressure.new_tensor(float("nan")),
        "reduced_constraint_count": pressure.new_tensor(float(reduced_constraints.shape[0])),
        "reduced_constraint_residual_l2": (
            torch.linalg.vector_norm(constraint_residual)
            if constraint_residual.numel()
            else pressure.new_tensor(0.0)
        ),
        "reduced_constraint_residual_max": (
            torch.max(torch.abs(constraint_residual))
            if constraint_residual.numel()
            else pressure.new_tensor(0.0)
        ),
        "partitioned_constraint_residual_l2": (
            torch.linalg.vector_norm(partitioned_constraint_residual)
            if partitioned_constraint_residual.numel()
            else pressure.new_tensor(0.0)
        ),
        "lambda_kirchhoff": pressure.new_tensor(float(lambda_kirchhoff)),
        "lambda_pressure_constraints": pressure.new_tensor(float(lambda_pressure_constraints)),
        "lambda_flow_residual": pressure.new_tensor(float(lambda_flow_residual)),
        "flow_residual_in_solver_l2": (
            torch.linalg.vector_norm(flow_residual)
            if flow_residual.numel()
            else pressure.new_tensor(0.0)
        ),
        "flow_residual_in_solver_rmse": (
            torch.sqrt(torch.mean(flow_residual**2))
            if flow_residual.numel()
            else pressure.new_tensor(0.0)
        ),
        "flow_residual_edge_count": pressure.new_tensor(float(flow_payload["edge_count"])),
        "flow_row_scale_used": flow_row_scale,
        "kirchhoff_objective_l2": torch.linalg.vector_norm(constrained_residual),
        "pressure_constraint_objective_l2": (
            torch.linalg.vector_norm(constraint_residual)
            if constraint_residual.numel()
            else pressure.new_tensor(0.0)
        ),
        "constrained_vs_partitioned_pressure_max_abs": torch.max(
            torch.abs(pressure - partitioned_pressure)
        ),
        "constrained_vs_partitioned_flow_max_abs": torch.max(
            torch.abs(edge_flow - partitioned_flow)
        ),
        "laplacian_scale_used": laplacian_scale,
    }

    return {
        "pressure_pa": pressure.to(dtype=torch.float32),
        "edge_flow_m3_s": edge_flow.to(dtype=torch.float32),
        "nodal_residual_m3_s": nodal_residual.to(dtype=torch.float32),
        "source_vector_m3_s": source_vector.to(dtype=torch.float32),
        "dc_solve_mode": dc_solve_mode,
        "solver_kind_used": solver_kind_used,
        "constraint_labels": constraint_labels if constraint_labels else ["partitioned_flow_gauge:gauge_only"],
        "gauge_node_index": gauge_node,
        "gauge_node_id": _node_label(data, gauge_node),
        "pressure_prescribed_node_ids": _node_label(data, gauge_node),
        "pressure_constraints_used_in_solve": constraints_status,
        "formulation_warning": formulation_warning,
        "diagnostics": {
            key: value.to(dtype=torch.float32)
            for key, value in diagnostics.items()
        },
    }


def _solve_baseline(
    data,
    conductance: torch.Tensor,
    solver_kind: str,
    device: torch.device,
    dc_solve_mode: str,
    arterial_flow_mode: str,
    pressure_constraints: list[str],
    alpha_pa: float | None,
    lambda_kirchhoff: float,
    lambda_pressure_constraints: float,
    lambda_flow_residual: float,
    flip_observed_flow_sign: bool = False,
    use_observed_flow_snr_weighting: bool = True,
) -> dict[str, object]:
    # Honor an explicitly requested reduced/partitioned solve mode for sweep
    # scripts, even if the default config still names the legacy hard solver.
    if dc_solve_mode in (
        "partitioned-flow-gauge",
        "reduced-hard-constrained-lstsq",
        "reduced-soft-constrained-lstsq",
    ) and solver_kind != "constrained_dc_equal_A_equal_V":
        solver_kind = str(solver_kind)

    if dc_solve_mode == "partitioned-flow-gauge":
        return _solve_partitioned_flow_gauge_pressure(
            data=data,
            conductance=conductance,
            arterial_flow_mode=arterial_flow_mode,
            device=device,
        )
    if dc_solve_mode in (
        "reduced-hard-constrained-lstsq",
        "reduced-soft-constrained-lstsq",
    ):
        return _solve_reduced_constrained_pressure(
            data=data,
            conductance=conductance,
            arterial_flow_mode=arterial_flow_mode,
            pressure_constraints=pressure_constraints,
            alpha_pa=alpha_pa,
            device=device,
            dc_solve_mode=dc_solve_mode,
            lambda_kirchhoff=lambda_kirchhoff,
            lambda_pressure_constraints=lambda_pressure_constraints,
            lambda_flow_residual=lambda_flow_residual,
            flip_observed_flow_sign=flip_observed_flow_sign,
            use_observed_flow_snr_weighting=use_observed_flow_snr_weighting,
        )
    if solver_kind == "constrained_dc_equal_A_equal_V":
        solve_result = constrained_dc_solve_equal_A_equal_V_torch(
            edge_index=data.edge_index.to(device=device),
            g_edge=conductance.to(device=device, dtype=torch.float64),
            num_nodes=int(len(data.node_id)),
            arterial_nodes=data.arterial_node_indices,
            arterial_flows=data.boundary_injection_m3_s[data.arterial_node_indices],
            venous_nodes=data.venous_node_indices,
            device=device,
        )
        venous_nodes = data.venous_node_indices.detach().cpu().numpy().astype(np.int64)
        gauge_node = int(venous_nodes[0]) if venous_nodes.size else int(data.reference_node)
        return {
            "pressure_pa": solve_result.pressure_pa.to(dtype=torch.float32),
            "edge_pressure_drop_pa": solve_result.edge_pressure_drop_pa.to(dtype=torch.float32),
            "edge_flow_m3_s": solve_result.edge_flow_m3_s.to(dtype=torch.float32),
            "nodal_residual_m3_s": solve_result.nodal_residual_m3_s.to(dtype=torch.float32),
            "source_vector_m3_s": data.boundary_injection_m3_s.to(dtype=torch.float32),
            "dc_solve_mode": "hard_equal_a_equal_v_flow_balance",
            "solver_kind_used": "constrained_dc_equal_A_equal_V",
            "constraint_labels": [
                "hard_total_inlet_equals_total_outlet",
                "hard_equal_arterial_pressure",
                "hard_equal_venous_pressure",
                "soft_arterial_inlet_distribution_match",
                "soft_free_node_kirchhoff_fit",
            ],
            "gauge_node_index": gauge_node,
            "gauge_node_id": _node_label(data, gauge_node),
            "pressure_prescribed_node_ids": _node_label(data, gauge_node),
            "partitioned_solver_used_lstsq": False,
            "pressure_constraints_used_in_solve": "hard_equal_a_equal_v",
            "formulation_warning": "",
            "diagnostics": {
                key: value.to(dtype=torch.float32)
                for key, value in solve_result.diagnostics.items()
            },
        }
    raise ValueError(f"Unsupported dc_solve_mode: {dc_solve_mode}")


def main() -> None:
    start_time = time.perf_counter()
    args = parse_args()
    config, config_path = _load_config(args.config)
    device = torch.device(args.device)
    pressure_constraints = _selected_pressure_constraints(args)

    graph_path = args.graph.expanduser().resolve()
    graph = load_graph(graph_path)
    data = build_real_gnn_data(graph_path, config)

    viscosity = float(args.viscosity_pa_s)
    if not math.isfinite(viscosity) or viscosity <= 0.0:
        raise ValueError("--viscosity-pa-s must be a positive finite value.")

    radius_m = data.radius_m.detach().cpu().numpy().astype(np.float64)
    length_m = data.length_m.detach().cpu().numpy().astype(np.float64)
    conductance_np = (
        np.pi * radius_m**4 / (8.0 * viscosity * np.maximum(length_m, 1.0e-30))
    )
    conductance = torch.tensor(conductance_np, dtype=torch.float32)

    solver_kind = str(
        config.get("physics", {}).get("solver_kind", "constrained_dc_equal_A_equal_V")
    )
    solve_result = _solve_baseline(
        data=data,
        conductance=conductance,
        solver_kind=solver_kind,
        device=device,
        dc_solve_mode=str(args.dc_solve_mode),
        arterial_flow_mode=str(args.arterial_flow_mode),
        pressure_constraints=pressure_constraints,
        alpha_pa=args.alpha_pa,
        lambda_kirchhoff=float(args.lambda_kirchhoff),
        lambda_pressure_constraints=float(args.lambda_pressure_constraints),
        lambda_flow_residual=float(args.lambda_flow_residual),
        flip_observed_flow_sign=bool(args.flip_observed_flow_sign),
        use_observed_flow_snr_weighting=not bool(args.no_observed_flow_snr_weighting),
    )

    pressure = solve_result["pressure_pa"].detach().cpu().numpy().astype(np.float64)
    source = data.edge_index[0].detach().cpu().numpy()
    target = data.edge_index[1].detach().cpu().numpy()

    p_source = pressure[source]
    p_target = pressure[target]

    # Orientation diagnostics. The solver-returned edge flow is the primary
    # Poiseuille prediction. The recomputed source-minus-target and
    # target-minus-source versions are kept for debugging sign conventions.
    pressure_drop_source_minus_target = p_source - p_target
    pressure_drop_target_minus_source = p_target - p_source

    q_pred_source_minus_target = conductance_np * pressure_drop_source_minus_target
    q_pred_target_minus_source = conductance_np * pressure_drop_target_minus_source

    q_pred_solver = solve_result["edge_flow_m3_s"].detach().cpu().numpy().astype(np.float64)
    if q_pred_solver.shape[0] != conductance_np.shape[0]:
        raise ValueError(
            "Solver returned edge_flow_m3_s with unexpected length: "
            f"{q_pred_solver.shape[0]} vs n_edges={conductance_np.shape[0]}"
        )

    # This is the authoritative predicted flow used for metrics.
    q_pred = q_pred_solver

    physical_flow_sign = np.ones_like(q_pred, dtype=np.float64)
    use_physical_flow_plot = bool(graph.graph.get("canonical_conversion_ready", False))
    if use_physical_flow_plot:
        for edge_idx, (u, v) in enumerate(data.edge_ids):
            edge_data = graph.edges[u, v]
            flow_from = edge_data.get("flow_from")
            flow_to = edge_data.get("flow_to")
            if flow_from == v and flow_to == u:
                physical_flow_sign[edge_idx] = -1.0
            else:
                physical_flow_sign[edge_idx] = 1.0
    q_pred_physical = q_pred * physical_flow_sign

    q_obs = _observed_flow_m3_s(data).astype(np.float64)
    if args.flip_observed_flow_sign:
        q_obs = -q_obs
    residual = q_pred - q_obs
    valid_mask = _valid_observed_mask(data, q_obs)
    # Keep SI units inside the solve and convert only for reported diagnostics.
    q_obs_nl_s = q_obs * NL_PER_M3
    q_pred_nl_s = q_pred * NL_PER_M3
    residual_nl_s = residual * NL_PER_M3

    q_scale = _flow_scale(q_obs, valid_mask)
    sign_eps_abs = (
        float(args.sign_eps_relative) * q_scale
        if math.isfinite(q_scale) and q_scale > 0.0
        else 1.0e-30
    )
    q_scale_nl_s = q_scale * NL_PER_M3 if math.isfinite(q_scale) else float("nan")

    sign_flip = _sign_flip_mask(q_pred, q_obs, valid_mask, sign_eps_abs)
    sign_flip_source_minus_target = _sign_flip_mask(
        q_pred_source_minus_target, q_obs, valid_mask, sign_eps_abs
    )
    sign_flip_target_minus_source = _sign_flip_mask(
        q_pred_target_minus_source, q_obs, valid_mask, sign_eps_abs
    )

    nodal_residual = solve_result["nodal_residual_m3_s"]
    source_vector = solve_result.get("source_vector_m3_s")
    if nodal_residual is None:
        nodal_residual_np = np.full(len(data.node_id), np.nan, dtype=np.float64)
        kirchhoff_residual = float("nan")
    else:
        nodal_residual_np = nodal_residual.detach().cpu().numpy().astype(np.float64)
        internal_mask = np.ones(len(data.node_id), dtype=bool)
        if data.arterial_node_indices.numel():
            internal_mask[data.arterial_node_indices.detach().cpu().numpy()] = False
        if data.venous_node_indices.numel():
            internal_mask[data.venous_node_indices.detach().cpu().numpy()] = False
        kirchhoff_residual = float(np.linalg.norm(nodal_residual_np[internal_mask]))
    if source_vector is None:
        source_vector_np = np.full(len(data.node_id), np.nan, dtype=np.float64)
    else:
        source_vector_np = source_vector.detach().cpu().numpy().astype(np.float64)

    arterial_idx_np = data.arterial_node_indices.detach().cpu().numpy().astype(np.int64)
    venous_idx_np = data.venous_node_indices.detach().cpu().numpy().astype(np.int64)
    arterial_dataset_flow_total = (
        float(np.sum(source_vector_np[arterial_idx_np]))
        if arterial_idx_np.size
        else 0.0
    )
    venous_dataset_flow_total = (
        float(np.sum(source_vector_np[venous_idx_np]))
        if venous_idx_np.size
        else 0.0
    )
    arterial_predicted_flow_total = (
        float(np.sum((nodal_residual_np + source_vector_np)[arterial_idx_np]))
        if arterial_idx_np.size
        else 0.0
    )
    venous_predicted_flow_total = (
        float(np.sum((nodal_residual_np + source_vector_np)[venous_idx_np]))
        if venous_idx_np.size
        else 0.0
    )
    arterial_boundary_mismatch_l2 = (
        float(np.linalg.norm(nodal_residual_np[arterial_idx_np]))
        if arterial_idx_np.size
        else 0.0
    )
    venous_boundary_mismatch_l2 = (
        float(np.linalg.norm(nodal_residual_np[venous_idx_np]))
        if venous_idx_np.size
        else 0.0
    )

    arterial_pressure_spread = (
        float(np.max(pressure[arterial_idx_np]) - np.min(pressure[arterial_idx_np]))
        if arterial_idx_np.size
        else float("nan")
    )
    venous_pressure_spread = (
        float(np.max(pressure[venous_idx_np]) - np.min(pressure[venous_idx_np]))
        if venous_idx_np.size
        else float("nan")
    )

    pressure_constraint_values: dict[str, float] = {
        "equal_a_pressure_spread_pa": arterial_pressure_spread,
        "equal_v_pressure_spread_pa": venous_pressure_spread,
        "equal_av_pressure_drop_residual_pa": float("nan"),
        "mean_a_minus_v_alpha_residual_pa": float("nan"),
        "gauge_pressure_abs_pa": float("nan"),
    }
    if venous_idx_np.size:
        pressure_constraint_values["gauge_pressure_abs_pa"] = float(abs(pressure[venous_idx_np[0]]))
    else:
        pressure_constraint_values["gauge_pressure_abs_pa"] = float(
            abs(pressure[int(data.reference_node)])
        )
    if arterial_idx_np.size >= 2 and venous_idx_np.size >= 2:
        pressure_constraint_values["equal_av_pressure_drop_residual_pa"] = float(
            (pressure[arterial_idx_np[0]] - pressure[venous_idx_np[0]])
            - (pressure[arterial_idx_np[1]] - pressure[venous_idx_np[1]])
        )
    if arterial_idx_np.size >= 2 and venous_idx_np.size >= 1 and args.alpha_pa is not None:
        pressure_constraint_values["mean_a_minus_v_alpha_residual_pa"] = float(
            0.5 * (pressure[arterial_idx_np[0]] + pressure[arterial_idx_np[1]])
            - pressure[venous_idx_np[0]]
            - float(args.alpha_pa)
        )
    actual_constraint_residuals = _constraint_residuals_pa(
        data=data,
        pressure_pa=pressure,
        pressure_constraints=pressure_constraints,
        alpha_pa=args.alpha_pa,
    )
    actual_constraint_values = np.asarray(
        [value for _, value in actual_constraint_residuals],
        dtype=np.float64,
    )
    pressure_constraints_used = str(
        solve_result.get("pressure_constraints_used_in_solve", False)
    )

    arterial_equality_residual_pa = (
        float(abs(pressure[arterial_idx_np[0]] - pressure[arterial_idx_np[1]]))
        if arterial_idx_np.size >= 2
        else float("nan")
    )
    venous_equality_residual_pa = (
        float(abs(pressure[venous_idx_np[0]] - pressure[venous_idx_np[1]]))
        if venous_idx_np.size >= 2
        else float("nan")
    )
    boundary_residual_rms_pa = _rms(actual_constraint_values)
    boundary_residual_max_pa = _safe_max_abs(actual_constraint_values)

    internal_residual_nl_s = nodal_residual_np[internal_mask] * NL_PER_M3
    kirchhoff_rms_per_internal_node_nl_s = _rms(internal_residual_nl_s)
    kirchhoff_mae_per_internal_node_nl_s = _mean_abs(internal_residual_nl_s)
    kirchhoff_p95_abs_nl_s = _percentile_abs(internal_residual_nl_s, 95.0)
    kirchhoff_max_abs_nl_s = _safe_max_abs(internal_residual_nl_s)
    kirchhoff_rms_normalized_median = (
        kirchhoff_rms_per_internal_node_nl_s / q_scale_nl_s
        if math.isfinite(kirchhoff_rms_per_internal_node_nl_s)
        and math.isfinite(q_scale_nl_s)
        and q_scale_nl_s > 0.0
        else float("nan")
    )
    flow_rmse_nl_s = _rmse(q_pred_nl_s, q_obs_nl_s, valid_mask)
    flow_mae_nl_s = _mae(q_pred_nl_s, q_obs_nl_s, valid_mask)
    flow_nrmse_median = (
        flow_rmse_nl_s / q_scale_nl_s
        if math.isfinite(flow_rmse_nl_s) and math.isfinite(q_scale_nl_s) and q_scale_nl_s > 0.0
        else float("nan")
    )
    pressure_min_pa = float(np.min(pressure)) if pressure.size else float("nan")
    pressure_max_pa = float(np.max(pressure)) if pressure.size else float("nan")
    pressure_mean_pa = _mean_or_nan(pressure)
    arterial_pressure_mean_pa = _mean_or_nan(pressure[arterial_idx_np])
    venous_pressure_mean_pa = _mean_or_nan(pressure[venous_idx_np])
    av_drop_1_pa = (
        float(pressure[arterial_idx_np[0]] - pressure[venous_idx_np[0]])
        if arterial_idx_np.size >= 1 and venous_idx_np.size >= 1
        else float("nan")
    )
    av_drop_2_pa = (
        float(pressure[arterial_idx_np[1]] - pressure[venous_idx_np[1]])
        if arterial_idx_np.size >= 2 and venous_idx_np.size >= 2
        else float("nan")
    )
    mean_av_drop_pa = _mean_or_nan(np.asarray([av_drop_1_pa, av_drop_2_pa], dtype=np.float64))
    n_internal_nodes = int(np.sum(internal_mask))

    sign_flip_count, sign_flip_fraction, sign_eligible_count = _sign_count_fraction(
        q_pred, q_obs, valid_mask, sign_eps_abs
    )
    (
        sign_flip_count_source_minus_target,
        sign_flip_fraction_source_minus_target,
        sign_eligible_count_source_minus_target,
    ) = _sign_count_fraction(q_pred_source_minus_target, q_obs, valid_mask, sign_eps_abs)
    (
        sign_flip_count_target_minus_source,
        sign_flip_fraction_target_minus_source,
        sign_eligible_count_target_minus_source,
    ) = _sign_count_fraction(q_pred_target_minus_source, q_obs, valid_mask, sign_eps_abs)

    solver_vs_recomputed_source_minus_target_max_abs = _safe_max_abs(
        q_pred_solver - q_pred_source_minus_target, valid_mask
    )
    solver_vs_recomputed_target_minus_source_max_abs = _safe_max_abs(
        q_pred_solver - q_pred_target_minus_source, valid_mask
    )

    output_dir = args.output_dir.expanduser().resolve()
    if args.run_name:
        output_dir = output_dir / args.run_name
    output_dir.mkdir(parents=True, exist_ok=True)
    runtime_seconds = time.perf_counter() - start_time
    solver_success = bool(
        np.isfinite(pressure).all()
        and np.isfinite(q_pred).all()
        and np.isfinite(nodal_residual_np).all()
    )

    edge_rows: list[dict[str, object]] = []
    for edge_idx, (u, v) in enumerate(data.edge_ids):
        sign_match = bool(
            valid_mask[edge_idx]
            and np.isfinite(q_obs[edge_idx])
            and np.isfinite(q_pred[edge_idx])
            and np.sign(q_obs[edge_idx]) == np.sign(q_pred[edge_idx])
        )
        edge_rows.append(
            {
                "edge_id": int(edge_idx),
                "source": str(u),
                "target": str(v),
                "source_node": str(u),
                "target_node": str(v),
                "radius_m": float(radius_m[edge_idx]),
                "length_m": float(length_m[edge_idx]),
                "vessel_radius_m": float(radius_m[edge_idx]),
                "vessel_length_m": float(length_m[edge_idx]),
                "G_poiseuille_m3_pa_s": float(conductance_np[edge_idx]),
                "poiseuille_conductance": float(conductance_np[edge_idx]),
                "p_source_pa": float(p_source[edge_idx]),
                "p_target_pa": float(p_target[edge_idx]),
                "pressure_drop_source_minus_target_pa": float(
                    pressure_drop_source_minus_target[edge_idx]
                ),
                "pressure_drop_target_minus_source_pa": float(
                    pressure_drop_target_minus_source[edge_idx]
                ),
                "q_obs_m3_s": float(q_obs[edge_idx]),
                "q_pred_m3_s": float(q_pred[edge_idx]),
                "q_pred_solver_m3_s": float(q_pred_solver[edge_idx]),
                "q_pred_source_minus_target_m3_s": float(
                    q_pred_source_minus_target[edge_idx]
                ),
                "q_pred_target_minus_source_m3_s": float(
                    q_pred_target_minus_source[edge_idx]
                ),
                "observed_flow_nl_s": float(q_obs_nl_s[edge_idx]),
                "predicted_flow_nl_s": float(q_pred_nl_s[edge_idx]),
                "predicted_flow_physical_nl_s": float(q_pred_physical[edge_idx] * NL_PER_M3)
                if use_physical_flow_plot
                else float("nan"),
                "flow_residual_nl_s": float(residual_nl_s[edge_idx]),
                "absolute_flow_residual_nl_s": float(abs(residual_nl_s[edge_idx])),
                "q_obs_over_scale": float(q_obs[edge_idx] / q_scale)
                if math.isfinite(q_scale) and q_scale > 0.0
                else float("nan"),
                "q_pred_over_scale": float(q_pred[edge_idx] / q_scale)
                if math.isfinite(q_scale) and q_scale > 0.0
                else float("nan"),
                "residual_m3_s": float(residual[edge_idx]),
                "observed_flow_valid": bool(valid_mask[edge_idx]),
                "valid_observed_flow": bool(valid_mask[edge_idx]),
                "sign_match": sign_match,
                "sign_flip": bool(sign_flip[edge_idx]),
                "sign_flip_source_minus_target": bool(
                    sign_flip_source_minus_target[edge_idx]
                ),
                "sign_flip_target_minus_source": bool(
                    sign_flip_target_minus_source[edge_idx]
                ),
                "tile_id": int(data.edge_tile_id[edge_idx])
                if hasattr(data, "edge_tile_id")
                else -1,
            }
        )

    predicted_net_flow_nl_s = (nodal_residual_np + source_vector_np) * NL_PER_M3
    boundary_injection_nl_s = source_vector_np * NL_PER_M3
    # Kirchhoff residuals are physical flow imbalances, reported in nL/s.
    kirchhoff_residual_nl_s = nodal_residual_np * NL_PER_M3
    arterial_idx = arterial_idx_np
    venous_idx = venous_idx_np
    arterial_set = set(arterial_idx.tolist())
    venous_set = set(venous_idx.tolist())
    boundary_set = arterial_set | venous_set
    node_rows: list[dict[str, object]] = []
    for node_index, node_id in enumerate(data.node_id):
        is_arterial = node_index in arterial_set
        is_venous = node_index in venous_set
        is_boundary = node_index in boundary_set
        is_internal = not is_boundary
        node_type = (
            "arterial"
            if is_arterial
            else "venous"
            if is_venous
            else "internal"
        )
        x_px, y_px = _node_coordinates_px(data, node_index)
        node_rows.append(
            {
                "node_index": int(node_index),
                "node_id": str(node_id),
                "node_type": node_type,
                "pressure_pa": float(pressure[node_index]),
                "boundary_injection_nl_s": float(boundary_injection_nl_s[node_index]),
                "predicted_net_flow_nl_s": float(predicted_net_flow_nl_s[node_index]),
                "kirchhoff_residual_nl_s": float(kirchhoff_residual_nl_s[node_index]),
                "x_px": x_px,
                "y_px": y_px,
                "is_arterial": bool(is_arterial),
                "is_venous": bool(is_venous),
                "is_boundary": bool(is_boundary),
                "is_internal": bool(is_internal),
            }
        )

    summary = {
        "run_name": args.run_name or "",
        "script_name": Path(__file__).name,
        "graph_path": str(graph_path),
        "config_path": str(config_path) if config_path is not None else "",
        "output_dir": str(output_dir),
        "solver_kind": str(solve_result.get("solver_kind_used", solver_kind)),
        "dc_solve_mode": str(solve_result.get("dc_solve_mode", args.dc_solve_mode)),
        "arterial_flow_mode": str(args.arterial_flow_mode),
        "pressure_constraints": "|".join(pressure_constraints),
        "pressure_constraint_labels": "|".join(solve_result["constraint_labels"]),
        "pressure_constraints_used_in_solve": pressure_constraints_used,
        "alpha_pa": float(args.alpha_pa) if args.alpha_pa is not None else float("nan"),
        "viscosity_pa_s": viscosity,
        "lambda_q": float(args.lambda_flow_residual),
        "lambda_k": float(args.lambda_kirchhoff),
        "lambda_b": float(args.lambda_pressure_constraints),
        "lambda_kirchhoff": float(args.lambda_kirchhoff),
        "lambda_pressure_constraints": float(args.lambda_pressure_constraints),
        "lambda_flow_residual": float(args.lambda_flow_residual),
        "flip_observed_flow_sign": bool(args.flip_observed_flow_sign),
        "use_observed_flow_snr_weighting": not bool(args.no_observed_flow_snr_weighting),
        **{
            key: float(value)
            for key, value in data.observed_flow_weight_stats.items()
            if isinstance(value, (int, float, bool))
        },
        "n_nodes": int(len(data.node_id)),
        "n_edges": int(data.n_edges),
        "n_observed_edges": int(np.sum(valid_mask)),
        "n_internal_nodes": n_internal_nodes,
        "arterial_node_count": int(data.arterial_node_indices.numel()),
        "venous_node_count": int(data.venous_node_indices.numel()),
        "reference_node_index": int(data.reference_node),
        "gauge_node_index": int(solve_result.get("gauge_node_index", -1)),
        "gauge_node_id": str(solve_result.get("gauge_node_id", "")),
        "pressure_prescribed_node_ids": str(
            solve_result.get("pressure_prescribed_node_ids", "")
        ),
        "solver_success": solver_success,
        "runtime_seconds": runtime_seconds,
        "observed_flow_scale_m3_s": q_scale,
        "observed_flow_scale_nl_s": q_scale_nl_s,
        "sign_eps_abs_m3_s": sign_eps_abs,
        "sign_eps_abs_nl_s": sign_eps_abs * NL_PER_M3,
        "sign_eps_relative": float(args.sign_eps_relative),
        "arterial_node_ids": "|".join(_node_label(data, idx) for idx in arterial_idx),
        "venous_node_ids": "|".join(_node_label(data, idx) for idx in venous_idx),
        "flow_rmse_nl_s": flow_rmse_nl_s,
        "flow_mae_nl_s": flow_mae_nl_s,
        "flow_nrmse_median": flow_nrmse_median,
        "observed_flow_rmse_m3_s": _rmse(q_pred, q_obs, valid_mask),
        "observed_flow_mae_m3_s": _mae(q_pred, q_obs, valid_mask),
        "observed_flow_relative_rmse": _relative_rmse(q_pred, q_obs, valid_mask),
        "normalized_rmse": _normalized_rmse(q_pred, q_obs, valid_mask),
        "kirchhoff_residual_l2_m3_s": kirchhoff_residual,
        "kirchhoff_rms_per_internal_node_nl_s": kirchhoff_rms_per_internal_node_nl_s,
        "kirchhoff_mae_per_internal_node_nl_s": kirchhoff_mae_per_internal_node_nl_s,
        "kirchhoff_p95_abs_nl_s": kirchhoff_p95_abs_nl_s,
        "kirchhoff_max_abs_nl_s": kirchhoff_max_abs_nl_s,
        "kirchhoff_rms_normalized_median": kirchhoff_rms_normalized_median,
        "pressure_min_pa": pressure_min_pa,
        "pressure_max_pa": pressure_max_pa,
        "pressure_range_pa": (
            pressure_max_pa - pressure_min_pa
            if math.isfinite(pressure_min_pa) and math.isfinite(pressure_max_pa)
            else float("nan")
        ),
        "pressure_mean_pa": pressure_mean_pa,
        "arterial_pressure_mean_pa": arterial_pressure_mean_pa,
        "venous_pressure_mean_pa": venous_pressure_mean_pa,
        "arterial_pressure_spread_pa": arterial_pressure_spread,
        "venous_pressure_spread_pa": venous_pressure_spread,
        "av_drop_1_pa": av_drop_1_pa,
        "av_drop_2_pa": av_drop_2_pa,
        "mean_av_drop_pa": mean_av_drop_pa,
        "arterial_equality_residual_pa": arterial_equality_residual_pa,
        "venous_equality_residual_pa": venous_equality_residual_pa,
        "boundary_residual_rms_pa": boundary_residual_rms_pa,
        "boundary_residual_max_pa": boundary_residual_max_pa,
        "constraint_residual_labels": "|".join(
            label for label, _ in actual_constraint_residuals
        ),
        "constraint_residual_values_pa": "|".join(
            f"{value:.12g}" for _, value in actual_constraint_residuals
        ),
        "sign_flipped_predicted_flow_count": sign_flip_count,
        "sign_flipped_predicted_flow_fraction": sign_flip_fraction,
        "sign_evaluable_edge_count": sign_eligible_count,
        "sign_flipped_source_minus_target_count": sign_flip_count_source_minus_target,
        "sign_flipped_source_minus_target_fraction": (
            sign_flip_fraction_source_minus_target
        ),
        "sign_evaluable_source_minus_target_edge_count": (
            sign_eligible_count_source_minus_target
        ),
        "sign_flipped_target_minus_source_count": sign_flip_count_target_minus_source,
        "sign_flipped_target_minus_source_fraction": (
            sign_flip_fraction_target_minus_source
        ),
        "sign_evaluable_target_minus_source_edge_count": (
            sign_eligible_count_target_minus_source
        ),
        "solver_vs_recomputed_source_minus_target_max_abs_m3_s": (
            solver_vs_recomputed_source_minus_target_max_abs
        ),
        "solver_vs_recomputed_target_minus_source_max_abs_m3_s": (
            solver_vs_recomputed_target_minus_source_max_abs
        ),
        "predicted_flow_l2_over_observed_flow_l2": (
            float(
                np.linalg.norm(q_pred[valid_mask])
                / max(np.linalg.norm(q_obs[valid_mask]), 1.0e-300)
            )
            if np.any(valid_mask)
            else float("nan")
        ),
        "arterial_dataset_flow_total_m3_s": arterial_dataset_flow_total,
        "venous_dataset_flow_total_m3_s": venous_dataset_flow_total,
        "arterial_predicted_flow_total_m3_s": arterial_predicted_flow_total,
        "venous_predicted_flow_total_m3_s": venous_predicted_flow_total,
        "arterial_boundary_mismatch_l2_m3_s": arterial_boundary_mismatch_l2,
        "venous_boundary_mismatch_l2_m3_s": venous_boundary_mismatch_l2,
        "arterial_dataset_flow_total_nl_s": arterial_dataset_flow_total * NL_PER_M3,
        "venous_dataset_flow_total_nl_s": venous_dataset_flow_total * NL_PER_M3,
        "arterial_predicted_flow_total_nl_s": arterial_predicted_flow_total * NL_PER_M3,
        "venous_predicted_flow_total_nl_s": venous_predicted_flow_total * NL_PER_M3,
        "arterial_boundary_mismatch_l2_nl_s": arterial_boundary_mismatch_l2 * NL_PER_M3,
        "venous_boundary_mismatch_l2_nl_s": venous_boundary_mismatch_l2 * NL_PER_M3,
        "unknown_pressure_node_count": float(
            solve_result["diagnostics"]
            .get("unknown_pressure_node_count", torch.tensor(float("nan")))
            .detach()
            .cpu()
        ),
        "active_equation_row_count": float(
            solve_result["diagnostics"]
            .get("active_equation_row_count", torch.tensor(float("nan")))
            .detach()
            .cpu()
        ),
        "total_injection_m3_s": float(
            solve_result["diagnostics"]
            .get("total_injection_m3_s", torch.tensor(float("nan")))
            .detach()
            .cpu()
        ),
        "net_injection_m3_s": float(
            solve_result["diagnostics"]
            .get("net_injection_m3_s", torch.tensor(float("nan")))
            .detach()
            .cpu()
        ),
        "partitioned_system_residual_l2": float(
            solve_result["diagnostics"]
            .get("partitioned_system_residual_l2", torch.tensor(float("nan")))
            .detach()
            .cpu()
        ),
        "partitioned_system_residual_max": float(
            solve_result["diagnostics"]
            .get("partitioned_system_residual_max", torch.tensor(float("nan")))
            .detach()
            .cpu()
        ),
        "partitioned_solver_used_lstsq": bool(
            float(
                solve_result["diagnostics"]
                .get("partitioned_solver_used_lstsq", torch.tensor(0.0))
                .detach()
                .cpu()
            )
        ),
        "reduced_constraint_count": float(
            solve_result["diagnostics"]
            .get("reduced_constraint_count", torch.tensor(float("nan")))
            .detach()
            .cpu()
        ),
        "reduced_constraint_residual_l2": float(
            solve_result["diagnostics"]
            .get("reduced_constraint_residual_l2", torch.tensor(float("nan")))
            .detach()
            .cpu()
        ),
        "reduced_constraint_residual_max": float(
            solve_result["diagnostics"]
            .get("reduced_constraint_residual_max", torch.tensor(float("nan")))
            .detach()
            .cpu()
        ),
        "partitioned_constraint_residual_l2": float(
            solve_result["diagnostics"]
            .get("partitioned_constraint_residual_l2", torch.tensor(float("nan")))
            .detach()
            .cpu()
        ),
        "laplacian_scale_used": float(
            solve_result["diagnostics"]
            .get("laplacian_scale_used", torch.tensor(float("nan")))
            .detach()
            .cpu()
        ),
        "flow_row_scale_used": float(
            solve_result["diagnostics"]
            .get("flow_row_scale_used", torch.tensor(float("nan")))
            .detach()
            .cpu()
        ),
        "flow_residual_in_solver_l2": float(
            solve_result["diagnostics"]
            .get("flow_residual_in_solver_l2", torch.tensor(float("nan")))
            .detach()
            .cpu()
        ),
        "flow_residual_in_solver_rmse": float(
            solve_result["diagnostics"]
            .get("flow_residual_in_solver_rmse", torch.tensor(float("nan")))
            .detach()
            .cpu()
        ),
        "flow_residual_edge_count": float(
            solve_result["diagnostics"]
            .get("flow_residual_edge_count", torch.tensor(float("nan")))
            .detach()
            .cpu()
        ),
        "kirchhoff_objective_l2": float(
            solve_result["diagnostics"]
            .get("kirchhoff_objective_l2", torch.tensor(float("nan")))
            .detach()
            .cpu()
        ),
        "pressure_constraint_objective_l2": float(
            solve_result["diagnostics"]
            .get("pressure_constraint_objective_l2", torch.tensor(float("nan")))
            .detach()
            .cpu()
        ),
        "gauge_row_excluded_from_lstsq": bool(
            float(
                solve_result["diagnostics"]
                .get("gauge_row_excluded_from_lstsq", torch.tensor(0.0))
                .detach()
                .cpu()
            )
        ),
        "gauge_eliminated_directly": bool(
            float(
                solve_result["diagnostics"]
                .get("gauge_eliminated_directly", torch.tensor(0.0))
                .detach()
                .cpu()
            )
        ),
        "constrained_vs_partitioned_pressure_max_abs": float(
            solve_result["diagnostics"]
            .get(
                "constrained_vs_partitioned_pressure_max_abs",
                torch.tensor(float("nan")),
            )
            .detach()
            .cpu()
        ),
        "constrained_vs_partitioned_flow_max_abs": float(
            solve_result["diagnostics"]
            .get(
                "constrained_vs_partitioned_flow_max_abs",
                torch.tensor(float("nan")),
            )
            .detach()
            .cpu()
        ),
        "formulation_warning": str(solve_result.get("formulation_warning", "")),
        **pressure_constraint_values,
        "hard_boundary_constraint_residual_l2": float(
            solve_result["diagnostics"]
            .get("hard_boundary_constraint_residual_l2", torch.tensor(float("nan")))
            .detach()
            .cpu()
        ),
        "hard_boundary_constraint_residual_max": float(
            solve_result["diagnostics"]
            .get("hard_boundary_constraint_residual_max", torch.tensor(float("nan")))
            .detach()
            .cpu()
        ),
        "constrained_equation_residual_l2": float(
            solve_result["diagnostics"]
            .get("constrained_equation_residual_l2", torch.tensor(float("nan")))
            .detach()
            .cpu()
        ),
        "constrained_equation_residual_max": float(
            solve_result["diagnostics"]
            .get("constrained_equation_residual_max", torch.tensor(float("nan")))
            .detach()
            .cpu()
        ),
        "pressure_solver_final_relative_residual": float(
            solve_result["diagnostics"]
            .get("pressure_solver_final_relative_residual", torch.tensor(float("nan")))
            .detach()
            .cpu()
        ),
    }

    _write_csv(output_dir / "edge_predictions.csv", edge_rows)
    _write_csv(output_dir / "node_predictions.csv", node_rows)
    _write_csv(output_dir / "summary.csv", [summary])
    write_yaml(output_dir / "summary.yaml", summary)

    # Legacy filenames retained for backward compatibility with older scripts.
    _write_csv(output_dir / "poiseuille_edge_predictions.csv", edge_rows)
    _write_csv(output_dir / "poiseuille_summary.csv", [summary])


if __name__ == "__main__":
    main()
