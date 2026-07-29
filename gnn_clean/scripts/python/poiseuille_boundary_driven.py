#!/usr/bin/env python
"""Flow-aware Poiseuille pressure-fit benchmark for the real-data GNN graph."""

from __future__ import annotations

import argparse
import csv
import math
import sys
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from physics_layer import build_weighted_laplacian
from real_data import MU, build_real_gnn_data
from utils import load_yaml


DEFAULT_GRAPH = PROJECT_ROOT / "datasets" / "emb1_mosaic_graph_analyzed.gpickle"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "flow_aware_poiseuille_pressure_fit"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("graph", nargs="?", type=Path, default=DEFAULT_GRAPH)
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--viscosity-pa-s", type=float, default=float(MU))
    parser.add_argument("--run-name", default="")
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--arterial-flow-mode",
        choices=("dataset", "none"),
        default="dataset",
    )
    parser.add_argument(
        "--pressure-constraint",
        action="append",
        choices=(
            "equal-a-equal-v",
            "equal-av-pressure-drop",
            "mean-a-minus-v-alpha-equal-v",
        ),
        default=None,
    )
    parser.add_argument("--alpha-pa", type=float, default=None)
    parser.add_argument("--lambda-q", type=float, default=1.0)
    parser.add_argument("--lambda-k", type=float, default=1.0)
    parser.add_argument("--lambda-b", type=float, default=1.0)
    parser.add_argument("--use-snr-weights", action="store_true", default=True)
    parser.add_argument(
        "--flow-scale-mode",
        choices=("median_abs", "rms", "none"),
        default="median_abs",
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


def _flow_scale(q_obs: np.ndarray, mask: np.ndarray, mode: str) -> float:
    values = q_obs[mask & np.isfinite(q_obs)]
    if values.size == 0:
        return float("nan")
    if mode == "none":
        return 1.0
    if mode == "median_abs":
        abs_values = np.abs(values)
        abs_values = abs_values[abs_values > 0.0]
        return float(np.median(abs_values)) if abs_values.size else 1.0
    if mode == "rms":
        return float(np.sqrt(np.mean(values**2)))
    raise ValueError(f"Unsupported flow-scale mode: {mode}")


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


def _selected_pressure_constraints(args: argparse.Namespace) -> list[str]:
    if args.pressure_constraint:
        return list(dict.fromkeys(args.pressure_constraint))
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


def _stack_weighted_rows(
    matrices: list[torch.Tensor],
    vectors: list[torch.Tensor],
    num_nodes: int,
    dtype: torch.dtype,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    rows = [m for m in matrices if m.numel()]
    rhs = [v for v in vectors if v.numel()]
    if rows:
        return torch.cat(rows, dim=0), torch.cat(rhs, dim=0)
    return (
        torch.zeros((0, num_nodes), dtype=dtype, device=device),
        torch.zeros((0,), dtype=dtype, device=device),
    )


def _validated_lambda(name: str, value: float) -> float:
    value = float(value)
    if not math.isfinite(value) or value < 0.0:
        raise ValueError(f"{name} must be a finite non-negative value.")
    return value


def _apply_block_weight(
    matrix: torch.Tensor,
    rhs: torch.Tensor,
    lambda_value: float,
    row_weights: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    lambda_value = _validated_lambda("lambda", lambda_value)
    if lambda_value == 0.0 or matrix.shape[0] == 0:
        return matrix[:0], rhs[:0]

    if row_weights is None:
        return math.sqrt(lambda_value) * matrix, math.sqrt(lambda_value) * rhs

    weights = row_weights.to(dtype=matrix.dtype, device=matrix.device).flatten()
    if weights.shape[0] != matrix.shape[0]:
        raise ValueError(
            "row_weights must have the same length as the number of rows in the block."
        )
    positive_mask = weights > 0.0
    if not torch.any(positive_mask):
        return matrix[:0], rhs[:0]

    matrix = matrix[positive_mask]
    rhs = rhs[positive_mask]
    weights = weights[positive_mask]
    row_scale = torch.sqrt(matrix.new_tensor(lambda_value) * weights).unsqueeze(1)
    return row_scale * matrix, row_scale.squeeze(1) * rhs


def _solve_flow_aware_pressure(
    data,
    conductance: torch.Tensor,
    arterial_flow_mode: str,
    pressure_constraints: list[str],
    alpha_pa: float | None,
    lambda_q: float,
    lambda_k: float,
    lambda_b: float,
    use_snr_weights: bool,
    flow_scale_mode: str,
    device: torch.device,
) -> dict[str, object]:
    dtype = torch.float64
    lambda_q = _validated_lambda("lambda_q", lambda_q)
    lambda_k = _validated_lambda("lambda_k", lambda_k)
    lambda_b = _validated_lambda("lambda_b", lambda_b)

    edge_index = data.edge_index.to(device=device)
    conductance = conductance.to(device=device, dtype=dtype)
    num_nodes = int(len(data.node_id))

    laplacian = build_weighted_laplacian(
        edge_index=edge_index,
        conductance=conductance,
        n_nodes=num_nodes,
    ).to(dtype=dtype, device=device)

    q_obs = _observed_flow_m3_s(data).astype(np.float64)
    valid_flow_mask = _valid_observed_mask(data, q_obs)
    q_scale = _flow_scale(q_obs, valid_flow_mask, flow_scale_mode)
    q_scale = q_scale if math.isfinite(q_scale) and q_scale > 0.0 else 1.0

    source_vector = torch.zeros(num_nodes, dtype=dtype, device=device)
    equation_mask = torch.ones(num_nodes, dtype=torch.bool, device=device)

    venous_nodes = data.venous_node_indices.to(device=device, dtype=torch.long).flatten()
    arterial_nodes = data.arterial_node_indices.to(device=device, dtype=torch.long).flatten()
    if venous_nodes.numel():
        equation_mask[venous_nodes] = False

    if arterial_flow_mode == "dataset":
        arterial_flows = data.boundary_injection_m3_s[data.arterial_node_indices].to(
            device=device, dtype=dtype
        ).flatten()
        if arterial_nodes.numel() != arterial_flows.numel():
            raise ValueError("Arterial nodes and flows must have matching lengths.")
        if arterial_nodes.numel():
            source_vector.index_add_(0, arterial_nodes, arterial_flows)
    elif arterial_flow_mode == "none":
        if arterial_nodes.numel():
            equation_mask[arterial_nodes] = False
    else:
        raise ValueError(f"Unsupported arterial-flow mode: {arterial_flow_mode}")

    kirchhoff_rows = torch.nonzero(equation_mask, as_tuple=False).flatten()
    kirchhoff_matrix = laplacian.index_select(0, kirchhoff_rows)
    kirchhoff_rhs = source_vector.index_select(0, kirchhoff_rows)

    constraint_rows, constraint_rhs, constraint_labels = _pressure_constraint_rows(
        data=data,
        num_nodes=num_nodes,
        pressure_constraints=pressure_constraints,
        alpha_pa=alpha_pa,
        dtype=dtype,
        device=device,
    )
    constraint_matrix = torch.stack(constraint_rows, dim=0)
    constraint_vector = torch.stack(constraint_rhs, dim=0)

    source = edge_index[0].detach().cpu().numpy()
    target = edge_index[1].detach().cpu().numpy()
    valid_edge_idx = np.flatnonzero(valid_flow_mask)

    flow_matrix = torch.zeros((len(valid_edge_idx), num_nodes), dtype=dtype, device=device)
    flow_rhs = torch.zeros(len(valid_edge_idx), dtype=dtype, device=device)
    weights_used = np.ones(len(valid_edge_idx), dtype=np.float64)
    if use_snr_weights:
        raw = data.dc_loss_weight.detach().cpu().numpy().astype(np.float64)
        raw = np.where(np.isfinite(raw) & (raw > 0.0), raw, 0.0)
        weights_used = np.square(raw[valid_edge_idx])
    for row_idx, edge_idx in enumerate(valid_edge_idx):
        coeff = float(conductance[edge_idx].detach().cpu()) / q_scale
        flow_matrix[row_idx, int(source[edge_idx])] = coeff
        flow_matrix[row_idx, int(target[edge_idx])] = -coeff
        flow_rhs[row_idx] = float(q_obs[edge_idx]) / q_scale

    flow_matrix_scaled, flow_rhs_scaled = _apply_block_weight(
        flow_matrix,
        flow_rhs,
        lambda_value=lambda_q,
        row_weights=torch.as_tensor(weights_used, dtype=dtype, device=device),
    )
    kirchhoff_matrix_scaled, kirchhoff_rhs_scaled = _apply_block_weight(
        kirchhoff_matrix,
        kirchhoff_rhs,
        lambda_value=lambda_k,
    )
    constraint_matrix_scaled, constraint_vector_scaled = _apply_block_weight(
        constraint_matrix,
        constraint_vector,
        lambda_value=lambda_b,
    )

    A, b = _stack_weighted_rows(
        matrices=[flow_matrix_scaled, kirchhoff_matrix_scaled, constraint_matrix_scaled],
        vectors=[flow_rhs_scaled, kirchhoff_rhs_scaled, constraint_vector_scaled],
        num_nodes=num_nodes,
        dtype=dtype,
        device=device,
    )
    if A.shape[0] == 0:
        raise ValueError(
            "At least one of the weighted residual blocks must be active. "
            "Check lambda_q, lambda_k, lambda_b, and the available observations."
        )
    lstsq_result = torch.linalg.lstsq(A, b)
    pressure = lstsq_result.solution

    edge_pressure_drop = pressure[edge_index[0]] - pressure[edge_index[1]]
    edge_flow = conductance * edge_pressure_drop
    nodal_residual = laplacian @ pressure - source_vector

    flow_pred_scaled = edge_flow[valid_edge_idx] / q_scale if len(valid_edge_idx) else edge_flow[:0]
    flow_obs_scaled = torch.as_tensor(
        q_obs[valid_edge_idx] / q_scale, dtype=dtype, device=device
    )
    flow_weight_sqrt = torch.as_tensor(
        np.sqrt(np.maximum(weights_used, 0.0)), dtype=dtype, device=device
    )
    flow_residual_scaled = flow_weight_sqrt * (flow_pred_scaled - flow_obs_scaled)
    constrained_residual = kirchhoff_matrix @ pressure - kirchhoff_rhs
    constraint_residual = constraint_matrix @ pressure - constraint_vector
    stacked_residual = A @ pressure - b

    diagnostics = {
        "pressure_solver_iterations_used": pressure.new_tensor(1.0),
        "pressure_solver_final_relative_residual": (
            torch.linalg.vector_norm(stacked_residual)
            / torch.linalg.vector_norm(b).clamp_min(1.0e-30)
        ),
        "pressure_solver_reached_max_iterations": pressure.new_tensor(0.0),
        "flow_residual_scaled_l2": torch.linalg.vector_norm(flow_residual_scaled),
        "flow_residual_scaled_max": (
            torch.max(torch.abs(flow_residual_scaled))
            if flow_residual_scaled.numel()
            else pressure.new_tensor(0.0)
        ),
        "hard_boundary_constraint_residual_l2": torch.linalg.vector_norm(constraint_residual),
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
    }

    return {
        "pressure_pa": pressure.to(dtype=torch.float32),
        "edge_flow_m3_s": edge_flow.to(dtype=torch.float32),
        "nodal_residual_m3_s": nodal_residual.to(dtype=torch.float32),
        "diagnostics": {k: v.to(dtype=torch.float32) for k, v in diagnostics.items()},
        "constraint_labels": constraint_labels,
        "flow_scale_m3_s": float(q_scale),
        "n_flow_rows": int(flow_matrix_scaled.shape[0]),
        "n_kirchhoff_rows": int(kirchhoff_matrix_scaled.shape[0]),
        "n_boundary_rows": int(constraint_matrix_scaled.shape[0]),
    }


def main() -> None:
    args = parse_args()
    config, config_path = _load_config(args.config)
    device = torch.device(args.device)
    pressure_constraints = _selected_pressure_constraints(args)

    graph_path = args.graph.expanduser().resolve()
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

    solve_result = _solve_flow_aware_pressure(
        data=data,
        conductance=conductance,
        arterial_flow_mode=str(args.arterial_flow_mode),
        pressure_constraints=pressure_constraints,
        alpha_pa=args.alpha_pa,
        lambda_q=float(args.lambda_q),
        lambda_k=float(args.lambda_k),
        lambda_b=float(args.lambda_b),
        use_snr_weights=bool(args.use_snr_weights),
        flow_scale_mode=str(args.flow_scale_mode),
        device=device,
    )

    pressure = solve_result["pressure_pa"].detach().cpu().numpy().astype(np.float64)
    source = data.edge_index[0].detach().cpu().numpy()
    target = data.edge_index[1].detach().cpu().numpy()
    p_source = pressure[source]
    p_target = pressure[target]

    pressure_drop_source_minus_target = p_source - p_target
    pressure_drop_target_minus_source = p_target - p_source

    q_pred_source_minus_target = conductance_np * pressure_drop_source_minus_target
    q_pred_target_minus_source = conductance_np * pressure_drop_target_minus_source
    q_pred_solver = solve_result["edge_flow_m3_s"].detach().cpu().numpy().astype(np.float64)
    q_pred = q_pred_solver

    q_obs = _observed_flow_m3_s(data).astype(np.float64)
    residual = q_pred - q_obs
    valid_mask = _valid_observed_mask(data, q_obs)

    q_scale = float(solve_result["flow_scale_m3_s"])
    sign_eps_abs = (
        float(args.sign_eps_relative) * q_scale
        if math.isfinite(q_scale) and q_scale > 0.0
        else 1.0e-30
    )

    sign_flip = _sign_flip_mask(q_pred, q_obs, valid_mask, sign_eps_abs)
    sign_flip_source_minus_target = _sign_flip_mask(
        q_pred_source_minus_target, q_obs, valid_mask, sign_eps_abs
    )
    sign_flip_target_minus_source = _sign_flip_mask(
        q_pred_target_minus_source, q_obs, valid_mask, sign_eps_abs
    )

    nodal_residual = solve_result["nodal_residual_m3_s"]
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

    edge_rows: list[dict[str, object]] = []
    for edge_idx, (u, v) in enumerate(data.edge_ids):
        edge_rows.append(
            {
                "edge_id": int(edge_idx),
                "source": str(u),
                "target": str(v),
                "radius_m": float(radius_m[edge_idx]),
                "length_m": float(length_m[edge_idx]),
                "G_poiseuille_m3_pa_s": float(conductance_np[edge_idx]),
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
                "q_obs_scaled": float(q_obs[edge_idx] / q_scale)
                if math.isfinite(q_scale) and q_scale > 0.0
                else float("nan"),
                "q_pred_scaled": float(q_pred[edge_idx] / q_scale)
                if math.isfinite(q_scale) and q_scale > 0.0
                else float("nan"),
                "residual_m3_s": float(residual[edge_idx]),
                "residual_scaled": float(residual[edge_idx] / q_scale)
                if math.isfinite(q_scale) and q_scale > 0.0
                else float("nan"),
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

    arterial_idx, venous_idx = _boundary_role_indices(data)
    summary_rows = [
        {
            "run_name": args.run_name or "",
            "script_name": Path(__file__).name,
            "graph_path": str(graph_path),
            "config_path": str(config_path) if config_path is not None else "",
            "output_dir": str(output_dir),
            "solver_kind": "flow_aware_lstsq",
            "arterial_flow_mode": str(args.arterial_flow_mode),
            "pressure_constraints": "|".join(pressure_constraints),
            "pressure_constraint_labels": "|".join(solve_result["constraint_labels"]),
            "alpha_pa": float(args.alpha_pa) if args.alpha_pa is not None else float("nan"),
            "viscosity_pa_s": viscosity,
            "lambda_q": float(args.lambda_q),
            "lambda_k": float(args.lambda_k),
            "lambda_b": float(args.lambda_b),
            "use_snr_weights": bool(args.use_snr_weights),
            "flow_scale_mode": str(args.flow_scale_mode),
            "flow_scale_m3_s": q_scale,
            "n_flow_rows": int(solve_result["n_flow_rows"]),
            "n_kirchhoff_rows": int(solve_result["n_kirchhoff_rows"]),
            "n_boundary_rows": int(solve_result["n_boundary_rows"]),
            "n_nodes": int(len(data.node_id)),
            "n_edges": int(data.n_edges),
            "n_observed_edges": int(np.sum(valid_mask)),
            "sign_eps_abs_m3_s": sign_eps_abs,
            "sign_eps_relative": float(args.sign_eps_relative),
            "arterial_node_ids": "|".join(_node_label(data, idx) for idx in arterial_idx),
            "venous_node_ids": "|".join(_node_label(data, idx) for idx in venous_idx),
            "observed_flow_rmse_m3_s": _rmse(q_pred, q_obs, valid_mask),
            "observed_flow_mae_m3_s": _mae(q_pred, q_obs, valid_mask),
            "observed_flow_relative_rmse": _relative_rmse(q_pred, q_obs, valid_mask),
            "normalized_rmse": _normalized_rmse(q_pred, q_obs, valid_mask),
            "kirchhoff_residual_l2_m3_s": kirchhoff_residual,
            "pressure_min_pa": float(np.min(pressure)) if pressure.size else float("nan"),
            "pressure_max_pa": float(np.max(pressure)) if pressure.size else float("nan"),
            "pressure_range_pa": (
                float(np.max(pressure) - np.min(pressure))
                if pressure.size
                else float("nan")
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
            "arterial_node_count": int(data.arterial_node_indices.numel()),
            "venous_node_count": int(data.venous_node_indices.numel()),
            "reference_node_index": int(data.reference_node),
            "flow_residual_scaled_l2": float(
                solve_result["diagnostics"]
                .get("flow_residual_scaled_l2", torch.tensor(float("nan")))
                .detach()
                .cpu()
            ),
            "flow_residual_scaled_max": float(
                solve_result["diagnostics"]
                .get("flow_residual_scaled_max", torch.tensor(float("nan")))
                .detach()
                .cpu()
            ),
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
    ]

    _write_csv(output_dir / "flow_aware_edge_predictions.csv", edge_rows)
    _write_csv(output_dir / "flow_aware_summary.csv", summary_rows)


if __name__ == "__main__":
    main()
