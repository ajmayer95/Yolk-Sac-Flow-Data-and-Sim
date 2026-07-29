#!/usr/bin/env python
"""Train a conductance-correction GNN with a differentiable pressure solver."""

from __future__ import annotations

import argparse
import copy
import csv
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from tqdm.auto import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from experiment import build_model, read_resolved_config
from gnn_losses import pressure_shape_loss
from physics_layer import (
    conductance_from_delta,
    solve_reduced_pressure,
)
from real_data import MU, build_real_gnn_data
from utils import resolve_device, set_random_seed, write_yaml


DEFAULT_GRAPH = PROJECT_ROOT / "datasets" / "emb1_mosaic_graph_analyzed.gpickle"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "gnn_flow"

PRESETS = {
    "solver_KB_outer_Qdelta": {
        "physics": {
            "pressure_solver_mode": "reduced-soft-constrained-lstsq",
            "pressure_solver_lambda_flow_residual": 0.0,
            "pressure_solver_lambda_kirchhoff": 1.0,
            "pressure_solver_lambda_pressure_constraints": 1.0,
        },
        "gnn_outer_losses": {
            "flow": 1.0,
            "kirchhoff": 0.0,
            "boundary": 0.0,
            "delta_l2": 1.0e-3,
            "delta_smooth": 0.0,
            "pressure_shape": 0.0,
        },
    },
    "solver_QKB_outer_Qdelta": {
        "physics": {
            "pressure_solver_mode": "reduced-soft-constrained-lstsq",
            "pressure_solver_lambda_flow_residual": 0.1,
            "pressure_solver_lambda_kirchhoff": 1.0,
            "pressure_solver_lambda_pressure_constraints": 1.0,
        },
        "gnn_outer_losses": {
            "flow": 1.0,
            "kirchhoff": 0.0,
            "boundary": 0.0,
            "delta_l2": 1.0e-3,
            "delta_smooth": 0.0,
            "pressure_shape": 0.0,
        },
    },
    "solver_QKB_outer_QKBdelta": {
        "physics": {
            "pressure_solver_mode": "reduced-soft-constrained-lstsq",
            "pressure_solver_lambda_flow_residual": 0.1,
            "pressure_solver_lambda_kirchhoff": 1.0,
            "pressure_solver_lambda_pressure_constraints": 1.0,
        },
        "gnn_outer_losses": {
            "flow": 1.0,
            "kirchhoff": 0.1,
            "boundary": 1.0,
            "delta_l2": 1.0e-3,
            "delta_smooth": 0.0,
            "pressure_shape": 0.0,
        },
    },
    "solver_QB_outer_QKdelta": {
        "physics": {
            "pressure_solver_mode": "reduced-soft-constrained-lstsq",
            "pressure_solver_lambda_flow_residual": 0.1,
            "pressure_solver_lambda_kirchhoff": 0.0,
            "pressure_solver_lambda_pressure_constraints": 1.0,
        },
        "gnn_outer_losses": {
            "flow": 1.0,
            "kirchhoff": 0.1,
            "boundary": 0.0,
            "delta_l2": 1.0e-3,
            "delta_smooth": 0.0,
            "pressure_shape": 0.0,
        },
    },
    "solver_Q_only_outer_Qdelta": {
        "physics": {
            "pressure_solver_mode": "reduced-soft-constrained-lstsq",
            "pressure_solver_lambda_flow_residual": 1.0,
            "pressure_solver_lambda_kirchhoff": 0.0,
            "pressure_solver_lambda_pressure_constraints": 0.0,
        },
        "gnn_outer_losses": {
            "flow": 1.0,
            "kirchhoff": 0.0,
            "boundary": 0.0,
            "delta_l2": 1.0e-3,
            "delta_smooth": 0.0,
            "pressure_shape": 0.0,
        },
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("graph", nargs="?", type=Path, default=DEFAULT_GRAPH)
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--run-name", default="")
    parser.add_argument("--preset", choices=sorted(PRESETS), default="solver_QKB_outer_QKBdelta")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--weight-decay", type=float, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--viscosity-pa-s", type=float, default=float(MU))
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
    parser.add_argument(
        "--flow-scale-mode",
        choices=("median_abs", "rms", "none"),
        default=None,
    )
    parser.add_argument(
        "--pressure-solver-mode",
        choices=("partitioned-flow-gauge", "reduced-soft-constrained-lstsq"),
        default=None,
    )
    parser.add_argument("--pressure-solver-lambda-kirchhoff", type=float, default=None)
    parser.add_argument("--pressure-solver-lambda-pressure-constraints", type=float, default=None)
    parser.add_argument("--pressure-solver-lambda-flow-residual", type=float, default=None)
    parser.add_argument(
        "--pressure-detach",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    parser.add_argument("--summary-only", action="store_true")
    parser.add_argument("--no-snr-weights", action="store_true")
    return parser.parse_args()


def default_config() -> dict:
    return {
        "model_name": "physics_informed_gnn",
        "K": 3,
        "model": {
            "hidden_dim": 64,
            "activation": "silu",
            "dropout": 0.0,
            "correction_bound": 2.0,
            "correction_min": -2.0,
            "correction_max": 2.0,
            "correction_parameterization": "tanh",
            "initialize_decoder_near_zero": True,
        },
        "training": {
            "seed": 0,
            "epochs": 10,
            "patience": 40,
            "learning_rate": 1.0e-3,
            "weight_decay": 1.0e-4,
            "gradient_clip_norm": 5.0,
            "optimizer": "adamw",
        },
        "data": {
            "include_boundary_nodes_in_pressure_solve": False,
            "split_fractions": {"train": 0.70, "validation": 0.15, "test": 0.15},
            "flow_normalization_reference_flux_nL_per_s": 1.0,
            "use_tilewise_flow_normalization": False,
        },
        "physics": {
            "arterial_flow_mode": "dataset",
            "pressure_constraints": ["equal-a-equal-v"],
            "alpha_pa": None,
            "use_snr_weights": True,
            "flow_scale_mode": "median_abs",
            "solver_device": "same",
            "pressure_detach": True,
            "pressure_solver_mode": "reduced-soft-constrained-lstsq",
            "pressure_solver_lambda_kirchhoff": 1.0,
            "pressure_solver_lambda_pressure_constraints": 1.0,
            "pressure_solver_lambda_flow_residual": 0.1,
            "pressure_shape_reference": "delta_zero_reference",
            "sign_eps_relative": 1.0e-6,
            "delta_saturation_atol": 5.0e-3,
        },
        "gnn_outer_losses": {
            "flow": 1.0,
            "kirchhoff": 0.1,
            "boundary": 1.0,
            "delta_l2": 1.0e-3,
            "delta_smooth": 1.0e-3,
            "pressure_shape": 0.0,
        },
        "output": {
            "save_history_csv": True,
            "save_exploration_diagnostics_csv": True,
        },
}


def deep_update(base: dict, update: dict) -> dict:
    merged = copy.deepcopy(base)
    for key, value in update.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_update(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def selected_pressure_constraints(config: dict) -> list[str]:
    constraints = list(config["physics"].get("pressure_constraints", ["equal-a-equal-v"]))
    normalized = ["gauge-only" if str(value) == "gauge_only" else str(value) for value in constraints]
    return list(dict.fromkeys(normalized))


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
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


def observed_flow_m3_s(data) -> torch.Tensor:
    return data.velocity_observed_m_s[:, 0, 0] * data.area_m2


def valid_observed_edge_mask(data) -> torch.Tensor:
    split_mask = data.train_mask | data.val_mask | data.test_mask
    return split_mask & torch.isfinite(observed_flow_m3_s(data))


def flow_scale_value(q_obs: torch.Tensor, mask: torch.Tensor, mode: str) -> torch.Tensor:
    valid = q_obs[mask & torch.isfinite(q_obs)]
    if valid.numel() == 0:
        return q_obs.new_tensor(1.0)
    if mode == "none":
        return q_obs.new_tensor(1.0)
    if mode == "median_abs":
        values = torch.abs(valid)
        values = values[values > 0.0]
        if values.numel() == 0:
            return q_obs.new_tensor(1.0)
        return torch.median(values)
    if mode == "rms":
        return torch.sqrt(torch.mean(valid**2)).clamp_min(1.0e-30)
    raise ValueError(f"Unsupported flow-scale mode: {mode}")


def maybe_initialize_decoder_near_zero(model: nn.Module, enabled: bool) -> None:
    if not enabled:
        return
    decoder = getattr(model, "delta_decoder", None)
    if not isinstance(decoder, nn.Sequential) or len(decoder) == 0:
        return
    last = decoder[-1]
    if isinstance(last, nn.Linear):
        nn.init.zeros_(last.weight)
        nn.init.zeros_(last.bias)


class DifferentiablePressureSolver(nn.Module):
    def __init__(self, config: dict):
        super().__init__()
        self.config = config

    def forward(
        self,
        data,
        conductance_corrected: torch.Tensor,
        delta_e: torch.Tensor | None = None,
        reference_pressure: torch.Tensor | None = None,
    ) -> dict[str, object]:
        output_device = conductance_corrected.device
        physics_cfg = self.config.get("physics", {})
        solver_device_name = str(physics_cfg.get("solver_device", "same")).lower()
        if solver_device_name == "same":
            device = output_device
        else:
            device = torch.device(solver_device_name)
        conductance_solver = conductance_corrected.to(device=device, dtype=torch.float64)
        solved = solve_reduced_pressure(
            data=data,
            conductance=conductance_solver,
            arterial_flow_mode=str(physics_cfg.get("arterial_flow_mode", "dataset")),
            pressure_solver_mode=str(
                physics_cfg.get("pressure_solver_mode", "reduced-soft-constrained-lstsq")
            ),
            pressure_constraints=selected_pressure_constraints(self.config),
            alpha_pa=physics_cfg.get("alpha_pa"),
            lambda_kirchhoff=float(physics_cfg.get("pressure_solver_lambda_kirchhoff", 1.0)),
            lambda_pressure_constraints=float(
                physics_cfg.get("pressure_solver_lambda_pressure_constraints", 0.0)
            ),
            lambda_flow_residual=float(
                physics_cfg.get("pressure_solver_lambda_flow_residual", 0.0)
            ),
            device=device,
        )

        pressure = solved["pressure_pa"].to(device=device, dtype=torch.float64)
        edge_pressure_drop = solved["edge_pressure_drop_pa"].to(device=device, dtype=torch.float64)
        if bool(physics_cfg.get("pressure_detach", False)):
            pressure = pressure.detach()
            edge_pressure_drop = edge_pressure_drop.detach()

        q_pred = conductance_corrected.to(device=device, dtype=torch.float64) * edge_pressure_drop
        nodal_residual = solved["nodal_residual_m3_s"].to(device=device, dtype=torch.float64)
        q_obs = observed_flow_m3_s(data).to(device=device, dtype=torch.float64)
        valid_edges = valid_observed_edge_mask(data).to(device=device)
        q_scale = flow_scale_value(
            q_obs=q_obs,
            mask=valid_edges,
            mode=str(physics_cfg.get("flow_scale_mode", "median_abs")),
        ).to(dtype=torch.float64).clamp_min(1.0e-30)
        flow_residual = q_pred[valid_edges] - q_obs[valid_edges]
        if bool(physics_cfg.get("use_snr_weights", True)):
            flow_row_weights = data.dc_loss_weight.to(device=device, dtype=torch.float64)[valid_edges] ** 2
        else:
            flow_row_weights = torch.ones(int(valid_edges.sum()), dtype=torch.float64, device=device)
        weighted_flow_num = torch.sum(flow_row_weights * flow_residual**2)
        weighted_flow_denom = torch.sum(flow_row_weights * q_obs[valid_edges] ** 2).clamp_min(1.0e-30)
        kirchhoff_rows = torch.nonzero(
            torch.arange(len(data.node_id), device=device) != int(solved["gauge_node_index"]),
            as_tuple=False,
        ).flatten()
        kirchhoff_relative_denom = torch.sum(
            solved["source_vector_m3_s"].to(device=device, dtype=torch.float64)[kirchhoff_rows] ** 2
        ).clamp_min(1.0e-30)
        pressure_shape_value = pressure_shape_loss(
            pressure.to(dtype=torch.float32),
            None if reference_pressure is None else reference_pressure.to(dtype=torch.float32, device=device),
        ).to(dtype=torch.float64)
        constraint_l2 = solved["diagnostics"]["pressure_solver_constraint_residual_l2"].to(
            device=device,
            dtype=torch.float64,
        )
        constraint_count = solved["diagnostics"]["reduced_constraint_count"].to(
            device=device,
            dtype=torch.float64,
        ).clamp_min(1.0)

        raw_losses = {
            "L_flow_raw": torch.mean(flow_residual**2)
            if flow_residual.numel()
            else pressure.new_tensor(0.0),
            "L_flow_relative": weighted_flow_num / weighted_flow_denom
            if flow_residual.numel()
            else pressure.new_tensor(0.0),
            "L_kirchhoff_raw": (
                solved["diagnostics"]["pressure_solver_kirchhoff_residual_l2"].to(
                    device=device,
                    dtype=torch.float64,
                )
                ** 2
            )
            / kirchhoff_rows.numel(),
            "L_kirchhoff_relative": (
                solved["diagnostics"]["pressure_solver_kirchhoff_residual_l2"].to(
                    device=device,
                    dtype=torch.float64,
                )
                ** 2
            )
            / kirchhoff_relative_denom,
            "L_boundary_raw": (constraint_l2**2) / constraint_count,
            "L_boundary_relative": (constraint_l2**2) / constraint_count,
            "L_pressure_shape": pressure_shape_value,
        }
        diagnostics = dict(solved["diagnostics"])
        diagnostics["flow_scale_m3_s"] = q_scale
        diagnostics["formulation_warning"] = solved["formulation_warning"]
        diagnostics["gauge_node_id"] = solved["gauge_node_id"]
        diagnostics["pressure_prescribed_node_ids"] = solved["pressure_prescribed_node_ids"]
        return {
            "pressure_pa": pressure.to(device=output_device, dtype=torch.float32),
            "q_pred_m3_s": q_pred.to(device=output_device, dtype=torch.float32),
            "edge_pressure_drop_pa": edge_pressure_drop.to(device=output_device, dtype=torch.float32),
            "nodal_residual_m3_s": nodal_residual.to(device=output_device, dtype=torch.float32),
            "raw_losses": {
                key: value.to(device=output_device, dtype=torch.float32)
                for key, value in raw_losses.items()
            },
            "diagnostics": {
                key: (
                    value.to(device=output_device, dtype=torch.float32)
                    if torch.is_tensor(value)
                    else value
                )
                for key, value in diagnostics.items()
            },
        }


def reference_pressure_for_solver(data, config: dict):
    mode = str(config["physics"].get("pressure_shape_reference", "delta_zero_reference"))
    if mode == "none":
        return None
    if mode == "reference":
        return data.reference_pressure_pa
    if mode == "delta_zero_reference":
        return data.delta_zero_reference_pressure_pa
    raise ValueError(f"Unsupported pressure_shape_reference: {mode}")


def collect_global_metrics(outputs, data, config: dict) -> dict[str, torch.Tensor]:
    delta = outputs["delta_e"]
    raw_delta = outputs["raw_delta_e"]
    conductance_corrected = outputs["Gcorr_e"]
    conductance_ratio = outputs["Gcorr_over_G0"]
    pressure = outputs["pressure_pa"]
    q_pred = outputs["q_pred_m3_s"]
    q_obs = observed_flow_m3_s(data).to(device=q_pred.device)
    valid_edges = valid_observed_edge_mask(data).to(device=q_pred.device)
    eps_abs = (
        outputs["raw_losses"]["flow_scale_m3_s"]
        if "flow_scale_m3_s" in outputs["raw_losses"]
        else outputs["solver_diagnostics"]["flow_scale_m3_s"]
    )
    eps_abs = eps_abs * float(config["physics"].get("sign_eps_relative", 1.0e-6))
    eligible = valid_edges & (torch.abs(q_obs) > eps_abs) & (torch.abs(q_pred) > eps_abs)
    sign_flip_fraction = (
        torch.mean((torch.sign(q_obs[eligible]) != torch.sign(q_pred[eligible])).to(q_pred.dtype))
        if bool(torch.any(eligible))
        else q_pred.new_tensor(float("nan"))
    )
    delta_min = float(config["model"].get("correction_min", -2.0))
    delta_max = float(config["model"].get("correction_max", 2.0))
    sat_tol = float(config["physics"].get("delta_saturation_atol", 5.0e-3))
    metrics = {
        "L_delta_l2": torch.mean(delta**2),
        "L_delta_abs_mean": torch.mean(torch.abs(delta)),
        "L_delta_smooth": (
            torch.mean(
                (delta[data.edge_neighbor_index[0]] - delta[data.edge_neighbor_index[1]]) ** 2
            )
            if data.edge_neighbor_index.numel()
            else delta.new_tensor(0.0)
        ),
        "pressure_range_pa": torch.max(pressure) - torch.min(pressure),
        "pressure_min_pa": torch.min(pressure),
        "pressure_max_pa": torch.max(pressure),
        "sign_flip_fraction": sign_flip_fraction,
        "delta_min": torch.min(delta),
        "delta_max": torch.max(delta),
        "delta_mean": torch.mean(delta),
        "delta_std": torch.std(delta, unbiased=False),
        "delta_saturation_fraction_min": torch.mean((delta <= delta_min + sat_tol).to(delta.dtype)),
        "delta_saturation_fraction_max": torch.mean((delta >= delta_max - sat_tol).to(delta.dtype)),
        "Gcorr_min": torch.min(conductance_corrected),
        "Gcorr_max": torch.max(conductance_corrected),
        "Gcorr_over_G0_min": torch.min(conductance_ratio),
        "Gcorr_over_G0_max": torch.max(conductance_ratio),
        "Gcorr_dynamic_range": torch.max(conductance_corrected)
        / torch.min(conductance_corrected).clamp_min(1.0e-30),
        "raw_delta_min": torch.min(raw_delta),
        "raw_delta_max": torch.max(raw_delta),
    }
    metrics.update(outputs["raw_losses"])
    metrics["pressure_solver_relative_residual"] = outputs["solver_diagnostics"][
        "pressure_solver_relative_residual"
    ]
    metrics["L_pressure_shape"] = pressure_shape_loss(
        pressure,
        reference_pressure_for_solver(data, config),
    )
    return metrics


def outer_loss_and_terms(outputs, data, config: dict, edge_mask: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    q_pred = outputs["q_pred_m3_s"]
    q_obs = observed_flow_m3_s(data).to(device=q_pred.device)
    weights = data.dc_loss_weight.to(device=q_pred.device) ** 2
    active = edge_mask.to(device=q_pred.device) & torch.isfinite(q_obs)
    if bool(torch.any(active)):
        residual = q_pred[active] - q_obs[active]
        numerator = torch.sum(weights[active] * residual**2)
        denominator = torch.sum(weights[active] * q_obs[active] ** 2).clamp_min(1.0e-30)
        flow_term = numerator / denominator
    else:
        flow_term = q_pred.new_tensor(0.0)

    terms = {
        "flow": flow_term,
        "kirchhoff": outputs["raw_losses"]["L_kirchhoff_relative"],
        "boundary": outputs["raw_losses"]["L_boundary_relative"],
        "delta_l2": outputs["global_metrics"]["L_delta_l2"],
        "delta_smooth": outputs["global_metrics"]["L_delta_smooth"],
        "pressure_shape": outputs["global_metrics"]["L_pressure_shape"],
    }
    weights_cfg = config["gnn_outer_losses"]
    total = q_pred.new_tensor(0.0)
    for key, value in terms.items():
        total = total + float(weights_cfg.get(key, 0.0)) * value
    return total, terms


def tensor_dict_to_float(payload: dict[str, object]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in payload.items():
        if torch.is_tensor(value):
            if value.numel() == 1:
                result[key] = float(value.detach().cpu())
            else:
                result[key] = value.detach().cpu().tolist()
        elif isinstance(value, dict):
            result[key] = tensor_dict_to_float(value)
        else:
            result[key] = value
    return result


def flatten_dict(payload: dict[str, object], prefix: str = "") -> dict[str, object]:
    flat: dict[str, object] = {}
    for key, value in payload.items():
        name = f"{prefix}.{key}" if prefix else str(key)
        if isinstance(value, dict):
            flat.update(flatten_dict(value, prefix=name))
        else:
            flat[name] = value
    return flat


def resolved_config_snapshot(config: dict) -> dict[str, object]:
    model_cfg = config["model"]
    training_cfg = config["training"]
    physics_cfg = config["physics"]
    return {
        "preset": "solver_QKB_outer_QKBdelta",
        "gnn_outer_losses": copy.deepcopy(config["gnn_outer_losses"]),
        "physics": {
            "arterial_flow_mode": physics_cfg.get("arterial_flow_mode"),
            "pressure_constraints": list(physics_cfg.get("pressure_constraints", [])),
            "alpha_pa": physics_cfg.get("alpha_pa"),
            "use_snr_weights": physics_cfg.get("use_snr_weights"),
            "flow_scale_mode": physics_cfg.get("flow_scale_mode"),
            "solver_device": physics_cfg.get("solver_device"),
            "pressure_solver_mode": physics_cfg.get("pressure_solver_mode"),
            "pressure_solver_lambda_kirchhoff": physics_cfg.get(
                "pressure_solver_lambda_kirchhoff"
            ),
            "pressure_solver_lambda_pressure_constraints": physics_cfg.get(
                "pressure_solver_lambda_pressure_constraints"
            ),
            "pressure_solver_lambda_flow_residual": physics_cfg.get(
                "pressure_solver_lambda_flow_residual"
            ),
            "pressure_shape_reference": physics_cfg.get("pressure_shape_reference"),
            "pressure_detach": bool(physics_cfg.get("pressure_detach", False)),
        },
        "correction_bounds": {
            "correction_min": model_cfg.get("correction_min"),
            "correction_max": model_cfg.get("correction_max"),
            "correction_bound": model_cfg.get("correction_bound"),
            "correction_parameterization": model_cfg.get("correction_parameterization"),
        },
        "optimization": {
            "learning_rate": training_cfg.get("learning_rate"),
            "weight_decay": training_cfg.get("weight_decay"),
            "gradient_clip_norm": training_cfg.get("gradient_clip_norm"),
            "optimizer": training_cfg.get("optimizer"),
            "epochs": training_cfg.get("epochs"),
            "patience": training_cfg.get("patience"),
        },
    }


def safe_correlation(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    finite = torch.isfinite(x) & torch.isfinite(y)
    if int(finite.sum()) < 2:
        return x.new_tensor(float("nan"))
    x = x[finite]
    y = y[finite]
    x = x - torch.mean(x)
    y = y - torch.mean(y)
    denom = torch.linalg.vector_norm(x) * torch.linalg.vector_norm(y)
    if float(denom.detach().cpu()) <= 1.0e-30:
        return x.new_tensor(float("nan"))
    return torch.sum(x * y) / denom


def relative_l2_change(current: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
    return torch.linalg.vector_norm(current - reference) / torch.linalg.vector_norm(reference).clamp_min(1.0e-30)


def flow_relative_loss_for_mask(
    q_pred: torch.Tensor,
    q_obs: torch.Tensor,
    weights: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    active = mask & torch.isfinite(q_obs) & torch.isfinite(q_pred) & torch.isfinite(weights)
    if not bool(torch.any(active)):
        return q_pred.new_tensor(0.0)
    residual = q_pred[active] - q_obs[active]
    numerator = torch.sum(weights[active] * residual**2)
    denominator = torch.sum(weights[active] * q_obs[active] ** 2).clamp_min(1.0e-30)
    return numerator / denominator


def sign_flip_fraction_for_mask(
    q_pred: torch.Tensor,
    q_obs: torch.Tensor,
    mask: torch.Tensor,
    eps_abs: torch.Tensor,
) -> torch.Tensor:
    eligible = mask & (torch.abs(q_obs) > eps_abs) & (torch.abs(q_pred) > eps_abs)
    if not bool(torch.any(eligible)):
        return q_pred.new_tensor(float("nan"))
    return torch.mean((torch.sign(q_obs[eligible]) != torch.sign(q_pred[eligible])).to(q_pred.dtype))


def build_ablation_metrics(
    name: str,
    q_pred: torch.Tensor,
    q_obs: torch.Tensor,
    weights: torch.Tensor,
    valid_edges: torch.Tensor,
    eps_abs: torch.Tensor,
) -> dict[str, torch.Tensor]:
    return {
        f"{name}_flow_relative_loss": flow_relative_loss_for_mask(
            q_pred=q_pred,
            q_obs=q_obs,
            weights=weights,
            mask=valid_edges,
        ),
        f"{name}_qcorr": safe_correlation(q_pred[valid_edges], q_obs[valid_edges]),
        f"{name}_sign_flip_fraction": sign_flip_fraction_for_mask(
            q_pred=q_pred,
            q_obs=q_obs,
            mask=valid_edges,
            eps_abs=eps_abs,
        ),
    }


def zero_delta_outputs(data, solver: DifferentiablePressureSolver, config: dict, device: torch.device) -> dict[str, object]:
    zero_delta = data.base_conductance.new_zeros(data.n_edges, device=device)
    conductance_zero, conductance_ratio_zero = conductance_from_delta(
        data.base_conductance.to(device=device),
        zero_delta,
    )
    solver_outputs = solver(
        data=data,
        conductance_corrected=conductance_zero,
        delta_e=zero_delta,
        reference_pressure=reference_pressure_for_solver(data, config),
    )
    return {
        "raw_delta_e": zero_delta,
        "delta_e": zero_delta,
        "delta_dc": zero_delta,
        "Gcorr_e": conductance_zero,
        "Gcorr_over_G0": conductance_ratio_zero,
        "pressure_pa": solver_outputs["pressure_pa"],
        "q_pred_m3_s": solver_outputs["q_pred_m3_s"],
        "edge_pressure_drop_pa": solver_outputs["edge_pressure_drop_pa"],
        "nodal_residual_m3_s": solver_outputs["nodal_residual_m3_s"],
        "raw_losses": solver_outputs["raw_losses"],
        "solver_diagnostics": solver_outputs["diagnostics"],
    }


def exploration_diagnostics_row(
    epoch: int,
    outputs: dict[str, object],
    baseline_outputs: dict[str, object],
    data,
    config: dict,
) -> dict[str, object]:
    device = outputs["delta_e"].device
    q_obs = observed_flow_m3_s(data).to(device=device)
    weights = data.dc_loss_weight.to(device=device) ** 2
    valid_edges = valid_observed_edge_mask(data).to(device=device)
    eps_abs = (
        outputs["solver_diagnostics"]["flow_scale_m3_s"]
        * float(config["physics"].get("sign_eps_relative", 1.0e-6))
    )
    delta = outputs["delta_e"]
    conductance_ratio = outputs["Gcorr_over_G0"]
    pressure = outputs["pressure_pa"]
    q_pred = outputs["q_pred_m3_s"]
    edge_pressure_drop = outputs["edge_pressure_drop_pa"]
    base_pressure = baseline_outputs["pressure_pa"]
    base_flow = baseline_outputs["q_pred_m3_s"]
    base_conductance = data.base_conductance.to(device=device)
    base_drop = baseline_outputs["edge_pressure_drop_pa"]
    sat_tol = float(config["physics"].get("delta_saturation_atol", 5.0e-3))
    delta_min_bound = float(config["model"].get("correction_min", -2.0))
    delta_max_bound = float(config["model"].get("correction_max", 2.0))

    current_with_zero_pressure = outputs["Gcorr_e"] * base_drop
    zero_with_current_pressure = base_conductance * edge_pressure_drop
    zero_with_zero_pressure = base_flow

    ref_pressure = getattr(data, "delta_zero_reference_pressure_pa", None)
    if ref_pressure is not None:
        pressure_reference_corr = safe_correlation(
            pressure,
            ref_pressure.to(device=device, dtype=pressure.dtype),
        )
    else:
        pressure_reference_corr = pressure.new_tensor(float("nan"))

    flow_corr = safe_correlation(q_pred[valid_edges], q_obs[valid_edges])
    pressure_change_norm = relative_l2_change(pressure, base_pressure)
    flow_change_norm = relative_l2_change(q_pred, base_flow)
    conductance_change_norm = relative_l2_change(outputs["Gcorr_e"], base_conductance)
    normalized_flow_change = flow_change_norm
    normalized_conductance_change = conductance_change_norm
    flow_over_conductance_ratio = normalized_flow_change / normalized_conductance_change.clamp_min(1.0e-30)
    conductance_pressure_drop_corr = safe_correlation(
        conductance_ratio,
        edge_pressure_drop - base_drop,
    )

    # These diagnostics separate "what the GNN changed" from "what the pressure solve
    # compensated away" by comparing the learned state against the delta=0 baseline.
    # The four ablations below let us ask whether fit changes are driven by
    # conductance updates, pressure updates, or cancellation between the two.
    row: dict[str, object] = {
        "epoch": epoch,
        "pressure_solver_mode": outputs["solver_diagnostics"].get("pressure_solver_mode"),
        "pressure_solver_lambda_kirchhoff": float(
            outputs["solver_diagnostics"]["pressure_solver_lambda_kirchhoff"].detach().cpu()
        ),
        "pressure_solver_lambda_pressure_constraints": float(
            outputs["solver_diagnostics"]["pressure_solver_lambda_pressure_constraints"].detach().cpu()
        ),
        "pressure_solver_lambda_flow_residual": float(
            outputs["solver_diagnostics"]["pressure_solver_lambda_flow_residual"].detach().cpu()
        ),
        "pressure_solver_kirchhoff_residual_l2": float(
            outputs["solver_diagnostics"]["pressure_solver_kirchhoff_residual_l2"].detach().cpu()
        ),
        "pressure_solver_flow_residual_l2": float(
            outputs["solver_diagnostics"]["pressure_solver_flow_residual_l2"].detach().cpu()
        ),
        "pressure_solver_flow_residual_rmse": float(
            outputs["solver_diagnostics"]["pressure_solver_flow_residual_rmse"].detach().cpu()
        ),
        "pressure_solver_constraint_residual_l2": float(
            outputs["solver_diagnostics"]["pressure_solver_constraint_residual_l2"].detach().cpu()
        ),
        "pressure_solver_constraint_residual_max": float(
            outputs["solver_diagnostics"]["pressure_solver_constraint_residual_max"].detach().cpu()
        ),
        "pressure_solver_pressure_range_pa": float(
            outputs["solver_diagnostics"]["pressure_solver_pressure_range_pa"].detach().cpu()
        ),
        "pressure_solver_flow_row_scale": float(
            outputs["solver_diagnostics"]["pressure_solver_flow_row_scale"].detach().cpu()
        ),
        "pressure_solver_laplacian_scale": float(
            outputs["solver_diagnostics"]["pressure_solver_laplacian_scale"].detach().cpu()
        ),
        "pressure_solver_used_lstsq": bool(
            float(outputs["solver_diagnostics"]["pressure_solver_used_lstsq"].detach().cpu())
        ),
        "delta_mean": float(torch.mean(delta).detach().cpu()),
        "delta_std": float(torch.std(delta, unbiased=False).detach().cpu()),
        "delta_min": float(torch.min(delta).detach().cpu()),
        "delta_max": float(torch.max(delta).detach().cpu()),
        "delta_saturation_fraction": float(
            torch.mean(
                ((delta <= delta_min_bound + sat_tol) | (delta >= delta_max_bound - sat_tol)).to(delta.dtype)
            ).detach().cpu()
        ),
        "conductance_ratio_mean": float(torch.mean(conductance_ratio).detach().cpu()),
        "conductance_ratio_std": float(torch.std(conductance_ratio, unbiased=False).detach().cpu()),
        "conductance_ratio_min": float(torch.min(conductance_ratio).detach().cpu()),
        "conductance_ratio_max": float(torch.max(conductance_ratio).detach().cpu()),
        "conductance_ratio_dynamic_range": float(
            (torch.max(conductance_ratio) / torch.min(conductance_ratio).clamp_min(1.0e-30)).detach().cpu()
        ),
        "pressure_min_pa": float(torch.min(pressure).detach().cpu()),
        "pressure_max_pa": float(torch.max(pressure).detach().cpu()),
        "pressure_range_pa": float((torch.max(pressure) - torch.min(pressure)).detach().cpu()),
        "pressure_vs_delta_zero_reference_corr": float(pressure_reference_corr.detach().cpu()),
        "flow_relative_loss": float(
            flow_relative_loss_for_mask(q_pred, q_obs, weights, valid_edges).detach().cpu()
        ),
        "sign_flip_fraction": float(
            sign_flip_fraction_for_mask(q_pred, q_obs, valid_edges, eps_abs).detach().cpu()
        ),
        "q_pred_vs_q_obs_corr": float(flow_corr.detach().cpu()),
        "pressure_change_norm_from_delta_zero": float(pressure_change_norm.detach().cpu()),
        "flow_change_norm_from_delta_zero": float(flow_change_norm.detach().cpu()),
        "conductance_change_norm_from_G0": float(conductance_change_norm.detach().cpu()),
        "normalized_flow_change": float(normalized_flow_change.detach().cpu()),
        "normalized_conductance_change": float(normalized_conductance_change.detach().cpu()),
        "flow_change_over_conductance_change": float(flow_over_conductance_ratio.detach().cpu()),
        "conductance_pressure_drop_change_corr": float(conductance_pressure_drop_corr.detach().cpu()),
    }
    row.update(
        tensor_dict_to_float(
            build_ablation_metrics(
                "current_delta_solved_pressure",
                q_pred,
                q_obs,
                weights,
                valid_edges,
                eps_abs,
            )
        )
    )
    row.update(
        tensor_dict_to_float(
            build_ablation_metrics(
                "zero_delta_solved_pressure",
                zero_with_current_pressure,
                q_obs,
                weights,
                valid_edges,
                eps_abs,
            )
        )
    )
    row.update(
        tensor_dict_to_float(
            build_ablation_metrics(
                "current_delta_delta_zero_pressure",
                current_with_zero_pressure,
                q_obs,
                weights,
                valid_edges,
                eps_abs,
            )
        )
    )
    row.update(
        tensor_dict_to_float(
            build_ablation_metrics(
                "zero_delta_delta_zero_pressure",
                zero_with_zero_pressure,
                q_obs,
                weights,
                valid_edges,
                eps_abs,
            )
        )
    )
    return row


def assert_finite(name: str, value: torch.Tensor) -> None:
    if not torch.isfinite(value).all():
        raise FloatingPointError(f"{name} contains NaN or Inf values.")


def gradients_are_finite(model: nn.Module) -> tuple[bool, list[str]]:
    bad: list[str] = []
    for name, parameter in model.named_parameters():
        if parameter.grad is None:
            continue
        if not torch.isfinite(parameter.grad).all():
            bad.append(name)
    return len(bad) == 0, bad


def parameters_are_finite(model: nn.Module) -> tuple[bool, list[str]]:
    bad: list[str] = []
    for name, parameter in model.named_parameters():
        if not torch.isfinite(parameter).all():
            bad.append(name)
    return len(bad) == 0, bad


def forward_model(model: nn.Module, solver: DifferentiablePressureSolver, data, config: dict) -> dict[str, object]:
    model_outputs = model(data)
    conductance_corrected, conductance_ratio = conductance_from_delta(
        data.base_conductance.to(device=model_outputs["delta_e"].device),
        model_outputs["delta_e"],
    )
    solver_outputs = solver(
        data=data,
        conductance_corrected=conductance_corrected,
        delta_e=model_outputs["delta_e"],
        reference_pressure=reference_pressure_for_solver(data, config),
    )
    source, target = data.edge_index
    velocity_dc = solver_outputs["q_pred_m3_s"] / data.area_m2.to(device=conductance_corrected.device).clamp_min(1.0e-30)
    velocity = data.velocity_observed_m_s.to(device=conductance_corrected.device).new_zeros(
        (data.n_edges, data.n_channels, 2)
    )
    velocity[:, 0, 0] = velocity_dc
    outputs = {
        **model_outputs,
        "Gcorr_e": conductance_corrected,
        "Gcorr_over_G0": conductance_ratio,
        "pressure_pa": solver_outputs["pressure_pa"],
        "q_pred_m3_s": solver_outputs["q_pred_m3_s"],
        "flow_m3_s": solver_outputs["q_pred_m3_s"],
        "edge_pressure_drop_pa": solver_outputs["edge_pressure_drop_pa"],
        "nodal_residual_m3_s": solver_outputs["nodal_residual_m3_s"],
        "velocity_m_s": velocity,
        "raw_losses": solver_outputs["raw_losses"],
        "solver_diagnostics": solver_outputs["diagnostics"],
    }
    outputs["raw_losses"]["flow_scale_m3_s"] = outputs["solver_diagnostics"]["flow_scale_m3_s"]
    outputs["global_metrics"] = collect_global_metrics(outputs, data, config)
    assert_finite("delta_e", outputs["delta_e"])
    assert_finite("Gcorr_e", outputs["Gcorr_e"])
    assert_finite("pressure_pa", outputs["pressure_pa"])
    assert_finite("q_pred_m3_s", outputs["q_pred_m3_s"])
    return outputs


def train_model(model: nn.Module, solver: DifferentiablePressureSolver, data, config: dict):
    device = next(model.parameters()).device
    data = data.to(device)
    optimizer_name = str(config["training"].get("optimizer", "adamw")).lower()
    optimizer_cls = torch.optim.AdamW if optimizer_name == "adamw" else torch.optim.Adam
    optimizer = optimizer_cls(
        model.parameters(),
        lr=float(config["training"]["learning_rate"]),
        weight_decay=float(config["training"].get("weight_decay", 0.0)),
    )
    max_epochs = int(config["training"].get("epochs", 250))
    patience = int(config["training"].get("patience", 40))
    grad_clip = float(config["training"].get("gradient_clip_norm", 5.0))

    history: list[dict[str, object]] = []
    exploration_history: list[dict[str, object]] = []
    best_state = copy.deepcopy(model.state_dict())
    best_val = float("inf")
    best_epoch = 0
    wait = 0
    model.eval()
    with torch.no_grad():
        baseline_outputs = zero_delta_outputs(data, solver, config, device)

    with tqdm(total=max_epochs, desc="GNN flow training", unit="epoch", dynamic_ncols=True) as progress:
        for epoch in range(1, max_epochs + 1):
            model.train()
            optimizer.zero_grad()
            train_outputs = forward_model(model, solver, data, config)
            train_loss, train_terms = outer_loss_and_terms(train_outputs, data, config, data.train_mask)
            train_loss.backward()
            finite_grads, bad_grads = gradients_are_finite(model)
            if not finite_grads:
                raise FloatingPointError(
                    "Non-finite gradients detected before optimizer step: "
                    + ", ".join(bad_grads[:10])
                )
            pre_step_state = copy.deepcopy(model.state_dict())
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
            finite_params, bad_params = parameters_are_finite(model)
            if not finite_params:
                model.load_state_dict(pre_step_state)
                raise FloatingPointError(
                    "Non-finite model parameters detected after optimizer step: "
                    + ", ".join(bad_params[:10])
                    + ". This often indicates a numerically unstable device path; "
                    "the default config now uses physics.solver_device=cpu to avoid "
                    "CUDA lstsq instabilities."
                )

            model.eval()
            with torch.no_grad():
                eval_outputs = forward_model(model, solver, data, config)
                val_loss, val_terms = outer_loss_and_terms(eval_outputs, data, config, data.val_mask)
                test_loss, test_terms = outer_loss_and_terms(eval_outputs, data, config, data.test_mask)

            row = {
                "epoch": epoch,
                "train_total": float(train_loss.detach().cpu()),
                "val_total": float(val_loss.detach().cpu()),
                "test_total": float(test_loss.detach().cpu()),
            }
            for prefix, terms in (
                ("train", train_terms),
                ("val", val_terms),
                ("test", test_terms),
            ):
                for key, value in terms.items():
                    row[f"{prefix}_{key}"] = float(value.detach().cpu())
            row["pressure_solver_relative_residual"] = float(
                eval_outputs["global_metrics"]["pressure_solver_relative_residual"].detach().cpu()
            )
            row["sign_flip_fraction"] = float(
                eval_outputs["global_metrics"]["sign_flip_fraction"].detach().cpu()
            )
            for key in (
                "pressure_solver_mode",
                "pressure_solver_lambda_kirchhoff",
                "pressure_solver_lambda_pressure_constraints",
                "pressure_solver_lambda_flow_residual",
                "pressure_solver_kirchhoff_residual_l2",
                "pressure_solver_flow_residual_l2",
                "pressure_solver_flow_residual_rmse",
                "pressure_solver_constraint_residual_l2",
                "pressure_solver_constraint_residual_max",
                "pressure_solver_pressure_range_pa",
                "pressure_solver_flow_row_scale",
                "pressure_solver_laplacian_scale",
                "pressure_solver_used_lstsq",
            ):
                value = eval_outputs["solver_diagnostics"].get(key)
                if value is not None:
                    row[key] = (
                        float(value.detach().cpu()) if torch.is_tensor(value) else value
                    )
            exploration_row = exploration_diagnostics_row(
                epoch=epoch,
                outputs=eval_outputs,
                baseline_outputs=baseline_outputs,
                data=data,
                config=config,
            )
            row.update(exploration_row)
            history.append(row)
            exploration_history.append(exploration_row)

            current_val = float(val_loss.detach().cpu())
            progress.set_postfix(
                train=f"{row['train_total']:.3e}",
                val=f"{row['val_total']:.3e}",
                best=f"{best_val if math.isfinite(best_val) else current_val:.3e}",
                sign=f"{row['sign_flip_fraction']:.3f}",
            )
            progress.update(1)

            if current_val < best_val:
                best_val = current_val
                best_epoch = epoch
                best_state = copy.deepcopy(model.state_dict())
                wait = 0
            else:
                wait += 1
                if wait >= patience:
                    progress.write(
                        f"Early stopping at epoch {epoch} with best validation loss at epoch {best_epoch}."
                    )
                    break

    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        final_outputs = forward_model(model, solver, data, config)
        train_loss, train_terms = outer_loss_and_terms(final_outputs, data, config, data.train_mask)
        val_loss, val_terms = outer_loss_and_terms(final_outputs, data, config, data.val_mask)
        test_loss, test_terms = outer_loss_and_terms(final_outputs, data, config, data.test_mask)

    summary = {
        "best_epoch": best_epoch,
        "best_val_total": best_val,
        "train_total": float(train_loss.detach().cpu()),
        "val_total": float(val_loss.detach().cpu()),
        "test_total": float(test_loss.detach().cpu()),
    }
    for prefix, terms in (("train", train_terms), ("val", val_terms), ("test", test_terms)):
        for key, value in terms.items():
            summary[f"{prefix}_{key}"] = float(value.detach().cpu())
    final_exploration = exploration_diagnostics_row(
        epoch=best_epoch,
        outputs=final_outputs,
        baseline_outputs=baseline_outputs,
        data=data,
        config=config,
    )
    summary.update({f"exploration_{key}": value for key, value in final_exploration.items() if key != "epoch"})
    return model, final_outputs, history, exploration_history, summary


def pressure_sanity_checks(model: nn.Module, solver: DifferentiablePressureSolver, data, config: dict) -> dict[str, object]:
    device = next(model.parameters()).device
    data = data.to(device)
    zero_delta = data.base_conductance.new_zeros(data.n_edges, device=device)
    base_conductance = data.base_conductance.to(device=device)
    zero_solver = solver(
        data=data,
        conductance_corrected=base_conductance,
        delta_e=zero_delta,
        reference_pressure=reference_pressure_for_solver(data, config),
    )
    p = zero_solver["pressure_pa"]
    src, dst = data.edge_index
    recomputed = base_conductance * (p[src] - p[dst])
    partitioned_cfg = copy.deepcopy(config)
    partitioned_cfg["physics"]["pressure_solver_mode"] = "partitioned-flow-gauge"
    partitioned_cfg["physics"]["pressure_solver_lambda_kirchhoff"] = 1.0
    partitioned_cfg["physics"]["pressure_solver_lambda_pressure_constraints"] = 0.0
    partitioned_cfg["physics"]["pressure_solver_lambda_flow_residual"] = 0.0
    partitioned_solver = DifferentiablePressureSolver(partitioned_cfg).to(device)
    partitioned_outputs = partitioned_solver(
        data=data,
        conductance_corrected=base_conductance,
        delta_e=zero_delta,
        reference_pressure=reference_pressure_for_solver(data, partitioned_cfg),
    )
    equivalence_cfg = copy.deepcopy(config)
    equivalence_cfg["physics"]["pressure_solver_mode"] = "reduced-soft-constrained-lstsq"
    equivalence_cfg["physics"]["pressure_solver_lambda_kirchhoff"] = 1.0
    equivalence_cfg["physics"]["pressure_solver_lambda_pressure_constraints"] = 0.0
    equivalence_cfg["physics"]["pressure_solver_lambda_flow_residual"] = 0.0
    equivalence_solver = DifferentiablePressureSolver(equivalence_cfg).to(device)
    equivalence_outputs = equivalence_solver(
        data=data,
        conductance_corrected=base_conductance,
        delta_e=zero_delta,
        reference_pressure=reference_pressure_for_solver(data, equivalence_cfg),
    )
    flow_aware_cfg = copy.deepcopy(equivalence_cfg)
    flow_aware_cfg["physics"]["pressure_solver_lambda_flow_residual"] = float(
        config["physics"].get("pressure_solver_lambda_flow_residual", 0.0)
    )
    flow_aware_solver = DifferentiablePressureSolver(flow_aware_cfg).to(device)
    flow_aware_outputs = flow_aware_solver(
        data=data,
        conductance_corrected=base_conductance,
        delta_e=zero_delta,
        reference_pressure=reference_pressure_for_solver(data, flow_aware_cfg),
    )
    q_obs = observed_flow_m3_s(data).to(device=device)
    valid_edges = valid_observed_edge_mask(data).to(device=device)
    weights = data.dc_loss_weight.to(device=device) ** 2
    return {
        "delta_zero_reproduces_solver_max_abs_flow_diff_m3_s": float(
            torch.max(torch.abs(recomputed - zero_solver["q_pred_m3_s"])).detach().cpu()
        ),
        "reduced_soft_matches_partitioned_pressure_max_abs": float(
            torch.max(
                torch.abs(equivalence_outputs["pressure_pa"] - partitioned_outputs["pressure_pa"])
            ).detach().cpu()
        ),
        "reduced_soft_matches_partitioned_flow_max_abs": float(
            torch.max(
                torch.abs(equivalence_outputs["q_pred_m3_s"] - partitioned_outputs["q_pred_m3_s"])
            ).detach().cpu()
        ),
        "flow_aware_minus_partitioned_flow_rmse": float(
            (
                flow_relative_loss_for_mask(
                    flow_aware_outputs["q_pred_m3_s"],
                    q_obs,
                    weights,
                    valid_edges,
                )
                - flow_relative_loss_for_mask(
                    partitioned_outputs["q_pred_m3_s"],
                    q_obs,
                    weights,
                    valid_edges,
                )
            ).detach().cpu()
        ),
        "flow_aware_minus_partitioned_kirchhoff_residual_l2": float(
            (
                flow_aware_outputs["diagnostics"]["pressure_solver_kirchhoff_residual_l2"]
                - partitioned_outputs["diagnostics"]["pressure_solver_kirchhoff_residual_l2"]
            ).detach().cpu()
        ),
        "pressure_solver_mode": str(config["physics"].get("pressure_solver_mode")),
        "pressure_solver_lambda_kirchhoff": float(
            config["physics"].get("pressure_solver_lambda_kirchhoff", 1.0)
        ),
        "pressure_solver_lambda_pressure_constraints": float(
            config["physics"].get("pressure_solver_lambda_pressure_constraints", 0.0)
        ),
        "pressure_solver_lambda_flow_residual": float(
            config["physics"].get("pressure_solver_lambda_flow_residual", 0.0)
        ),
    }


def edge_rows(outputs, data, config: dict) -> list[dict[str, object]]:
    q_obs = observed_flow_m3_s(data).detach().cpu().numpy()
    q_pred = outputs["q_pred_m3_s"].detach().cpu().numpy()
    pressure = outputs["pressure_pa"].detach().cpu().numpy()
    delta = outputs["delta_e"].detach().cpu().numpy()
    raw_delta = outputs["raw_delta_e"].detach().cpu().numpy()
    base_conductance = data.base_conductance.detach().cpu().numpy()
    conductance = outputs["Gcorr_e"].detach().cpu().numpy()
    conductance_ratio = outputs["Gcorr_over_G0"].detach().cpu().numpy()
    drop = outputs["edge_pressure_drop_pa"].detach().cpu().numpy()
    weights = data.dc_loss_weight.detach().cpu().numpy()
    valid = valid_observed_edge_mask(data).detach().cpu().numpy().astype(bool)
    q_scale = float(outputs["solver_diagnostics"]["flow_scale_m3_s"].detach().cpu())
    eps_abs = q_scale * float(config["physics"].get("sign_eps_relative", 1.0e-6))
    src = data.edge_index[0].detach().cpu().numpy()
    dst = data.edge_index[1].detach().cpu().numpy()
    rows = []
    for edge_idx, (u, v) in enumerate(data.edge_ids):
        sign_flip = (
            valid[edge_idx]
            and abs(q_obs[edge_idx]) > eps_abs
            and abs(q_pred[edge_idx]) > eps_abs
            and np.sign(q_obs[edge_idx]) != np.sign(q_pred[edge_idx])
        )
        rows.append(
            {
                "edge_id": int(edge_idx),
                "source": str(u),
                "target": str(v),
                "source_node": str(u),
                "target_node": str(v),
                "source_index": int(src[edge_idx]),
                "target_index": int(dst[edge_idx]),
                "tile_id": int(data.edge_tile_id[edge_idx]),
                "radius_m": float(data.radius_m[edge_idx].detach().cpu()),
                "length_m": float(data.length_m[edge_idx].detach().cpu()),
                "area_m2": float(data.area_m2[edge_idx].detach().cpu()),
                "q_obs_m3_s": float(q_obs[edge_idx]),
                "q_pred_m3_s": float(q_pred[edge_idx]),
                "observed_flow_nl_s": float(q_obs[edge_idx] * 1.0e12),
                "predicted_flow_nl_s": float(q_pred[edge_idx] * 1.0e12),
                "flow_residual_nl_s": float((q_pred[edge_idx] - q_obs[edge_idx]) * 1.0e12),
                "q_residual_m3_s": float(q_pred[edge_idx] - q_obs[edge_idx]),
                "q_obs_over_scale": float(q_obs[edge_idx] / q_scale) if q_scale > 0.0 else float("nan"),
                "q_pred_over_scale": float(q_pred[edge_idx] / q_scale) if q_scale > 0.0 else float("nan"),
                "pressure_drop_pa": float(drop[edge_idx]),
                "p_source_pa": float(pressure[src[edge_idx]]),
                "p_target_pa": float(pressure[dst[edge_idx]]),
                "G0_m3_pa_s": float(base_conductance[edge_idx]),
                "Gcorr_m3_pa_s": float(conductance[edge_idx]),
                "Gcorr_over_G0": float(conductance_ratio[edge_idx]),
                "raw_delta_e": float(raw_delta[edge_idx]),
                "delta_e": float(delta[edge_idx]),
                "snr_weight": float(weights[edge_idx]),
                "valid_observed_flow": bool(valid[edge_idx]),
                "sign_flip": bool(sign_flip),
            }
        )
    return rows


def node_rows(outputs, data) -> list[dict[str, object]]:
    pressure = outputs["pressure_pa"].detach().cpu().numpy()
    residual = outputs["nodal_residual_m3_s"].detach().cpu().numpy()
    arterial = set(data.arterial_node_indices.detach().cpu().numpy().tolist())
    venous = set(data.venous_node_indices.detach().cpu().numpy().tolist())
    rows = []
    for idx, node_id in enumerate(data.node_id):
        is_arterial = idx in arterial
        is_venous = idx in venous
        is_boundary = is_arterial or is_venous
        is_internal = not is_boundary
        if is_arterial:
            role = "arterial"
        elif is_venous:
            role = "venous"
        else:
            role = "internal"
        coords = getattr(data, "node_xy_px", None)
        if coords is None:
            x_px = float("nan")
            y_px = float("nan")
        else:
            try:
                coord_row = np.asarray(coords[idx], dtype=np.float64).reshape(-1)
            except Exception:
                coord_row = np.asarray([], dtype=np.float64)
            x_px = float(coord_row[0]) if coord_row.size >= 2 else float("nan")
            y_px = float(coord_row[1]) if coord_row.size >= 2 else float("nan")
        rows.append(
            {
                "node_index": int(idx),
                "node_id": str(node_id),
                "node_type": role,
                "boundary_role": role,
                "pressure_pa": float(pressure[idx]),
                "boundary_injection_nl_s": float(
                    data.boundary_injection_m3_s[idx].detach().cpu() * 1.0e12
                ),
                "predicted_net_flow_nl_s": float(
                    (residual[idx] + float(data.boundary_injection_m3_s[idx].detach().cpu())) * 1.0e12
                ),
                "kirchhoff_residual_nl_s": float(residual[idx] * 1.0e12),
                "kirchhoff_residual_m3_s": float(residual[idx]),
                "boundary_injection_m3_s": float(data.boundary_injection_m3_s[idx].detach().cpu()),
                "x_px": x_px,
                "y_px": y_px,
                "is_arterial": bool(is_arterial),
                "is_venous": bool(is_venous),
                "is_boundary": bool(is_boundary),
                "is_internal": bool(is_internal),
            }
        )
    return rows


def build_summary(
    args: argparse.Namespace,
    config: dict,
    data,
    outputs,
    training_summary: dict[str, object],
    sanity: dict[str, object],
    output_dir: Path,
    config_snapshot: dict[str, object],
) -> dict[str, object]:
    summary = {
        "script_name": Path(__file__).name,
        "graph_path": str(args.graph.expanduser().resolve()),
        "output_dir": str(output_dir),
        "run_name": args.run_name,
        "preset": args.preset,
        "n_nodes": int(len(data.node_id)),
        "n_edges": int(data.n_edges),
        "n_observed_edges": int(valid_observed_edge_mask(data).sum().detach().cpu()),
        "device": str(args.device),
        "viscosity_pa_s": float(args.viscosity_pa_s),
        "pressure_constraints": "|".join(selected_pressure_constraints(config)),
        "arterial_flow_mode": str(config["physics"].get("arterial_flow_mode")),
        "use_snr_weights": bool(config["physics"].get("use_snr_weights", True)),
        "flow_scale_mode": str(config["physics"].get("flow_scale_mode")),
        "pressure_shape_reference": str(config["physics"].get("pressure_shape_reference")),
        "solver_device": str(config["physics"].get("solver_device")),
        "pressure_detach": bool(config["physics"].get("pressure_detach", False)),
        "lambda_q": float(config["gnn_outer_losses"].get("flow", 0.0)),
        "lambda_k": float(config["gnn_outer_losses"].get("kirchhoff", 0.0)),
        "lambda_b": float(config["gnn_outer_losses"].get("boundary", 0.0)),
        "lambda_delta": float(config["gnn_outer_losses"].get("delta_l2", 0.0)),
        **training_summary,
        **tensor_dict_to_float(outputs["raw_losses"]),
        **tensor_dict_to_float(outputs["global_metrics"]),
        **sanity,
    }
    solver_diag = tensor_dict_to_float(outputs["solver_diagnostics"])
    for key, value in solver_diag.items():
        if key == "constraint_labels":
            summary["constraint_labels"] = "|".join(value)
        else:
            summary[key] = value
    physics_cfg = config["physics"]
    outer_losses = config["gnn_outer_losses"]
    summary["weighted_pressure_solver_objective_proxy"] = (
        float(physics_cfg.get("pressure_solver_lambda_flow_residual", 0.0)) * summary["L_flow_relative"]
        + float(physics_cfg.get("pressure_solver_lambda_kirchhoff", 0.0)) * summary["L_kirchhoff_relative"]
        + float(physics_cfg.get("pressure_solver_lambda_pressure_constraints", 0.0))
        * summary["L_boundary_relative"]
    )
    summary["weighted_outer_objective_proxy"] = (
        float(outer_losses.get("flow", 0.0)) * summary["train_flow"]
        + float(outer_losses.get("kirchhoff", 0.0)) * summary["train_kirchhoff"]
        + float(outer_losses.get("boundary", 0.0)) * summary["train_boundary"]
        + float(outer_losses.get("delta_l2", 0.0)) * summary["train_delta_l2"]
        + float(outer_losses.get("delta_smooth", 0.0)) * summary["train_delta_smooth"]
        + float(outer_losses.get("pressure_shape", 0.0)) * summary["train_pressure_shape"]
    )
    summary.update(
        {
            f"resolved_{key.replace('.', '_')}": value
            for key, value in flatten_dict(config_snapshot).items()
            if not isinstance(value, list)
        }
    )
    return summary


def prepare_config(args: argparse.Namespace) -> dict:
    config = default_config()
    config = deep_update(config, PRESETS[args.preset])
    if args.config is not None:
        config = deep_update(config, read_resolved_config(args.config.expanduser().resolve()))
    if args.epochs is not None:
        config["training"]["epochs"] = int(args.epochs)
    if args.learning_rate is not None:
        config["training"]["learning_rate"] = float(args.learning_rate)
    if args.weight_decay is not None:
        config["training"]["weight_decay"] = float(args.weight_decay)
    if args.seed is not None:
        config["training"]["seed"] = int(args.seed)
    if args.arterial_flow_mode is not None:
        config["physics"]["arterial_flow_mode"] = str(args.arterial_flow_mode)
    if args.pressure_constraint is not None:
        config["physics"]["pressure_constraints"] = list(args.pressure_constraint)
    if args.alpha_pa is not None:
        config["physics"]["alpha_pa"] = float(args.alpha_pa)
    if args.flow_scale_mode is not None:
        config["physics"]["flow_scale_mode"] = str(args.flow_scale_mode)
    if args.pressure_solver_mode is not None:
        config["physics"]["pressure_solver_mode"] = str(args.pressure_solver_mode)
    if args.pressure_solver_lambda_kirchhoff is not None:
        config["physics"]["pressure_solver_lambda_kirchhoff"] = float(
            args.pressure_solver_lambda_kirchhoff
        )
    if args.pressure_solver_lambda_pressure_constraints is not None:
        config["physics"]["pressure_solver_lambda_pressure_constraints"] = float(
            args.pressure_solver_lambda_pressure_constraints
        )
    if args.pressure_solver_lambda_flow_residual is not None:
        config["physics"]["pressure_solver_lambda_flow_residual"] = float(
            args.pressure_solver_lambda_flow_residual
        )
    if args.pressure_detach is not None:
        config["physics"]["pressure_detach"] = bool(args.pressure_detach)
    if args.no_snr_weights:
        config["physics"]["use_snr_weights"] = False
    return config


def output_directory(args: argparse.Namespace) -> Path:
    path = args.output_dir.expanduser().resolve()
    if args.run_name:
        return path / args.run_name
    return path / args.preset


def overwrite_base_conductance_from_viscosity(data, viscosity_pa_s: float) -> None:
    if not math.isfinite(viscosity_pa_s) or viscosity_pa_s <= 0.0:
        raise ValueError("--viscosity-pa-s must be positive and finite.")
    conductance = (
        math.pi
        * data.radius_m.detach().cpu().numpy().astype(np.float64) ** 4
        / (8.0 * viscosity_pa_s * np.maximum(data.length_m.detach().cpu().numpy().astype(np.float64), 1.0e-30))
    )
    data.base_conductance = torch.tensor(conductance, dtype=torch.float32)


def main() -> None:
    start_time = time.perf_counter()
    args = parse_args()
    config = prepare_config(args)
    config_snapshot = resolved_config_snapshot(config)
    config_snapshot["preset"] = args.preset
    set_random_seed(int(config["training"]["seed"]))
    device = resolve_device(args.device)
    graph_path = args.graph.expanduser().resolve()
    data = build_real_gnn_data(graph_path, config)
    overwrite_base_conductance_from_viscosity(data, float(args.viscosity_pa_s))

    model = build_model(data, config).to(device)
    maybe_initialize_decoder_near_zero(
        model,
        bool(config["model"].get("initialize_decoder_near_zero", True)),
    )
    solver = DifferentiablePressureSolver(config).to(device)

    print("Resolved config snapshot:")
    print(config_snapshot)

    model, outputs, history, exploration_history, training_summary = train_model(
        model,
        solver,
        data,
        config,
    )
    sanity = pressure_sanity_checks(model, solver, data, config)
    out_dir = output_directory(args)
    out_dir.mkdir(parents=True, exist_ok=True)

    summary = build_summary(
        args,
        config,
        data,
        outputs,
        training_summary,
        sanity,
        out_dir,
        config_snapshot,
    )
    summary["runtime_seconds"] = time.perf_counter() - start_time
    summary["solver_success"] = True
    write_yaml(out_dir / "config_used.yaml", config)
    write_yaml(out_dir / "resolved_config_snapshot.yaml", config_snapshot)
    write_yaml(out_dir / "summary.yaml", summary)
    write_csv(out_dir / "summary.csv", [summary])
    if not args.summary_only:
        write_csv(out_dir / "edge_predictions.csv", edge_rows(outputs, data, config))
        write_csv(out_dir / "node_predictions.csv", node_rows(outputs, data))
        if bool(config["output"].get("save_history_csv", True)):
            write_csv(out_dir / "training_history.csv", history)
        if bool(config["output"].get("save_exploration_diagnostics_csv", True)):
            write_csv(out_dir / "exploration_diagnostics.csv", exploration_history)
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "config": config,
                "resolved_config_snapshot": config_snapshot,
                "summary": summary,
            },
            out_dir / "model_checkpoint.pt",
        )


if __name__ == "__main__":
    main()
