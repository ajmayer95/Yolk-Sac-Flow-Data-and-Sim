"""Dataset construction, differentiable physics, and generic GNN training."""

from __future__ import annotations

import copy
import math
from dataclasses import dataclass, fields
from pathlib import Path

import numpy as np
import torch

from distensibility.io import load_dataset
from gnn_losses import combined_loss


@dataclass
class GNNData:
    node_id: np.ndarray
    edge_index: torch.Tensor
    node_features: torch.Tensor
    edge_features: torch.Tensor
    radius_m: torch.Tensor
    length_m: torch.Tensor
    area_m2: torch.Tensor
    base_conductance: torch.Tensor
    boundary_injection_m3_s: torch.Tensor
    velocity_observed_m_s: torch.Tensor
    velocity_normalized: torch.Tensor
    velocity_center_m_s: torch.Tensor
    velocity_scale_m_s: torch.Tensor
    train_mask: torch.Tensor
    val_mask: torch.Tensor
    test_mask: torch.Tensor
    reference_node: int
    n_harmonics: int
    n_channels: int
    n_edges: int

    def to(self, device):
        values = {}
        for item in fields(self):
            value = getattr(self, item.name)
            values[item.name] = value.to(device) if torch.is_tensor(value) else value
        return GNNData(**values)


def harmonic_count(mode: str) -> int:
    return {"dc_only": 0, "dc_h1": 1, "dc_h1_h2": 2}[mode]


def _finite_stats(values, train_mask):
    selected = values[train_mask]
    center = np.nanmean(selected, axis=0)
    scale = np.nanstd(selected, axis=0)
    center = np.where(np.isfinite(center), center, 0.0)
    scale = np.where(np.isfinite(scale) & (scale > 1.0e-12), scale, 1.0)
    return center, scale


def build_gnn_data(dataset_path: Path, harmonic_mode: str, split_config: dict):
    dataset = load_dataset(dataset_path)
    n_harmonics = harmonic_count(harmonic_mode)
    n_channels = 1 + n_harmonics
    edge_index = np.stack(
        [dataset.edge_source_index, dataset.edge_target_index], axis=0
    )
    degree = np.zeros(dataset.n_nodes, dtype=float)
    np.add.at(degree, dataset.edge_source_index, 1.0)
    np.add.at(degree, dataset.edge_target_index, 1.0)
    xy = dataset.node_xy_px.astype(float)
    finite_xy = np.isfinite(xy).all(axis=1)
    xy_center = np.nanmean(xy[finite_xy], axis=0)
    xy_scale = max(
        float(np.nanmax(xy[finite_xy], axis=0)[0] - np.nanmin(xy[finite_xy], axis=0)[0]),
        float(np.nanmax(xy[finite_xy], axis=0)[1] - np.nanmin(xy[finite_xy], axis=0)[1]),
        1.0,
    )
    xy_norm = np.where(np.isfinite(xy), (xy - xy_center) / xy_scale, 0.0)

    injection = np.zeros(dataset.n_nodes, dtype=np.complex128)
    with np.load(dataset.path, allow_pickle=False) as archive:
        boundary_flow = np.asarray(archive["boundary_flow_m3_s"])
    for row, node in enumerate(dataset.boundary_node_index):
        injection[int(node)] += boundary_flow[row, 0]
    injection = injection.real
    is_source = np.zeros(dataset.n_nodes, dtype=float)
    is_sink = np.zeros(dataset.n_nodes, dtype=float)
    for node, kind in zip(dataset.boundary_node_index, dataset.boundary_type):
        if str(kind) == "source":
            is_source[int(node)] = 1.0
        elif str(kind) == "sink":
            is_sink[int(node)] = 1.0
    node_features = np.column_stack(
        [
            xy_norm,
            degree,
            injection * 1.0e12,
            is_source,
            is_sink,
        ]
    )
    node_center = np.mean(node_features, axis=0)
    node_scale = np.maximum(np.std(node_features, axis=0), 1.0e-12)
    node_features = (node_features - node_center) / node_scale

    conductance = np.pi * dataset.edge_radius_m**4 / (
        8.0 * dataset.viscosity_pa_s * dataset.edge_length_m
    )
    velocity_conductance = conductance / dataset.edge_area_m2
    edge_features = np.column_stack(
        [
            np.log(np.maximum(dataset.edge_radius_m, 1.0e-30)),
            np.log(np.maximum(dataset.edge_length_m, 1.0e-30)),
            np.log(np.maximum(velocity_conductance, 1.0e-30)),
        ]
    )
    edge_center = np.mean(edge_features, axis=0)
    edge_scale = np.maximum(np.std(edge_features, axis=0), 1.0e-12)
    edge_features = (edge_features - edge_center) / edge_scale

    velocity = dataset.velocity_observed_m_s[:, :n_channels]
    velocity_ri = np.stack([velocity.real, velocity.imag], axis=-1)
    train_mask = dataset.edge_split_code == int(split_config["train_split_code"])
    val_mask = dataset.edge_split_code == int(split_config["val_split_code"])
    test_mask = dataset.edge_split_code == int(split_config["test_split_code"])
    center, scale = _finite_stats(velocity_ri, train_mask)
    velocity_normalized = (velocity_ri - center[None, :, :]) / scale[None, :, :]
    velocity_normalized = np.where(
        np.isfinite(velocity_normalized), velocity_normalized, 0.0
    )
    sink_nodes = dataset.boundary_node_index[
        np.asarray(dataset.boundary_type).astype(str) == "sink"
    ]
    reference = int(sink_nodes[0]) if len(sink_nodes) else int(dataset.boundary_node_index[0])
    return GNNData(
        node_id=dataset.node_id,
        edge_index=torch.tensor(edge_index, dtype=torch.long),
        node_features=torch.tensor(node_features, dtype=torch.float32),
        edge_features=torch.tensor(edge_features, dtype=torch.float32),
        radius_m=torch.tensor(dataset.edge_radius_m, dtype=torch.float32),
        length_m=torch.tensor(dataset.edge_length_m, dtype=torch.float32),
        area_m2=torch.tensor(dataset.edge_area_m2, dtype=torch.float32),
        base_conductance=torch.tensor(conductance, dtype=torch.float32),
        boundary_injection_m3_s=torch.tensor(injection, dtype=torch.float32),
        velocity_observed_m_s=torch.tensor(velocity_ri, dtype=torch.float32),
        velocity_normalized=torch.tensor(velocity_normalized, dtype=torch.float32),
        velocity_center_m_s=torch.tensor(center, dtype=torch.float32),
        velocity_scale_m_s=torch.tensor(scale, dtype=torch.float32),
        train_mask=torch.tensor(train_mask, dtype=torch.bool),
        val_mask=torch.tensor(val_mask, dtype=torch.bool),
        test_mask=torch.tensor(test_mask, dtype=torch.bool),
        reference_node=reference,
        n_harmonics=n_harmonics,
        n_channels=n_channels,
        n_edges=dataset.n_edges,
    )


def graph_laplacian_matvec(pressure, conductance, edge_index, reference_node):
    source, target = edge_index
    drop = pressure[source] - pressure[target]
    flow = conductance * drop
    out = torch.zeros_like(pressure)
    out.index_add_(0, source, flow)
    out.index_add_(0, target, -flow)
    out[reference_node] = pressure[reference_node]
    return out


def solve_pressure_cg(data, conductance, iterations: int, tolerance: float):
    """Stable differentiable damped-Jacobi pressure solve.

    The historical function name is retained as an internal API, but a fixed
    damped iteration is used because gradients through adaptive CG coefficients
    can become unstable during early neural-network training.
    """
    scale = conductance.detach().median().clamp_min(1.0e-30)
    normalized_conductance = conductance / scale
    rhs = data.boundary_injection_m3_s / scale
    rhs = rhs.clone()
    rhs[data.reference_node] = 0.0
    source, target = data.edge_index
    diagonal = torch.zeros_like(rhs)
    diagonal.index_add_(0, source, normalized_conductance)
    diagonal.index_add_(0, target, normalized_conductance)
    diagonal = diagonal.clamp_min(1.0e-8)
    diagonal = diagonal.clone()
    diagonal[data.reference_node] = 1.0
    pressure = torch.zeros_like(rhs)
    damping = 0.65
    initial_norm = torch.linalg.vector_norm(rhs).detach().clamp_min(1.0e-30)
    for _ in range(int(iterations)):
        residual = rhs - graph_laplacian_matvec(
            pressure,
            normalized_conductance,
            data.edge_index,
            data.reference_node,
        )
        pressure = pressure + damping * residual / diagonal
        pressure = pressure.clone()
        pressure[data.reference_node] = 0.0
        if (
            float(torch.linalg.vector_norm(residual).detach().cpu())
            <= float(tolerance) * float(initial_norm.cpu())
        ):
            break
    return pressure


def forward_model(model, data, config):
    raw = model(data)
    use_physics = bool(config["loss_toggles"]["use_physics_layer"])
    pressure = None
    conductance = data.base_conductance * torch.exp(raw["delta_dc"])
    if raw.get("predicted_pressure_pa") is not None:
        # Purely data-driven nodal pressure decoder. Geometry enters only in
        # the fixed Poiseuille readout used to reconstruct observed velocity.
        pressure = raw["predicted_pressure_pa"]
        source, target = data.edge_index
        flow = data.base_conductance * (
            pressure[source] - pressure[target]
        )
        velocity_dc = flow / data.area_m2
        dc_normalized = (
            velocity_dc - data.velocity_center_m_s[0, 0]
        ) / data.velocity_scale_m_s[0, 0]
        velocity_normalized = data.velocity_normalized.new_zeros(
            (data.n_edges, data.n_channels, 2)
        )
        velocity_normalized[:, 0, 0] = dc_normalized
        if data.n_harmonics:
            velocity_normalized[:, 1:, :] = raw[
                "harmonic_output_normalized"
            ][:, : data.n_harmonics, :]
        conductance = data.base_conductance
    elif use_physics:
        pressure = solve_pressure_cg(
            data,
            conductance,
            config["training"]["pressure_solver_iterations"],
            config["training"]["pressure_solver_tolerance"],
        )
        source, target = data.edge_index
        flow = conductance * (pressure[source] - pressure[target])
        velocity_dc = flow / data.area_m2
        dc_normalized = (
            velocity_dc - data.velocity_center_m_s[0, 0]
        ) / data.velocity_scale_m_s[0, 0]
        velocity_normalized = data.velocity_normalized.new_zeros(
            (data.n_edges, data.n_channels, 2)
        )
        velocity_normalized[:, 0, 0] = dc_normalized
        if data.n_harmonics:
            velocity_normalized[:, 1:, :] = raw[
                "harmonic_output_normalized"
            ][:, : data.n_harmonics, :]
    else:
        velocity_normalized = raw["direct_velocity_normalized"][
            :, : data.n_channels, :
        ]
        # DC velocity is real by definition; do not let a direct baseline
        # create an unconstrained imaginary DC channel.
        velocity_normalized = velocity_normalized.clone()
        velocity_normalized[:, 0, 1] = 0.0
    velocity_physical = (
        velocity_normalized * data.velocity_scale_m_s[None, :, :]
        + data.velocity_center_m_s[None, :, :]
    )
    return {
        "velocity_normalized": velocity_normalized,
        "velocity_m_s": velocity_physical,
        "delta_dc": raw["delta_dc"],
        "harmonic_corrections": raw["harmonic_output_normalized"],
        "pressure_pa": pressure,
        "harmonic_pressure_pa": None,
        "conductance_m3_pa_s": conductance,
    }


def reconstruction_metrics(outputs, data, mask):
    metrics = {}
    predicted = outputs["velocity_m_s"]
    observed = data.velocity_observed_m_s
    for channel in range(data.n_channels):
        residual = predicted[mask, channel] - observed[mask, channel]
        truth = observed[mask, channel]
        rmse = torch.sqrt(torch.mean(residual**2))
        scale = torch.sqrt(torch.mean(truth**2)).clamp_min(1.0e-30)
        label = ("dc", "h1", "h2")[channel]
        metrics[f"{label}_rmse_m_s"] = float(rmse.detach().cpu())
        metrics[f"{label}_relative_rmse"] = float((rmse / scale).detach().cpu())
    return metrics


def train_model(model, data, config, checkpoint_path: Path):
    device = next(model.parameters()).device
    data = data.to(device)
    training = config["training"]
    optimizer_class = (
        torch.optim.AdamW
        if training["optimizer"] == "adamw"
        else torch.optim.Adam
    )
    optimizer = optimizer_class(
        model.parameters(),
        lr=float(training["learning_rate"]),
        weight_decay=float(training["weight_decay"]),
    )
    best_state = None
    best_val = float("inf")
    wait = 0
    history = []
    for epoch in range(1, int(training["epochs"]) + 1):
        model.train()
        optimizer.zero_grad()
        outputs = forward_model(model, data, config)
        loss, terms = combined_loss(outputs, data, data.train_mask, config)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            model.parameters(), float(training["gradient_clip"])
        )
        optimizer.step()

        model.eval()
        with torch.no_grad():
            eval_outputs = forward_model(model, data, config)
            val_loss, _ = combined_loss(
                eval_outputs, data, data.val_mask, config
            )
        row = {
            "epoch": epoch,
            "train_loss": float(loss.detach().cpu()),
            "val_loss": float(val_loss.detach().cpu()),
        }
        row.update(
            {
                name: float(value.detach().cpu())
                for name, value in terms.items()
            }
        )
        history.append(row)
        val_value = row["val_loss"]
        if val_value < best_val:
            best_val = val_value
            best_state = copy.deepcopy(model.state_dict())
            wait = 0
            if config["output"]["save_checkpoint"]:
                checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
                torch.save(best_state, checkpoint_path)
        else:
            wait += 1
        if epoch == 1 or epoch % 10 == 0:
            print(
                f"epoch={epoch} train={row['train_loss']:.5g} "
                f"val={row['val_loss']:.5g}"
            )
        if wait >= int(training["patience"]):
            break
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        final_outputs = forward_model(model, data, config)
    split_metrics = {
        "train": reconstruction_metrics(final_outputs, data, data.train_mask),
        "validation": reconstruction_metrics(
            final_outputs, data, data.val_mask
        ),
        "test": reconstruction_metrics(final_outputs, data, data.test_mask),
    }
    delta = final_outputs["delta_dc"]
    bound = float(config["model"]["correction_bound"])
    correction_metrics = {
        "mean": float(delta.mean().cpu()),
        "std": float(delta.std().cpu()),
        "min": float(delta.min().cpu()),
        "max": float(delta.max().cpu()),
        "percent_near_bound": float(
            (delta.abs() >= 0.95 * bound).float().mean().mul(100).cpu()
        ),
    }
    pressure_penalty = 0.0
    if final_outputs["pressure_pa"] is not None:
        source, target = data.edge_index
        pressure_penalty = float(
            (
                (
                    final_outputs["pressure_pa"][source]
                    - final_outputs["pressure_pa"][target]
                )
                ** 2
            )
            .mean()
            .cpu()
        )
    return model, final_outputs, history, {
        "splits": split_metrics,
        "corrections": correction_metrics,
        "pressure_variation_penalty": pressure_penalty,
        "best_validation_loss": best_val,
        "epochs_completed": len(history),
    }
