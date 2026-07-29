"""Loss terms for conductance-only real-data GNN training."""

from __future__ import annotations

import torch


def masked_weighted_mse(
    predicted: torch.Tensor,
    observed: torch.Tensor,
    mask: torch.Tensor,
    weights: torch.Tensor | None = None,
) -> torch.Tensor:
    if not bool(mask.any()):
        return predicted.new_tensor(0.0)
    residual = predicted[mask] - observed[mask]
    if weights is None:
        return torch.mean(residual**2)
    local_weights = weights[mask].clamp_min(1.0e-12)
    return torch.mean((residual * local_weights) ** 2)


def velocity_loss(outputs, data, edge_mask: torch.Tensor) -> torch.Tensor:
    return masked_weighted_mse(
        outputs["velocity_m_s"][:, 0, 0],
        data.velocity_observed_m_s[:, 0, 0],
        edge_mask,
        data.dc_loss_weight,
    )


def observed_flow_m3_s(data) -> torch.Tensor:
    return data.velocity_observed_m_s[:, 0, 0] * data.area_m2


def flow_loss(outputs, data, edge_mask: torch.Tensor) -> torch.Tensor:
    predicted = outputs["flow_m3_s"]
    observed = observed_flow_m3_s(data)
    if not bool(edge_mask.any()):
        return predicted.new_tensor(0.0)
    residual = predicted[edge_mask] - observed[edge_mask]
    weights = data.dc_loss_weight[edge_mask].clamp_min(1.0e-12)
    denom = torch.sum(weights * observed[edge_mask] ** 2).clamp_min(1.0e-30)
    return torch.sum(weights * residual**2) / denom


def alpha_pressure_drop_value(pressure: torch.Tensor, data) -> torch.Tensor:
    if pressure is None:
        return data.base_conductance.new_tensor(float("nan"))
    if data.arterial_node_indices.numel() == 0 or data.venous_node_indices.numel() == 0:
        return pressure.new_tensor(float("nan"))
    arterial_mean = pressure[data.arterial_node_indices].mean()
    venous_mean = pressure[data.venous_node_indices].mean()
    return arterial_mean - venous_mean


def alpha_pressure_drop_loss(outputs, data, alpha_target_pa: float) -> tuple[torch.Tensor, torch.Tensor]:
    alpha_value = alpha_pressure_drop_value(outputs.get("pressure_pa"), data)
    if not torch.isfinite(alpha_value):
        zero = data.base_conductance.new_tensor(0.0)
        return zero, zero
    residual = alpha_value - alpha_value.new_tensor(float(alpha_target_pa))
    return residual**2, residual


def delta_regularization(delta_e: torch.Tensor) -> torch.Tensor:
    return torch.mean(delta_e**2)


def delta_smoothness(delta_e: torch.Tensor, edge_neighbor_index: torch.Tensor) -> torch.Tensor:
    if edge_neighbor_index.numel() == 0:
        return delta_e.new_tensor(0.0)
    edge_a, edge_b = edge_neighbor_index
    return torch.mean((delta_e[edge_a] - delta_e[edge_b]) ** 2)


def pressure_shape_loss(pressure: torch.Tensor, reference_pressure: torch.Tensor) -> torch.Tensor:
    if pressure is None or reference_pressure is None:
        if pressure is not None:
            return pressure.new_tensor(0.0)
        if reference_pressure is not None:
            return reference_pressure.new_tensor(0.0)
        return torch.tensor(0.0)
    finite = torch.isfinite(pressure) & torch.isfinite(reference_pressure)
    if not bool(finite.any()):
        return pressure.new_tensor(0.0)
    p = pressure[finite]
    p0 = reference_pressure[finite]
    p = p - torch.mean(p)
    p0 = p0 - torch.mean(p0)
    p_norm = torch.linalg.vector_norm(p).clamp_min(1.0e-30)
    p0_norm = torch.linalg.vector_norm(p0).clamp_min(1.0e-30)
    return torch.mean(((p / p_norm) - (p0 / p0_norm)) ** 2)


def pressure_shape_correlation(pressure: torch.Tensor, reference_pressure: torch.Tensor) -> torch.Tensor:
    if pressure is None or reference_pressure is None:
        if pressure is not None:
            return pressure.new_tensor(float("nan"))
        if reference_pressure is not None:
            return reference_pressure.new_tensor(float("nan"))
        return torch.tensor(float("nan"))
    finite = torch.isfinite(pressure) & torch.isfinite(reference_pressure)
    if finite.sum() < 2:
        return pressure.new_tensor(float("nan"))
    p = pressure[finite] - torch.mean(pressure[finite])
    p0 = reference_pressure[finite] - torch.mean(reference_pressure[finite])
    denom = torch.linalg.vector_norm(p) * torch.linalg.vector_norm(p0)
    if float(denom.detach().cpu()) <= 1.0e-30:
        return pressure.new_tensor(float("nan"))
    return torch.sum(p * p0) / denom


def combined_loss(outputs, data, edge_mask, config):
    loss_cfg = config["loss"]
    alpha_target_pa = float(loss_cfg.get("alpha_target_pa", 0.0))
    target_signal = str(config.get("data", {}).get("target_signal", "dc_velocity_only"))
    use_alpha = bool(config.get("physics", {}).get("use_alpha_pressure_drop_constraint", True))
    if target_signal == "dc_flow_only":
        primary_loss = flow_loss(outputs, data, edge_mask)
        primary_key = "flow_loss"
    else:
        primary_loss = velocity_loss(outputs, data, edge_mask)
        primary_key = "velocity_loss"
    pressure_ref = getattr(data, "reference_pressure_pa", None)
    pressure_shape_term = pressure_shape_loss(outputs.get("pressure_pa"), pressure_ref)
    pressure_shape_corr = pressure_shape_correlation(outputs.get("pressure_pa"), pressure_ref)
    terms = {
        "velocity_loss": velocity_loss(outputs, data, edge_mask),
        "flow_loss": flow_loss(outputs, data, edge_mask),
        "primary_loss": primary_loss,
        "primary_loss_key": primary_key,
        "alpha_pressure_drop_loss": outputs["delta_e"].new_tensor(0.0),
        "delta_regularization_loss": delta_regularization(outputs["delta_e"]),
        "delta_smoothness_loss": delta_smoothness(
            outputs["delta_e"],
            data.edge_neighbor_index,
        ),
        "alpha_pressure_drop_value": outputs["delta_e"].new_tensor(float("nan")),
        "alpha_residual": outputs["delta_e"].new_tensor(float("nan")),
        "pressure_shape_loss": pressure_shape_term,
        "pressure_shape_corr": pressure_shape_corr,
    }
    if use_alpha and float(loss_cfg.get("lambda_alpha", 0.0)) != 0.0:
        alpha_loss, alpha_residual = alpha_pressure_drop_loss(outputs, data, alpha_target_pa)
        terms["alpha_pressure_drop_loss"] = alpha_loss
        terms["alpha_residual"] = alpha_residual
        terms["alpha_pressure_drop_value"] = alpha_pressure_drop_value(
            outputs.get("pressure_pa"),
            data,
        )
    total = (
        float(loss_cfg.get("lambda_flow", loss_cfg.get("lambda_velocity", 1.0)))
        * primary_loss
        + float(loss_cfg.get("lambda_pressure_shape", 0.0)) * terms["pressure_shape_loss"]
        + float(loss_cfg.get("lambda_alpha", 0.0)) * terms["alpha_pressure_drop_loss"]
        + float(loss_cfg.get("lambda_delta", 0.0)) * terms["delta_regularization_loss"]
        + float(loss_cfg.get("lambda_delta_smooth", 0.0)) * terms["delta_smoothness_loss"]
    )
    return total, terms
