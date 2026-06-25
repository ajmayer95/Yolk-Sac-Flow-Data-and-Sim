"""Reusable, toggleable GNN loss terms."""

from __future__ import annotations

import torch


def masked_reconstruction_loss(predicted, observed, mask):
    """Mean squared normalized residual over selected edges/channels."""
    if not bool(mask.any()):
        return predicted.new_tensor(0.0)
    residual = predicted - observed
    return (residual[mask] ** 2).mean()


def velocity_reconstruction_loss(predicted, observed, edge_mask):
    return masked_reconstruction_loss(
        predicted[:, 0, :], observed[:, 0, :], edge_mask
    )


def correction_penalty(delta):
    return (delta**2).mean()


def pressure_variation_penalty(pressure, edge_index):
    source, target = edge_index
    return ((pressure[source] - pressure[target]) ** 2).mean()


def harmonic_velocity_loss(predicted, observed, edge_mask, n_harmonics):
    if n_harmonics == 0:
        return predicted.new_tensor(0.0)
    active = edge_mask[:, None].expand(-1, n_harmonics)
    return masked_reconstruction_loss(
        predicted[:, 1 : 1 + n_harmonics, :],
        observed[:, 1 : 1 + n_harmonics, :],
        active,
    )


def harmonic_correction_penalty(corrections):
    if corrections.numel() == 0:
        return corrections.new_tensor(0.0)
    return (corrections**2).mean()


def harmonic_pressure_penalty(pressure_harmonic, edge_index):
    if pressure_harmonic is None or pressure_harmonic.numel() == 0:
        device = edge_index.device
        return torch.tensor(0.0, device=device)
    source, target = edge_index
    drop = pressure_harmonic[source] - pressure_harmonic[target]
    return (drop**2).mean()


def combined_loss(outputs, data, edge_mask, config):
    """Build the configured training objective and its component log."""
    toggles = config["loss_toggles"]
    weights = config["loss"]
    zero = data.velocity_normalized.new_tensor(0.0)
    terms = {
        "velocity_loss": zero,
        "correction_loss": zero,
        "pressure_loss": zero,
        "harmonic_loss": zero,
        "harmonic_correction_loss": zero,
        "harmonic_pressure_loss": zero,
    }
    if toggles["use_velocity_loss"]:
        terms["velocity_loss"] = velocity_reconstruction_loss(
            outputs["velocity_normalized"], data.velocity_normalized, edge_mask
        )
    if toggles["use_correction_loss"]:
        terms["correction_loss"] = correction_penalty(outputs["delta_dc"])
    if toggles["use_pressure_loss"] and outputs.get("pressure_pa") is not None:
        terms["pressure_loss"] = pressure_variation_penalty(
            outputs["pressure_pa"], data.edge_index
        )
    if toggles["use_harmonic_loss"]:
        terms["harmonic_loss"] = harmonic_velocity_loss(
            outputs["velocity_normalized"],
            data.velocity_normalized,
            edge_mask,
            data.n_harmonics,
        )
    if toggles["use_correction_loss"]:
        terms["harmonic_correction_loss"] = harmonic_correction_penalty(
            outputs["harmonic_corrections"]
        )
    if (
        toggles["use_pressure_loss"]
        and outputs.get("harmonic_pressure_pa") is not None
    ):
        terms["harmonic_pressure_loss"] = harmonic_pressure_penalty(
            outputs["harmonic_pressure_pa"], data.edge_index
        )
    total = (
        terms["velocity_loss"]
        + float(weights["lambda_delta"]) * terms["correction_loss"]
        + float(weights["lambda_pressure"]) * terms["pressure_loss"]
        + float(weights["lambda_harmonic"]) * terms["harmonic_loss"]
        + float(weights["lambda_delta_harmonic"])
        * terms["harmonic_correction_loss"]
        + float(weights["lambda_pressure_harmonic"])
        * terms["harmonic_pressure_loss"]
    )
    return total, terms
