"""Evaluation utilities for the modular GNN edge-flow workflow.

This module contains non-training evaluation helpers:
- masked train/validation edge splits
- RMSE/NRMSE calculations
- prediction metrics
- mass-conservation residual summaries
"""

from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
import torch

from .constants import nL_per_m3
from .data import MosaicData

def split_masks(
    data: MosaicData,
    val_fraction: float,
    seed: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Split valid observed edges into train and validation masks."""
    valid_idx = torch.where(data.valid_mask)[0].cpu().numpy()
    rng = np.random.default_rng(int(seed))
    rng.shuffle(valid_idx)

    n_val = max(1, int(round(float(val_fraction) * len(valid_idx)))) if len(valid_idx) else 0
    val_idx = set(int(i) for i in valid_idx[:n_val])

    train = data.valid_mask.clone()
    val = torch.zeros_like(data.valid_mask)

    for i in val_idx:
        train[i] = False
        val[i] = True

    return train, val


def rmse_nrmse_nls(
    q_hat: torch.Tensor,
    q_obs: torch.Tensor,
    mask: torch.Tensor,
) -> Tuple[float, float]:
    """Return RMSE and normalized RMSE in nL/s."""
    if not bool(mask.any()):
        return float("nan"), float("nan")

    pred = q_hat.detach()
    obs = q_obs.detach()

    resid = (pred - obs) * nL_per_m3
    obs_nls = obs * nL_per_m3

    rmse = torch.sqrt((resid[mask] ** 2).mean())
    obs_rms = torch.sqrt((obs_nls[mask] ** 2).mean()).clamp_min(1e-30)

    return float(rmse.cpu()), float((rmse / obs_rms).cpu())


@torch.no_grad()
def evaluate_arrays(
    data: MosaicData,
    q_hat: torch.Tensor,
    mask: torch.Tensor,
) -> dict:
    """Compute scalar prediction metrics for one edge mask."""
    q = data.q_obs.detach().cpu().numpy() * nL_per_m3
    pred = q_hat.detach().cpu().numpy() * nL_per_m3
    m = mask.detach().cpu().numpy().astype(bool)

    if not np.any(m):
        return {
            "n": 0,
            "RMSE_nL_s": np.nan,
            "normalized_RMSE": np.nan,
            "MAE_nL_s": np.nan,
            "pearson_corr": np.nan,
            "R2": np.nan,
        }

    resid = pred[m] - q[m]
    rmse = float(np.sqrt(np.mean(resid ** 2)))
    mae = float(np.mean(np.abs(resid)))
    obs_rms = float(np.sqrt(np.mean(q[m] ** 2)))

    if np.std(q[m]) <= 1e-30 or np.std(pred[m]) <= 1e-30:
        corr = np.nan
    else:
        corr = float(np.corrcoef(q[m], pred[m])[0, 1])

    ss_res = float(np.sum((q[m] - pred[m]) ** 2))
    ss_tot = float(np.sum((q[m] - np.mean(q[m])) ** 2))

    return {
        "n": int(np.sum(m)),
        "RMSE_nL_s": rmse,
        "normalized_RMSE": float(rmse / max(obs_rms, 1e-30)),
        "MAE_nL_s": mae,
        "pearson_corr": corr,
        "R2": float(1.0 - ss_res / ss_tot) if ss_tot > 1e-30 else np.nan,
    }


@torch.no_grad()
def mass_residual_rmse(data: MosaicData, q_hat: torch.Tensor) -> float:
    """Return node-wise mass-conservation residual RMSE in nL/s."""
    src, dst = data.edge_index.cpu()

    net = torch.zeros(data.x_node.shape[0], dtype=torch.float32)
    q_cpu = q_hat.detach().cpu() * nL_per_m3

    net.index_add_(0, src, q_cpu)
    net.index_add_(0, dst, -q_cpu)

    residual = net - data.source_sink.cpu() * nL_per_m3
    return float(torch.sqrt((residual ** 2).mean()).cpu())


@torch.no_grad()
def mass_residual_norm(data: MosaicData, q_hat: torch.Tensor) -> float:
    """Return L2 norm of node-wise mass-conservation residuals in nL/s."""
    src, dst = data.edge_index.cpu()

    net = torch.zeros(data.x_node.shape[0], dtype=torch.float32)
    q_cpu = q_hat.detach().cpu() * nL_per_m3

    net.index_add_(0, src, q_cpu)
    net.index_add_(0, dst, -q_cpu)

    residual = net - data.source_sink.cpu() * nL_per_m3
    return float(torch.linalg.vector_norm(residual).cpu())


def metrics_rows(
    data: MosaicData,
    outputs: Dict[str, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
    train_mask: torch.Tensor,
    val_mask: torch.Tensor,
) -> list[dict]:
    """Create metrics table rows for each model and split."""
    rows = []

    for name, (q_hat, _p, _g) in outputs.items():
        for split, mask in (
            ("train", train_mask),
            ("val", val_mask),
            ("all", data.valid_mask),
        ):
            if split == "val" and not bool(val_mask.any()):
                continue

            row = {"model": name, "split": split}
            row.update(evaluate_arrays(data, q_hat, mask))
            row["mass_residual_RMSE_nL_s"] = mass_residual_rmse(data, q_hat)
            rows.append(row)

    return rows