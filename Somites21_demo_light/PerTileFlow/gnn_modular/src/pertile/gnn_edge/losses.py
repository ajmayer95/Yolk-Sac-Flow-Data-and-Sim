"""Loss functions for the modular GNN edge-flow workflow.

This module contains differentiable objective terms used during training.

It owns:
- DC flow MSE in nL/s
- harmonic auxiliary flow losses
- harmonic loss weighting by user-specified lambda values

It should not contain model definitions, pressure solves, training loops,
evaluation metrics, plotting, or file writing.
"""

from __future__ import annotations

from typing import Tuple

import torch

from .constants import nL_per_m3

def loss_mse_nls(
    q_hat: torch.Tensor,
    q_obs: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """Mean squared DC flow error in nL/s over a masked edge set."""
    if not bool(mask.any()):
        return q_hat.new_tensor(0.0)
    resid = (q_hat - q_obs) * nL_per_m3
    return (resid[mask] ** 2).mean()


def harmonic_lambdas(
    args,
    n_harmonics: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Return per-harmonic loss weights from command-line/config args."""
    vals = [float(args.lambda_h1), float(args.lambda_h2)]
    return torch.tensor(vals[:n_harmonics], device=device, dtype=dtype)


def loss_harmonics_nls(
    q_hat_h: torch.Tensor,
    q_obs_h: torch.Tensor,
    valid_h: torch.Tensor,
    weight_h: torch.Tensor,
    edge_mask: torch.Tensor,
    args,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """SNR-weighted harmonic flow loss in nL/s.

    q_hat_h and q_obs_h have shape:

        [n_edges, n_harmonics, 2]

    where the final dimension stores real and imaginary components.
    """
    if q_hat_h.shape[1] == 0:
        zero = q_hat_h.new_tensor(0.0)
        return zero, q_hat_h.new_zeros((0,))

    active = valid_h & edge_mask[:, None]
    losses = []

    resid_nls = (q_hat_h - q_obs_h) * nL_per_m3
    weighted_sq = (resid_nls * weight_h[:, :, None].clamp_min(1e-12)) ** 2

    for hi in range(q_hat_h.shape[1]):
        mask = active[:, hi]
        if bool(mask.any()):
            losses.append(weighted_sq[:, hi, :][mask].mean())
        else:
            losses.append(q_hat_h.new_tensor(0.0))

    per_h = torch.stack(losses)
    lam = harmonic_lambdas(args, q_hat_h.shape[1], q_hat_h.device, q_hat_h.dtype)
    return (lam * per_h).sum(), per_h