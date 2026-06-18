"""Training loops for the modular GNN edge-flow workflow.

This module owns optimization logic for:
- physics-embedded conductance-correction GNN
- direct-flow baseline GNN
"""

from __future__ import annotations

from typing import Iterable, List, Optional, Tuple

import torch

try:
    from tqdm.auto import tqdm
except Exception:  # pragma: no cover
    tqdm = None

from .data import MosaicData
from .evaluate import rmse_nrmse_nls
from .losses import loss_harmonics_nls, loss_mse_nls
from .models import DirectFlowGNN, EdgeCorrectionGNN
from .physics import physics_forward


def train_physics_model(
    data: MosaicData,
    train_mask: torch.Tensor,
    args,
    device: torch.device,
    label: str,
    val_mask: Optional[torch.Tensor] = None,
) -> Tuple[EdgeCorrectionGNN, List[dict], List[dict]]:
    """Train the physics-embedded conductance-correction GNN."""
    model = EdgeCorrectionGNN(
        data.x_node.shape[1],
        data.x_edge.shape[1],
        hidden_dim=args.hidden_dim,
        n_layers=args.layers,
        n_harmonics=data.q_harmonic_obs.shape[1],
    ).to(device)

    opt_cls = torch.optim.AdamW if args.optimizer == "adamw" else torch.optim.Adam
    opt = opt_cls(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    history: List[dict] = []
    validation_history: List[dict] = []

    d = data.to(device)
    train_mask = train_mask.to(device)
    val_mask_dev = val_mask.to(device) if val_mask is not None else None
    has_validation = val_mask_dev is not None and bool(val_mask_dev.any())

    iterator: Iterable[int] = range(1, args.epochs + 1)
    if args.use_tqdm and tqdm is not None:
        iterator = tqdm(iterator, desc=f"{label}/physics", dynamic_ncols=True)

    for epoch in iterator:
        opt.zero_grad()

        delta = model(d)
        q_hat, _, _ = physics_forward(d, delta, args.jitter)

        q_loss = loss_mse_nls(q_hat, d.q_obs, train_mask)

        q_hat_h = model.forward_harmonics(d)
        harmonic_loss, harmonic_losses = loss_harmonics_nls(
            q_hat_h,
            d.q_harmonic_obs,
            d.harmonic_valid_mask,
            d.harmonic_loss_weight,
            train_mask,
            args,
        )

        delta_loss = (delta ** 2).mean()
        loss = q_loss + harmonic_loss + float(args.lambda_delta) * delta_loss

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        opt.step()

        row = {
            "epoch": epoch,
            "model": "physics_gnn",
            "loss": float(loss.detach().cpu()),
            "q_loss": float(q_loss.detach().cpu()),
            "harmonic_loss": float(harmonic_loss.detach().cpu()),
            "h1_loss": (
                float(harmonic_losses[0].detach().cpu())
                if harmonic_losses.numel() > 0
                else float("nan")
            ),
            "h2_loss": (
                float(harmonic_losses[1].detach().cpu())
                if harmonic_losses.numel() > 1
                else float("nan")
            ),
            "delta_loss": float(delta_loss.detach().cpu()),
        }
        history.append(row)

        if has_validation:
            with torch.no_grad():
                eval_delta = model(d)
                eval_q_hat, _, _ = physics_forward(d, eval_delta, args.jitter)
                eval_q_hat_h = model.forward_harmonics(d)

                train_loss = loss_mse_nls(eval_q_hat, d.q_obs, train_mask)
                val_loss = loss_mse_nls(eval_q_hat, d.q_obs, val_mask_dev)

                train_h_loss, _ = loss_harmonics_nls(
                    eval_q_hat_h,
                    d.q_harmonic_obs,
                    d.harmonic_valid_mask,
                    d.harmonic_loss_weight,
                    train_mask,
                    args,
                )
                val_h_loss, _ = loss_harmonics_nls(
                    eval_q_hat_h,
                    d.q_harmonic_obs,
                    d.harmonic_valid_mask,
                    d.harmonic_loss_weight,
                    val_mask_dev,
                    args,
                )

                train_rmse, train_nrmse = rmse_nrmse_nls(
                    eval_q_hat,
                    d.q_obs,
                    train_mask,
                )
                val_rmse, val_nrmse = rmse_nrmse_nls(
                    eval_q_hat,
                    d.q_obs,
                    val_mask_dev,
                )

            validation_history.append(
                {
                    "epoch": epoch,
                    "train_loss": float((train_loss + train_h_loss).cpu()),
                    "val_loss": float((val_loss + val_h_loss).cpu()),
                    "train_q_loss": float(train_loss.cpu()),
                    "val_q_loss": float(val_loss.cpu()),
                    "train_harmonic_loss": float(train_h_loss.cpu()),
                    "val_harmonic_loss": float(val_h_loss.cpu()),
                    "train_rmse": train_rmse,
                    "val_rmse": val_rmse,
                    "train_nrmse": train_nrmse,
                    "val_nrmse": val_nrmse,
                }
            )

        if hasattr(iterator, "set_postfix"):
            postfix = {
                "loss": f"{row['loss']:.3e}",
                "q": f"{row['q_loss']:.3e}",
            }
            if has_validation and validation_history:
                postfix["val"] = f"{validation_history[-1]['val_loss']:.3e}"
            iterator.set_postfix(**postfix)

    return model, history, validation_history


def train_direct_model(
    data: MosaicData,
    train_mask: torch.Tensor,
    args,
    device: torch.device,
    label: str,
) -> Tuple[DirectFlowGNN, List[dict]]:
    """Train the direct-flow GNN baseline."""
    model = DirectFlowGNN(
        data.x_node.shape[1],
        data.x_edge.shape[1],
        hidden_dim=args.hidden_dim,
        n_layers=args.layers,
    ).to(device)

    opt_cls = torch.optim.AdamW if args.optimizer == "adamw" else torch.optim.Adam
    opt = opt_cls(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    history: List[dict] = []

    d = data.to(device)
    train_mask = train_mask.to(device)

    iterator: Iterable[int] = range(1, args.epochs + 1)
    if args.use_tqdm and tqdm is not None:
        iterator = tqdm(iterator, desc=f"{label}/direct", dynamic_ncols=True)

    for epoch in iterator:
        opt.zero_grad()

        q_hat = model(d)
        loss = loss_mse_nls(q_hat, d.q_obs, train_mask)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        opt.step()

        row = {
            "epoch": epoch,
            "model": "direct_gnn",
            "loss": float(loss.detach().cpu()),
            "q_loss": float(loss.detach().cpu()),
            "delta_loss": float("nan"),
        }
        history.append(row)

        if hasattr(iterator, "set_postfix"):
            iterator.set_postfix(loss=f"{row['loss']:.3e}")

    return model, history