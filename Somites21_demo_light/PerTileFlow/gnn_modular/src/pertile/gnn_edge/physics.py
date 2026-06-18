"""Physics layer for the modular GNN edge-flow workflow.

This module contains the differentiable DC resistive-network solve.

The physics branch predicts an edge correction delta and converts it into

    G_hat[e] = G_pois[e] * exp(delta[e])

Then it solves the grounded graph Laplacian pressure problem and reconstructs

    Q_hat[e] = G_hat[e] * (p_src - p_dst)

Internal units are SI. Flow values are converted to nL/s elsewhere for losses,
metrics, tables, and plots.
"""

from __future__ import annotations

from typing import Tuple

import torch

from .data import MosaicData

def solve_pressures(data: MosaicData, g_hat: torch.Tensor, jitter: float) -> torch.Tensor:
    """Solve grounded DC pressure field for edge conductances.

    Parameters
    ----------
    data:
        Tensor container with graph topology and source/sink injections.
    g_hat:
        Edge conductances in SI units.
    jitter:
        Small diagonal regularization added to the reduced Laplacian.

    Returns
    -------
    torch.Tensor
        Node pressure vector with the reference node fixed to zero.
    """
    n_nodes = data.x_node.shape[0]
    out_device = g_hat.device
    solve_device = torch.device("cpu") if out_device.type == "mps" else out_device
    g_solve = g_hat.to(solve_device)
    src, dst = data.edge_index.to(solve_device)

    L = torch.zeros((n_nodes, n_nodes), device=solve_device, dtype=g_solve.dtype)
    L.index_put_((src, src), g_solve, accumulate=True)
    L.index_put_((dst, dst), g_solve, accumulate=True)
    L.index_put_((src, dst), -g_solve, accumulate=True)
    L.index_put_((dst, src), -g_solve, accumulate=True)

    ref = int(data.ref_node_index)
    keep = torch.ones(n_nodes, dtype=torch.bool, device=solve_device)
    keep[ref] = False

    L_red = L[keep][:, keep]
    rhs = data.source_sink.to(device=solve_device, dtype=g_solve.dtype)[keep]
    L_red = L_red + torch.eye(
        L_red.shape[0],
        device=solve_device,
        dtype=g_solve.dtype,
    ) * float(jitter)

    p_red = torch.linalg.solve(L_red, rhs)

    p = torch.zeros(n_nodes, device=solve_device, dtype=g_solve.dtype)
    p[keep] = p_red
    return p.to(out_device)


def physics_forward(
    data: MosaicData,
    delta: torch.Tensor,
    jitter: float,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Run the physics branch from conductance correction to predicted flow."""
    g_hat = data.g_pois.to(delta.device) * torch.exp(delta)
    p = solve_pressures(data, g_hat, jitter=jitter)
    src, dst = data.edge_index
    q_hat = g_hat * (p[src] - p[dst])
    return q_hat, p, g_hat


def poisson_baseline(
    data: MosaicData,
    jitter: float,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute the uncorrected Poiseuille baseline with delta = 0."""
    delta = torch.zeros_like(data.g_pois)
    return physics_forward(data, delta, jitter)
