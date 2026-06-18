"""GNN model definitions for edge-flow prediction.

This module contains neural network architectures only.

It owns:
- message-passing layers
- conductance-correction GNN
- direct-flow baseline GNN
"""

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn

from .data import MosaicData
from .constants import nL_per_m3

class EdgeMPNNLayer(nn.Module):
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.msg = nn.Sequential(
            nn.Linear(3 * hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.node_upd = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.edge_upd = nn.Sequential(
            nn.Linear(3 * hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.node_norm = nn.LayerNorm(hidden_dim)
        self.edge_norm = nn.LayerNorm(hidden_dim)

    def forward(self, h: torch.Tensor, e: torch.Tensor,
                edge_index: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        src, dst = edge_index
        m_fwd = self.msg(torch.cat([h[src], h[dst], e], dim=-1))
        m_rev = self.msg(torch.cat([h[dst], h[src], e], dim=-1))
        agg = torch.zeros_like(h)
        agg.index_add_(0, dst, m_fwd)
        agg.index_add_(0, src, m_rev)
        h_new = self.node_norm(h + self.node_upd(torch.cat([h, agg], dim=-1)))
        e_new = self.edge_norm(e + self.edge_upd(torch.cat([h_new[src], h_new[dst], e], dim=-1)))
        return h_new, e_new


class EdgeCorrectionGNN(nn.Module):
    def __init__(self, node_dim: int, edge_dim: int,
                 hidden_dim: int = 64, n_layers: int = 2,
                 n_harmonics: int = 0):
        super().__init__()
        self.n_harmonics = int(n_harmonics)
        self.node_enc = nn.Sequential(nn.Linear(node_dim, hidden_dim), nn.SiLU())
        self.edge_enc = nn.Sequential(nn.Linear(edge_dim, hidden_dim), nn.SiLU())
        self.layers = nn.ModuleList([EdgeMPNNLayer(hidden_dim) for _ in range(n_layers)])
        self.delta_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
        )
        self.harmonic_head = (
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.SiLU(),
                nn.Linear(hidden_dim, 2 * self.n_harmonics),
            )
            if self.n_harmonics > 0 else None
        )

    def encode_edges(self, data: MosaicData) -> torch.Tensor:
        h = self.node_enc(data.x_node)
        e = self.edge_enc(data.x_edge)
        for layer in self.layers:
            h, e = layer(h, e, data.edge_index)
        return e

    def forward(self, data: MosaicData) -> torch.Tensor:
        e = self.encode_edges(data)
        return self.delta_head(e).squeeze(-1).clamp(-8.0, 8.0)

    def forward_harmonics(self, data: MosaicData) -> torch.Tensor:
        if self.harmonic_head is None:
            return data.x_edge.new_zeros((data.x_edge.shape[0], 0, 2))
        e = self.encode_edges(data)
        out = self.harmonic_head(e).reshape(data.x_edge.shape[0], self.n_harmonics, 2)
        return out / nL_per_m3


class DirectFlowGNN(nn.Module):
    def __init__(self, node_dim: int, edge_dim: int,
                 hidden_dim: int = 64, n_layers: int = 2):
        super().__init__()
        self.node_enc = nn.Sequential(nn.Linear(node_dim, hidden_dim), nn.SiLU())
        self.edge_enc = nn.Sequential(nn.Linear(edge_dim, hidden_dim), nn.SiLU())
        self.layers = nn.ModuleList([EdgeMPNNLayer(hidden_dim) for _ in range(n_layers)])
        self.q_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, data: MosaicData) -> torch.Tensor:
        h = self.node_enc(data.x_node)
        e = self.edge_enc(data.x_edge)
        for layer in self.layers:
            h, e = layer(h, e, data.edge_index)
        return self.q_head(e).squeeze(-1) / nL_per_m3
