"""Neural baseline models with interfaces compatible with the physics GNN."""

from __future__ import annotations

import torch
import torch.nn as nn

from .gnn import activation, mlp


class VanillaGCN(nn.Module):
    """Data-driven node GCN that decodes DC pressure and optional harmonics."""

    def __init__(
        self,
        node_dim: int,
        edge_dim: int,
        hidden_dim: int,
        K: int,
        n_channels: int,
        activation_name: str = "relu",
        dropout: float = 0.0,
        correction_bound: float = 8.0,
        **_kwargs,
    ):
        super().__init__()
        if K < 1:
            raise ValueError("VanillaGCN requires K >= 1")
        self.K = int(K)
        self.n_channels = int(n_channels)
        self.n_harmonics = max(self.n_channels - 1, 0)
        self.node_encoder = nn.Linear(node_dim, hidden_dim)
        self.layers = nn.ModuleList(
            [nn.Linear(hidden_dim, hidden_dim) for _ in range(self.K)]
        )
        self.act = activation(activation_name)
        self.dropout = nn.Dropout(float(dropout))
        self.pressure_decoder = mlp(
            hidden_dim,
            hidden_dim,
            1,
            activation_name,
        )
        self.harmonic_decoder = (
            mlp(
                2 * hidden_dim + edge_dim,
                hidden_dim,
                2 * self.n_harmonics,
                activation_name,
            )
            if self.n_harmonics
            else None
        )

    def forward(self, data):
        source, target = data.edge_index
        node = self.act(self.node_encoder(data.node_features))
        for layer in self.layers:
            aggregate = torch.zeros_like(node)
            aggregate.index_add_(0, target, node[source])
            aggregate.index_add_(0, source, node[target])
            degree = torch.zeros(
                (node.shape[0], 1), device=node.device, dtype=node.dtype
            )
            ones = torch.ones(
                (source.shape[0], 1), device=node.device, dtype=node.dtype
            )
            degree.index_add_(0, source, ones)
            degree.index_add_(0, target, ones)
            node = self.dropout(
                self.act(layer((node + aggregate) / (1.0 + degree)))
            )
        pressure = self.pressure_decoder(node).squeeze(-1)
        # Nodal pressure has an arbitrary additive gauge. Pin the same sink
        # reference used by the physics model so saved fields are comparable.
        pressure = pressure - pressure[data.reference_node]
        harmonic = data.edge_features.new_zeros(
            (data.n_edges, self.n_harmonics, 2)
        )
        if self.harmonic_decoder is not None:
            harmonic = self.harmonic_decoder(
                torch.cat(
                    [node[source], node[target], data.edge_features], dim=-1
                )
            ).reshape(data.n_edges, self.n_harmonics, 2)
        return {
            "delta_dc": data.edge_features.new_zeros(data.n_edges),
            "predicted_pressure_pa": pressure,
            "harmonic_output_normalized": harmonic,
        }


class EdgeLocalMLP(nn.Module):
    """K=0 model using only endpoint and local edge features."""

    def __init__(
        self,
        node_dim: int,
        edge_dim: int,
        hidden_dim: int,
        n_channels: int,
        activation_name: str = "relu",
        correction_bound: float = 8.0,
        harmonic_correction_bound: float = 8.0,
        **_kwargs,
    ):
        super().__init__()
        self.n_channels = int(n_channels)
        self.n_harmonics = max(self.n_channels - 1, 0)
        self.correction_bound = float(correction_bound)
        self.harmonic_correction_bound = float(harmonic_correction_bound)
        self.network = mlp(
            2 * node_dim + edge_dim,
            hidden_dim,
            1 + 2 * self.n_channels,
            activation_name,
        )

    def forward(self, data):
        source, target = data.edge_index
        decoded = self.network(
            torch.cat(
                [
                    data.node_features[source],
                    data.node_features[target],
                    data.edge_features,
                ],
                dim=-1,
            )
        )
        harmonic = decoded[:, 3:].reshape(
            data.n_edges, self.n_harmonics, 2
        )
        return {
            "delta_dc": decoded[:, 0].clamp(
                -self.correction_bound, self.correction_bound
            ),
            "direct_velocity_normalized": decoded[:, 1:].reshape(
                data.n_edges, self.n_channels, 2
            ),
            "harmonic_output_normalized": harmonic.clamp(
                -self.harmonic_correction_bound,
                self.harmonic_correction_bound,
            ),
        }
