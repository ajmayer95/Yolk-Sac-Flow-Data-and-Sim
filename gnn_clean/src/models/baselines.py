"""Neural baseline models with interfaces compatible with the physics GNN."""

from __future__ import annotations

import torch
import torch.nn as nn

from physics_layer import apply_bounded_delta
from .gnn import activation, mlp


class VanillaGCN(nn.Module):
    """Node-message-passing baseline that decodes edge conductance corrections."""

    def __init__(
        self,
        node_dim: int,
        edge_dim: int,
        hidden_dim: int,
        K: int,
        activation_name: str = "relu",
        dropout: float = 0.0,
        correction_bound: float = 8.0,
        correction_min: float | None = None,
        correction_max: float | None = None,
        correction_parameterization: str = "tanh",
        **_kwargs,
    ):
        super().__init__()
        del edge_dim
        if K < 1:
            raise ValueError("VanillaGCN requires K >= 1")
        self.K = int(K)
        self.correction_min = (
            -float(correction_bound) if correction_min is None else float(correction_min)
        )
        self.correction_max = (
            float(correction_bound) if correction_max is None else float(correction_max)
        )
        self.correction_parameterization = str(correction_parameterization)
        self.node_encoder = nn.Linear(node_dim, hidden_dim)
        self.layers = nn.ModuleList(
            [nn.Linear(hidden_dim, hidden_dim) for _ in range(self.K)]
        )
        self.act = activation(activation_name)
        self.dropout = nn.Dropout(float(dropout))
        self.edge_decoder = mlp(2 * hidden_dim, hidden_dim, 1, activation_name)

    def forward(self, data):
        source, target = data.edge_index
        node = self.act(self.node_encoder(data.node_features))
        for layer in self.layers:
            aggregate = torch.zeros_like(node)
            aggregate.index_add_(0, target, node[source])
            aggregate.index_add_(0, source, node[target])
            degree = torch.zeros((node.shape[0], 1), device=node.device, dtype=node.dtype)
            ones = torch.ones((source.shape[0], 1), device=node.device, dtype=node.dtype)
            degree.index_add_(0, source, ones)
            degree.index_add_(0, target, ones)
            node = self.dropout(self.act(layer((node + aggregate) / (1.0 + degree))))
        raw_delta = self.edge_decoder(torch.cat([node[source], node[target]], dim=-1)).squeeze(-1)
        delta = apply_bounded_delta(
            raw_delta,
            delta_min=self.correction_min,
            delta_max=self.correction_max,
            parameterization=self.correction_parameterization,
        )
        return {
            "raw_delta_e": raw_delta,
            "delta_e": delta,
            "delta_dc": delta,
        }


class EdgeLocalMLP(nn.Module):
    """K=0 model using only endpoint and local edge features."""

    def __init__(
        self,
        node_dim: int,
        edge_dim: int,
        hidden_dim: int,
        activation_name: str = "relu",
        correction_bound: float = 8.0,
        correction_min: float | None = None,
        correction_max: float | None = None,
        correction_parameterization: str = "tanh",
        **_kwargs,
    ):
        super().__init__()
        self.correction_min = (
            -float(correction_bound) if correction_min is None else float(correction_min)
        )
        self.correction_max = (
            float(correction_bound) if correction_max is None else float(correction_max)
        )
        self.correction_parameterization = str(correction_parameterization)
        self.network = mlp(2 * node_dim + edge_dim, hidden_dim, 1, activation_name)

    def forward(self, data):
        source, target = data.edge_index
        raw_delta = self.network(
            torch.cat(
                [
                    data.node_features[source],
                    data.node_features[target],
                    data.edge_features,
                ],
                dim=-1,
            )
        ).squeeze(-1)
        delta = apply_bounded_delta(
            raw_delta,
            delta_min=self.correction_min,
            delta_max=self.correction_max,
            parameterization=self.correction_parameterization,
        )
        return {
            "raw_delta_e": raw_delta,
            "delta_e": delta,
            "delta_dc": delta,
        }


class DirectEdgeDelta(nn.Module):
    """No-GNN inverse baseline with one learnable correction per edge."""

    def __init__(
        self,
        n_edges: int,
        correction_bound: float = 8.0,
        correction_min: float | None = None,
        correction_max: float | None = None,
        correction_parameterization: str = "tanh",
        **_kwargs,
    ):
        super().__init__()
        self.correction_min = (
            -float(correction_bound) if correction_min is None else float(correction_min)
        )
        self.correction_max = (
            float(correction_bound) if correction_max is None else float(correction_max)
        )
        self.correction_parameterization = str(correction_parameterization)
        self.raw_delta = nn.Parameter(torch.zeros(int(n_edges), dtype=torch.float32))

    def forward(self, data):
        raw_delta = self.raw_delta.to(device=data.edge_features.device, dtype=data.edge_features.dtype)
        delta = apply_bounded_delta(
            raw_delta,
            delta_min=self.correction_min,
            delta_max=self.correction_max,
            parameterization=self.correction_parameterization,
        )
        return {
            "raw_delta_e": raw_delta,
            "delta_e": delta,
            "delta_dc": delta,
        }
