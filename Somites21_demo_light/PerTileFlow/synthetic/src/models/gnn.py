"""Physics-informed edge-correction GNN models."""

from __future__ import annotations

import torch
import torch.nn as nn


def activation(name: str) -> nn.Module:
    choices = {
        "relu": nn.ReLU,
        "silu": nn.SiLU,
        "gelu": nn.GELU,
    }
    if name not in choices:
        raise ValueError(f"Unsupported activation: {name}")
    return choices[name]()


def mlp(input_dim: int, hidden_dim: int, output_dim: int, name: str):
    return nn.Sequential(
        nn.Linear(input_dim, hidden_dim),
        activation(name),
        nn.Linear(hidden_dim, output_dim),
    )


class DirectedMessagePassing(nn.Module):
    """Residual directed edge/node message-passing layer."""

    def __init__(self, hidden_dim: int, activation_name: str, dropout: float):
        super().__init__()
        self.message = mlp(3 * hidden_dim, hidden_dim, hidden_dim, activation_name)
        self.node_update = mlp(
            2 * hidden_dim, hidden_dim, hidden_dim, activation_name
        )
        self.edge_update = mlp(
            3 * hidden_dim, hidden_dim, hidden_dim, activation_name
        )
        self.node_norm = nn.LayerNorm(hidden_dim)
        self.edge_norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(float(dropout))

    def forward(self, node, edge_forward, edge_reverse, edge_index):
        source, target = edge_index
        message_forward = self.message(
            torch.cat([node[source], node[target], edge_forward], dim=-1)
        )
        message_reverse = self.message(
            torch.cat([node[target], node[source], edge_reverse], dim=-1)
        )
        aggregate = torch.zeros_like(node)
        aggregate.index_add_(0, target, message_forward)
        aggregate.index_add_(0, source, message_reverse)
        node = self.node_norm(
            node
            + self.dropout(self.node_update(torch.cat([node, aggregate], dim=-1)))
        )
        edge_forward = self.edge_norm(
            edge_forward
            + self.dropout(
                self.edge_update(
                    torch.cat(
                        [node[source], node[target], edge_forward], dim=-1
                    )
                )
            )
        )
        edge_reverse = self.edge_norm(
            edge_reverse
            + self.dropout(
                self.edge_update(
                    torch.cat(
                        [node[target], node[source], edge_reverse], dim=-1
                    )
                )
            )
        )
        return node, edge_forward, edge_reverse


class PhysicsInformedGNN(nn.Module):
    """Encode-message-decode GNN with symmetric conductance corrections."""

    def __init__(
        self,
        node_dim: int,
        edge_dim: int,
        hidden_dim: int,
        K: int,
        n_harmonics: int,
        activation_name: str = "relu",
        dropout: float = 0.0,
        correction_bound: float = 8.0,
        harmonic_correction_bound: float = 8.0,
    ):
        super().__init__()
        if K < 1:
            raise ValueError("PhysicsInformedGNN requires K >= 1")
        self.K = int(K)
        self.n_harmonics = int(n_harmonics)
        self.correction_bound = float(correction_bound)
        self.harmonic_correction_bound = float(harmonic_correction_bound)
        self.node_encoder = nn.Sequential(
            nn.Linear(node_dim, hidden_dim), activation(activation_name)
        )
        self.edge_encoder = nn.Sequential(
            nn.Linear(edge_dim, hidden_dim), activation(activation_name)
        )
        self.layers = nn.ModuleList(
            [
                DirectedMessagePassing(hidden_dim, activation_name, dropout)
                for _ in range(self.K)
            ]
        )
        decoder_input = 3 * hidden_dim
        self.delta_decoder = mlp(
            decoder_input, hidden_dim, 1, activation_name
        )
        self.harmonic_decoder = (
            mlp(
                decoder_input,
                hidden_dim,
                2 * self.n_harmonics,
                activation_name,
            )
            if self.n_harmonics
            else None
        )

    def forward(self, data):
        node = self.node_encoder(data.node_features)
        edge_forward = self.edge_encoder(data.edge_features)
        edge_reverse = self.edge_encoder(data.edge_features)
        for layer in self.layers:
            node, edge_forward, edge_reverse = layer(
                node, edge_forward, edge_reverse, data.edge_index
            )
        source, target = data.edge_index
        forward_input = torch.cat(
            [node[source], node[target], edge_forward], dim=-1
        )
        reverse_input = torch.cat(
            [node[target], node[source], edge_reverse], dim=-1
        )
        delta = 0.5 * (
            self.delta_decoder(forward_input)
            + self.delta_decoder(reverse_input)
        )
        delta = delta.squeeze(-1).clamp(
            -self.correction_bound, self.correction_bound
        )
        harmonic = data.edge_features.new_zeros(
            (data.n_edges, self.n_harmonics, 2)
        )
        if self.harmonic_decoder is not None:
            harmonic = 0.5 * (
                self.harmonic_decoder(forward_input)
                + self.harmonic_decoder(reverse_input)
            )
            harmonic = harmonic.reshape(data.n_edges, self.n_harmonics, 2)
            harmonic = harmonic.clamp(
                -self.harmonic_correction_bound,
                self.harmonic_correction_bound,
            )
        return {
            "delta_dc": delta,
            "harmonic_output_normalized": harmonic,
        }
