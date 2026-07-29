"""Physics-informed edge-correction GNN models."""

from __future__ import annotations

import torch
import torch.nn as nn

from physics_layer import apply_bounded_delta


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
                    torch.cat([node[source], node[target], edge_forward], dim=-1)
                )
            )
        )
        edge_reverse = self.edge_norm(
            edge_reverse
            + self.dropout(
                self.edge_update(
                    torch.cat([node[target], node[source], edge_reverse], dim=-1)
                )
            )
        )
        return node, edge_forward, edge_reverse


class PhysicsInformedGNN(nn.Module):
    """Encode-message-decode GNN that predicts only edge conductance corrections."""

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
        predict_gamma: bool = False,
        gamma_min: float = -0.5,
        gamma_max: float = 0.5,
        gamma_parameterization: str = "tanh",
    ):
        super().__init__()
        if K < 0:
            raise ValueError("PhysicsInformedGNN requires K >= 0")
        self.K = int(K)
        self.correction_min = (
            -float(correction_bound) if correction_min is None else float(correction_min)
        )
        self.correction_max = (
            float(correction_bound) if correction_max is None else float(correction_max)
        )
        self.correction_parameterization = str(correction_parameterization)
        self.predict_gamma = bool(predict_gamma)
        self.gamma_min = float(gamma_min)
        self.gamma_max = float(gamma_max)
        self.gamma_parameterization = str(gamma_parameterization)
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
        self.delta_decoder = mlp(decoder_input, hidden_dim, 1, activation_name)
        self.gamma_decoder = (
            mlp(decoder_input, hidden_dim, 1, activation_name)
            if self.predict_gamma
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
        forward_input = torch.cat([node[source], node[target], edge_forward], dim=-1)
        reverse_input = torch.cat([node[target], node[source], edge_reverse], dim=-1)
        raw_delta = 0.5 * (
            self.delta_decoder(forward_input) + self.delta_decoder(reverse_input)
        ).squeeze(-1)
        delta = apply_bounded_delta(
            raw_delta,
            delta_min=self.correction_min,
            delta_max=self.correction_max,
            parameterization=self.correction_parameterization,
        )
        outputs = {
            "raw_delta_e": raw_delta,
            "delta_e": delta,
            "delta_dc": delta,
        }
        if self.predict_gamma and self.gamma_decoder is not None:
            raw_gamma = 0.5 * (
                self.gamma_decoder(forward_input) + self.gamma_decoder(reverse_input)
            ).squeeze(-1)
            gamma = apply_bounded_delta(
                raw_gamma,
                delta_min=self.gamma_min,
                delta_max=self.gamma_max,
                parameterization=self.gamma_parameterization,
            )
            outputs["raw_gamma_e"] = raw_gamma
            outputs["gamma_e"] = gamma
        return outputs
