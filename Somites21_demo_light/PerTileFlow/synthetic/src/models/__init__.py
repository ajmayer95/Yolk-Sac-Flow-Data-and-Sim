"""Classical and learned model implementations."""

from .bayesian_mosaic import solve as solve_bayesian_mosaic
from .bayesian_tile import solve as solve_bayesian_tile
from .baselines import EdgeLocalMLP, VanillaGCN
from .gnn import PhysicsInformedGNN
from .linear_mosaic import solve as solve_linear_mosaic
from .linear_tile import solve as solve_linear_tile

__all__ = [
    "EdgeLocalMLP",
    "PhysicsInformedGNN",
    "VanillaGCN",
    "solve_bayesian_mosaic",
    "solve_bayesian_tile",
    "solve_linear_mosaic",
    "solve_linear_tile",
]
