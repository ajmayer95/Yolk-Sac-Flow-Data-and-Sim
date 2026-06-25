"""Tile-specific Bayesian solver with marginalized boundary pressures."""

from __future__ import annotations

from distensibility.io import VascularDataset
from distensibility.physics import SpatialProblem
from distensibility.pressure import PressureConditioning

from ._shared import SolverResult, solve_problem


def solve(
    dataset: VascularDataset,
    problem: SpatialProblem,
    config: dict,
    pressure_conditioning: PressureConditioning | None = None,
) -> SolverResult:
    """Marginalize tile-boundary pressures and infer D0/alpha."""
    return solve_problem(
        dataset,
        problem,
        config,
        method="bayesian",
        pressure_conditioning=pressure_conditioning,
    )
