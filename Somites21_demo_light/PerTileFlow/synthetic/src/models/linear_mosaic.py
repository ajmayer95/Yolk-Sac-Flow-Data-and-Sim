"""Whole-mosaic deterministic profile-likelihood solver."""

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
    """Profile external mosaic pressures and infer D0/alpha."""
    return solve_problem(
        dataset,
        problem,
        config,
        method="linear",
        pressure_conditioning=pressure_conditioning,
    )
