"""Minimal analysis package surface for real_v3.

This vendored copy exists to support the real-data GNN and outer-loop
calibration workflows without importing the full top-level pertile analysis
stack on package import.
"""

from .transmission_line import (
    TransmissionLineResult,
    solve_transmission_line,
    plot_transmission_line_result,
    compute_eta_from_harmonics,
    compute_eta_from_qt,
)

__all__ = [
    "TransmissionLineResult",
    "solve_transmission_line",
    "plot_transmission_line_result",
    "compute_eta_from_harmonics",
    "compute_eta_from_qt",
]
