"""Minimal edge-geometry helpers for the clean DC/AC workflows."""

from __future__ import annotations


def edge_geometry(edge_data: dict, px_size_m: float) -> tuple[float, float]:
    """Return `(radius_m, length_m)` for one edge.

    Preference order matches the original workflow:
    1. `radius_px_true`
    2. `radius`
    3. `radius_adapted_m`

    Values below `1e-3` are treated as already being in meters; larger
    values are interpreted as pixels and converted with `px_size_m`.
    """

    radius_raw = edge_data.get("radius_px_true")
    if radius_raw is None:
        radius_raw = edge_data.get("radius")
    if radius_raw is None:
        radius_raw = edge_data.get("radius_adapted_m", 1.0)
    if hasattr(radius_raw, "item"):
        radius_raw = radius_raw.item()
    radius_value = float(radius_raw)
    radius_m = radius_value if radius_value < 1.0e-3 else radius_value * px_size_m

    length_raw = edge_data.get("length_true") or edge_data.get("length", 1.0)
    if hasattr(length_raw, "item"):
        length_raw = length_raw.item()
    length_value = float(length_raw)
    length_m = length_value if length_value < 1.0e-3 else length_value * px_size_m

    return radius_m, length_m
