#!/usr/bin/env python
"""Plot the edgewise distensibility field on the harmonized graph."""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from harmonic_utils import edge_distensibility_values, edge_geometry_m  # noqa: E402
from real_data import load_graph  # noqa: E402

DEFAULT_GRAPH = PROJECT_ROOT / "datasets" / "harmonized_scaled_dataset.gpickle"
DEFAULT_OUTPUT = (
    PROJECT_ROOT
    / "outputs"
    / "ac"
    / "00_ideal_models"
    / "distensibility_sweep"
    / "figures"
    / "distensibility_field_D0_1e-1_alpha_2.png"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--graph-path", type=Path, default=DEFAULT_GRAPH)
    parser.add_argument("--D0", type=float, default=1.0e-1)
    parser.add_argument("--alpha", type=float, default=2.0)
    parser.add_argument("--output-path", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def graph_positions(graph) -> dict[str, tuple[float, float]]:
    positions: dict[str, tuple[float, float]] = {}
    for node_id, node_data in graph.nodes(data=True):
        try:
            x = float(node_data.get("x", node_data.get("graph_x")))
            y = float(node_data.get("y", node_data.get("graph_y")))
        except (TypeError, ValueError):
            continue
        if math.isfinite(x) and math.isfinite(y):
            positions[str(node_id)] = (x, y)
    return positions


def main() -> None:
    args = parse_args()

    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection
    from matplotlib.colors import Normalize

    graph = load_graph(args.graph_path)
    positions = graph_positions(graph)

    segments: list[np.ndarray] = []
    radii_m: list[float] = []
    edge_labels: list[tuple[str, str]] = []
    for u, v, edge_data in graph.edges(data=True):
        u_key = str(u)
        v_key = str(v)
        if u_key not in positions or v_key not in positions:
            continue
        radius_m, _ = edge_geometry_m(edge_data)
        segments.append(np.asarray([positions[u_key], positions[v_key]], dtype=np.float64))
        radii_m.append(float(radius_m))
        edge_labels.append((u_key, v_key))

    if not segments:
        raise RuntimeError("No edge geometry could be assembled for plotting.")

    radii_arr = np.asarray(radii_m, dtype=np.float64)
    distensibility, reference_radius_m = edge_distensibility_values(
        radii_arr,
        d0=float(args.D0),
        alpha=float(args.alpha),
    )

    x_values = np.concatenate([segment[:, 0] for segment in segments])
    y_values = np.concatenate([segment[:, 1] for segment in segments])
    linewidths = 0.7 + 2.3 * np.sqrt(np.clip(radii_arr / np.nanmax(radii_arr), 0.0, 1.0))

    fig, ax = plt.subplots(figsize=(8.8, 7.6), constrained_layout=True)
    collection = LineCollection(
        segments,
        cmap="viridis",
        norm=Normalize(vmin=float(np.nanmin(distensibility)), vmax=float(np.nanmax(distensibility))),
        linewidths=linewidths,
        zorder=2,
    )
    collection.set_array(distensibility)
    ax.add_collection(collection)

    ax.set_xlim(float(np.nanmin(x_values)), float(np.nanmax(x_values)))
    ax.set_ylim(float(np.nanmax(y_values)), float(np.nanmin(y_values)))
    ax.set_aspect("equal", adjustable="box")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)

    title = (
        f"Distensibility field on harmonized graph\n"
        f"D0 = {args.D0:.1e} Pa^-1, alpha = {args.alpha:g}, "
        f"reference radius = {reference_radius_m * 1.0e6:.2f} um"
    )
    ax.set_title(title)

    cbar = fig.colorbar(collection, ax=ax, shrink=0.92, pad=0.02)
    cbar.set_label("Distensibility D_e [Pa^-1]")

    summary_text = (
        f"edges: {len(edge_labels)}\n"
        f"min: {float(np.nanmin(distensibility)):.2e}\n"
        f"median: {float(np.nanmedian(distensibility)):.2e}\n"
        f"max: {float(np.nanmax(distensibility)):.2e}"
    )
    ax.text(
        0.02,
        0.02,
        summary_text,
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=9,
        bbox={"boxstyle": "round,pad=0.35", "facecolor": "white", "edgecolor": "#d0d0d0", "alpha": 0.92},
    )

    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(f"[saved] {args.output_path}")


if __name__ == "__main__":
    main()
