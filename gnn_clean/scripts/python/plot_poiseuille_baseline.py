#!/usr/bin/env python
"""Plot Step 0 Poiseuille baseline fields and summary metrics."""

from __future__ import annotations

import argparse
import math
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import LogNorm
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "dc"
    / "00_ideal_models"
    / "poiseuille_only_baseline"
    / "default_partitioned"
)
DEFAULT_OUTPUT_DIR = DEFAULT_INPUT_DIR / "figures"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--dpi", type=int, default=300)
    return parser.parse_args()


def configure_matplotlib() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.grid": False,
            "axes.titlesize": 14,
            "axes.labelsize": 11,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "savefig.facecolor": "white",
        }
    )


def save_figure(fig: plt.Figure, path: Path, dpi: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def numeric(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    result = df.copy()
    for column in columns:
        if column in result.columns:
            result[column] = pd.to_numeric(result[column], errors="coerce")
    return result


def bool_mask(df: pd.DataFrame, column: str) -> pd.Series:
    if column not in df.columns:
        return pd.Series(False, index=df.index)
    return df[column].astype(str).str.lower().isin({"true", "1", "yes"})


def robust_symmetric_limits(values: list[float], percentile: float = 95.0) -> tuple[float, float]:
    finite = np.asarray([value for value in values if math.isfinite(value)], dtype=float)
    if finite.size == 0:
        return (-1.0, 1.0)
    bound = float(np.percentile(np.abs(finite), percentile))
    if not math.isfinite(bound) or bound <= 0.0:
        bound = float(np.max(np.abs(finite)))
    if not math.isfinite(bound) or bound <= 0.0:
        bound = 1.0
    return (-bound, bound)


def robust_limits(values: list[float], lower: float = 5.0, upper: float = 95.0) -> tuple[float, float]:
    finite = np.asarray([value for value in values if math.isfinite(value)], dtype=float)
    if finite.size == 0:
        return (0.0, 1.0)
    vmin = float(np.percentile(finite, lower))
    vmax = float(np.percentile(finite, upper))
    if not math.isfinite(vmin) or not math.isfinite(vmax):
        return (float(np.nanmin(finite)), float(np.nanmax(finite)))
    if math.isclose(vmin, vmax):
        pad = max(abs(vmin) * 0.05, 1.0e-12)
        return (vmin - pad, vmax + pad)
    return (vmin, vmax)


def log_widths(values: np.ndarray) -> np.ndarray:
    return 0.5 + 2.0 * np.clip(np.log10(np.clip(np.abs(values), 1.0e-6, None)) + 3.0, 0.0, 3.0) / 3.0


def first_populated_numeric_column(
    df: pd.DataFrame,
    columns: list[str],
) -> tuple[np.ndarray, str]:
    for column in columns:
        if column not in df.columns:
            continue
        values = pd.to_numeric(df[column], errors="coerce").to_numpy(dtype=float)
        if np.isfinite(values).any():
            return values, column
    return np.full((len(df),), np.nan, dtype=float), columns[0]


def load_run(input_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    nodes = pd.read_csv(input_dir / "node_predictions.csv")
    edges = pd.read_csv(input_dir / "edge_predictions.csv")
    summary = pd.read_csv(input_dir / "summary.csv").iloc[0]
    nodes = numeric(
        nodes,
        ["node_index", "pressure_pa", "kirchhoff_residual_nl_s", "x_px", "y_px"],
    )
    edges = numeric(
        edges,
        [
            "edge_id",
            "predicted_flow_nl_s",
            "predicted_flow_physical_nl_s",
            "observed_flow_nl_s",
            "flow_residual_nl_s",
            "absolute_flow_residual_nl_s",
        ],
    )
    return nodes, edges, summary


def node_lookup(nodes: pd.DataFrame) -> tuple[dict[str, tuple[float, float]], dict[str, tuple[float, float]]]:
    by_id: dict[str, tuple[float, float]] = {}
    by_index: dict[str, tuple[float, float]] = {}
    for _, row in nodes.iterrows():
        x = float(row.get("x_px", float("nan")))
        y = float(row.get("y_px", float("nan")))
        if not (math.isfinite(x) and math.isfinite(y)):
            continue
        by_id[str(row.get("node_id", ""))] = (x, y)
        by_index[str(int(row["node_index"]))] = (x, y)
    return by_id, by_index


def build_edge_segments(edges: pd.DataFrame, nodes: pd.DataFrame) -> list[np.ndarray]:
    by_id, by_index = node_lookup(nodes)
    segments: list[np.ndarray] = []
    for _, row in edges.iterrows():
        source_key = str(row.get("source_node", row.get("source", "")))
        target_key = str(row.get("target_node", row.get("target", "")))
        a = by_id.get(source_key) or by_index.get(source_key)
        b = by_id.get(target_key) or by_index.get(target_key)
        if a is None or b is None:
            continue
        segments.append(np.asarray([[a[0], a[1]], [b[0], b[1]]], dtype=float))
    return segments


def transform_mosaic_coords(
    x: pd.Series | np.ndarray,
    y: pd.Series | np.ndarray,
    x_bounds: tuple[float, float],
    y_bounds: tuple[float, float],
) -> tuple[np.ndarray, np.ndarray]:
    """Plot the mosaic in the positive frame after the requested reorientation."""
    x_arr = np.asarray(x, dtype=float)
    y_arr = np.asarray(y, dtype=float)
    x_min, _ = x_bounds
    y_min, _ = y_bounds
    # The prior plotting orientation already applied one rotate+mirror transform.
    # Applying another 90-degree clockwise rotation followed by a vertical-axis
    # mirror reduces to the raw mosaic orientation, shifted into the positive frame.
    transformed_x = x_arr - x_min
    transformed_y = y_arr - y_min
    return transformed_x, transformed_y


def transform_segments(
    segments: list[np.ndarray],
    x_bounds: tuple[float, float],
    y_bounds: tuple[float, float],
) -> list[np.ndarray]:
    transformed: list[np.ndarray] = []
    for segment in segments:
        tx, ty = transform_mosaic_coords(segment[:, 0], segment[:, 1], x_bounds, y_bounds)
        transformed.append(np.column_stack([tx, ty]))
    return transformed


def bounds_from_nodes(nodes: pd.DataFrame) -> tuple[tuple[float, float], tuple[float, float]]:
    coords = nodes[["x_px", "y_px"]].apply(pd.to_numeric, errors="coerce")
    return (
        (float(coords["x_px"].min()), float(coords["x_px"].max())),
        (float(coords["y_px"].min()), float(coords["y_px"].max())),
    )


def decorate_axes(ax: plt.Axes, x_bounds: tuple[float, float], y_bounds: tuple[float, float]) -> None:
    ax.set_xlim((0.0, x_bounds[1] - x_bounds[0]))
    ax.set_ylim((0.0, y_bounds[1] - y_bounds[0]))
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)


def draw_boundary_markers(
    ax: plt.Axes,
    nodes: pd.DataFrame,
    x_bounds: tuple[float, float],
    y_bounds: tuple[float, float],
) -> None:
    arterial = nodes[bool_mask(nodes, "is_arterial")]
    venous = nodes[bool_mask(nodes, "is_venous")]
    if not arterial.empty:
        ax.scatter(
            *transform_mosaic_coords(arterial["x_px"], arterial["y_px"], x_bounds, y_bounds),
            marker="^",
            color="black",
            s=18,
            zorder=3,
        )
    if not venous.empty:
        ax.scatter(
            *transform_mosaic_coords(venous["x_px"], venous["y_px"], x_bounds, y_bounds),
            marker="s",
            color="black",
            s=16,
            zorder=3,
        )


def plot_flow_field(
    nodes: pd.DataFrame,
    edges: pd.DataFrame,
    output_dir: Path,
    dpi: int,
) -> None:
    x_bounds, y_bounds = bounds_from_nodes(nodes)
    segments = transform_segments(build_edge_segments(edges, nodes), x_bounds, y_bounds)
    values, value_column = first_populated_numeric_column(
        edges,
        [
            "predicted_flow_physical_nl_s",
            "predicted_flow_nl_s",
            "q_pred_m3_s",
        ],
    )
    keep = [idx for idx, (segment, value) in enumerate(zip(segments, values)) if np.isfinite(segment).all() and math.isfinite(value)]
    segments = [segments[idx] for idx in keep]
    values = values[keep] if len(keep) else np.asarray([], dtype=float)

    fig, ax = plt.subplots(figsize=(6.8, 6.2), constrained_layout=True)
    if segments:
        collection = LineCollection(segments, cmap="coolwarm", linewidths=1.15, zorder=2)
        collection.set_array(values)
        collection.set_clim(*robust_symmetric_limits(values.tolist()))
        ax.add_collection(collection)
        cbar = fig.colorbar(collection, ax=ax, shrink=0.88, pad=0.02)
        cbar_label = "Predicted flow (nL/s)"
        if value_column == "q_pred_m3_s":
            values = values * 1.0e12
            collection.set_array(values)
            collection.set_clim(*robust_symmetric_limits(values.tolist()))
        cbar.set_label(cbar_label)
    draw_boundary_markers(ax, nodes, x_bounds, y_bounds)
    decorate_axes(ax, x_bounds, y_bounds)
    ax.set_title("Step 0 Flow Field")
    save_figure(fig, output_dir / "flow_field.png", dpi)


def plot_flow_magnitude_field(
    nodes: pd.DataFrame,
    edges: pd.DataFrame,
    output_dir: Path,
    dpi: int,
) -> None:
    x_bounds, y_bounds = bounds_from_nodes(nodes)
    segments = transform_segments(build_edge_segments(edges, nodes), x_bounds, y_bounds)
    values, value_column = first_populated_numeric_column(
        edges,
        [
            "predicted_flow_physical_nl_s",
            "predicted_flow_nl_s",
            "q_pred_m3_s",
        ],
    )
    if value_column == "q_pred_m3_s":
        values = values * 1.0e12
    values = np.abs(values)
    keep = [idx for idx, (segment, value) in enumerate(zip(segments, values)) if np.isfinite(segment).all() and math.isfinite(value)]
    segments = [segments[idx] for idx in keep]
    values = values[keep] if len(keep) else np.asarray([], dtype=float)

    fig, ax = plt.subplots(figsize=(6.8, 6.2), constrained_layout=True)
    if segments:
        finite_positive = values[np.isfinite(values) & (values > 0.0)]
        background = LineCollection(
            segments,
            colors="#d0cbc4",
            linewidths=0.5,
            alpha=0.35,
            zorder=1,
        )
        ax.add_collection(background)
        collection = LineCollection(
            segments,
            cmap="coolwarm",
            norm=LogNorm(
                vmin=max(float(np.nanpercentile(finite_positive, 1.0)), 1.0e-3),
                vmax=max(float(np.nanpercentile(finite_positive, 99.5)), 1.0e-3),
            ),
            linewidths=log_widths(values),
            zorder=2,
        )
        collection.set_array(np.clip(values, 1.0e-12, None))
        ax.add_collection(collection)
        cbar = fig.colorbar(collection, ax=ax, shrink=0.88, pad=0.02)
        cbar.set_label("Predicted flow amplitude |Q| (nL/s)")
        node_x, node_y = transform_mosaic_coords(nodes["x_px"], nodes["y_px"], x_bounds, y_bounds)
        ax.scatter(node_x, node_y, s=3, c="#5f5f5f", linewidths=0.0, zorder=3)
    draw_boundary_markers(ax, nodes, x_bounds, y_bounds)
    decorate_axes(ax, x_bounds, y_bounds)
    ax.set_title("Step 0 Flow Magnitude")
    save_figure(fig, output_dir / "flow_magnitude_field.png", dpi)


def plot_pressure_field(
    nodes: pd.DataFrame,
    edges: pd.DataFrame,
    output_dir: Path,
    dpi: int,
) -> None:
    x_bounds, y_bounds = bounds_from_nodes(nodes)
    segments = transform_segments(build_edge_segments(edges, nodes), x_bounds, y_bounds)
    pressure_values = pd.to_numeric(nodes["pressure_pa"], errors="coerce").to_numpy(dtype=float)

    fig, ax = plt.subplots(figsize=(6.8, 6.2), constrained_layout=True)
    if segments:
        ax.add_collection(LineCollection(segments, colors="#d0d0d0", linewidths=0.55, zorder=1))
    node_x, node_y = transform_mosaic_coords(nodes["x_px"], nodes["y_px"], x_bounds, y_bounds)
    scatter = ax.scatter(
        node_x,
        node_y,
        c=pressure_values,
        cmap="viridis",
        s=12,
        zorder=2,
    )
    limits = robust_limits(pressure_values[np.isfinite(pressure_values)].tolist(), lower=2.5, upper=97.5)
    scatter.set_clim(*limits)
    cbar = fig.colorbar(scatter, ax=ax, shrink=0.88, pad=0.02)
    cbar.set_label("Pressure [Pa]")
    draw_boundary_markers(ax, nodes, x_bounds, y_bounds)
    decorate_axes(ax, x_bounds, y_bounds)
    ax.set_title("Step 0 Pressure Field")
    save_figure(fig, output_dir / "pressure_field.png", dpi)


def plot_metric_bars(summary: pd.Series, output_dir: Path, dpi: int) -> None:
    metrics = {
        "Flow RMSE": float(pd.to_numeric(summary.get("flow_rmse_nl_s"), errors="coerce")),
        "Kirchhoff RMS": float(
            pd.to_numeric(summary.get("kirchhoff_rms_per_internal_node_nl_s"), errors="coerce")
        ),
    }
    labels = list(metrics.keys())
    values = [metrics[label] for label in labels]

    fig, ax = plt.subplots(figsize=(6.4, 4.6), constrained_layout=True)
    colors = ["#1f77b4", "#d62728"]
    bars = ax.bar(labels, values, color=colors, width=0.58)
    ax.set_ylabel("Metric value (nL/s)")
    ax.set_title("Step 0 Summary Metrics")
    ax.grid(True, axis="y", alpha=0.25)
    ax.set_axisbelow(True)
    for bar, value in zip(bars, values):
        if math.isfinite(value):
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                bar.get_height(),
                f"{value:.3g}",
                ha="center",
                va="bottom",
                fontsize=10,
            )
    save_figure(fig, output_dir / "flow_kirchhoff_metrics.png", dpi)


def main() -> None:
    args = parse_args()
    configure_matplotlib()
    input_dir = args.input_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    nodes, edges, summary = load_run(input_dir)

    plot_flow_field(nodes, edges, output_dir, args.dpi)
    plot_flow_magnitude_field(nodes, edges, output_dir, args.dpi)
    plot_pressure_field(nodes, edges, output_dir, args.dpi)
    plot_metric_bars(summary, output_dir, args.dpi)


if __name__ == "__main__":
    main()
