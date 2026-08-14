#!/usr/bin/env python
"""Plot the Poiseuille boundary-weight calibration sweep."""

from __future__ import annotations

import argparse
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
DEFAULT_INPUT_CSV = (
    PROJECT_ROOT
    / "outputs"
    / "dc"
    / "01_boundary_parameter_calibration"
    / "boundary_weight_summary.csv"
)
DEFAULT_OUTPUT_DIR = (
    PROJECT_ROOT / "outputs" / "dc" / "01_boundary_parameter_calibration" / "figures"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-csv", type=Path, default=DEFAULT_INPUT_CSV)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--lambda-b",
        type=float,
        default=None,
        help="If provided, also plot the flow/flow-magnitude/pressure fields for this lambda_B run.",
    )
    parser.add_argument(
        "--input-root",
        type=Path,
        default=DEFAULT_INPUT_CSV.parent,
        help="Root directory containing per-lambda_B run folders such as lambda_b_100.",
    )
    return parser.parse_args()


def load_summary(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if df.empty:
        raise ValueError(f"No rows found in {path}")
    numeric_columns = [
        "lambda_b",
        "flow_rmse_nl_s",
        "kirchhoff_rms_per_internal_node_nl_s",
        "boundary_residual_rms_pa",
        "boundary_residual_max_pa",
        "pressure_range_pa",
    ]
    for column in numeric_columns:
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors="coerce")
    df = df.sort_values("lambda_b").reset_index(drop=True)
    return df


def save_figure(fig: plt.Figure, output_dir: Path, stem: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_dir / f"{stem}.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def run_name_for_lambda(lambda_b: float) -> str:
    if float(lambda_b).is_integer():
        return f"lambda_b_{int(lambda_b)}"
    return f"lambda_b_{str(lambda_b).replace('.', 'p')}"


def base_axes(title: str, x_label: str, y_label: str) -> tuple[plt.Figure, plt.Axes]:
    fig, ax = plt.subplots(figsize=(7.5, 5.0), constrained_layout=True)
    ax.set_title(title)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_xscale("log")
    ax.grid(True, which="both", alpha=0.3)
    return fig, ax


def plot_single_metric(
    df: pd.DataFrame,
    output_dir: Path,
    y_column: str,
    stem: str,
    title: str,
    y_label: str,
) -> None:
    fig, ax = base_axes(title, r"$\lambda_B$", y_label)
    ax.plot(
        df["lambda_b"],
        df[y_column],
        marker="o",
        linewidth=2.0,
        color="#1f77b4",
    )
    save_figure(fig, output_dir, stem)


def plot_boundary_residuals(df: pd.DataFrame, output_dir: Path) -> None:
    fig, ax = base_axes(
        "Boundary Residual vs Boundary Weight",
        r"$\lambda_B$",
        "Residual (Pa)",
    )
    ax.plot(
        df["lambda_b"],
        df["boundary_residual_rms_pa"],
        marker="o",
        linewidth=2.0,
        label="RMS",
        color="#1f77b4",
    )
    ax.plot(
        df["lambda_b"],
        df["boundary_residual_max_pa"],
        marker="s",
        linewidth=2.0,
        label="Max abs",
        color="#ff7f0e",
    )
    ax.legend(frameon=False)
    save_figure(fig, output_dir, "boundary_residual_vs_lambda_b")


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


def bounds_from_nodes(nodes: pd.DataFrame) -> tuple[tuple[float, float], tuple[float, float]]:
    coords = nodes[["x_px", "y_px"]].apply(pd.to_numeric, errors="coerce")
    return (
        (float(coords["x_px"].min()), float(coords["x_px"].max())),
        (float(coords["y_px"].min()), float(coords["y_px"].max())),
    )


def transform_mosaic_coords(
    x: pd.Series | np.ndarray,
    y: pd.Series | np.ndarray,
    x_bounds: tuple[float, float],
    y_bounds: tuple[float, float],
) -> tuple[np.ndarray, np.ndarray]:
    x_arr = np.asarray(x, dtype=float)
    y_arr = np.asarray(y, dtype=float)
    x_min, _ = x_bounds
    y_min, _ = y_bounds
    return x_arr - x_min, y_arr - y_min


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
            zorder=4,
        )
    if not venous.empty:
        ax.scatter(
            *transform_mosaic_coords(venous["x_px"], venous["y_px"], x_bounds, y_bounds),
            marker="s",
            color="black",
            s=16,
            zorder=4,
        )


def node_lookup(nodes: pd.DataFrame) -> tuple[dict[str, tuple[float, float]], dict[str, tuple[float, float]]]:
    by_id: dict[str, tuple[float, float]] = {}
    by_index: dict[str, tuple[float, float]] = {}
    for _, row in nodes.iterrows():
        x = float(row.get("x_px", float("nan")))
        y = float(row.get("y_px", float("nan")))
        if not (np.isfinite(x) and np.isfinite(y)):
            continue
        by_id[str(row.get("node_id", ""))] = (x, y)
        if "node_index" in row and np.isfinite(float(row["node_index"])):
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


def first_populated_numeric_column(df: pd.DataFrame, columns: list[str]) -> tuple[np.ndarray, str]:
    for column in columns:
        if column not in df.columns:
            continue
        values = pd.to_numeric(df[column], errors="coerce").to_numpy(dtype=float)
        if np.isfinite(values).any():
            return values, column
    return np.full((len(df),), np.nan, dtype=float), columns[0]


def log_widths(values: np.ndarray) -> np.ndarray:
    return 0.5 + 2.0 * np.clip(np.log10(np.clip(np.abs(values), 1.0e-6, None)) + 3.0, 0.0, 3.0) / 3.0


def load_run(run_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    nodes = pd.read_csv(run_dir / "node_predictions.csv")
    edges = pd.read_csv(run_dir / "edge_predictions.csv")
    nodes = numeric(nodes, ["node_index", "pressure_pa", "x_px", "y_px"])
    edges = numeric(
        edges,
        ["predicted_flow_nl_s", "predicted_flow_physical_nl_s", "q_pred_m3_s"],
    )
    return nodes, edges


def plot_lambda_flow_field(nodes: pd.DataFrame, edges: pd.DataFrame, output_dir: Path, stem: str) -> None:
    x_bounds, y_bounds = bounds_from_nodes(nodes)
    segments = transform_segments(build_edge_segments(edges, nodes), x_bounds, y_bounds)
    values, value_column = first_populated_numeric_column(
        edges,
        ["predicted_flow_physical_nl_s", "predicted_flow_nl_s", "q_pred_m3_s"],
    )
    if value_column == "q_pred_m3_s":
        values = values * 1.0e12
    keep = [idx for idx, (segment, value) in enumerate(zip(segments, values)) if np.isfinite(segment).all() and np.isfinite(value)]
    segments = [segments[idx] for idx in keep]
    values = values[keep] if keep else np.asarray([], dtype=float)

    fig, ax = plt.subplots(figsize=(6.8, 6.2), constrained_layout=True)
    if segments:
        collection = LineCollection(segments, cmap="coolwarm", linewidths=1.15, zorder=2)
        collection.set_array(values)
        vmax = max(float(np.nanpercentile(np.abs(values[np.isfinite(values)]), 95.0)), 1.0e-12)
        collection.set_clim(-vmax, vmax)
        ax.add_collection(collection)
        cbar = fig.colorbar(collection, ax=ax, shrink=0.88, pad=0.02)
        cbar.set_label("Predicted flow (nL/s)")
    draw_boundary_markers(ax, nodes, x_bounds, y_bounds)
    decorate_axes(ax, x_bounds, y_bounds)
    ax.set_title("Boundary Calibration Flow Field")
    save_figure(fig, output_dir, stem)


def plot_lambda_flow_magnitude_field(nodes: pd.DataFrame, edges: pd.DataFrame, output_dir: Path, stem: str) -> None:
    x_bounds, y_bounds = bounds_from_nodes(nodes)
    segments = transform_segments(build_edge_segments(edges, nodes), x_bounds, y_bounds)
    values, value_column = first_populated_numeric_column(
        edges,
        ["predicted_flow_physical_nl_s", "predicted_flow_nl_s", "q_pred_m3_s"],
    )
    if value_column == "q_pred_m3_s":
        values = values * 1.0e12
    values = np.abs(values)
    keep = [idx for idx, (segment, value) in enumerate(zip(segments, values)) if np.isfinite(segment).all() and np.isfinite(value)]
    segments = [segments[idx] for idx in keep]
    values = values[keep] if keep else np.asarray([], dtype=float)

    fig, ax = plt.subplots(figsize=(6.8, 6.2), constrained_layout=True)
    if segments:
        finite_positive = values[np.isfinite(values) & (values > 0.0)]
        background = LineCollection(segments, colors="#d0cbc4", linewidths=0.5, alpha=0.35, zorder=1)
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
    ax.set_title("Boundary Calibration Flow Magnitude")
    save_figure(fig, output_dir, stem)


def plot_lambda_pressure_field(nodes: pd.DataFrame, edges: pd.DataFrame, output_dir: Path, stem: str) -> None:
    x_bounds, y_bounds = bounds_from_nodes(nodes)
    segments = transform_segments(build_edge_segments(edges, nodes), x_bounds, y_bounds)
    pressure_values = pd.to_numeric(nodes["pressure_pa"], errors="coerce").to_numpy(dtype=float)

    fig, ax = plt.subplots(figsize=(6.8, 6.2), constrained_layout=True)
    if segments:
        ax.add_collection(LineCollection(segments, colors="#d0d0d0", linewidths=0.55, zorder=1))
    node_x, node_y = transform_mosaic_coords(nodes["x_px"], nodes["y_px"], x_bounds, y_bounds)
    scatter = ax.scatter(node_x, node_y, c=pressure_values, cmap="viridis", s=12, zorder=2)
    finite = pressure_values[np.isfinite(pressure_values)]
    if finite.size:
        scatter.set_clim(float(np.nanpercentile(finite, 2.5)), float(np.nanpercentile(finite, 97.5)))
    cbar = fig.colorbar(scatter, ax=ax, shrink=0.88, pad=0.02)
    cbar.set_label("Pressure [Pa]")
    draw_boundary_markers(ax, nodes, x_bounds, y_bounds)
    decorate_axes(ax, x_bounds, y_bounds)
    ax.set_title("Boundary Calibration Pressure Field")
    save_figure(fig, output_dir, stem)


def plot_lambda_fields(input_root: Path, output_dir: Path, lambda_b: float) -> None:
    run_dir = input_root / run_name_for_lambda(lambda_b)
    if not run_dir.exists():
        raise FileNotFoundError(f"Could not find run directory for lambda_B={lambda_b}: {run_dir}")
    nodes, edges = load_run(run_dir)
    label = run_name_for_lambda(lambda_b)
    plot_lambda_flow_field(nodes, edges, output_dir, f"{label}_flow_field")
    plot_lambda_flow_magnitude_field(nodes, edges, output_dir, f"{label}_flow_magnitude_field")
    plot_lambda_pressure_field(nodes, edges, output_dir, f"{label}_pressure_field")


def main() -> None:
    args = parse_args()
    df = load_summary(args.input_csv.expanduser().resolve())
    output_dir = args.output_dir.expanduser().resolve()

    plot_boundary_residuals(df, output_dir)
    plot_single_metric(
        df,
        output_dir,
        y_column="flow_rmse_nl_s",
        stem="flow_rmse_vs_lambda_b",
        title="Flow RMSE vs Boundary Weight",
        y_label="Flow RMSE (nL/s)",
    )
    plot_single_metric(
        df,
        output_dir,
        y_column="kirchhoff_rms_per_internal_node_nl_s",
        stem="kirchhoff_rms_vs_lambda_b",
        title="Kirchhoff RMS vs Boundary Weight",
        y_label="Kirchhoff RMS per internal node (nL/s)",
    )
    plot_single_metric(
        df,
        output_dir,
        y_column="pressure_range_pa",
        stem="pressure_range_vs_lambda_b",
        title="Pressure Range vs Boundary Weight",
        y_label="Pressure range (Pa)",
    )
    if args.lambda_b is not None:
        plot_lambda_fields(args.input_root.expanduser().resolve(), output_dir, float(args.lambda_b))


if __name__ == "__main__":
    main()
