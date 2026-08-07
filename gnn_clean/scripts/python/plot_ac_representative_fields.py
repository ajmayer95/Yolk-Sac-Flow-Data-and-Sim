#!/usr/bin/env python
"""Plot AC representative fields for selected harmonic distensibility profiles."""

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
from matplotlib.colors import LogNorm, Normalize
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT_ROOT = PROJECT_ROOT / "outputs" / "ac" / "03_distensibility_alpha_profiles" / "H1"
DEFAULT_MODEL_NAME = "taylor_ideal"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, default=DEFAULT_INPUT_ROOT)
    parser.add_argument("--representatives-csv", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--model-name", type=str, default=DEFAULT_MODEL_NAME)
    parser.add_argument("--dpi", type=int, default=300)
    return parser.parse_args()


def configure_matplotlib() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.grid": False,
            "axes.titlesize": 14,
            "axes.labelsize": 12,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "savefig.facecolor": "white",
        }
    )


def save(fig: plt.Figure, path: Path, dpi: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def field_axes(title: str) -> tuple[plt.Figure, plt.Axes]:
    fig, ax = plt.subplots(figsize=(6.8, 6.2), constrained_layout=True)
    ax.set_title(title)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    return fig, ax


def safe_float(value: object) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


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
    _, x_max = x_bounds
    _, y_max = y_bounds
    return y_max - y_arr, x_max - x_arr


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


def decorate_field_axes(ax: plt.Axes, x_bounds: tuple[float, float], y_bounds: tuple[float, float]) -> None:
    ax.set_xlim((0.0, y_bounds[1] - y_bounds[0]))
    ax.set_ylim((0.0, x_bounds[1] - x_bounds[0]))
    ax.set_aspect("equal")


def node_lookup(nodes: pd.DataFrame) -> tuple[dict[str, tuple[float, float]], dict[str, tuple[float, float]]]:
    by_id: dict[str, tuple[float, float]] = {}
    by_index: dict[str, tuple[float, float]] = {}
    for _, row in nodes.iterrows():
        x = safe_float(row.get("x_px"))
        y = safe_float(row.get("y_px"))
        if not (math.isfinite(x) and math.isfinite(y)):
            continue
        by_id[str(row.get("node_id", ""))] = (x, y)
        node_index = safe_float(row.get("node_index"))
        if math.isfinite(node_index):
            by_index[str(int(node_index))] = (x, y)
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


def draw_boundary_markers_field(
    ax: plt.Axes,
    nodes: pd.DataFrame,
    x_bounds: tuple[float, float],
    y_bounds: tuple[float, float],
) -> None:
    node_type = nodes.get("node_type", pd.Series("", index=nodes.index)).astype(str).str.lower()
    arterial = nodes[node_type.eq("arterial")]
    venous = nodes[node_type.eq("venous")]
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


def log_widths(values: np.ndarray) -> np.ndarray:
    return 0.5 + 2.0 * np.clip(np.log10(np.clip(np.abs(values), 1.0e-6, None)) + 3.0, 0.0, 3.0) / 3.0


def load_field_tables(run_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    nodes = pd.read_csv(run_dir / "node_predictions.csv")
    edges = pd.read_csv(run_dir / "edge_predictions.csv")
    for column in ("node_index", "x_px", "y_px", "pressure_amplitude_pa", "pressure_phase_deg"):
        if column in nodes.columns:
            nodes[column] = pd.to_numeric(nodes[column], errors="coerce")
    for column in (
        "predicted_flow_real_nl_s",
        "predicted_flow_amplitude_nl_s",
        "predicted_flow_phase_deg",
        "source_index",
        "target_index",
    ):
        if column in edges.columns:
            edges[column] = pd.to_numeric(edges[column], errors="coerce")
    return nodes, edges


def label_text(alpha: float, d0: float) -> str:
    return rf"$\alpha = {alpha:g}$, $D_0 = {d0:.3g}$"


def plot_field_set(
    run_dir: Path,
    output_dir: Path,
    stem_prefix: str,
    title_prefix: str,
    dpi: int,
) -> None:
    nodes, edges = load_field_tables(run_dir)
    x_bounds, y_bounds = bounds_from_nodes(nodes)
    segments = transform_segments(build_edge_segments(edges, nodes), x_bounds, y_bounds)
    keep = np.asarray([np.isfinite(segment).all() for segment in segments], dtype=bool) if segments else np.asarray([], dtype=bool)
    segments = [segment for segment, ok in zip(segments, keep) if ok]

    flow_real_values = pd.to_numeric(edges["predicted_flow_real_nl_s"], errors="coerce").to_numpy(dtype=float)
    flow_amp_values = pd.to_numeric(edges["predicted_flow_amplitude_nl_s"], errors="coerce").to_numpy(dtype=float)
    if keep.size:
        flow_real_values = flow_real_values[keep]
        flow_amp_values = flow_amp_values[keep]
    pressure_amp_values = pd.to_numeric(nodes["pressure_amplitude_pa"], errors="coerce").to_numpy(dtype=float)
    pressure_phase_values = pd.to_numeric(nodes["pressure_phase_deg"], errors="coerce").to_numpy(dtype=float)
    node_x, node_y = transform_mosaic_coords(nodes["x_px"], nodes["y_px"], x_bounds, y_bounds)

    fig, ax = field_axes(f"{title_prefix} Flow Field")
    if segments:
        collection = LineCollection(segments, cmap="coolwarm", linewidths=1.15, zorder=2)
        finite = flow_real_values[np.isfinite(flow_real_values)]
        vmax = max(float(np.nanpercentile(np.abs(finite), 95.0)), 1.0e-12) if finite.size else 1.0
        collection.set_array(flow_real_values)
        collection.set_clim(-vmax, vmax)
        ax.add_collection(collection)
        cbar = fig.colorbar(collection, ax=ax, shrink=0.88, pad=0.02)
        cbar.set_label("Predicted flow real component (nL/s)")
    draw_boundary_markers_field(ax, nodes, x_bounds, y_bounds)
    decorate_field_axes(ax, x_bounds, y_bounds)
    save(fig, output_dir / f"{stem_prefix}_flow_field.png", dpi=dpi)

    fig, ax = field_axes(f"{title_prefix} Flow Amplitude")
    if segments:
        finite_positive = flow_amp_values[np.isfinite(flow_amp_values) & (flow_amp_values > 0.0)]
        background = LineCollection(segments, colors="#d0cbc4", linewidths=0.5, alpha=0.35, zorder=1)
        ax.add_collection(background)
        collection = LineCollection(
            segments,
            cmap="coolwarm",
            norm=LogNorm(
                vmin=max(float(np.nanpercentile(finite_positive, 1.0)), 1.0e-3) if finite_positive.size else 1.0e-3,
                vmax=max(float(np.nanpercentile(finite_positive, 99.5)), 1.0e-3) if finite_positive.size else 1.0,
            ),
            linewidths=log_widths(flow_amp_values if flow_amp_values.size else np.asarray([1.0])),
            zorder=2,
        )
        collection.set_array(np.clip(flow_amp_values if flow_amp_values.size else np.asarray([1.0]), 1.0e-12, None))
        ax.add_collection(collection)
        cbar = fig.colorbar(collection, ax=ax, shrink=0.88, pad=0.02)
        cbar.set_label("Predicted flow amplitude |Q| (nL/s)")
        ax.scatter(node_x, node_y, s=3, c="#5f5f5f", linewidths=0.0, zorder=3)
    draw_boundary_markers_field(ax, nodes, x_bounds, y_bounds)
    decorate_field_axes(ax, x_bounds, y_bounds)
    save(fig, output_dir / f"{stem_prefix}_flow_amplitude_field.png", dpi=dpi)

    fig, ax = field_axes(f"{title_prefix} Pressure Field")
    if segments:
        ax.add_collection(LineCollection(segments, colors="#d0d0d0", linewidths=0.55, zorder=1))
    scatter = ax.scatter(node_x, node_y, c=pressure_amp_values, cmap="viridis", s=12, zorder=2)
    finite_pressure = pressure_amp_values[np.isfinite(pressure_amp_values)]
    if finite_pressure.size:
        scatter.set_clim(float(np.nanpercentile(finite_pressure, 2.5)), float(np.nanpercentile(finite_pressure, 97.5)))
    cbar = fig.colorbar(scatter, ax=ax, shrink=0.88, pad=0.02)
    cbar.set_label("Pressure amplitude (Pa)")
    draw_boundary_markers_field(ax, nodes, x_bounds, y_bounds)
    decorate_field_axes(ax, x_bounds, y_bounds)
    save(fig, output_dir / f"{stem_prefix}_pressure_field.png", dpi=dpi)

    fig, ax = field_axes(f"{title_prefix} Pressure Phase Field")
    if segments:
        ax.add_collection(LineCollection(segments, colors="#d0d0d0", linewidths=0.55, zorder=1))
    scatter = ax.scatter(
        node_x,
        node_y,
        c=pressure_phase_values,
        cmap="twilight_shifted",
        norm=Normalize(vmin=-180.0, vmax=180.0),
        s=12,
        zorder=2,
    )
    cbar = fig.colorbar(scatter, ax=ax, shrink=0.88, pad=0.02)
    cbar.set_label("Pressure phase (deg)")
    draw_boundary_markers_field(ax, nodes, x_bounds, y_bounds)
    decorate_field_axes(ax, x_bounds, y_bounds)
    save(fig, output_dir / f"{stem_prefix}_pressure_phase_field.png", dpi=dpi)


def main() -> None:
    args = parse_args()
    configure_matplotlib()

    input_root = args.input_root.expanduser().resolve()
    representatives_csv = (
        args.representatives_csv.expanduser().resolve()
        if args.representatives_csv is not None
        else input_root / "representative_configurations.csv"
    )
    output_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir is not None
        else input_root / "figures" / "representative_fields" / args.model_name
    )

    reps = pd.read_csv(representatives_csv)
    reps = reps[
        reps["model_name"].astype(str).eq(str(args.model_name))
        & reps["profile_run_dir"].notna()
    ].copy()
    if reps.empty:
        raise ValueError(f"No representative rows found for model_name={args.model_name!r} in {representatives_csv}")

    reps["alpha"] = pd.to_numeric(reps["alpha"], errors="coerce")
    reps["representative_D0"] = pd.to_numeric(reps["representative_D0"], errors="coerce")
    reps = reps.sort_values(["representative_label", "alpha"])

    for _, row in reps.iterrows():
        parent_run_dir = Path(str(row["profile_run_dir"])).expanduser().resolve()
        model_run_dir = parent_run_dir / "models" / str(args.model_name)
        if not model_run_dir.exists():
            raise FileNotFoundError(f"Model run directory not found: {model_run_dir}")
        rep_label = str(row["representative_label"])
        alpha = float(row["alpha"])
        d0 = float(row["representative_D0"])
        stem = f"{rep_label}_alpha_{int(round(alpha))}"
        title = f"{rep_label}, {label_text(alpha, d0)}"
        plot_field_set(model_run_dir, output_dir, stem, title, args.dpi)

    print(f"[ok] Wrote AC representative field plots to {output_dir}")


if __name__ == "__main__":
    main()
