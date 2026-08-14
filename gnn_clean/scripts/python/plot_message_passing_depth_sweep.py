#!/usr/bin/env python
"""Plot DC Step 4 message-passing-depth sweep results."""

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
import numpy as np
import pandas as pd

from run_message_passing_depth_sweep import (
    K_VALUES,
    _bool_mask,
    build_segments,
    edge_df,
    node_df,
    robust_limits,
    robust_symmetric_limits,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT_ROOT = PROJECT_ROOT / "outputs" / "dc" / "04_message_passing_sensitivity"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, default=DEFAULT_INPUT_ROOT)
    parser.add_argument("--dpi", type=int, default=300)
    return parser.parse_args()


def transform_mosaic_coords(
    x: pd.Series | np.ndarray,
    y: pd.Series | np.ndarray,
    x_bounds: tuple[float, float],
    y_bounds: tuple[float, float],
) -> tuple[np.ndarray, np.ndarray]:
    """Rotate the mosaic the opposite way from the previous Step 4 update."""
    x_arr = np.asarray(x, dtype=float)
    y_arr = np.asarray(y, dtype=float)
    x_min, _ = x_bounds
    y_min, _ = y_bounds
    transformed_x = x_arr - x_min
    transformed_y = y_arr - y_min
    return transformed_x, transformed_y


def transform_segment_collection(
    segments: list[np.ndarray],
    x_bounds: tuple[float, float],
    y_bounds: tuple[float, float],
) -> list[np.ndarray]:
    transformed: list[np.ndarray] = []
    for segment in segments:
        tx, ty = transform_mosaic_coords(segment[:, 0], segment[:, 1], x_bounds, y_bounds)
        transformed.append(np.column_stack([tx, ty]))
    return transformed


def plot_metric_vs_K(summary_df: pd.DataFrame, y_col: str, y_label: str, title: str, path: Path, dpi: int) -> None:
    fig, ax = plt.subplots(figsize=(6.5, 4.6), constrained_layout=True)
    xs = summary_df["K"].to_numpy(dtype=float)
    ys = pd.to_numeric(summary_df[y_col], errors="coerce").to_numpy(dtype=float)
    ax.plot(xs, ys, color="#1f77b4", linewidth=2.2, marker="o", markersize=6)
    ax.set_xlabel("Message-passing layers K")
    ax.set_ylabel(y_label)
    ax.set_title(title)
    ax.grid(True, axis="y", alpha=0.3)
    ax.set_xticks(xs)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def run_dirs_by_k(input_root: Path, summary_df: pd.DataFrame) -> dict[int, Path]:
    result: dict[int, Path] = {}
    for _, row in summary_df.iterrows():
        k_value = int(float(row["K"]))
        run_name = str(row["run_name"])
        run_dir = input_root / run_name
        if not run_dir.exists():
            raise FileNotFoundError(f"Missing run directory for K={k_value}: {run_dir}")
        result[k_value] = run_dir
    missing = [k for k in K_VALUES if k not in result]
    if missing:
        raise ValueError(f"Summary is missing K values: {missing}")
    return result


def plot_pressure_maps(run_dirs: dict[int, Path], path: Path, dpi: int) -> None:
    payloads = []
    all_pressures: list[float] = []
    xmins: list[float] = []
    xmaxs: list[float] = []
    ymins: list[float] = []
    ymaxs: list[float] = []
    for k_value in K_VALUES:
        run_dir = run_dirs[k_value]
        nodes = node_df(run_dir)
        edges = edge_df(run_dir)
        payloads.append((k_value, nodes, edges))
        vals = pd.to_numeric(nodes["pressure_pa"], errors="coerce")
        all_pressures.extend(vals[np.isfinite(vals)].tolist())
        coords = nodes[["x_px", "y_px"]].apply(pd.to_numeric, errors="coerce")
        xmins.append(coords["x_px"].min())
        xmaxs.append(coords["x_px"].max())
        ymins.append(coords["y_px"].min())
        ymaxs.append(coords["y_px"].max())
    limits = robust_limits(all_pressures, lower=5.0, upper=95.0)
    if limits is None:
        return
    vmin, vmax = limits
    xlim = (min(xmins), max(xmaxs))
    ylim = (min(ymins), max(ymaxs))
    transformed_xlim = (0.0, xlim[1] - xlim[0])
    transformed_ylim = (0.0, ylim[1] - ylim[0])
    fig, axes = plt.subplots(1, len(K_VALUES), figsize=(3.1 * len(K_VALUES), 3.8), constrained_layout=True)
    scatter = None
    for ax, (k_value, nodes, edges) in zip(axes, payloads):
        segments = transform_segment_collection(build_segments(nodes, edges), xlim, ylim)
        if segments:
            ax.add_collection(LineCollection(segments, colors="#cdcdcd", linewidths=0.6, zorder=1))
        node_x, node_y = transform_mosaic_coords(nodes["x_px"], nodes["y_px"], xlim, ylim)
        scatter = ax.scatter(node_x, node_y, c=nodes["pressure_pa"], cmap="coolwarm", vmin=vmin, vmax=vmax, s=18, zorder=2)
        arterial = nodes[_bool_mask(nodes, "is_arterial")]
        venous = nodes[_bool_mask(nodes, "is_venous")]
        if not arterial.empty:
            arterial_x, arterial_y = transform_mosaic_coords(arterial["x_px"], arterial["y_px"], xlim, ylim)
            ax.scatter(arterial_x, arterial_y, marker="^", color="black", s=24, zorder=3)
        if not venous.empty:
            venous_x, venous_y = transform_mosaic_coords(venous["x_px"], venous["y_px"], xlim, ylim)
            ax.scatter(venous_x, venous_y, marker="s", color="black", s=20, zorder=3)
        ax.set_title(f"K = {k_value}")
        ax.set_xlim(transformed_xlim)
        ax.set_ylim(transformed_ylim)
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)
    if scatter is not None:
        cbar = fig.colorbar(scatter, ax=axes.ravel().tolist(), shrink=0.9, pad=0.02)
        cbar.set_label("Pressure (Pa)")
    fig.suptitle("Pressure maps by message-passing depth", fontsize=13)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def plot_delta_maps(run_dirs: dict[int, Path], path: Path, dpi: int) -> None:
    payloads = []
    xmins: list[float] = []
    xmaxs: list[float] = []
    ymins: list[float] = []
    ymaxs: list[float] = []
    for k_value in K_VALUES:
        run_dir = run_dirs[k_value]
        nodes = node_df(run_dir)
        edges = edge_df(run_dir)
        payloads.append((k_value, nodes, edges))
        coords = nodes[["x_px", "y_px"]].apply(pd.to_numeric, errors="coerce")
        xmins.append(coords["x_px"].min())
        xmaxs.append(coords["x_px"].max())
        ymins.append(coords["y_px"].min())
        ymaxs.append(coords["y_px"].max())
    xlim = (min(xmins), max(xmaxs))
    ylim = (min(ymins), max(ymaxs))
    transformed_xlim = (0.0, xlim[1] - xlim[0])
    transformed_ylim = (0.0, ylim[1] - ylim[0])
    fig, axes = plt.subplots(1, len(K_VALUES), figsize=(3.1 * len(K_VALUES), 3.8), constrained_layout=True)
    collection = None
    for ax, (k_value, nodes, edges) in zip(axes, payloads):
        lookup = nodes.set_index("node_index")[["x_px", "y_px"]]
        segments: list[np.ndarray] = []
        values: list[float] = []
        for _, edge_row in edges.iterrows():
            try:
                a = lookup.loc[int(edge_row["source_index"])].to_numpy(dtype=float)
                b = lookup.loc[int(edge_row["target_index"])].to_numpy(dtype=float)
            except Exception:
                continue
            if not np.isfinite(a).all() or not np.isfinite(b).all():
                continue
            tx, ty = transform_mosaic_coords(np.array([a[0], b[0]], dtype=float), np.array([a[1], b[1]], dtype=float), xlim, ylim)
            segments.append(np.column_stack([tx, ty]))
            values.append(float(edge_row["delta_e"]))
        if segments:
            collection = LineCollection(segments, cmap="coolwarm", linewidths=1.15, zorder=2)
            collection.set_array(np.asarray(values, dtype=float))
            collection.set_clim(-0.5, 0.5)
            ax.add_collection(collection)
        arterial = nodes[_bool_mask(nodes, "is_arterial")]
        venous = nodes[_bool_mask(nodes, "is_venous")]
        if not arterial.empty:
            arterial_x, arterial_y = transform_mosaic_coords(arterial["x_px"], arterial["y_px"], xlim, ylim)
            ax.scatter(arterial_x, arterial_y, marker="^", color="black", s=18, zorder=3)
        if not venous.empty:
            venous_x, venous_y = transform_mosaic_coords(venous["x_px"], venous["y_px"], xlim, ylim)
            ax.scatter(venous_x, venous_y, marker="s", color="black", s=16, zorder=3)
        ax.set_title(f"K = {k_value}")
        ax.set_xlim(transformed_xlim)
        ax.set_ylim(transformed_ylim)
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)
    if collection is not None:
        cbar = fig.colorbar(collection, ax=axes.ravel().tolist(), shrink=0.9, pad=0.02)
        cbar.set_label("delta_e")
    fig.suptitle("Conductance-correction maps by message-passing depth", fontsize=13)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def plot_flow_residual_maps(run_dirs: dict[int, Path], path: Path, dpi: int) -> None:
    payloads = []
    xmins: list[float] = []
    xmaxs: list[float] = []
    ymins: list[float] = []
    ymaxs: list[float] = []
    all_abs_values: list[float] = []
    for k_value in K_VALUES:
        run_dir = run_dirs[k_value]
        nodes = node_df(run_dir)
        edges = edge_df(run_dir)
        payloads.append((k_value, nodes, edges))
        coords = nodes[["x_px", "y_px"]].apply(pd.to_numeric, errors="coerce")
        xmins.append(coords["x_px"].min())
        xmaxs.append(coords["x_px"].max())
        ymins.append(coords["y_px"].min())
        ymaxs.append(coords["y_px"].max())
        values = pd.to_numeric(edges.get("flow_residual_nl_s"), errors="coerce")
        finite = values[np.isfinite(values)]
        if not finite.empty:
            all_abs_values.extend(np.abs(finite.to_numpy(dtype=float)).tolist())
    limits = robust_symmetric_limits(all_abs_values, percentile=95.0)
    if limits is None:
        return
    vmin, vmax = limits
    xlim = (min(xmins), max(xmaxs))
    ylim = (min(ymins), max(ymaxs))
    transformed_xlim = (0.0, xlim[1] - xlim[0])
    transformed_ylim = (0.0, ylim[1] - ylim[0])
    fig, axes = plt.subplots(1, len(K_VALUES), figsize=(3.1 * len(K_VALUES), 3.8), constrained_layout=True)
    collection = None
    for ax, (k_value, nodes, edges) in zip(axes, payloads):
        lookup = nodes.set_index("node_index")[["x_px", "y_px"]]
        segments: list[np.ndarray] = []
        values: list[float] = []
        for _, edge_row in edges.iterrows():
            try:
                a = lookup.loc[int(edge_row["source_index"])].to_numpy(dtype=float)
                b = lookup.loc[int(edge_row["target_index"])].to_numpy(dtype=float)
            except Exception:
                continue
            value = float(pd.to_numeric(edge_row.get("flow_residual_nl_s"), errors="coerce"))
            if not np.isfinite(a).all() or not np.isfinite(b).all() or not np.isfinite(value):
                continue
            tx, ty = transform_mosaic_coords(np.array([a[0], b[0]], dtype=float), np.array([a[1], b[1]], dtype=float), xlim, ylim)
            segments.append(np.column_stack([tx, ty]))
            values.append(value)
        if segments:
            collection = LineCollection(segments, cmap="coolwarm", linewidths=1.15, zorder=2)
            collection.set_array(np.asarray(values, dtype=float))
            collection.set_clim(vmin, vmax)
            ax.add_collection(collection)
        arterial = nodes[_bool_mask(nodes, "is_arterial")]
        venous = nodes[_bool_mask(nodes, "is_venous")]
        if not arterial.empty:
            arterial_x, arterial_y = transform_mosaic_coords(arterial["x_px"], arterial["y_px"], xlim, ylim)
            ax.scatter(arterial_x, arterial_y, marker="^", color="black", s=18, zorder=3)
        if not venous.empty:
            venous_x, venous_y = transform_mosaic_coords(venous["x_px"], venous["y_px"], xlim, ylim)
            ax.scatter(venous_x, venous_y, marker="s", color="black", s=16, zorder=3)
        ax.set_title(f"K = {k_value}")
        ax.set_xlim(transformed_xlim)
        ax.set_ylim(transformed_ylim)
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)
    if collection is not None:
        cbar = fig.colorbar(collection, ax=axes.ravel().tolist(), shrink=0.9, pad=0.02)
        cbar.set_label("Flow residual (nL/s)")
    fig.suptitle("Flow-residual maps by message-passing depth", fontsize=13)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def plot_kirchhoff_residual_maps(run_dirs: dict[int, Path], path: Path, dpi: int) -> None:
    payloads = []
    xmins: list[float] = []
    xmaxs: list[float] = []
    ymins: list[float] = []
    ymaxs: list[float] = []
    all_abs_values: list[float] = []
    for k_value in K_VALUES:
        run_dir = run_dirs[k_value]
        nodes = node_df(run_dir)
        edges = edge_df(run_dir)
        payloads.append((k_value, nodes, edges))
        coords = nodes[["x_px", "y_px"]].apply(pd.to_numeric, errors="coerce")
        xmins.append(coords["x_px"].min())
        xmaxs.append(coords["x_px"].max())
        ymins.append(coords["y_px"].min())
        ymaxs.append(coords["y_px"].max())
        values = pd.to_numeric(nodes.get("kirchhoff_residual_nl_s"), errors="coerce")
        finite = values[np.isfinite(values)]
        if not finite.empty:
            all_abs_values.extend(np.abs(finite.to_numpy(dtype=float)).tolist())
    limits = robust_symmetric_limits(all_abs_values, percentile=95.0)
    if limits is None:
        return
    vmin, vmax = limits
    xlim = (min(xmins), max(xmaxs))
    ylim = (min(ymins), max(ymaxs))
    transformed_xlim = (0.0, xlim[1] - xlim[0])
    transformed_ylim = (0.0, ylim[1] - ylim[0])
    fig, axes = plt.subplots(1, len(K_VALUES), figsize=(3.1 * len(K_VALUES), 3.8), constrained_layout=True)
    scatter = None
    for ax, (k_value, nodes, edges) in zip(axes, payloads):
        segments = transform_segment_collection(build_segments(nodes, edges), xlim, ylim)
        if segments:
            ax.add_collection(LineCollection(segments, colors="#cdcdcd", linewidths=0.6, zorder=1))
        node_x, node_y = transform_mosaic_coords(nodes["x_px"], nodes["y_px"], xlim, ylim)
        values = pd.to_numeric(nodes.get("kirchhoff_residual_nl_s"), errors="coerce").to_numpy(dtype=float)
        scatter = ax.scatter(node_x, node_y, c=values, cmap="coolwarm", vmin=vmin, vmax=vmax, s=18, zorder=2)
        arterial = nodes[_bool_mask(nodes, "is_arterial")]
        venous = nodes[_bool_mask(nodes, "is_venous")]
        if not arterial.empty:
            arterial_x, arterial_y = transform_mosaic_coords(arterial["x_px"], arterial["y_px"], xlim, ylim)
            ax.scatter(arterial_x, arterial_y, marker="^", color="black", s=24, zorder=3)
        if not venous.empty:
            venous_x, venous_y = transform_mosaic_coords(venous["x_px"], venous["y_px"], xlim, ylim)
            ax.scatter(venous_x, venous_y, marker="s", color="black", s=20, zorder=3)
        ax.set_title(f"K = {k_value}")
        ax.set_xlim(transformed_xlim)
        ax.set_ylim(transformed_ylim)
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)
    if scatter is not None:
        cbar = fig.colorbar(scatter, ax=axes.ravel().tolist(), shrink=0.9, pad=0.02)
        cbar.set_label("Kirchhoff residual (nL/s)")
    fig.suptitle("Kirchhoff-residual maps by message-passing depth", fontsize=13)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    input_root = args.input_root.expanduser().resolve()
    summary_path = input_root / "summary.csv"
    if not summary_path.exists():
        raise FileNotFoundError(
            f"Missing Step 04 summary CSV: {summary_path}. Run the Step 04 solver first."
        )
    summary_df = pd.read_csv(summary_path)
    if summary_df.empty:
        raise ValueError(f"Step 04 summary CSV is empty: {summary_path}")
    summary_df["K"] = pd.to_numeric(summary_df["K"], errors="coerce")
    summary_df = summary_df.sort_values("K").reset_index(drop=True)
    run_dirs = run_dirs_by_k(input_root, summary_df)

    plot_metric_vs_K(
        summary_df,
        "flow_rmse_nl_s",
        "Flow RMSE (nL/s)",
        "Flow agreement versus message-passing depth",
        input_root / "flow_rmse_vs_K.png",
        args.dpi,
    )
    plot_metric_vs_K(
        summary_df,
        "kirchhoff_rms_per_internal_node_nl_s",
        "Kirchhoff RMS per internal node (nL/s)",
        "Conservation consistency versus message-passing depth",
        input_root / "kirchhoff_rms_vs_K.png",
        args.dpi,
    )
    plot_pressure_maps(run_dirs, input_root / "pressure_maps_by_K.png", args.dpi)
    plot_flow_residual_maps(run_dirs, input_root / "flow_residual_maps_by_K.png", args.dpi)
    plot_kirchhoff_residual_maps(run_dirs, input_root / "kirchhoff_residual_maps_by_K.png", args.dpi)
    plot_delta_maps(run_dirs, input_root / "conductance_correction_maps_by_K.png", args.dpi)
    print(f"[ok] Wrote Step 04 message-passing figures to {input_root}")


if __name__ == "__main__":
    main()
