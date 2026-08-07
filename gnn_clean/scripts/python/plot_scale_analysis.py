#!/usr/bin/env python
"""Plot flow-scaled error mosaics for harmonized AC and DC representative runs."""

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


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DC_REPRESENTATIVES = (
    PROJECT_ROOT / "outputs" / "dc" / "02_physics_weight_sweep" / "representative_configurations.csv"
)
DEFAULT_DC_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "dc" / "06_scale_analysis"
DEFAULT_AC_ROOT = PROJECT_ROOT / "outputs" / "ac" / "02_physics_weight_sweep"
DEFAULT_AC_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "ac" / "04_scale_analysis"
AC_MODEL_ORDER = ("full_ideal", "taylor_ideal", "taylor_dc_transferred")
EPSILON_FLOW_NL_S = 1.0e-6
TARGET_LABELS = ("B1", "F1", "K1")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dc-representatives", type=Path, default=DEFAULT_DC_REPRESENTATIVES)
    parser.add_argument("--ac-root", type=Path, default=DEFAULT_AC_ROOT)
    parser.add_argument("--dc-output-dir", type=Path, default=DEFAULT_DC_OUTPUT_DIR)
    parser.add_argument("--ac-output-dir", type=Path, default=DEFAULT_AC_OUTPUT_DIR)
    parser.add_argument("--dpi", type=int, default=300)
    return parser.parse_args()


def configure_plot() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.grid": False,
            "axes.titlesize": 13,
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


def build_segments(nodes: pd.DataFrame, edges: pd.DataFrame) -> list[np.ndarray]:
    lookup = nodes.set_index("node_index")[["x_px", "y_px"]]
    segments: list[np.ndarray] = []
    for _, row in edges.iterrows():
        try:
            a = lookup.loc[int(row["source_index"])].to_numpy(dtype=float)
            b = lookup.loc[int(row["target_index"])].to_numpy(dtype=float)
        except Exception:
            continue
        if not np.isfinite(a).all() or not np.isfinite(b).all():
            continue
        segments.append(np.vstack([a, b]))
    return segments


def transform_mosaic_coords(
    x: pd.Series | np.ndarray,
    y: pd.Series | np.ndarray,
    x_bounds: tuple[float, float],
    y_bounds: tuple[float, float],
) -> tuple[np.ndarray, np.ndarray]:
    x_arr = np.asarray(x, dtype=float)
    y_arr = np.asarray(y, dtype=float)
    x_min, x_max = x_bounds
    y_min, y_max = y_bounds
    transformed_x = y_max - y_arr
    transformed_y = x_max - x_arr
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


def filter_target_labels(df: pd.DataFrame, label_column: str = "plot_label") -> pd.DataFrame:
    if label_column not in df.columns:
        return df.iloc[0:0].copy()
    return df[df[label_column].astype(str).isin(TARGET_LABELS)].copy()


def ordered_target_labels(values: list[str]) -> list[str]:
    present = {str(value) for value in values}
    return [label for label in TARGET_LABELS if label in present]


def load_dc_run(run_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    nodes = pd.read_csv(run_dir / "node_predictions.csv")
    edges = pd.read_csv(run_dir / "edge_predictions.csv")
    nodes = numeric(
        nodes,
        ["node_index", "x_px", "y_px", "kirchhoff_residual_nl_s"],
    )
    edges = numeric(
        edges,
        ["source_index", "target_index", "predicted_flow_nl_s", "observed_flow_nl_s", "flow_residual_nl_s"],
    )
    return nodes, edges


def load_ac_run(run_dir: Path, model_name: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    model_dir = run_dir / "models" / model_name
    nodes = pd.read_csv(model_dir / "node_predictions.csv")
    edges = pd.read_csv(model_dir / "edge_predictions.csv")
    nodes = numeric(
        nodes,
        ["node_index", "x_px", "y_px", "kirchhoff_residual_abs_nl_s"],
    )
    edges = numeric(
        edges,
        [
            "source_index",
            "target_index",
            "predicted_flow_amplitude_nl_s",
            "observed_flow_amplitude_nl_s",
            "flow_residual_abs_nl_s",
        ],
    )
    return nodes, edges


def compute_node_throughput(
    nodes: pd.DataFrame,
    edges: pd.DataFrame,
    edge_flow_column: str,
) -> pd.Series:
    throughput = pd.Series(0.0, index=nodes["node_index"].astype(int).to_numpy(dtype=int))
    for _, row in edges.iterrows():
        src = row.get("source_index")
        dst = row.get("target_index")
        flow = row.get(edge_flow_column)
        if not (pd.notna(src) and pd.notna(dst) and pd.notna(flow)):
            continue
        magnitude = abs(float(flow))
        if not math.isfinite(magnitude):
            continue
        throughput.loc[int(src)] = throughput.get(int(src), 0.0) + 0.5 * magnitude
        throughput.loc[int(dst)] = throughput.get(int(dst), 0.0) + 0.5 * magnitude
    return throughput


def choose_edge_scale(row: pd.Series, predicted_col: str, observed_col: str) -> float:
    candidates = [
        abs(float(value))
        for value in (row.get(predicted_col), row.get(observed_col))
        if pd.notna(value) and math.isfinite(float(value))
    ]
    if not candidates:
        return EPSILON_FLOW_NL_S
    return max(max(candidates), EPSILON_FLOW_NL_S)


def prepare_dc_scaled_maps(run_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    nodes, edges = load_dc_run(run_dir)
    node_throughput = compute_node_throughput(nodes, edges, "predicted_flow_nl_s")
    nodes = nodes.copy()
    edges = edges.copy()
    nodes["flow_through_node_nl_s"] = nodes["node_index"].astype(int).map(node_throughput).fillna(0.0)
    nodes["scaled_kirchhoff_error"] = (
        pd.to_numeric(nodes["kirchhoff_residual_nl_s"], errors="coerce").abs()
        / nodes["flow_through_node_nl_s"].clip(lower=EPSILON_FLOW_NL_S)
    )
    edge_scale = edges.apply(
        lambda row: choose_edge_scale(row, "predicted_flow_nl_s", "observed_flow_nl_s"),
        axis=1,
    )
    edges["flow_scale_nl_s"] = edge_scale
    edges["scaled_flow_error"] = (
        pd.to_numeric(edges["flow_residual_nl_s"], errors="coerce").abs() / edge_scale
    )
    return nodes, edges


def prepare_ac_scaled_maps(run_dir: Path, model_name: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    nodes, edges = load_ac_run(run_dir, model_name)
    node_throughput = compute_node_throughput(nodes, edges, "predicted_flow_amplitude_nl_s")
    nodes = nodes.copy()
    edges = edges.copy()
    nodes["flow_through_node_nl_s"] = nodes["node_index"].astype(int).map(node_throughput).fillna(0.0)
    nodes["scaled_kirchhoff_error"] = (
        pd.to_numeric(nodes["kirchhoff_residual_abs_nl_s"], errors="coerce")
        / nodes["flow_through_node_nl_s"].clip(lower=EPSILON_FLOW_NL_S)
    )
    edge_scale = edges.apply(
        lambda row: choose_edge_scale(
            row,
            "predicted_flow_amplitude_nl_s",
            "observed_flow_amplitude_nl_s",
        ),
        axis=1,
    )
    edges["flow_scale_nl_s"] = edge_scale
    edges["scaled_flow_error"] = (
        pd.to_numeric(edges["flow_residual_abs_nl_s"], errors="coerce") / edge_scale
    )
    return nodes, edges


def draw_node_panel(
    ax: plt.Axes,
    nodes: pd.DataFrame,
    edges: pd.DataFrame,
    value_column: str,
    xlim: tuple[float, float],
    ylim: tuple[float, float],
    vmax: float,
    marker_size: float = 4.0,
) -> plt.Axes:
    segments = transform_segments(build_segments(nodes, edges), xlim, ylim)
    if segments:
        ax.add_collection(LineCollection(segments, colors="#cfcfcf", linewidths=0.55, zorder=1))
    node_x, node_y = transform_mosaic_coords(nodes["x_px"], nodes["y_px"], xlim, ylim)
    scatter = ax.scatter(
        node_x,
        node_y,
        c=pd.to_numeric(nodes[value_column], errors="coerce"),
        cmap="magma",
        vmin=0.0,
        vmax=vmax,
        s=marker_size,
        zorder=2,
    )
    arterial = nodes[nodes["boundary_role"].astype(str) == "arterial"]
    venous = nodes[nodes["boundary_role"].astype(str) == "venous"]
    if "is_arterial" in nodes.columns:
        arterial = nodes[bool_mask(nodes, "is_arterial")]
    if "is_venous" in nodes.columns:
        venous = nodes[bool_mask(nodes, "is_venous")]
    if not arterial.empty:
        ax.scatter(*transform_mosaic_coords(arterial["x_px"], arterial["y_px"], xlim, ylim), marker="^", color="black", s=10, zorder=3)
    if not venous.empty:
        ax.scatter(*transform_mosaic_coords(venous["x_px"], venous["y_px"], xlim, ylim), marker="s", color="black", s=9, zorder=3)
    ax.set_xlim((0.0, ylim[1] - ylim[0]))
    ax.set_ylim((0.0, xlim[1] - xlim[0]))
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    return scatter


def draw_edge_panel(
    ax: plt.Axes,
    nodes: pd.DataFrame,
    edges: pd.DataFrame,
    value_column: str,
    xlim: tuple[float, float],
    ylim: tuple[float, float],
    vmax: float,
) -> plt.Axes | None:
    lookup = nodes.set_index("node_index")[["x_px", "y_px"]]
    segments: list[np.ndarray] = []
    values: list[float] = []
    for _, edge_row in edges.iterrows():
        try:
            a = lookup.loc[int(edge_row["source_index"])].to_numpy(dtype=float)
            b = lookup.loc[int(edge_row["target_index"])].to_numpy(dtype=float)
        except Exception:
            continue
        value = float(pd.to_numeric(edge_row.get(value_column), errors="coerce"))
        if not np.isfinite(a).all() or not np.isfinite(b).all() or not math.isfinite(value):
            continue
        tx, ty = transform_mosaic_coords(np.array([a[0], b[0]]), np.array([a[1], b[1]]), xlim, ylim)
        segments.append(np.column_stack([tx, ty]))
        values.append(value)
    collection = None
    if segments:
        collection = LineCollection(segments, cmap="magma", linewidths=1.0, zorder=2)
        collection.set_array(np.asarray(values, dtype=float))
        collection.set_clim(0.0, vmax)
        ax.add_collection(collection)
    arterial = nodes[nodes["boundary_role"].astype(str) == "arterial"]
    venous = nodes[nodes["boundary_role"].astype(str) == "venous"]
    if "is_arterial" in nodes.columns:
        arterial = nodes[bool_mask(nodes, "is_arterial")]
    if "is_venous" in nodes.columns:
        venous = nodes[bool_mask(nodes, "is_venous")]
    if not arterial.empty:
        ax.scatter(*transform_mosaic_coords(arterial["x_px"], arterial["y_px"], xlim, ylim), marker="^", color="black", s=10, zorder=3)
    if not venous.empty:
        ax.scatter(*transform_mosaic_coords(venous["x_px"], venous["y_px"], xlim, ylim), marker="s", color="black", s=9, zorder=3)
    ax.set_xlim((0.0, ylim[1] - ylim[0]))
    ax.set_ylim((0.0, xlim[1] - xlim[0]))
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    return collection


def plot_dc_scale_analysis(rep_path: Path, output_dir: Path, dpi: int) -> None:
    rep_df = pd.read_csv(rep_path)
    rep_df = rep_df[rep_df["selected_representative"] == True].copy()  # noqa: E712
    rep_df = filter_target_labels(rep_df)
    rep_df["selection_rank_within_regime"] = pd.to_numeric(
        rep_df["selection_rank_within_regime"], errors="coerce"
    )
    rep_df = rep_df.sort_values(["selection_category", "selection_rank_within_regime", "plot_label"])
    panels: list[dict[str, object]] = []
    node_values: list[float] = []
    edge_values: list[float] = []
    xmins: list[float] = []
    xmaxs: list[float] = []
    ymins: list[float] = []
    ymaxs: list[float] = []
    for _, row in rep_df.iterrows():
        run_dir = Path(str(row["output_dir"]))
        nodes, edges = prepare_dc_scaled_maps(run_dir)
        panels.append({"label": str(row["plot_label"]), "run_name": str(row["run_name"]), "nodes": nodes, "edges": edges})
        node_values.extend(pd.to_numeric(nodes["scaled_kirchhoff_error"], errors="coerce").dropna().tolist())
        edge_values.extend(pd.to_numeric(edges["scaled_flow_error"], errors="coerce").dropna().tolist())
        coords = nodes[["x_px", "y_px"]].apply(pd.to_numeric, errors="coerce")
        xmins.append(float(coords["x_px"].min()))
        xmaxs.append(float(coords["x_px"].max()))
        ymins.append(float(coords["y_px"].min()))
        ymaxs.append(float(coords["y_px"].max()))
    if not panels:
        return
    xlim = (min(xmins), max(xmaxs))
    ylim = (min(ymins), max(ymaxs))
    node_vmax = robust_symmetric_limits(node_values)[1]
    edge_vmax = robust_symmetric_limits(edge_values)[1]
    fig, axes = plt.subplots(len(panels), 1, figsize=(3.8, 2.45 * len(panels)), squeeze=False)
    fig.subplots_adjust(left=0.12, right=0.84, top=0.93, bottom=0.04, hspace=0.12)
    node_artist = None
    for row_idx, payload in enumerate(panels):
        nodes = payload["nodes"]
        edges = payload["edges"]
        node_artist = draw_node_panel(
            axes[row_idx, 0],
            nodes,
            edges,
            "scaled_kirchhoff_error",
            xlim,
            ylim,
            node_vmax,
            marker_size=3.0,
        )
        axes[row_idx, 0].set_ylabel(str(payload["label"]), rotation=0, labelpad=22, va="center", ha="right", fontsize=11, fontweight="bold")
    if node_artist is not None:
        cax = fig.add_axes([0.87, 0.18, 0.03, 0.64])
        cbar = fig.colorbar(node_artist, cax=cax)
        cbar.set_label("|Kirchhoff residual| / local flow")
    fig.suptitle("DC representative mosaics: scaled Kirchhoff error", fontsize=13, y=0.985)
    save_figure(fig, output_dir / "dc_scaled_kirchhoff_error_mosaics.png", dpi)

    fig, axes = plt.subplots(len(panels), 1, figsize=(3.8, 2.45 * len(panels)), squeeze=False)
    fig.subplots_adjust(left=0.12, right=0.84, top=0.93, bottom=0.04, hspace=0.12)
    edge_artist = None
    for row_idx, payload in enumerate(panels):
        nodes = payload["nodes"]
        edges = payload["edges"]
        edge_artist = draw_edge_panel(
            axes[row_idx, 0],
            nodes,
            edges,
            "scaled_flow_error",
            xlim,
            ylim,
            edge_vmax,
        )
        axes[row_idx, 0].set_ylabel(str(payload["label"]), rotation=0, labelpad=22, va="center", ha="right", fontsize=11, fontweight="bold")
    if edge_artist is not None:
        cax = fig.add_axes([0.87, 0.18, 0.03, 0.64])
        cbar = fig.colorbar(edge_artist, cax=cax)
        cbar.set_label("|Flow residual| / edge flow")
    fig.suptitle("DC representative mosaics: scaled flow error", fontsize=13, y=0.985)
    save_figure(fig, output_dir / "dc_scaled_flow_error_mosaics.png", dpi)


def plot_ac_scale_analysis(ac_root: Path, output_dir: Path, dpi: int) -> None:
    for harmonic_dir in sorted(path for path in ac_root.iterdir() if path.is_dir()):
        rep_path = harmonic_dir / "ac_physics_weight_representatives.csv"
        if not rep_path.exists():
            continue
        rep_df = pd.read_csv(rep_path)
        rep_df = rep_df[rep_df["selected_representative"] == True].copy()  # noqa: E712
        rep_df = filter_target_labels(rep_df)
        rep_df["selection_rank_within_regime"] = pd.to_numeric(
            rep_df["selection_rank_within_regime"], errors="coerce"
        )
        rep_df = rep_df.sort_values(["plot_label", "model_name"])
        labels = ordered_target_labels(rep_df["plot_label"].dropna().astype(str).unique().tolist())
        available_models = [name for name in AC_MODEL_ORDER if name in set(rep_df["model_name"].astype(str))]
        panels: dict[tuple[str, str], tuple[pd.DataFrame, pd.DataFrame]] = {}
        node_values: list[float] = []
        edge_values: list[float] = []
        xmins: list[float] = []
        xmaxs: list[float] = []
        ymins: list[float] = []
        ymaxs: list[float] = []
        for _, row in rep_df.iterrows():
            label = str(row["plot_label"])
            model_name = str(row["model_name"])
            run_dir = harmonic_dir / str(row["run_name"])
            nodes, edges = prepare_ac_scaled_maps(run_dir, model_name)
            panels[(label, model_name)] = (nodes, edges)
            node_values.extend(pd.to_numeric(nodes["scaled_kirchhoff_error"], errors="coerce").dropna().tolist())
            edge_values.extend(pd.to_numeric(edges["scaled_flow_error"], errors="coerce").dropna().tolist())
            coords = nodes[["x_px", "y_px"]].apply(pd.to_numeric, errors="coerce")
            xmins.append(float(coords["x_px"].min()))
            xmaxs.append(float(coords["x_px"].max()))
            ymins.append(float(coords["y_px"].min()))
            ymaxs.append(float(coords["y_px"].max()))
        if not panels:
            continue
        xlim = (min(xmins), max(xmaxs))
        ylim = (min(ymins), max(ymaxs))
        node_vmax = robust_symmetric_limits(node_values)[1]
        edge_vmax = robust_symmetric_limits(edge_values)[1]

        for value_column, title_stub, filename_stub, vmax, is_node in (
            ("scaled_kirchhoff_error", "Scaled Kirchhoff error", "scaled_kirchhoff_error", node_vmax, True),
            ("scaled_flow_error", "Scaled flow error", "scaled_flow_error", edge_vmax, False),
        ):
            fig, axes = plt.subplots(
                len(labels),
                len(available_models),
                figsize=(2.85 * len(available_models), 2.35 * len(labels)),
                squeeze=False,
            )
            fig.subplots_adjust(left=0.08, right=0.84, top=0.84, bottom=0.04, wspace=0.08, hspace=0.10)
            artist = None
            for row_idx, label in enumerate(labels):
                for col_idx, model_name in enumerate(available_models):
                    ax = axes[row_idx, col_idx]
                    payload = panels.get((label, model_name))
                    if payload is None:
                        ax.axis("off")
                        continue
                    nodes, edges = payload
                    if is_node:
                        artist = draw_node_panel(
                            ax,
                            nodes,
                            edges,
                            value_column,
                            xlim,
                            ylim,
                            vmax,
                            marker_size=3.0,
                        )
                    else:
                        artist = draw_edge_panel(ax, nodes, edges, value_column, xlim, ylim, vmax)
                    if row_idx == 0:
                        ax.set_title(model_name.replace("_", " "), fontsize=11)
                    if col_idx == 0:
                        ax.set_ylabel(label, rotation=0, labelpad=20, va="center", ha="right", fontsize=11, fontweight="bold")
            if artist is not None:
                cax = fig.add_axes([0.87, 0.16, 0.025, 0.68])
                cbar = fig.colorbar(artist, cax=cax)
                if is_node:
                    cbar.set_label("|Kirchhoff residual| / local flow")
                else:
                    cbar.set_label("|Flow residual| / edge flow")
            harmonic_name = harmonic_dir.name
            fig.suptitle(f"{harmonic_name} representative mosaics: {title_stub.lower()}", fontsize=13, y=0.985)
            harmonic_output = output_dir / harmonic_name
            save_figure(fig, harmonic_output / f"{filename_stub}_mosaics.png", dpi)


def main() -> None:
    args = parse_args()
    configure_plot()
    plot_dc_scale_analysis(args.dc_representatives.expanduser().resolve(), args.dc_output_dir.expanduser().resolve(), int(args.dpi))
    plot_ac_scale_analysis(args.ac_root.expanduser().resolve(), args.ac_output_dir.expanduser().resolve(), int(args.dpi))


if __name__ == "__main__":
    main()
