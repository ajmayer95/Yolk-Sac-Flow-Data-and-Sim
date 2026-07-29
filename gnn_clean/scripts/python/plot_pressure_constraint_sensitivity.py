#!/usr/bin/env python
"""Plot Step 3 pressure-constraint sensitivity outputs from existing CSV files."""

from __future__ import annotations

import argparse
import math
import os
from itertools import product
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

from pressure_constraint_sensitivity_lib import (
    CONSTRAINT_DISPLAY,
    CONSTRAINT_ORDER,
    DEFAULT_OUTPUT_ROOT,
    representative_label,
)


REGIME_ORDER = (
    "flow_prioritized",
    "balanced",
    "conservation_prioritized",
    "correction_regularized",
)
REGIME_COLORS = {
    "flow_prioritized": "#1f77b4",
    "balanced": "#2ca02c",
    "conservation_prioritized": "#d62728",
    "correction_regularized": "#ff7f0e",
}
REGIME_LABELS = {
    "flow_prioritized": "Flow-prioritized",
    "balanced": "Balanced",
    "conservation_prioritized": "Conservation-prioritized",
    "correction_regularized": "Correction-regularized",
}
RANK_MARKERS = {1: "o", 2: "s", 3: "^", 4: "p"}
RANK_LABELS = {1: "Rank 1", 2: "Rank 2", 3: "Rank 3", 4: "Rank 4"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--dpi", type=int, default=300)
    return parser.parse_args()


def configure_plot() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.grid": True,
            "grid.color": "#dddddd",
            "grid.alpha": 0.35,
            "grid.linewidth": 0.8,
            "axes.titlesize": 13,
            "axes.labelsize": 11.5,
            "xtick.labelsize": 9.5,
            "ytick.labelsize": 9.5,
            "legend.fontsize": 9.5,
            "savefig.facecolor": "white",
        }
    )


def load_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if df.empty:
        raise ValueError(f"No rows found in {path}")
    return df


def numeric(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    result = df.copy()
    for column in columns:
        if column in result.columns:
            result[column] = pd.to_numeric(result[column], errors="coerce")
    return result


def node_df(run_dir: Path) -> pd.DataFrame:
    df = pd.read_csv(run_dir / "node_predictions.csv")
    return numeric(
        df,
        ["node_index", "x_px", "y_px", "pressure_pa", "kirchhoff_residual_nl_s"],
    )


def edge_df(run_dir: Path) -> pd.DataFrame:
    df = pd.read_csv(run_dir / "edge_predictions.csv")
    return numeric(df, ["source_index", "target_index", "delta_e", "flow_residual_nl_s"])


def aligned_pressure(df: pd.DataFrame, gauge_node_id: str | None) -> pd.Series:
    if "boundary_role" in df.columns:
        venous = df[df["boundary_role"].astype(str) == "venous"]
    else:
        venous = df.iloc[0:0].copy()
    if not venous.empty:
        ref = float(pd.to_numeric(venous["pressure_pa"], errors="coerce").mean())
    else:
        mask = df["node_id"].astype(str) == str(gauge_node_id or "")
        ref = float(pd.to_numeric(df.loc[mask, "pressure_pa"], errors="coerce").mean())
    return pd.to_numeric(df["pressure_pa"], errors="coerce") - ref


def representative_style_df(all_runs: pd.DataFrame) -> pd.DataFrame:
    gnn = all_runs[all_runs["model_family"] == "gnn"].copy()
    reps = (
        gnn[
            [
                "representative_label",
                "parent_step2_run_name",
                "selection_category",
                "selection_rank_within_regime",
                "lambda_q",
                "lambda_k",
                "lambda_delta",
            ]
        ]
        .drop_duplicates()
        .copy()
    )
    reps["representative_label"] = reps["representative_label"].fillna("")
    reps = reps[reps["representative_label"].astype(str) != ""].copy()
    if reps.empty:
        return reps
    reps["selection_rank_within_regime"] = pd.to_numeric(
        reps["selection_rank_within_regime"], errors="coerce"
    )
    reps["plot_label"] = reps["representative_label"]
    reps["marker"] = reps["selection_rank_within_regime"].map(RANK_MARKERS)
    reps["regime_color"] = reps["selection_category"].map(REGIME_COLORS)
    reps["regime_color_name"] = reps["selection_category"].map(REGIME_LABELS)
    reps = reps.sort_values(["selection_category", "selection_rank_within_regime", "plot_label"])
    return reps


def write_style_lookup(style_df: pd.DataFrame, output_dir: Path) -> None:
    out = style_df[
        [
            "plot_label",
            "parent_step2_run_name",
            "selection_category",
            "selection_rank_within_regime",
            "lambda_q",
            "lambda_k",
            "lambda_delta",
            "marker",
            "regime_color_name",
        ]
    ].rename(columns={"parent_step2_run_name": "run_name"})
    out.to_csv(output_dir / "step3_configuration_plot_labels.csv", index=False)


def style_for_row(row: pd.Series) -> dict[str, object]:
    regime = str(row.get("selection_category", ""))
    rank = int(float(row.get("selection_rank_within_regime", 1)))
    return {
        "color": REGIME_COLORS.get(regime, "#333333"),
        "marker": RANK_MARKERS.get(rank, "o"),
        "label": str(row.get("representative_label") or representative_label(regime, rank)),
        "regime": regime,
        "rank": rank,
    }


def legend_handles() -> list[Line2D]:
    handles: list[Line2D] = []
    for regime in REGIME_ORDER:
        handles.append(
            Line2D(
                [0],
                [0],
                color=REGIME_COLORS[regime],
                linewidth=2.2,
                marker="o",
                markersize=5,
                label=REGIME_LABELS[regime],
            )
        )
    for rank in sorted(RANK_MARKERS):
        handles.append(
            Line2D(
                [0],
                [0],
                linestyle="None",
                color="#303030",
                marker=RANK_MARKERS[rank],
                markersize=7,
                label=RANK_LABELS[rank],
            )
        )
    handles.append(
        Line2D(
            [0],
            [0],
            color="#222222",
            linewidth=2.2,
            marker="o",
            markersize=5,
            label="GNN",
        )
    )
    handles.append(
        Line2D(
            [0],
            [0],
            color="#8a8a8a",
            linewidth=1.8,
            linestyle="--",
            marker="D",
            markersize=5,
            label="Poiseuille baseline",
        )
    )
    return handles


def endpoint_offset(regime: str, rank: int) -> tuple[float, float]:
    offsets = {
        ("flow_prioritized", 1): (8, 8),
        ("flow_prioritized", 2): (8, -8),
        ("flow_prioritized", 3): (8, 12),
        ("flow_prioritized", 4): (8, -12),
        ("balanced", 1): (8, 8),
        ("balanced", 2): (8, -8),
        ("balanced", 3): (8, 12),
        ("balanced", 4): (8, -12),
        ("conservation_prioritized", 1): (8, 8),
        ("conservation_prioritized", 2): (8, -8),
        ("conservation_prioritized", 3): (8, 12),
        ("conservation_prioritized", 4): (8, -12),
        ("correction_regularized", 1): (8, 8),
        ("correction_regularized", 2): (8, -8),
        ("correction_regularized", 3): (8, 12),
        ("correction_regularized", 4): (8, -12),
    }
    return offsets.get((regime, rank), (8, 8))


def scalar_sensitivity(
    all_runs: pd.DataFrame,
    metric: str,
    ylabel: str,
    output_path: Path,
    title: str,
) -> None:
    fig, ax = plt.subplots(figsize=(9.2, 5.4), constrained_layout=True)
    order = list(CONSTRAINT_ORDER)
    x_lookup = {name: idx for idx, name in enumerate(order)}
    gnn = all_runs[all_runs["model_family"] == "gnn"].copy()
    pois = all_runs[all_runs["model_family"] == "poiseuille_baseline"].copy()

    gnn["selection_rank_within_regime"] = pd.to_numeric(
        gnn["selection_rank_within_regime"], errors="coerce"
    )
    for _, group_df in gnn.groupby("representative_label"):
        if group_df.empty:
            continue
        group_df = group_df.sort_values(
            by="pressure_constraint_type",
            key=lambda s: s.map(x_lookup),
        )
        style = style_for_row(group_df.iloc[0])
        xs = [x_lookup[name] for name in group_df["pressure_constraint_type"]]
        ys = pd.to_numeric(group_df[metric], errors="coerce")
        ax.plot(
            xs,
            ys,
            color=style["color"],
            linewidth=2.0,
            alpha=0.68,
            marker=style["marker"],
            markersize=6.5,
            markerfacecolor=style["color"],
            markeredgecolor="white",
            markeredgewidth=0.6,
            zorder=3,
        )
        finite = np.isfinite(ys.to_numpy(dtype=float))
        if finite.any():
            last_idx = max(idx for idx, flag in enumerate(finite) if flag)
            dx, dy = endpoint_offset(str(style["regime"]), int(style["rank"]))
            ax.annotate(
                str(style["label"]),
                xy=(xs[last_idx], ys.iloc[last_idx]),
                xytext=(dx, dy),
                textcoords="offset points",
                fontsize=9,
                color=style["color"],
                fontweight="bold",
                zorder=5,
            )

    for _, group_df in pois.groupby(["lambda_q", "lambda_k"]):
        group_df = group_df.sort_values(
            by="pressure_constraint_type",
            key=lambda s: s.map(x_lookup),
        )
        xs = [x_lookup[name] for name in group_df["pressure_constraint_type"]]
        ys = pd.to_numeric(group_df[metric], errors="coerce")
        ax.plot(
            xs,
            ys,
            color="#8a8a8a",
            alpha=0.42,
            linewidth=1.4,
            linestyle="--",
            marker="D",
            markersize=4.5,
            markerfacecolor="#b5b5b5",
            markeredgecolor="#777777",
            markeredgewidth=0.5,
            zorder=1,
        )

    ax.set_title(title)
    ax.set_xticks(range(len(order)), [CONSTRAINT_DISPLAY[name] for name in order], rotation=18, ha="right")
    ax.set_ylabel(ylabel)
    ax.set_xlabel("Pressure constraint")
    ax.grid(True, axis="y", alpha=0.3)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.legend(handles=legend_handles(), frameon=False, loc="center left", bbox_to_anchor=(1.01, 0.5))
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


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
    """Rotate 90 degrees clockwise, then mirror across the vertical axis.

    The transform is applied within the shared mosaic bounds so the subplot grid,
    titles, labels, and colorbars stay unchanged while each map panel is reoriented.
    """
    x_arr = np.asarray(x, dtype=float)
    y_arr = np.asarray(y, dtype=float)
    x_min, x_max = x_bounds
    y_min, y_max = y_bounds
    transformed_x = y_max - y_arr
    transformed_y = x_max - x_arr
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


def robust_symmetric_limits(values: list[float], percentile: float = 95.0) -> tuple[float, float] | None:
    finite = np.asarray([value for value in values if math.isfinite(value)], dtype=float)
    if finite.size == 0:
        return None
    bound = float(np.percentile(np.abs(finite), percentile))
    if not math.isfinite(bound) or bound <= 0.0:
        bound = float(np.max(np.abs(finite)))
    if not math.isfinite(bound) or bound <= 0.0:
        bound = 1.0
    return (-bound, bound)


def _bool_mask(df: pd.DataFrame, column: str) -> pd.Series:
    if column not in df.columns:
        return pd.Series(False, index=df.index)
    return df[column].astype(str).str.lower().isin({"true", "1"})


def pressure_maps_by_regime(all_runs: pd.DataFrame, output_dir: Path, dpi: int) -> None:
    gnn_runs = all_runs[all_runs["model_family"] == "gnn"].copy()
    for regime in REGIME_ORDER:
        regime_df = gnn_runs[gnn_runs["selection_category"] == regime].copy()
        if regime_df.empty:
            continue
        labels = sorted(
            regime_df["representative_label"].dropna().unique().tolist(),
            key=lambda x: (str(x)[0], int(str(x)[1:]) if str(x)[1:].isdigit() else 0),
        )
        n_rows = len(labels)
        fig, axes = plt.subplots(
            n_rows,
            len(CONSTRAINT_ORDER),
            figsize=(3.0 * len(CONSTRAINT_ORDER), 2.65 * n_rows),
            constrained_layout=True,
            squeeze=False,
        )
        all_vals: list[float] = []
        panels: dict[tuple[str, str], tuple[pd.DataFrame, pd.DataFrame, pd.Series]] = {}
        xmins: list[float] = []
        xmaxs: list[float] = []
        ymins: list[float] = []
        ymaxs: list[float] = []
        for rep_label in labels:
            rep_df = regime_df[regime_df["representative_label"] == rep_label]
            run_map = {row["pressure_constraint_type"]: row for _, row in rep_df.iterrows()}
            if any(name not in run_map for name in CONSTRAINT_ORDER):
                continue
            for constraint in CONSTRAINT_ORDER:
                row = run_map[constraint]
                nodes = node_df(Path(row["output_dir"]))
                edges = edge_df(Path(row["output_dir"]))
                aligned = aligned_pressure(nodes, row.get("gauge_node_id", ""))
                panels[(rep_label, constraint)] = (nodes, edges, aligned)
                finite_vals = aligned[np.isfinite(aligned)]
                all_vals.extend(finite_vals.tolist())
                coords = nodes[["x_px", "y_px"]].apply(pd.to_numeric, errors="coerce")
                if not coords.empty:
                    xmins.append(coords["x_px"].min())
                    xmaxs.append(coords["x_px"].max())
                    ymins.append(coords["y_px"].min())
                    ymaxs.append(coords["y_px"].max())
        if not all_vals:
            plt.close(fig)
            continue
        vmin = float(np.min(all_vals))
        vmax = float(np.max(all_vals))
        xlim = (min(xmins), max(xmaxs))
        ylim = (min(ymins), max(ymaxs))
        scatter = None
        transformed_xlim = (0.0, ylim[1] - ylim[0])
        transformed_ylim = (0.0, xlim[1] - xlim[0])
        for row_idx, rep_label in enumerate(labels):
            for col_idx, constraint in enumerate(CONSTRAINT_ORDER):
                ax = axes[row_idx, col_idx]
                payload = panels.get((rep_label, constraint))
                if payload is None:
                    ax.axis("off")
                    continue
                nodes, edges, aligned = payload
                segments = transform_segment_collection(build_segments(nodes, edges), xlim, ylim)
                if segments:
                    ax.add_collection(
                        LineCollection(segments, colors="#cfcfcf", linewidths=0.6, zorder=1)
                    )
                node_x, node_y = transform_mosaic_coords(nodes["x_px"], nodes["y_px"], xlim, ylim)
                scatter = ax.scatter(
                    node_x,
                    node_y,
                    c=aligned,
                    cmap="coolwarm",
                    vmin=vmin,
                    vmax=vmax,
                    s=16,
                    zorder=2,
                )
                arterial = nodes[_bool_mask(nodes, "is_arterial")]
                venous = nodes[_bool_mask(nodes, "is_venous")]
                if not arterial.empty:
                    arterial_x, arterial_y = transform_mosaic_coords(
                        arterial["x_px"], arterial["y_px"], xlim, ylim
                    )
                    ax.scatter(
                        arterial_x,
                        arterial_y,
                        marker="^",
                        color="black",
                        s=22,
                        zorder=3,
                    )
                if not venous.empty:
                    venous_x, venous_y = transform_mosaic_coords(
                        venous["x_px"], venous["y_px"], xlim, ylim
                    )
                    ax.scatter(
                        venous_x,
                        venous_y,
                        marker="s",
                        color="black",
                        s=18,
                        zorder=3,
                    )
                ax.set_xlim(transformed_xlim)
                ax.set_ylim(transformed_ylim)
                ax.set_aspect("equal")
                ax.set_xticks([])
                ax.set_yticks([])
                for spine in ax.spines.values():
                    spine.set_visible(False)
                if row_idx == 0:
                    ax.set_title(CONSTRAINT_DISPLAY[constraint], pad=8)
                if col_idx == 0:
                    ax.set_ylabel(rep_label, rotation=0, labelpad=22, va="center", ha="right", fontsize=11, fontweight="bold")
        if scatter is not None:
            cbar = fig.colorbar(scatter, ax=axes.ravel().tolist(), shrink=0.92, pad=0.02)
            cbar.set_label("Aligned pressure (Pa)")
        fig.suptitle(f"Pressure maps: {REGIME_LABELS[regime]}", fontsize=14)
        fig.savefig(output_dir / f"pressure_maps_{regime}.png", dpi=dpi, bbox_inches="tight")
        plt.close(fig)


def correction_maps_by_regime(all_runs: pd.DataFrame, output_dir: Path, dpi: int) -> None:
    gnn_runs = all_runs[all_runs["model_family"] == "gnn"].copy()
    for regime in REGIME_ORDER:
        regime_df = gnn_runs[gnn_runs["selection_category"] == regime].copy()
        if regime_df.empty:
            continue
        labels = sorted(
            regime_df["representative_label"].dropna().unique().tolist(),
            key=lambda x: (str(x)[0], int(str(x)[1:]) if str(x)[1:].isdigit() else 0),
        )
        n_rows = len(labels)
        fig, axes = plt.subplots(
            n_rows,
            len(CONSTRAINT_ORDER),
            figsize=(3.0 * len(CONSTRAINT_ORDER), 2.65 * n_rows),
            constrained_layout=True,
            squeeze=False,
        )
        panels: dict[tuple[str, str], tuple[pd.DataFrame, pd.DataFrame]] = {}
        xmins: list[float] = []
        xmaxs: list[float] = []
        ymins: list[float] = []
        ymaxs: list[float] = []
        for rep_label in labels:
            rep_df = regime_df[regime_df["representative_label"] == rep_label]
            run_map = {row["pressure_constraint_type"]: row for _, row in rep_df.iterrows()}
            if any(name not in run_map for name in CONSTRAINT_ORDER):
                continue
            for constraint in CONSTRAINT_ORDER:
                row = run_map[constraint]
                nodes = node_df(Path(row["output_dir"]))
                edges = edge_df(Path(row["output_dir"]))
                panels[(rep_label, constraint)] = (nodes, edges)
                coords = nodes[["x_px", "y_px"]].apply(pd.to_numeric, errors="coerce")
                if not coords.empty:
                    xmins.append(coords["x_px"].min())
                    xmaxs.append(coords["x_px"].max())
                    ymins.append(coords["y_px"].min())
                    ymaxs.append(coords["y_px"].max())
        xlim = (min(xmins), max(xmaxs))
        ylim = (min(ymins), max(ymaxs))
        transformed_xlim = (0.0, ylim[1] - ylim[0])
        transformed_ylim = (0.0, xlim[1] - xlim[0])
        collection = None
        for row_idx, rep_label in enumerate(labels):
            for col_idx, constraint in enumerate(CONSTRAINT_ORDER):
                ax = axes[row_idx, col_idx]
                payload = panels.get((rep_label, constraint))
                if payload is None:
                    ax.axis("off")
                    continue
                nodes, edges = payload
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
                    tx, ty = transform_mosaic_coords(
                        np.array([a[0], b[0]], dtype=float),
                        np.array([a[1], b[1]], dtype=float),
                        xlim,
                        ylim,
                    )
                    segments.append(np.column_stack([tx, ty]))
                    values.append(float(edge_row["delta_e"]))
                if segments:
                    collection = LineCollection(
                        segments,
                        cmap="coolwarm",
                        linewidths=1.15,
                        zorder=2,
                    )
                    collection.set_array(np.asarray(values, dtype=float))
                    collection.set_clim(-0.5, 0.5)
                    ax.add_collection(collection)
                arterial = nodes[_bool_mask(nodes, "is_arterial")]
                venous = nodes[_bool_mask(nodes, "is_venous")]
                if not arterial.empty:
                    arterial_x, arterial_y = transform_mosaic_coords(
                        arterial["x_px"], arterial["y_px"], xlim, ylim
                    )
                    ax.scatter(arterial_x, arterial_y, marker="^", color="black", s=18, zorder=3)
                if not venous.empty:
                    venous_x, venous_y = transform_mosaic_coords(
                        venous["x_px"], venous["y_px"], xlim, ylim
                    )
                    ax.scatter(venous_x, venous_y, marker="s", color="black", s=16, zorder=3)
                ax.set_xlim(transformed_xlim)
                ax.set_ylim(transformed_ylim)
                ax.set_aspect("equal")
                ax.set_xticks([])
                ax.set_yticks([])
                for spine in ax.spines.values():
                    spine.set_visible(False)
                if row_idx == 0:
                    ax.set_title(CONSTRAINT_DISPLAY[constraint], pad=8)
                if col_idx == 0:
                    ax.set_ylabel(rep_label, rotation=0, labelpad=22, va="center", ha="right", fontsize=11, fontweight="bold")
        if collection is not None:
            cbar = fig.colorbar(collection, ax=axes.ravel().tolist(), shrink=0.92, pad=0.02)
            cbar.set_label("delta_e")
        fig.suptitle(f"Correction maps: {REGIME_LABELS[regime]}", fontsize=14)
        fig.savefig(output_dir / f"correction_maps_{regime}.png", dpi=dpi, bbox_inches="tight")
        plt.close(fig)


def flow_error_maps_by_regime(all_runs: pd.DataFrame, output_dir: Path, dpi: int) -> None:
    gnn_runs = all_runs[all_runs["model_family"] == "gnn"].copy()
    for regime in REGIME_ORDER:
        regime_df = gnn_runs[gnn_runs["selection_category"] == regime].copy()
        if regime_df.empty:
            continue
        labels = sorted(
            regime_df["representative_label"].dropna().unique().tolist(),
            key=lambda x: (str(x)[0], int(str(x)[1:]) if str(x)[1:].isdigit() else 0),
        )
        n_rows = len(labels)
        fig, axes = plt.subplots(
            n_rows,
            len(CONSTRAINT_ORDER),
            figsize=(3.0 * len(CONSTRAINT_ORDER), 2.65 * n_rows),
            constrained_layout=True,
            squeeze=False,
        )
        panels: dict[tuple[str, str], tuple[pd.DataFrame, pd.DataFrame]] = {}
        xmins: list[float] = []
        xmaxs: list[float] = []
        ymins: list[float] = []
        ymaxs: list[float] = []
        all_abs_vals: list[float] = []
        for rep_label in labels:
            rep_df = regime_df[regime_df["representative_label"] == rep_label]
            run_map = {row["pressure_constraint_type"]: row for _, row in rep_df.iterrows()}
            if any(name not in run_map for name in CONSTRAINT_ORDER):
                continue
            for constraint in CONSTRAINT_ORDER:
                row = run_map[constraint]
                nodes = node_df(Path(row["output_dir"]))
                edges = edge_df(Path(row["output_dir"]))
                panels[(rep_label, constraint)] = (nodes, edges)
                coords = nodes[["x_px", "y_px"]].apply(pd.to_numeric, errors="coerce")
                if not coords.empty:
                    xmins.append(coords["x_px"].min())
                    xmaxs.append(coords["x_px"].max())
                    ymins.append(coords["y_px"].min())
                    ymaxs.append(coords["y_px"].max())
                residual = pd.to_numeric(edges.get("flow_residual_nl_s"), errors="coerce")
                finite = residual[np.isfinite(residual)]
                if not finite.empty:
                    all_abs_vals.extend(np.abs(finite.to_numpy(dtype=float)).tolist())
        if not all_abs_vals:
            plt.close(fig)
            continue
        limits = robust_symmetric_limits(all_abs_vals, percentile=95.0)
        if limits is None:
            plt.close(fig)
            continue
        vmin, vmax = limits
        xlim = (min(xmins), max(xmaxs))
        ylim = (min(ymins), max(ymaxs))
        transformed_xlim = (0.0, ylim[1] - ylim[0])
        transformed_ylim = (0.0, xlim[1] - xlim[0])
        collection = None
        for row_idx, rep_label in enumerate(labels):
            for col_idx, constraint in enumerate(CONSTRAINT_ORDER):
                ax = axes[row_idx, col_idx]
                payload = panels.get((rep_label, constraint))
                if payload is None:
                    ax.axis("off")
                    continue
                nodes, edges = payload
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
                    tx, ty = transform_mosaic_coords(
                        np.array([a[0], b[0]], dtype=float),
                        np.array([a[1], b[1]], dtype=float),
                        xlim,
                        ylim,
                    )
                    segments.append(np.column_stack([tx, ty]))
                    values.append(value)
                if segments:
                    collection = LineCollection(
                        segments,
                        cmap="coolwarm",
                        linewidths=1.15,
                        zorder=2,
                    )
                    collection.set_array(np.asarray(values, dtype=float))
                    collection.set_clim(vmin, vmax)
                    ax.add_collection(collection)
                arterial = nodes[_bool_mask(nodes, "is_arterial")]
                venous = nodes[_bool_mask(nodes, "is_venous")]
                if not arterial.empty:
                    arterial_x, arterial_y = transform_mosaic_coords(
                        arterial["x_px"], arterial["y_px"], xlim, ylim
                    )
                    ax.scatter(arterial_x, arterial_y, marker="^", color="black", s=18, zorder=3)
                if not venous.empty:
                    venous_x, venous_y = transform_mosaic_coords(
                        venous["x_px"], venous["y_px"], xlim, ylim
                    )
                    ax.scatter(venous_x, venous_y, marker="s", color="black", s=16, zorder=3)
                ax.set_xlim(transformed_xlim)
                ax.set_ylim(transformed_ylim)
                ax.set_aspect("equal")
                ax.set_xticks([])
                ax.set_yticks([])
                for spine in ax.spines.values():
                    spine.set_visible(False)
                if row_idx == 0:
                    ax.set_title(CONSTRAINT_DISPLAY[constraint], pad=8)
                if col_idx == 0:
                    ax.set_ylabel(rep_label, rotation=0, labelpad=22, va="center", ha="right", fontsize=11, fontweight="bold")
        if collection is not None:
            cbar = fig.colorbar(collection, ax=axes.ravel().tolist(), shrink=0.92, pad=0.02)
            cbar.set_label("Flow residual (nL/s)")
        fig.suptitle(f"Flow-error maps: {REGIME_LABELS[regime]}", fontsize=14)
        fig.savefig(output_dir / f"flow_error_maps_{regime}.png", dpi=dpi, bbox_inches="tight")
        plt.close(fig)


def kirchhoff_error_maps_by_regime(all_runs: pd.DataFrame, output_dir: Path, dpi: int) -> None:
    gnn_runs = all_runs[all_runs["model_family"] == "gnn"].copy()
    for regime in REGIME_ORDER:
        regime_df = gnn_runs[gnn_runs["selection_category"] == regime].copy()
        if regime_df.empty:
            continue
        labels = sorted(
            regime_df["representative_label"].dropna().unique().tolist(),
            key=lambda x: (str(x)[0], int(str(x)[1:]) if str(x)[1:].isdigit() else 0),
        )
        n_rows = len(labels)
        fig, axes = plt.subplots(
            n_rows,
            len(CONSTRAINT_ORDER),
            figsize=(3.0 * len(CONSTRAINT_ORDER), 2.65 * n_rows),
            constrained_layout=True,
            squeeze=False,
        )
        panels: dict[tuple[str, str], tuple[pd.DataFrame, pd.DataFrame, pd.Series]] = {}
        xmins: list[float] = []
        xmaxs: list[float] = []
        ymins: list[float] = []
        ymaxs: list[float] = []
        all_abs_vals: list[float] = []
        for rep_label in labels:
            rep_df = regime_df[regime_df["representative_label"] == rep_label]
            run_map = {row["pressure_constraint_type"]: row for _, row in rep_df.iterrows()}
            if any(name not in run_map for name in CONSTRAINT_ORDER):
                continue
            for constraint in CONSTRAINT_ORDER:
                row = run_map[constraint]
                nodes = node_df(Path(row["output_dir"]))
                edges = edge_df(Path(row["output_dir"]))
                residual = pd.to_numeric(nodes.get("kirchhoff_residual_nl_s"), errors="coerce")
                panels[(rep_label, constraint)] = (nodes, edges, residual)
                finite = residual[np.isfinite(residual)]
                if not finite.empty:
                    all_abs_vals.extend(np.abs(finite.to_numpy(dtype=float)).tolist())
                coords = nodes[["x_px", "y_px"]].apply(pd.to_numeric, errors="coerce")
                if not coords.empty:
                    xmins.append(coords["x_px"].min())
                    xmaxs.append(coords["x_px"].max())
                    ymins.append(coords["y_px"].min())
                    ymaxs.append(coords["y_px"].max())
        if not all_abs_vals:
            plt.close(fig)
            continue
        limits = robust_symmetric_limits(all_abs_vals, percentile=95.0)
        if limits is None:
            plt.close(fig)
            continue
        vmin, vmax = limits
        xlim = (min(xmins), max(xmaxs))
        ylim = (min(ymins), max(ymaxs))
        transformed_xlim = (0.0, ylim[1] - ylim[0])
        transformed_ylim = (0.0, xlim[1] - xlim[0])
        scatter = None
        for row_idx, rep_label in enumerate(labels):
            for col_idx, constraint in enumerate(CONSTRAINT_ORDER):
                ax = axes[row_idx, col_idx]
                payload = panels.get((rep_label, constraint))
                if payload is None:
                    ax.axis("off")
                    continue
                nodes, edges, residual = payload
                segments = transform_segment_collection(build_segments(nodes, edges), xlim, ylim)
                if segments:
                    ax.add_collection(
                        LineCollection(segments, colors="#cfcfcf", linewidths=0.6, zorder=1)
                    )
                node_x, node_y = transform_mosaic_coords(nodes["x_px"], nodes["y_px"], xlim, ylim)
                scatter = ax.scatter(
                    node_x,
                    node_y,
                    c=residual,
                    cmap="coolwarm",
                    vmin=vmin,
                    vmax=vmax,
                    s=16,
                    zorder=2,
                )
                arterial = nodes[_bool_mask(nodes, "is_arterial")]
                venous = nodes[_bool_mask(nodes, "is_venous")]
                if not arterial.empty:
                    arterial_x, arterial_y = transform_mosaic_coords(
                        arterial["x_px"], arterial["y_px"], xlim, ylim
                    )
                    ax.scatter(
                        arterial_x,
                        arterial_y,
                        marker="^",
                        color="black",
                        s=22,
                        zorder=3,
                    )
                if not venous.empty:
                    venous_x, venous_y = transform_mosaic_coords(
                        venous["x_px"], venous["y_px"], xlim, ylim
                    )
                    ax.scatter(
                        venous_x,
                        venous_y,
                        marker="s",
                        color="black",
                        s=18,
                        zorder=3,
                    )
                ax.set_xlim(transformed_xlim)
                ax.set_ylim(transformed_ylim)
                ax.set_aspect("equal")
                ax.set_xticks([])
                ax.set_yticks([])
                for spine in ax.spines.values():
                    spine.set_visible(False)
                if row_idx == 0:
                    ax.set_title(CONSTRAINT_DISPLAY[constraint], pad=8)
                if col_idx == 0:
                    ax.set_ylabel(rep_label, rotation=0, labelpad=22, va="center", ha="right", fontsize=11, fontweight="bold")
        if scatter is not None:
            cbar = fig.colorbar(scatter, ax=axes.ravel().tolist(), shrink=0.92, pad=0.02)
            cbar.set_label("Kirchhoff residual (nL/s)")
        fig.suptitle(f"Kirchhoff-error maps: {REGIME_LABELS[regime]}", fontsize=14)
        fig.savefig(output_dir / f"kirchhoff_error_maps_{regime}.png", dpi=dpi, bbox_inches="tight")
        plt.close(fig)


def matrix_heatmap(
    pairwise_df: pd.DataFrame,
    value_col: str,
    output_path: Path,
    title: str,
) -> None:
    constraints = list(CONSTRAINT_ORDER)
    mat = np.eye(len(constraints), dtype=float)
    for i, left in enumerate(constraints):
        for j, right in enumerate(constraints):
            if i == j:
                continue
            mask = (
                ((pairwise_df["constraint_left"] == left) & (pairwise_df["constraint_right"] == right))
                | ((pairwise_df["constraint_left"] == right) & (pairwise_df["constraint_right"] == left))
            )
            values = pd.to_numeric(pairwise_df.loc[mask, value_col], errors="coerce").dropna()
            mat[i, j] = float(values.median()) if not values.empty else float("nan")
    fig, ax = plt.subplots(figsize=(6.2, 5.4), constrained_layout=True)
    im = ax.imshow(mat, cmap="coolwarm", vmin=-1.0, vmax=1.0)
    ax.set_xticks(range(len(constraints)), [CONSTRAINT_DISPLAY[c] for c in constraints], rotation=28, ha="right")
    ax.set_yticks(range(len(constraints)), [CONSTRAINT_DISPLAY[c] for c in constraints])
    ax.set_title(title)
    for i, j in product(range(len(constraints)), repeat=2):
        value = mat[i, j]
        if math.isfinite(value):
            ax.text(j, i, f"{value:.2f}", ha="center", va="center", color="black", fontsize=9)
    fig.colorbar(im, ax=ax, shrink=0.88)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def regime_matrix_heatmaps(
    pairwise_df: pd.DataFrame,
    all_runs: pd.DataFrame,
    value_col: str,
    prefix: str,
    title_prefix: str,
    output_dir: Path,
) -> None:
    label_to_regime = (
        all_runs[all_runs["model_family"] == "gnn"][["representative_label", "selection_category"]]
        .drop_duplicates()
        .set_index("representative_label")["selection_category"]
        .to_dict()
    )
    df = pairwise_df.copy()
    if "representative_label" in df.columns:
        df["selection_category"] = df["representative_label"].map(label_to_regime)
    for regime in REGIME_ORDER:
        subset = df[df.get("selection_category", "") == regime].copy()
        if subset.empty:
            continue
        matrix_heatmap(
            subset,
            value_col,
            output_dir / f"{prefix}_{regime}.png",
            f"{title_prefix}: {REGIME_LABELS[regime]}",
        )


def main() -> None:
    args = parse_args()
    configure_plot()
    input_root = args.input_root.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve() if args.output_dir else input_root / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)

    all_runs = load_csv(input_root / "pressure_constraint_all_runs.csv")
    pressure_pairwise = load_csv(input_root / "pressure_field_pairwise_metrics.csv")
    correction_pairwise = load_csv(input_root / "correction_field_pairwise_metrics.csv")
    all_runs = numeric(
        all_runs,
        [
            "lambda_q",
            "lambda_k",
            "lambda_delta",
            "flow_rmse_nl_s",
            "kirchhoff_rms_per_internal_node_nl_s",
            "pressure_range_pa",
            "boundary_residual_rms_pa",
            "selection_rank_within_regime",
        ],
    )
    if "representative_label" not in all_runs.columns:
        all_runs["representative_label"] = [
            representative_label(row.get("selection_category", ""), row.get("selection_rank_within_regime", 0))
            for _, row in all_runs.iterrows()
        ]
    style_df = representative_style_df(all_runs)
    write_style_lookup(style_df, output_dir)

    pressure_maps_by_regime(all_runs, output_dir, args.dpi)
    flow_error_maps_by_regime(all_runs, output_dir, args.dpi)
    kirchhoff_error_maps_by_regime(all_runs, output_dir, args.dpi)
    correction_maps_by_regime(all_runs, output_dir, args.dpi)

    pressure_gnn = pressure_pairwise[pressure_pairwise["model_family"] == "gnn"].copy() if "model_family" in pressure_pairwise.columns else pressure_pairwise.copy()
    matrix_heatmap(
        pressure_gnn,
        "pressure_pearson_aligned",
        output_dir / "pressure_correlation_matrix_median.png",
        "Median aligned-pressure correlation",
    )
    matrix_heatmap(
        correction_pairwise,
        "delta_pearson",
        output_dir / "correction_correlation_matrix_median.png",
        "Median delta correlation",
    )
    regime_matrix_heatmaps(
        pressure_gnn,
        all_runs,
        "pressure_pearson_aligned",
        "pressure_correlation_matrix",
        "Aligned-pressure correlation",
        output_dir,
    )
    regime_matrix_heatmaps(
        correction_pairwise,
        all_runs,
        "delta_pearson",
        "correction_correlation_matrix",
        "Delta correlation",
        output_dir,
    )

    scalar_sensitivity(
        all_runs,
        "flow_rmse_nl_s",
        "Flow RMSE (nL/s)",
        output_dir / "flow_rmse_by_pressure_constraint.png",
        "Flow error sensitivity to pressure constraints",
    )
    scalar_sensitivity(
        all_runs,
        "kirchhoff_rms_per_internal_node_nl_s",
        "Kirchhoff RMS per internal node (nL/s)",
        output_dir / "kirchhoff_rms_by_pressure_constraint.png",
        "Conservation sensitivity to pressure constraints",
    )
    scalar_sensitivity(
        all_runs,
        "pressure_range_pa",
        "Pressure range (Pa)",
        output_dir / "pressure_range_by_pressure_constraint.png",
        "Pressure range sensitivity to pressure constraints",
    )
    scalar_sensitivity(
        all_runs,
        "boundary_residual_rms_pa",
        "Pressure residual (Pa)",
        output_dir / "pressure_residual_by_pressure_constraint.png",
        "Pressure residual sensitivity to pressure constraints",
    )


if __name__ == "__main__":
    main()
