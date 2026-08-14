#!/usr/bin/env python
"""Plot Step 5 radius-refinement outputs."""

from __future__ import annotations

import argparse
import math
import os
import sys
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import TwoSlopeNorm
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from radius_correction_experiment_lib import (
    CONDITION_DISPLAY,
    DEFAULT_OUTPUT_ROOT,
    SHARED_CONDITIONS,
    STRATEGY_DISPLAY,
    STRATEGY_ORDER,
    condition_dir,
    normalize_bool,
    safe_float,
    shared_condition_dir,
)

MAIN_CONDITIONS = ("p_original", "p_corrected", "g_fixed", "g_retrained")
CONDITION_COLORS = {
    "p_original": "#6C7A89",
    "p_corrected": "#2A9D8F",
    "g_fixed": "#E9C46A",
    "g_retrained": "#E76F51",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--output-dir", type=Path, default=None)
    return parser.parse_args()


def configure_plot() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.grid": True,
            "grid.color": "#DDDDDD",
            "grid.alpha": 0.35,
            "grid.linewidth": 0.8,
            "axes.titlesize": 12,
            "axes.labelsize": 11,
            "xtick.labelsize": 9.5,
            "ytick.labelsize": 9.5,
            "legend.fontsize": 9,
            "savefig.facecolor": "white",
        }
    )


def save(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def condition_path(input_root: Path, strategy_name: str, condition_name: str, filename: str) -> Path:
    base = (
        shared_condition_dir(input_root, condition_name)
        if condition_name in SHARED_CONDITIONS
        else condition_dir(input_root, strategy_name, condition_name)
    )
    return base / filename


def load_summary(input_root: Path) -> pd.DataFrame:
    return pd.read_csv(input_root / "radius_refinement_summary.csv")


def load_edge_csv(input_root: Path, strategy_name: str, condition_name: str) -> pd.DataFrame:
    df = pd.read_csv(condition_path(input_root, strategy_name, condition_name, "edge_predictions.csv")).copy()
    if "source_node" not in df.columns and "source" in df.columns:
        df["source_node"] = df["source"]
    if "target_node" not in df.columns and "target" in df.columns:
        df["target_node"] = df["target"]
    numeric_columns = (
        "edge_id",
        "source_index",
        "target_index",
        "observed_flow_nl_s",
        "predicted_flow_nl_s",
        "flow_residual_nl_s",
        "delta_e",
        "original_delta",
        "fixed_weight_delta",
        "retrained_delta",
        "radius_percent_change",
        "snr",
        "tile_id",
    )
    for column in numeric_columns:
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors="coerce")
    for column in ("valid_observed_flow", "selected_for_radius_correction", "sign_flip"):
        if column in df.columns:
            df[column] = df[column].map(lambda value: normalize_bool(value, False))
    return df


def load_node_csv(input_root: Path, strategy_name: str, condition_name: str) -> pd.DataFrame:
    df = pd.read_csv(condition_path(input_root, strategy_name, condition_name, "node_predictions.csv")).copy()
    numeric_columns = (
        "node_index",
        "x_px",
        "y_px",
        "pressure_pa",
        "kirchhoff_residual_nl_s",
        "boundary_injection_nl_s",
        "predicted_net_flow_nl_s",
    )
    for column in numeric_columns:
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors="coerce")
    for column in ("is_internal", "is_boundary", "is_arterial", "is_venous"):
        if column in df.columns:
            df[column] = df[column].map(lambda value: normalize_bool(value, False))
    return df


def reference_nodes(input_root: Path) -> pd.DataFrame:
    df = load_node_csv(input_root, STRATEGY_ORDER[0], "p_original")
    return df[["node_index", "x_px", "y_px"]].copy()


def ensure_coords(node_df: pd.DataFrame, ref_df: pd.DataFrame) -> pd.DataFrame:
    if {"x_px", "y_px"} <= set(node_df.columns):
        if not (node_df["x_px"].isna().all() or node_df["y_px"].isna().all()):
            return node_df
    return node_df.drop(columns=["x_px", "y_px"], errors="ignore").merge(
        ref_df,
        on="node_index",
        how="left",
    )


def edge_segments_and_values(
    node_df: pd.DataFrame,
    edge_df: pd.DataFrame,
    value_column: str,
) -> tuple[np.ndarray, np.ndarray]:
    lookup = node_df.set_index("node_index")[["x_px", "y_px"]]
    segments: list[list[list[float]]] = []
    values: list[float] = []
    for _, row in edge_df.iterrows():
        try:
            src = int(row["source_index"])
            dst = int(row["target_index"])
        except Exception:
            continue
        if src not in lookup.index or dst not in lookup.index:
            continue
        x0, y0 = float(lookup.at[src, "x_px"]), float(lookup.at[src, "y_px"])
        x1, y1 = float(lookup.at[dst, "x_px"]), float(lookup.at[dst, "y_px"])
        value = safe_float(row.get(value_column))
        if not all(math.isfinite(v) for v in (x0, y0, x1, y1, value)):
            continue
        segments.append([[x0, y0], [x1, y1]])
        values.append(value)
    if not segments:
        return np.zeros((0, 2, 2), dtype=np.float64), np.zeros(0, dtype=np.float64)
    return np.asarray(segments, dtype=np.float64), np.asarray(values, dtype=np.float64)


def compute_flow_rmse_from_edges(edge_df: pd.DataFrame) -> float:
    if "valid_observed_flow" not in edge_df.columns or "flow_residual_nl_s" not in edge_df.columns:
        return float("nan")
    residual = pd.to_numeric(edge_df["flow_residual_nl_s"], errors="coerce").to_numpy(dtype=np.float64)
    valid = edge_df["valid_observed_flow"].to_numpy(dtype=bool)
    residual = residual[valid & np.isfinite(residual)]
    return float(np.sqrt(np.mean(residual**2))) if residual.size else float("nan")


def compute_kirchhoff_rms_from_nodes(node_df: pd.DataFrame) -> float:
    if "kirchhoff_residual_nl_s" not in node_df.columns:
        return float("nan")
    if "is_internal" in node_df.columns:
        use = node_df[node_df["is_internal"].astype(bool)].copy()
    else:
        use = node_df.copy()
    values = pd.to_numeric(use["kirchhoff_residual_nl_s"], errors="coerce").to_numpy(dtype=np.float64)
    values = values[np.isfinite(values)]
    return float(np.sqrt(np.mean(values**2))) if values.size else float("nan")


def compute_pressure_range_from_nodes(node_df: pd.DataFrame) -> float:
    values = pd.to_numeric(node_df.get("pressure_pa"), errors="coerce").to_numpy(dtype=np.float64)
    values = values[np.isfinite(values)]
    return float(np.max(values) - np.min(values)) if values.size else float("nan")


def compute_sign_flip_fraction(edge_df: pd.DataFrame) -> float:
    if "sign_flip" not in edge_df.columns:
        return float("nan")
    if "valid_observed_flow" in edge_df.columns:
        use = edge_df[edge_df["valid_observed_flow"].astype(bool)].copy()
    else:
        use = edge_df.copy()
    if use.empty:
        return float("nan")
    return float(use["sign_flip"].astype(bool).mean())


def hydrate_summary(
    summary_row: pd.Series,
    edge_df: pd.DataFrame,
    node_df: pd.DataFrame,
) -> dict[str, object]:
    summary = dict(summary_row)
    if not math.isfinite(safe_float(summary.get("flow_rmse_nl_s"))):
        summary["flow_rmse_nl_s"] = compute_flow_rmse_from_edges(edge_df)
    if not math.isfinite(safe_float(summary.get("kirchhoff_rms_per_internal_node_nl_s"))):
        summary["kirchhoff_rms_per_internal_node_nl_s"] = compute_kirchhoff_rms_from_nodes(node_df)
    if not math.isfinite(safe_float(summary.get("pressure_range_pa"))):
        summary["pressure_range_pa"] = compute_pressure_range_from_nodes(node_df)
    if not math.isfinite(safe_float(summary.get("sign_flip_fraction"))):
        summary["sign_flip_fraction"] = compute_sign_flip_fraction(edge_df)
    return summary


def metric_bar_plot(
    strategy_name: str,
    summaries: dict[str, dict[str, object]],
    metric: str,
    ylabel: str,
    title: str,
    output_path: Path,
    zero_for_poiseuille: bool = False,
) -> None:
    labels = [CONDITION_DISPLAY[name] for name in MAIN_CONDITIONS]
    values = []
    for condition_name in MAIN_CONDITIONS:
        value = safe_float(summaries[condition_name].get(metric))
        if zero_for_poiseuille and condition_name.startswith("p_"):
            value = 0.0
        values.append(value)
    fig, ax = plt.subplots(figsize=(7.0, 4.2), constrained_layout=True)
    bars = ax.bar(
        labels,
        values,
        color=[CONDITION_COLORS[name] for name in MAIN_CONDITIONS],
        edgecolor="#333333",
        linewidth=0.8,
    )
    if zero_for_poiseuille:
        for idx, condition_name in enumerate(MAIN_CONDITIONS):
            if condition_name.startswith("p_"):
                bars[idx].set_hatch("//")
    ax.set_ylabel(ylabel)
    ax.set_title(f"{title}: {STRATEGY_DISPLAY[strategy_name]}")
    ax.grid(True, axis="y", alpha=0.3)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    save(fig, output_path)


def boxplot_abs_metric(
    strategy_name: str,
    condition_payloads: dict[str, dict[str, object]],
    source: str,
    column: str,
    ylabel: str,
    title: str,
    output_path: Path,
) -> None:
    data = []
    labels = []
    for condition_name in MAIN_CONDITIONS:
        table = condition_payloads[condition_name][source]
        if source == "node" and column == "kirchhoff_residual_nl_s" and "is_internal" in table.columns:
            table = table[table["is_internal"].astype(bool)].copy()
        values = pd.to_numeric(table.get(column), errors="coerce").abs().dropna().to_numpy()
        data.append(values)
        labels.append(CONDITION_DISPLAY[condition_name])
    fig, ax = plt.subplots(figsize=(7.4, 4.4), constrained_layout=True)
    bp = ax.boxplot(data, tick_labels=labels, showfliers=False, patch_artist=True)
    for patch, condition_name in zip(bp["boxes"], MAIN_CONDITIONS):
        patch.set_facecolor(CONDITION_COLORS[condition_name])
        patch.set_alpha(0.85)
    ax.set_ylabel(ylabel)
    ax.set_title(f"{title}: {STRATEGY_DISPLAY[strategy_name]}")
    ax.grid(True, axis="y", alpha=0.3)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    save(fig, output_path)


def plot_predicted_vs_observed_flow(
    strategy_name: str,
    condition_payloads: dict[str, dict[str, object]],
    output_path: Path,
) -> None:
    points_by_condition: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    all_values: list[float] = []
    for condition_name in MAIN_CONDITIONS:
        edge_df = condition_payloads[condition_name]["edge"]
        observed = pd.to_numeric(edge_df.get("observed_flow_nl_s"), errors="coerce").to_numpy(
            dtype=np.float64
        )
        predicted = pd.to_numeric(
            edge_df.get("predicted_flow_nl_s"), errors="coerce"
        ).to_numpy(dtype=np.float64)
        valid = edge_df["valid_observed_flow"].to_numpy(dtype=bool)
        mask = valid & np.isfinite(observed) & np.isfinite(predicted)
        points_by_condition[condition_name] = (observed[mask], predicted[mask])
        if np.any(mask):
            all_values.extend(observed[mask].tolist())
            all_values.extend(predicted[mask].tolist())
    if not all_values:
        return
    lo = float(np.min(all_values))
    hi = float(np.max(all_values))
    fig, axes = plt.subplots(2, 2, figsize=(9.2, 8.0), constrained_layout=True)
    for ax, condition_name in zip(axes.ravel(), MAIN_CONDITIONS):
        observed, predicted = points_by_condition[condition_name]
        ax.scatter(
            observed,
            predicted,
            s=9,
            alpha=0.35,
            color=CONDITION_COLORS[condition_name],
            linewidths=0,
        )
        ax.plot([lo, hi], [lo, hi], color="#222222", linestyle="--", linewidth=1.0)
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        ax.set_title(CONDITION_DISPLAY[condition_name])
        ax.set_xlabel("Observed flow (nL/s)")
        ax.set_ylabel("Predicted flow (nL/s)")
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
    fig.suptitle(f"Predicted versus observed flow: {STRATEGY_DISPLAY[strategy_name]}", fontsize=13)
    save(fig, output_path)


def plot_delta_reference_scatter(
    strategy_name: str,
    edge_comparison: pd.DataFrame,
    output_path: Path,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.4), constrained_layout=True)
    selected = edge_comparison["selected_for_radius_correction"].astype(bool).to_numpy(dtype=bool)
    panels = [
        ("fixed_weight_delta", "Original delta vs G-fixed delta"),
        ("retrained_delta", "Original delta vs G-retrained delta"),
    ]
    all_vals: list[float] = []
    for column, _ in panels:
        x = pd.to_numeric(edge_comparison.get("original_delta"), errors="coerce").to_numpy(dtype=np.float64)
        y = pd.to_numeric(edge_comparison.get(column), errors="coerce").to_numpy(dtype=np.float64)
        mask = np.isfinite(x) & np.isfinite(y)
        if np.any(mask):
            all_vals.extend(x[mask].tolist())
            all_vals.extend(y[mask].tolist())
    if not all_vals:
        return
    lim = float(max(abs(np.min(all_vals)), abs(np.max(all_vals))))
    lim = max(lim, 1.0e-6)
    for ax, (column, title) in zip(axes, panels):
        x = pd.to_numeric(edge_comparison.get("original_delta"), errors="coerce").to_numpy(dtype=np.float64)
        y = pd.to_numeric(edge_comparison.get(column), errors="coerce").to_numpy(dtype=np.float64)
        unselected_mask = (~selected) & np.isfinite(x) & np.isfinite(y)
        selected_mask = selected & np.isfinite(x) & np.isfinite(y)
        ax.scatter(x[unselected_mask], y[unselected_mask], s=9, alpha=0.18, color="#7A7A7A", linewidths=0, label="Unselected")
        ax.scatter(x[selected_mask], y[selected_mask], s=11, alpha=0.65, color="#C2410C", linewidths=0, label="Selected")
        ax.plot([-lim, lim], [-lim, lim], color="#111111", linestyle="--", linewidth=1.0)
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
        ax.set_xlabel("Original delta")
        ax.set_ylabel(column.replace("_", " "))
        ax.set_title(title)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
    axes[1].legend(frameon=False, loc="upper left")
    fig.suptitle(f"Edgewise delta comparison: {STRATEGY_DISPLAY[strategy_name]}", fontsize=13)
    save(fig, output_path)


def ecdf(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return np.zeros(0, dtype=np.float64), np.zeros(0, dtype=np.float64)
    xs = np.sort(finite)
    ys = np.arange(1, len(xs) + 1, dtype=np.float64) / len(xs)
    return xs, ys


def plot_abs_delta_ecdf(
    strategy_name: str,
    edge_comparison: pd.DataFrame,
    selected_only: bool,
    output_path: Path,
) -> None:
    mask = edge_comparison["selected_for_radius_correction"].astype(bool)
    if not selected_only:
        mask = ~mask
    fig, ax = plt.subplots(figsize=(6.8, 4.4), constrained_layout=True)
    series_specs = [
        ("original_delta", "#6C7A89", "Original delta"),
        ("fixed_weight_delta", "#E9C46A", "G-fixed delta"),
        ("retrained_delta", "#E76F51", "G-retrained delta"),
    ]
    for column, color, label in series_specs:
        values = pd.to_numeric(edge_comparison.loc[mask, column], errors="coerce").abs().to_numpy(
            dtype=np.float64
        )
        xs, ys = ecdf(values)
        if xs.size:
            ax.step(xs, ys, where="post", color=color, linewidth=2.0, label=label)
    ax.set_xlabel("Absolute delta")
    ax.set_ylabel("ECDF")
    subset_label = "selected edges" if selected_only else "unselected edges"
    ax.set_title(f"Absolute delta ECDF on {subset_label}: {STRATEGY_DISPLAY[strategy_name]}")
    ax.legend(frameon=False, loc="lower right")
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    save(fig, output_path)


def finite_limits(arrays: list[np.ndarray], symmetric: bool = False) -> tuple[float, float] | None:
    finite_parts = [arr[np.isfinite(arr)] for arr in arrays if arr.size]
    finite_parts = [arr for arr in finite_parts if arr.size]
    if not finite_parts:
        return None
    values = np.concatenate(finite_parts)
    lo = float(np.min(values))
    hi = float(np.max(values))
    if symmetric:
        bound = max(abs(lo), abs(hi), 1.0e-12)
        return (-bound, bound)
    if math.isclose(lo, hi):
        pad = max(abs(lo) * 0.05, 1.0e-12)
        return (lo - pad, hi + pad)
    return (lo, hi)


def draw_edge_map(
    ax: plt.Axes,
    node_df: pd.DataFrame,
    edge_df: pd.DataFrame,
    value_column: str,
    cmap: str,
    limits: tuple[float, float],
) -> LineCollection | None:
    segments, values = edge_segments_and_values(node_df, edge_df, value_column)
    if not len(segments):
        return None
    if limits[0] < 0.0 < limits[1]:
        norm = TwoSlopeNorm(vmin=limits[0], vcenter=0.0, vmax=limits[1])
        lc = LineCollection(segments, cmap=cmap, norm=norm, linewidths=1.4)
    else:
        lc = LineCollection(segments, cmap=cmap, linewidths=1.4)
        lc.set_clim(limits[0], limits[1])
    lc.set_array(values)
    ax.add_collection(lc)
    return lc


def plot_multi_condition_edge_map(
    strategy_name: str,
    condition_payloads: dict[str, dict[str, object]],
    value_column: str,
    title: str,
    colorbar_label: str,
    cmap: str,
    output_path: Path,
    delta_zero_for_poiseuille: bool = False,
) -> None:
    arrays: list[np.ndarray] = []
    x_min = y_min = float("inf")
    x_max = y_max = float("-inf")
    edge_tables: dict[str, pd.DataFrame] = {}
    node_tables: dict[str, pd.DataFrame] = {}
    for condition_name in MAIN_CONDITIONS:
        node_df = condition_payloads[condition_name]["node"]
        edge_df = condition_payloads[condition_name]["edge"].copy()
        if delta_zero_for_poiseuille and condition_name.startswith("p_"):
            edge_df[value_column] = 0.0
        edge_tables[condition_name] = edge_df
        node_tables[condition_name] = node_df
        values = pd.to_numeric(edge_df.get(value_column), errors="coerce").to_numpy(dtype=np.float64)
        arrays.append(values)
        coords = node_df[["x_px", "y_px"]].to_numpy(dtype=np.float64)
        if coords.size:
            x_min = min(x_min, float(np.nanmin(coords[:, 0])))
            x_max = max(x_max, float(np.nanmax(coords[:, 0])))
            y_min = min(y_min, float(np.nanmin(coords[:, 1])))
            y_max = max(y_max, float(np.nanmax(coords[:, 1])))
    limits = finite_limits(arrays, symmetric=True)
    if limits is None:
        return
    fig, axes = plt.subplots(1, 4, figsize=(13.6, 4.0), constrained_layout=True)
    artist = None
    for ax, condition_name in zip(axes, MAIN_CONDITIONS):
        artist = draw_edge_map(
            ax,
            node_tables[condition_name],
            edge_tables[condition_name],
            value_column=value_column,
            cmap=cmap,
            limits=limits,
        )
        ax.set_title(CONDITION_DISPLAY[condition_name])
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)
    if artist is not None:
        cbar = fig.colorbar(artist, ax=axes.ravel().tolist(), shrink=0.88, pad=0.02)
        cbar.set_label(colorbar_label)
    fig.suptitle(f"{title}: {STRATEGY_DISPLAY[strategy_name]}", fontsize=13)
    save(fig, output_path)


def plot_multi_condition_node_map(
    strategy_name: str,
    condition_payloads: dict[str, dict[str, object]],
    value_column: str,
    title: str,
    colorbar_label: str,
    cmap: str,
    output_path: Path,
    symmetric: bool,
) -> None:
    arrays: list[np.ndarray] = []
    x_min = y_min = float("inf")
    x_max = y_max = float("-inf")
    for condition_name in MAIN_CONDITIONS:
        node_df = condition_payloads[condition_name]["node"]
        values = pd.to_numeric(node_df.get(value_column), errors="coerce").to_numpy(dtype=np.float64)
        arrays.append(values)
        coords = node_df[["x_px", "y_px"]].to_numpy(dtype=np.float64)
        if coords.size:
            x_min = min(x_min, float(np.nanmin(coords[:, 0])))
            x_max = max(x_max, float(np.nanmax(coords[:, 0])))
            y_min = min(y_min, float(np.nanmin(coords[:, 1])))
            y_max = max(y_max, float(np.nanmax(coords[:, 1])))
    limits = finite_limits(arrays, symmetric=symmetric)
    if limits is None:
        return
    fig, axes = plt.subplots(1, 4, figsize=(13.6, 4.0), constrained_layout=True)
    artist = None
    for ax, condition_name in zip(axes, MAIN_CONDITIONS):
        node_df = condition_payloads[condition_name]["node"]
        values = pd.to_numeric(node_df.get(value_column), errors="coerce").to_numpy(dtype=np.float64)
        coords = node_df[["x_px", "y_px"]].to_numpy(dtype=np.float64)
        finite = np.isfinite(values)
        if np.any(finite):
            if symmetric and limits[0] < 0.0 < limits[1]:
                norm = TwoSlopeNorm(vmin=limits[0], vcenter=0.0, vmax=limits[1])
                artist = ax.scatter(
                    coords[finite, 0],
                    coords[finite, 1],
                    c=values[finite],
                    cmap=cmap,
                    norm=norm,
                    s=14,
                    linewidths=0,
                )
            else:
                artist = ax.scatter(
                    coords[finite, 0],
                    coords[finite, 1],
                    c=values[finite],
                    cmap=cmap,
                    vmin=limits[0],
                    vmax=limits[1],
                    s=14,
                    linewidths=0,
                )
        ax.set_title(CONDITION_DISPLAY[condition_name])
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)
    if artist is not None:
        cbar = fig.colorbar(artist, ax=axes.ravel().tolist(), shrink=0.88, pad=0.02)
        cbar.set_label(colorbar_label)
    fig.suptitle(f"{title}: {STRATEGY_DISPLAY[strategy_name]}", fontsize=13)
    save(fig, output_path)


def plot_selection_overlap(input_root: Path, output_path: Path) -> None:
    ref_nodes = reference_nodes(input_root)
    nodes = ensure_coords(load_node_csv(input_root, STRATEGY_ORDER[0], "p_original"), ref_nodes)
    edges = load_edge_csv(input_root, STRATEGY_ORDER[0], "p_original")
    overlap = pd.read_csv(input_root / "edge_selection_overlap.csv")
    merged = edges.merge(overlap[["edge_id", "category"]], on="edge_id", how="left")
    colors = {
        "neither": "#D0D7DE",
        "targeted_only": "#1D4ED8",
        "low_snr_only": "#D97706",
        "overlap": "#B91C1C",
    }
    fig, ax = plt.subplots(figsize=(6.8, 5.8), constrained_layout=True)
    for category, color in colors.items():
        subset = merged[merged["category"].fillna("neither") == category].copy()
        segments, _ = edge_segments_and_values(nodes, subset.assign(dummy_value=0.0), "dummy_value")
        if len(segments):
            ax.add_collection(
                LineCollection(
                    segments,
                    colors=color,
                    linewidths=2.0 if category != "neither" else 0.8,
                    alpha=0.95 if category != "neither" else 0.25,
                    label=category.replace("_", " "),
                )
            )
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(frameon=False, loc="upper right")
    ax.set_title("Targeted versus low-SNR edge selection")
    ax.set_aspect("equal", adjustable="box")
    ax.invert_yaxis()
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    save(fig, output_path)


def strategy_payloads(
    input_root: Path,
    summary_df: pd.DataFrame,
    strategy_name: str,
) -> tuple[dict[str, dict[str, object]], pd.DataFrame]:
    ref_nodes = reference_nodes(input_root)
    subset = summary_df[summary_df["strategy_name"] == strategy_name].copy()
    payloads: dict[str, dict[str, object]] = {}
    for condition_name in ("p_original", "p_corrected", "g_fixed", "g_retrained", "g_original"):
        edge_df = load_edge_csv(input_root, strategy_name, condition_name)
        node_df = ensure_coords(load_node_csv(input_root, strategy_name, condition_name), ref_nodes)
        row = subset[subset["condition_name"] == condition_name]
        if row.empty:
            summary_row = pd.Series(dtype=object)
        else:
            summary_row = row.iloc[0]
        payloads[condition_name] = {
            "edge": edge_df,
            "node": node_df,
            "summary": hydrate_summary(summary_row, edge_df, node_df),
        }
    edge_comparison = pd.read_csv(input_root / f"radius_refinement_edge_comparison_{strategy_name}.csv")
    if "selected_for_radius_correction" in edge_comparison.columns:
        edge_comparison["selected_for_radius_correction"] = edge_comparison[
            "selected_for_radius_correction"
        ].map(lambda value: normalize_bool(value, False))
    return payloads, edge_comparison


def plot_strategy_suite(
    input_root: Path,
    output_dir: Path,
    summary_df: pd.DataFrame,
    strategy_name: str,
) -> None:
    payloads, edge_comparison = strategy_payloads(input_root, summary_df, strategy_name)
    summaries = {name: payloads[name]["summary"] for name in payloads}

    metric_bar_plot(
        strategy_name,
        summaries,
        metric="flow_rmse_nl_s",
        ylabel="Flow RMSE (nL/s)",
        title="Flow RMSE across conditions",
        output_path=output_dir / f"grouped_bar_flow_rmse_{strategy_name}.png",
    )
    metric_bar_plot(
        strategy_name,
        summaries,
        metric="kirchhoff_rms_per_internal_node_nl_s",
        ylabel="Kirchhoff RMS per internal node (nL/s)",
        title="Kirchhoff RMS across conditions",
        output_path=output_dir / f"grouped_bar_kirchhoff_rms_{strategy_name}.png",
    )
    metric_bar_plot(
        strategy_name,
        summaries,
        metric="sign_flip_fraction",
        ylabel="Sign-flip fraction",
        title="Sign-flip fraction across conditions",
        output_path=output_dir / f"grouped_bar_sign_flip_fraction_{strategy_name}.png",
    )
    metric_bar_plot(
        strategy_name,
        summaries,
        metric="pressure_range_pa",
        ylabel="Pressure range (Pa)",
        title="Pressure range across conditions",
        output_path=output_dir / f"grouped_bar_pressure_range_{strategy_name}.png",
    )
    metric_bar_plot(
        strategy_name,
        summaries,
        metric="delta_rms_all_edges",
        ylabel="All-edge delta RMS",
        title="All-edge delta RMS across conditions",
        output_path=output_dir / f"grouped_bar_delta_rms_all_edges_{strategy_name}.png",
        zero_for_poiseuille=True,
    )
    metric_bar_plot(
        strategy_name,
        summaries,
        metric="delta_rms_selected_edges",
        ylabel="Selected-edge delta RMS",
        title="Selected-edge delta RMS across conditions",
        output_path=output_dir / f"grouped_bar_delta_rms_selected_edges_{strategy_name}.png",
        zero_for_poiseuille=True,
    )

    boxplot_abs_metric(
        strategy_name,
        payloads,
        source="edge",
        column="flow_residual_nl_s",
        ylabel="Absolute flow residual (nL/s)",
        title="Absolute flow residual distribution",
        output_path=output_dir / f"boxplot_abs_flow_residual_{strategy_name}.png",
    )
    boxplot_abs_metric(
        strategy_name,
        payloads,
        source="node",
        column="kirchhoff_residual_nl_s",
        ylabel="Absolute Kirchhoff residual (nL/s)",
        title="Absolute Kirchhoff residual distribution",
        output_path=output_dir / f"boxplot_abs_kirchhoff_residual_{strategy_name}.png",
    )

    plot_predicted_vs_observed_flow(
        strategy_name,
        payloads,
        output_path=output_dir / f"scatter_predicted_vs_observed_flow_{strategy_name}.png",
    )
    plot_delta_reference_scatter(
        strategy_name,
        edge_comparison=edge_comparison,
        output_path=output_dir / f"scatter_original_vs_fixed_retrained_delta_{strategy_name}.png",
    )
    plot_abs_delta_ecdf(
        strategy_name,
        edge_comparison=edge_comparison,
        selected_only=True,
        output_path=output_dir / f"ecdf_abs_delta_selected_edges_{strategy_name}.png",
    )
    plot_abs_delta_ecdf(
        strategy_name,
        edge_comparison=edge_comparison,
        selected_only=False,
        output_path=output_dir / f"ecdf_abs_delta_unselected_edges_{strategy_name}.png",
    )

    plot_multi_condition_node_map(
        strategy_name,
        payloads,
        value_column="pressure_pa",
        title="Pressure maps across conditions",
        colorbar_label="Pressure (Pa)",
        cmap="viridis",
        output_path=output_dir / f"pressure_maps_four_conditions_{strategy_name}.png",
        symmetric=False,
    )
    plot_multi_condition_edge_map(
        strategy_name,
        payloads,
        value_column="flow_residual_nl_s",
        title="Flow-residual maps across conditions",
        colorbar_label="Flow residual (nL/s)",
        cmap="coolwarm",
        output_path=output_dir / f"flow_residual_maps_four_conditions_{strategy_name}.png",
    )
    plot_multi_condition_node_map(
        strategy_name,
        payloads,
        value_column="kirchhoff_residual_nl_s",
        title="Kirchhoff-residual maps across conditions",
        colorbar_label="Kirchhoff residual (nL/s)",
        cmap="coolwarm",
        output_path=output_dir / f"kirchhoff_residual_maps_four_conditions_{strategy_name}.png",
        symmetric=True,
    )
    plot_multi_condition_edge_map(
        strategy_name,
        payloads,
        value_column="delta_e",
        title="Conductance-correction maps across conditions",
        colorbar_label="delta",
        cmap="coolwarm",
        output_path=output_dir / f"delta_maps_four_conditions_{strategy_name}.png",
        delta_zero_for_poiseuille=True,
    )


def main() -> None:
    args = parse_args()
    configure_plot()
    input_root = args.input_root.expanduser().resolve()
    output_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir is not None
        else input_root / "figures"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_df = load_summary(input_root)

    for strategy_name in STRATEGY_ORDER:
        plot_strategy_suite(
            input_root=input_root,
            output_dir=output_dir,
            summary_df=summary_df,
            strategy_name=strategy_name,
        )

    plot_selection_overlap(
        input_root=input_root,
        output_path=output_dir / "edge_selection_targeted_vs_low_snr.png",
    )


if __name__ == "__main__":
    main()
