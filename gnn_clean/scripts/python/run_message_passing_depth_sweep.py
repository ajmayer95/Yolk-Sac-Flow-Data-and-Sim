#!/usr/bin/env python
"""Run and summarize the Step 4 GNN message-passing-depth sweep."""

from __future__ import annotations

import argparse
import csv
import math
import subprocess
import sys
from pathlib import Path

os_environ_note = "MPLCONFIGDIR"

import os

os.environ.setdefault(os_environ_note, "/tmp/matplotlib")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from analyze_physics_weight_sweep import compute_from_gnn_run, safe_float
from utils import load_yaml, write_yaml


DEFAULT_STEP2_ROOT = PROJECT_ROOT / "outputs" / "dc" / "02_physics_weight_sweep"
DEFAULT_REPRESENTATIVE_CSV = DEFAULT_STEP2_ROOT / "representative_configurations.csv"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "outputs" / "dc" / "04_message_passing_sensitivity"
GNN_SCRIPT = PROJECT_ROOT / "scripts" / "python" / "gnn_flow.py"
K_VALUES = (0, 1, 2, 3, 4)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--representative-csv", type=Path, default=DEFAULT_REPRESENTATIVE_CSV)
    parser.add_argument("--step2-root", type=Path, default=DEFAULT_STEP2_ROOT)
    parser.add_argument("--graph", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--representative-labels", nargs="*", default=None)
    parser.add_argument("--python-bin", default=sys.executable)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--aggregate-only", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def select_best_balanced(rep_csv: Path) -> pd.Series:
    df = pd.read_csv(rep_csv)
    balanced = df[df["selection_category"] == "balanced"].copy()
    if balanced.empty:
        raise ValueError("No balanced configuration found in representative_configurations.csv")
    if "selection_rank_within_regime" in balanced.columns:
        balanced["selection_rank_within_regime"] = pd.to_numeric(
            balanced["selection_rank_within_regime"], errors="coerce"
        )
        balanced = balanced.sort_values(
            ["selection_rank_within_regime", "selection_score", "run_name"],
            na_position="last",
        )
    elif "selection_score" in balanced.columns:
        balanced = balanced.sort_values(["selection_score", "run_name"], na_position="last")
    else:
        balanced = balanced.sort_values(["run_name"])
    return balanced.iloc[0]


def select_representatives(rep_csv: Path, labels: list[str] | None) -> list[pd.Series]:
    df = pd.read_csv(rep_csv)
    if labels:
        selected: list[pd.Series] = []
        missing: list[str] = []
        for label in labels:
            subset = df[df["plot_label"] == label].copy()
            if subset.empty:
                missing.append(label)
                continue
            selected.append(subset.iloc[0])
        if missing:
            raise ValueError(
                "Representative labels not found in representative_configurations.csv: "
                + ", ".join(sorted(missing))
            )
        return selected
    return [select_best_balanced(rep_csv)]


def selected_step2_config_path(step2_root: Path, run_name: str) -> Path:
    return step2_root / "_generated_configs" / f"{run_name}.yaml"


def deep_update(base: dict, update: dict) -> dict:
    merged = dict(base)
    for key, value in update.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_update(merged[key], value)
        else:
            merged[key] = value
    return merged


def build_override(step2_config: dict, K_value: int, seed: int, epochs: int) -> dict:
    return deep_update(
        step2_config,
        {
            "K": int(K_value),
            "training": {
                "seed": int(seed),
                "epochs": int(epochs),
            },
        },
    )


def build_run_name(parent_run_name: str, K_value: int) -> str:
    return f"{parent_run_name}__K_{int(K_value)}"


def gnn_command(
    python_bin: str,
    graph: Path,
    output_root: Path,
    run_name: str,
    config_path: Path,
    device: str,
    epochs: int,
    seed: int,
) -> list[str]:
    return [
        str(python_bin),
        str(GNN_SCRIPT),
        str(graph),
        "--output-dir",
        str(output_root),
        "--run-name",
        run_name,
        "--preset",
        "solver_QKB_outer_QKBdelta",
        "--device",
        str(device),
        "--epochs",
        str(int(epochs)),
        "--seed",
        str(int(seed)),
        "--config",
        str(config_path),
        "--no-pressure-detach",
    ]


def run_is_complete(run_dir: Path) -> bool:
    required = ("summary.csv", "summary.yaml", "node_predictions.csv", "edge_predictions.csv")
    return all((run_dir / name).exists() for name in required)


def cleanup_nonessential_outputs(run_dir: Path) -> None:
    for name in ("model_checkpoint.pt", "training_history.csv", "exploration_diagnostics.csv"):
        path = run_dir / name
        if path.exists():
            path.unlink()


def compute_sign_flip_fraction(edge_rows: list[dict[str, str]]) -> float:
    valid = [row for row in edge_rows if str(row.get("valid_observed_flow", "")).lower() == "true"]
    if not valid:
        return float("nan")
    flips = sum(1 for row in valid if str(row.get("sign_flip", "")).lower() == "true")
    return flips / len(valid)


def summarize_run(
    run_dir: Path,
    parent_row: pd.Series,
    K_value: int,
) -> dict[str, object]:
    metadata = {
        "run_name": run_dir.name,
        "model_family": "gnn",
        "lambda_q": float(parent_row["lambda_q"]),
        "lambda_k": float(parent_row["lambda_k"]),
        "lambda_b": 100.0,
        "lambda_delta": float(parent_row["lambda_delta"]),
        "message_passing_layers": int(K_value),
    }
    row = compute_from_gnn_run(run_dir, metadata)
    with (run_dir / "summary.csv").open("r", newline="", encoding="utf-8") as handle:
        summary_csv = next(csv.DictReader(handle))
    with (run_dir / "edge_predictions.csv").open("r", newline="", encoding="utf-8") as handle:
        edge_rows = list(csv.DictReader(handle))
    row.update(
        {
            "K": int(K_value),
            "selected_step2_run_name": str(parent_row["run_name"]),
            "selection_category": str(parent_row["selection_category"]),
            "selection_rank_within_regime": int(float(parent_row["selection_rank_within_regime"])),
            "best_epoch": safe_float(summary_csv.get("best_epoch")),
            "best_val_total": safe_float(summary_csv.get("best_val_total")),
            "pressure_constraint_labels": summary_csv.get("pressure_constraints", ""),
            "pressure_solver_constraint_residual_l2": safe_float(
                summary_csv.get("pressure_solver_constraint_residual_l2")
            ),
            "pressure_solver_constraint_residual_max": safe_float(
                summary_csv.get("pressure_solver_constraint_residual_max")
            ),
            "sign_flip_fraction": compute_sign_flip_fraction(edge_rows),
            "solver_success": str(summary_csv.get("solver_success", "true")).lower() == "true",
        }
    )
    return row


def node_df(run_dir: Path) -> pd.DataFrame:
    df = pd.read_csv(run_dir / "node_predictions.csv")
    for column in ("node_index", "x_px", "y_px", "pressure_pa", "kirchhoff_residual_nl_s"):
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors="coerce")
    return df


def edge_df(run_dir: Path) -> pd.DataFrame:
    df = pd.read_csv(run_dir / "edge_predictions.csv")
    for column in ("source_index", "target_index", "delta_e", "flow_residual_nl_s"):
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors="coerce")
    return df


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
    """Rotate 90 degrees clockwise, then mirror across the vertical axis."""
    x_arr = np.asarray(x, dtype=float)
    y_arr = np.asarray(y, dtype=float)
    _, x_max = x_bounds
    _, y_max = y_bounds
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


def robust_limits(values: list[float], lower: float = 5.0, upper: float = 95.0) -> tuple[float, float] | None:
    finite = np.asarray([value for value in values if math.isfinite(value)], dtype=float)
    if finite.size == 0:
        return None
    vmin = float(np.percentile(finite, lower))
    vmax = float(np.percentile(finite, upper))
    if not math.isfinite(vmin) or not math.isfinite(vmax):
        return None
    if math.isclose(vmin, vmax):
        pad = max(abs(vmin) * 0.05, 1.0e-12)
        return (vmin - pad, vmax + pad)
    return (vmin, vmax)


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


def plot_metric_vs_K(summary_df: pd.DataFrame, y_col: str, y_label: str, title: str, path: Path) -> None:
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
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_pressure_maps(run_dirs_by_k: dict[int, Path], path: Path) -> None:
    payloads = []
    all_pressures: list[float] = []
    xmins: list[float] = []
    xmaxs: list[float] = []
    ymins: list[float] = []
    ymaxs: list[float] = []
    for K_value in K_VALUES:
        run_dir = run_dirs_by_k[K_value]
        nodes = node_df(run_dir)
        edges = edge_df(run_dir)
        payloads.append((K_value, nodes, edges))
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
    transformed_xlim = (0.0, ylim[1] - ylim[0])
    transformed_ylim = (0.0, xlim[1] - xlim[0])
    fig, axes = plt.subplots(1, len(K_VALUES), figsize=(3.1 * len(K_VALUES), 3.8), constrained_layout=True)
    scatter = None
    for ax, (K_value, nodes, edges) in zip(axes, payloads):
        segments = transform_segment_collection(build_segments(nodes, edges), xlim, ylim)
        if segments:
            ax.add_collection(LineCollection(segments, colors="#cdcdcd", linewidths=0.6, zorder=1))
        node_x, node_y = transform_mosaic_coords(nodes["x_px"], nodes["y_px"], xlim, ylim)
        scatter = ax.scatter(
            node_x,
            node_y,
            c=nodes["pressure_pa"],
            cmap="coolwarm",
            vmin=vmin,
            vmax=vmax,
            s=18,
            zorder=2,
        )
        arterial = nodes[_bool_mask(nodes, "is_arterial")]
        venous = nodes[_bool_mask(nodes, "is_venous")]
        if not arterial.empty:
            arterial_x, arterial_y = transform_mosaic_coords(
                arterial["x_px"], arterial["y_px"], xlim, ylim
            )
            ax.scatter(arterial_x, arterial_y, marker="^", color="black", s=24, zorder=3)
        if not venous.empty:
            venous_x, venous_y = transform_mosaic_coords(
                venous["x_px"], venous["y_px"], xlim, ylim
            )
            ax.scatter(venous_x, venous_y, marker="s", color="black", s=20, zorder=3)
        ax.set_title(f"K = {K_value}")
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
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_delta_maps(run_dirs_by_k: dict[int, Path], path: Path) -> None:
    payloads = []
    xmins: list[float] = []
    xmaxs: list[float] = []
    ymins: list[float] = []
    ymaxs: list[float] = []
    for K_value in K_VALUES:
        run_dir = run_dirs_by_k[K_value]
        nodes = node_df(run_dir)
        edges = edge_df(run_dir)
        payloads.append((K_value, nodes, edges))
        coords = nodes[["x_px", "y_px"]].apply(pd.to_numeric, errors="coerce")
        xmins.append(coords["x_px"].min())
        xmaxs.append(coords["x_px"].max())
        ymins.append(coords["y_px"].min())
        ymaxs.append(coords["y_px"].max())
    xlim = (min(xmins), max(xmaxs))
    ylim = (min(ymins), max(ymaxs))
    transformed_xlim = (0.0, ylim[1] - ylim[0])
    transformed_ylim = (0.0, xlim[1] - xlim[0])
    fig, axes = plt.subplots(1, len(K_VALUES), figsize=(3.1 * len(K_VALUES), 3.8), constrained_layout=True)
    collection = None
    for ax, (K_value, nodes, edges) in zip(axes, payloads):
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
            collection = LineCollection(segments, cmap="coolwarm", linewidths=1.15, zorder=2)
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
        ax.set_title(f"K = {K_value}")
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
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_flow_residual_maps(run_dirs_by_k: dict[int, Path], path: Path) -> None:
    payloads = []
    xmins: list[float] = []
    xmaxs: list[float] = []
    ymins: list[float] = []
    ymaxs: list[float] = []
    all_abs_values: list[float] = []
    for K_value in K_VALUES:
        run_dir = run_dirs_by_k[K_value]
        nodes = node_df(run_dir)
        edges = edge_df(run_dir)
        payloads.append((K_value, nodes, edges))
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
    transformed_xlim = (0.0, ylim[1] - ylim[0])
    transformed_ylim = (0.0, xlim[1] - xlim[0])
    fig, axes = plt.subplots(1, len(K_VALUES), figsize=(3.1 * len(K_VALUES), 3.8), constrained_layout=True)
    collection = None
    for ax, (K_value, nodes, edges) in zip(axes, payloads):
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
            collection = LineCollection(segments, cmap="coolwarm", linewidths=1.15, zorder=2)
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
        ax.set_title(f"K = {K_value}")
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
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_kirchhoff_residual_maps(run_dirs_by_k: dict[int, Path], path: Path) -> None:
    payloads = []
    xmins: list[float] = []
    xmaxs: list[float] = []
    ymins: list[float] = []
    ymaxs: list[float] = []
    all_abs_values: list[float] = []
    for K_value in K_VALUES:
        run_dir = run_dirs_by_k[K_value]
        nodes = node_df(run_dir)
        edges = edge_df(run_dir)
        payloads.append((K_value, nodes, edges))
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
    transformed_xlim = (0.0, ylim[1] - ylim[0])
    transformed_ylim = (0.0, xlim[1] - xlim[0])
    fig, axes = plt.subplots(1, len(K_VALUES), figsize=(3.1 * len(K_VALUES), 3.8), constrained_layout=True)
    scatter = None
    for ax, (K_value, nodes, edges) in zip(axes, payloads):
        segments = transform_segment_collection(build_segments(nodes, edges), xlim, ylim)
        if segments:
            ax.add_collection(LineCollection(segments, colors="#cdcdcd", linewidths=0.6, zorder=1))
        node_x, node_y = transform_mosaic_coords(nodes["x_px"], nodes["y_px"], xlim, ylim)
        values = pd.to_numeric(nodes.get("kirchhoff_residual_nl_s"), errors="coerce").to_numpy(dtype=float)
        scatter = ax.scatter(
            node_x,
            node_y,
            c=values,
            cmap="coolwarm",
            vmin=vmin,
            vmax=vmax,
            s=18,
            zorder=2,
        )
        arterial = nodes[_bool_mask(nodes, "is_arterial")]
        venous = nodes[_bool_mask(nodes, "is_venous")]
        if not arterial.empty:
            arterial_x, arterial_y = transform_mosaic_coords(
                arterial["x_px"], arterial["y_px"], xlim, ylim
            )
            ax.scatter(arterial_x, arterial_y, marker="^", color="black", s=24, zorder=3)
        if not venous.empty:
            venous_x, venous_y = transform_mosaic_coords(
                venous["x_px"], venous["y_px"], xlim, ylim
            )
            ax.scatter(venous_x, venous_y, marker="s", color="black", s=20, zorder=3)
        ax.set_title(f"K = {K_value}")
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
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def aggregate_and_plot(output_root: Path, selected_row: pd.Series, graph_path: Path) -> tuple[pd.DataFrame, dict[str, object]]:
    rows: list[dict[str, object]] = []
    run_dirs_by_k: dict[int, Path] = {}
    for K_value in K_VALUES:
        run_name = build_run_name(str(selected_row["run_name"]), K_value)
        run_dir = output_root / run_name
        if not run_is_complete(run_dir):
            raise FileNotFoundError(f"Missing completed run outputs for {run_name}")
        rows.append(summarize_run(run_dir, selected_row, K_value))
        run_dirs_by_k[K_value] = run_dir
    summary_df = pd.DataFrame(rows).sort_values("K").reset_index(drop=True)
    summary_df.to_csv(output_root / "summary.csv", index=False)

    best_idx = summary_df["best_val_total"].astype(float).idxmin()
    best_row = summary_df.loc[int(best_idx)].to_dict()
    min_val = float(summary_df["best_val_total"].min())
    plateau_mask = summary_df["best_val_total"].astype(float) <= (1.01 * min_val)
    plateau_k = int(summary_df.loc[plateau_mask, "K"].min()) if plateau_mask.any() else int(best_row["K"])
    summary_yaml = {
        "experiment": {
            "graph": str(graph_path.expanduser().resolve()),
            "selected_step2_run_name": str(selected_row["run_name"]),
            "selection_category": str(selected_row["selection_category"]),
            "selection_rank_within_regime": int(float(selected_row["selection_rank_within_regime"])),
            "lambda_q": float(selected_row["lambda_q"]),
            "lambda_k": float(selected_row["lambda_k"]),
            "lambda_delta": float(selected_row["lambda_delta"]),
            "lambda_b": 100.0,
            "pressure_constraints": ["equal-a-equal-v"],
            "K_values": list(K_VALUES),
        },
        "best_run_by_validation": best_row,
        "smallest_K_within_1pct_of_best_val": plateau_k,
    }
    write_yaml(output_root / "summary.yaml", summary_yaml)

    plot_metric_vs_K(
        summary_df,
        "flow_rmse_nl_s",
        "Flow RMSE (nL/s)",
        "Flow agreement versus message-passing depth",
        output_root / "flow_rmse_vs_K.png",
    )
    plot_metric_vs_K(
        summary_df,
        "kirchhoff_rms_per_internal_node_nl_s",
        "Kirchhoff RMS per internal node (nL/s)",
        "Conservation consistency versus message-passing depth",
        output_root / "kirchhoff_rms_vs_K.png",
    )
    plot_pressure_maps(run_dirs_by_k, output_root / "pressure_maps_by_K.png")
    plot_flow_residual_maps(run_dirs_by_k, output_root / "flow_residual_maps_by_K.png")
    plot_kirchhoff_residual_maps(run_dirs_by_k, output_root / "kirchhoff_residual_maps_by_K.png")
    plot_delta_maps(run_dirs_by_k, output_root / "conductance_correction_maps_by_K.png")
    return summary_df, summary_yaml


def output_root_for_representative(base_output_root: Path, selected_row: pd.Series, multi: bool) -> Path:
    if not multi:
        return base_output_root
    label = str(selected_row.get("plot_label", "")).strip() or str(selected_row["run_name"])
    return base_output_root / label


def main() -> None:
    args = parse_args()
    base_output_root = args.output_root.expanduser().resolve()
    base_output_root.mkdir(parents=True, exist_ok=True)
    selected_rows = select_representatives(
        args.representative_csv.expanduser().resolve(),
        list(args.representative_labels) if args.representative_labels else None,
    )
    multi = len(selected_rows) > 1

    for selected_row in selected_rows:
        output_root = output_root_for_representative(base_output_root, selected_row, multi)
        output_root.mkdir(parents=True, exist_ok=True)

        step2_config_path = selected_step2_config_path(
            args.step2_root.expanduser().resolve(),
            str(selected_row["run_name"]),
        )
        if not step2_config_path.exists():
            raise FileNotFoundError(
                f"Missing Step 2 config for selected run {selected_row['run_name']}: {step2_config_path}"
            )

        if args.aggregate_only:
            aggregate_and_plot(output_root, selected_row, args.graph)
            continue

        step2_config = load_yaml(step2_config_path)
        generated_config_root = output_root / "_generated_configs"
        generated_config_root.mkdir(parents=True, exist_ok=True)

        for K_value in K_VALUES:
            run_name = build_run_name(str(selected_row["run_name"]), K_value)
            run_dir = output_root / run_name
            run_dir.mkdir(parents=True, exist_ok=True)
            config_path = generated_config_root / f"{run_name}.yaml"
            write_yaml(config_path, build_override(step2_config, K_value, args.seed, args.epochs))
            if run_is_complete(run_dir) and not args.overwrite:
                print(f"Skipping completed run: {run_name}")
                continue
            cmd = gnn_command(
                args.python_bin,
                args.graph.expanduser().resolve(),
                output_root,
                run_name,
                config_path,
                args.device,
                args.epochs,
                args.seed,
            )
            print("Command:", " ".join(str(part) for part in cmd))
            if not args.dry_run:
                subprocess.run(cmd, check=True)
                cleanup_nonessential_outputs(run_dir)

        if not args.dry_run:
            aggregate_and_plot(output_root, selected_row, args.graph)


if __name__ == "__main__":
    main()
