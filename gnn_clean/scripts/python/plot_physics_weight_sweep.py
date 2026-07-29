#!/usr/bin/env python
"""Plot aggregated Step 2 physics-weight sweep results."""

from __future__ import annotations

import argparse
import csv
import math
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT_ROOT = PROJECT_ROOT / "outputs" / "dc" / "02_physics_weight_sweep"
DEFAULT_FIGURES_DIRNAME = "figures"

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
REGIME_PREFIX = {
    "flow_prioritized": "F",
    "balanced": "B",
    "conservation_prioritized": "K",
    "correction_regularized": "C",
}
OBSOLETE_OUTPUTS = (
    "flow_kirchhoff_pareto_annotated.png",
    "flow_kirchhoff_pareto.pdf",
    "flow_kirchhoff_pareto_labeled.pdf",
    "flow_rmse_vs_delta_rms.pdf",
    "kirchhoff_rms_vs_delta_rms.pdf",
    "flow_rmse_vs_log_lambda_q_over_k.pdf",
    "kirchhoff_rms_vs_log_lambda_q_over_k.pdf",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, default=DEFAULT_INPUT_ROOT)
    parser.add_argument("--all-runs-csv", type=Path, default=None)
    parser.add_argument("--gnn-summary-csv", type=Path, default=None)
    parser.add_argument("--poiseuille-summary-csv", type=Path, default=None)
    parser.add_argument("--representatives-csv", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument(
        "--clean-obsolete",
        action="store_true",
        help="Remove obsolete annotated/PDF plotting outputs before writing new PNG files.",
    )
    return parser.parse_args()


def resolve_paths(args: argparse.Namespace) -> dict[str, Path]:
    input_root = args.input_root.expanduser().resolve()
    return {
        "input_root": input_root,
        "all_runs_csv": (
            args.all_runs_csv.expanduser().resolve()
            if args.all_runs_csv is not None
            else input_root / "physics_weight_all_runs.csv"
        ),
        "gnn_summary_csv": (
            args.gnn_summary_csv.expanduser().resolve()
            if args.gnn_summary_csv is not None
            else input_root / "physics_weight_gnn_summary.csv"
        ),
        "poiseuille_summary_csv": (
            args.poiseuille_summary_csv.expanduser().resolve()
            if args.poiseuille_summary_csv is not None
            else input_root / "physics_weight_poiseuille_summary.csv"
        ),
        "representatives_csv": (
            args.representatives_csv.expanduser().resolve()
            if args.representatives_csv is not None
            else input_root / "representative_configurations.csv"
        ),
        "output_dir": (
            args.output_dir.expanduser().resolve()
            if args.output_dir is not None
            else input_root / DEFAULT_FIGURES_DIRNAME
        ),
    }


def load_csv(path: Path, *, allow_empty: bool = False) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Required CSV not found: {path}")
    try:
        df = pd.read_csv(path)
    except pd.errors.EmptyDataError:
        if allow_empty:
            return pd.DataFrame()
        raise ValueError(f"No columns found in {path}") from None
    if df.empty:
        if allow_empty:
            return df
        raise ValueError(f"No rows found in {path}")
    return df


def configure_matplotlib() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.grid": True,
            "grid.color": "#d9d9d9",
            "grid.linewidth": 0.8,
            "grid.alpha": 0.65,
            "axes.titlesize": 14,
            "axes.labelsize": 12,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.fontsize": 10,
            "savefig.facecolor": "white",
        }
    )


def numeric(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    result = df.copy()
    for column in columns:
        if column in result.columns:
            result[column] = pd.to_numeric(result[column], errors="coerce")
    return result


def filter_valid(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    if df.empty:
        return df.copy()
    missing = [column for column in columns if column not in df.columns]
    if missing:
        return pd.DataFrame(columns=df.columns)
    return numeric(df, columns).dropna(subset=columns).copy()


def save(fig: plt.Figure, path: Path, dpi: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def maybe_clean_obsolete(output_dir: Path) -> None:
    for name in OBSOLETE_OUTPUTS:
        path = output_dir / name
        if path.exists():
            path.unlink()


def label_for_regime(regime: str) -> str:
    return REGIME_LABELS.get(regime, regime.replace("_", " ").title())


def weighting_regime_legend_handles(include_pareto: bool, include_baseline: bool) -> list[Line2D]:
    handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="None",
            markerfacecolor=REGIME_COLORS[regime],
            markeredgecolor="none",
            markersize=7,
            alpha=0.9,
            label=label_for_regime(regime),
        )
        for regime in REGIME_ORDER
    ]
    if include_baseline:
        handles.append(
            Line2D(
                [0],
                [0],
                marker="D",
                linestyle="None",
                markerfacecolor="#9a9a9a",
                markeredgecolor="#5f5f5f",
                markersize=7,
                alpha=0.8,
                label="Poiseuille baseline",
            )
        )
    if include_pareto:
        handles.append(
            Line2D(
                [0],
                [0],
                marker="o",
                linestyle="None",
                markerfacecolor="none",
                markeredgecolor="#202020",
                markeredgewidth=1.2,
                markersize=8.5,
                label="GNN Pareto front",
            )
        )
    return handles


def build_label_lookup(rep_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for regime in REGIME_ORDER:
        subset = rep_df[rep_df["selection_category"] == regime].copy()
        if subset.empty:
            continue
        subset = subset.sort_values(["selection_rank_within_regime", "run_name"])
        prefix = REGIME_PREFIX[regime]
        for _, row in subset.iterrows():
            rank = int(row["selection_rank_within_regime"])
            rows.append(
                {
                    "plot_label": f"{prefix}{rank}",
                    "run_name": row["run_name"],
                    "selection_category": row["selection_category"],
                    "selection_rank_within_regime": rank,
                    "selection_score": row.get("selection_score"),
                    "selection_score_formula": row.get("selection_score_formula"),
                    "selection_score_regime": row.get("selection_score_regime"),
                    "lambda_q": row["lambda_q"],
                    "lambda_k": row["lambda_k"],
                    "lambda_delta": row["lambda_delta"],
                    "flow_rmse_nl_s": row["flow_rmse_nl_s"],
                    "kirchhoff_rms_per_internal_node_nl_s": row[
                        "kirchhoff_rms_per_internal_node_nl_s"
                    ],
                    "delta_rms": row["delta_rms"],
                }
            )
    return pd.DataFrame(rows)


def merge_representative_labels(rep_df: pd.DataFrame, label_df: pd.DataFrame) -> pd.DataFrame:
    if rep_df.empty:
        return rep_df.copy()
    return rep_df.merge(label_df[["run_name", "plot_label"]], on="run_name", how="left")


def prepare_data(
    all_df: pd.DataFrame,
    gnn_df: pd.DataFrame,
    pois_df: pd.DataFrame,
    rep_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    common_numeric = [
        "lambda_q",
        "lambda_k",
        "lambda_delta",
        "flow_rmse_nl_s",
        "kirchhoff_rms_per_internal_node_nl_s",
        "delta_rms",
        "pareto_rank",
        "selection_rank_within_regime",
    ]
    gnn_all = filter_valid(
        all_df[all_df["model_family"] == "gnn"].copy(),
        ["flow_rmse_nl_s", "kirchhoff_rms_per_internal_node_nl_s"],
    )
    gnn_all = numeric(gnn_all, common_numeric)
    gnn_all["selected_representative"] = gnn_all["selected_representative"].fillna(False).astype(bool)
    gnn_all["is_pareto_front"] = gnn_all["is_pareto_front"].fillna(False).astype(bool)
    gnn_all["log_q_over_k"] = np.log10(gnn_all["lambda_q"] / gnn_all["lambda_k"])
    gnn_all["jittered_log_q_over_k"] = gnn_all["log_q_over_k"] + deterministic_jitter(
        gnn_all["run_name"], magnitude=0.045
    )

    gnn_summary = filter_valid(
        gnn_df.copy(),
        ["flow_rmse_nl_s", "kirchhoff_rms_per_internal_node_nl_s"],
    )
    gnn_summary = numeric(gnn_summary, common_numeric)
    gnn_summary["selected_representative"] = (
        gnn_summary["selected_representative"].fillna(False).astype(bool)
    )
    gnn_summary["is_pareto_front"] = gnn_summary["is_pareto_front"].fillna(False).astype(bool)
    gnn_summary["log_q_over_k"] = np.log10(gnn_summary["lambda_q"] / gnn_summary["lambda_k"])
    gnn_summary["jittered_log_q_over_k"] = gnn_summary["log_q_over_k"] + deterministic_jitter(
        gnn_summary["run_name"], magnitude=0.045
    )

    pois_summary = filter_valid(
        pois_df.copy(),
        ["flow_rmse_nl_s", "kirchhoff_rms_per_internal_node_nl_s"],
    )
    pois_summary = numeric(pois_summary, common_numeric)

    rep_prepped = filter_valid(
        rep_df.copy(),
        ["flow_rmse_nl_s", "kirchhoff_rms_per_internal_node_nl_s"],
    )
    rep_prepped = numeric(rep_prepped, common_numeric)
    rep_prepped["log_q_over_k"] = np.log10(rep_prepped["lambda_q"] / rep_prepped["lambda_k"])
    rep_prepped["jittered_log_q_over_k"] = rep_prepped["log_q_over_k"] + deterministic_jitter(
        rep_prepped["run_name"], magnitude=0.045
    )
    return gnn_all, gnn_summary, pois_summary, rep_prepped


def deterministic_jitter(values: pd.Series, magnitude: float) -> np.ndarray:
    offsets: list[float] = []
    for value in values.astype(str):
        checksum = sum(ord(char) for char in value) % 17
        offsets.append(((checksum / 16.0) - 0.5) * 2.0 * magnitude)
    return np.asarray(offsets, dtype=np.float64)


def safe_float(value: object) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return result if math.isfinite(result) else float("nan")


def mad_inlier_mask(values: pd.Series, threshold: float = 3.5) -> pd.Series:
    numeric_values = pd.to_numeric(values, errors="coerce")
    median = float(numeric_values.median())
    abs_dev = (numeric_values - median).abs()
    mad = float(abs_dev.median())
    if not math.isfinite(mad) or mad <= 0.0:
        return pd.Series(True, index=values.index)
    robust_sigma = 1.4826 * mad
    if not math.isfinite(robust_sigma) or robust_sigma <= 0.0:
        return pd.Series(True, index=values.index)
    robust_z = abs_dev / robust_sigma
    return (robust_z <= threshold).fillna(False)


def _edge_endpoints(row: dict[str, str]) -> tuple[str, str]:
    source = str(row.get("source_node") or row.get("source") or "").strip()
    target = str(row.get("target_node") or row.get("target") or "").strip()
    return source, target


def compute_normalized_kirchhoff_violation(output_dir: Path) -> float:
    node_path = output_dir / "node_predictions.csv"
    edge_path = output_dir / "edge_predictions.csv"
    if not node_path.exists() or not edge_path.exists():
        return float("nan")

    boundary_role_by_node: dict[str, str] = {}
    with node_path.open("r", newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            node_id = str(row.get("node_id", "")).strip()
            if not node_id:
                continue
            boundary_role_by_node[node_id] = str(row.get("boundary_role", "internal")).strip() or "internal"

    net_by_node: dict[str, float] = {}
    abs_sum_by_node: dict[str, float] = {}
    with edge_path.open("r", newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            source, target = _edge_endpoints(row)
            q_pred = safe_float(row.get("q_pred_m3_s"))
            if not source or not target or not math.isfinite(q_pred):
                continue
            net_by_node[source] = net_by_node.get(source, 0.0) - q_pred
            net_by_node[target] = net_by_node.get(target, 0.0) + q_pred
            abs_q = abs(q_pred)
            abs_sum_by_node[source] = abs_sum_by_node.get(source, 0.0) + abs_q
            abs_sum_by_node[target] = abs_sum_by_node.get(target, 0.0) + abs_q

    normalized_values: list[float] = []
    for node_id, role in boundary_role_by_node.items():
        if role != "internal":
            continue
        abs_sum = abs_sum_by_node.get(node_id, 0.0)
        if abs_sum <= 0.0:
            continue
        normalized_values.append(net_by_node.get(node_id, 0.0) / abs_sum)
    if not normalized_values:
        return float("nan")
    return math.sqrt(sum(value * value for value in normalized_values) / len(normalized_values))


def attach_normalized_kirchhoff_violation(df: pd.DataFrame) -> pd.DataFrame:
    if "output_dir" not in df.columns:
        result = df.copy()
        result["normalized_kirchhoff_violation"] = float("nan")
        return result
    result = df.copy()
    cache: dict[str, float] = {}
    values: list[float] = []
    for output_dir_value in result["output_dir"]:
        output_dir = str(output_dir_value)
        if output_dir not in cache:
            cache[output_dir] = compute_normalized_kirchhoff_violation(Path(output_dir))
        values.append(cache[output_dir])
    result["normalized_kirchhoff_violation"] = values
    return result


def selected_subset(df: pd.DataFrame) -> pd.DataFrame:
    if "selected_representative" not in df.columns:
        return df.iloc[0:0].copy()
    return df[df["selected_representative"] == True].copy()


def plot_nonselected_gnn(ax: plt.Axes, df: pd.DataFrame, x: str, y: str) -> None:
    nonselected = df[df["selected_representative"] != True].copy()
    for regime in REGIME_ORDER:
        subset = nonselected[nonselected["weighting_regime"] == regime]
        if subset.empty:
            continue
        ax.scatter(
            subset[x],
            subset[y],
            s=34,
            alpha=0.38,
            color=REGIME_COLORS[regime],
            edgecolors="none",
            zorder=1,
        )


def overlay_pareto_highlights(
    ax: plt.Axes,
    gnn_df: pd.DataFrame,
    x: str,
    y: str,
    global_front_mask: pd.Series | None = None,
) -> None:
    front = gnn_df[gnn_df["is_pareto_front"] == True].copy()
    if not front.empty:
        ax.scatter(
            front[x],
            front[y],
            s=64,
            facecolors="none",
            edgecolors="#202020",
            linewidths=1.2,
            zorder=3,
        )
    if global_front_mask is not None and bool(global_front_mask.any()):
        global_front = gnn_df[global_front_mask].copy()
        ax.scatter(
            global_front[x],
            global_front[y],
            s=84,
            facecolors="none",
            edgecolors="#8a8a8a",
            linewidths=0.9,
            linestyles="dashed",
            zorder=2.8,
        )


def build_tradeoff_envelope_curve(
    front: pd.DataFrame,
    pois_df: pd.DataFrame,
    x: str,
    y: str,
) -> tuple[np.ndarray, np.ndarray] | None:
    if front.empty or pois_df.empty:
        return None
    front_coords = (
        front[[x, y]]
        .apply(pd.to_numeric, errors="coerce")
        .replace([np.inf, -np.inf], np.nan)
        .dropna()
        .sort_values(x)
        .drop_duplicates(subset=[x, y])
    )
    pois_coords = (
        pois_df[[x, y]]
        .apply(pd.to_numeric, errors="coerce")
        .replace([np.inf, -np.inf], np.nan)
        .dropna()
    )
    if front_coords.empty or pois_coords.empty:
        return None
    x_values = front_coords[x].to_numpy(dtype=float)
    y_values = front_coords[y].to_numpy(dtype=float)
    x_limit = float(pois_coords[x].mean())
    y_limit = float(pois_coords[y].mean())
    x_min = float(np.min(x_values))
    b_min = max(-x_min + 1.0e-6, 1.0e-6)
    b_candidates = np.concatenate(
        [
            np.array([0.0], dtype=float),
            np.geomspace(1.0e-5, 10.0, 400),
        ]
    )
    best_params: tuple[float, float, float] | None = None
    best_loss = float("inf")
    for b in b_candidates:
        if b < 0.0:
            continue
        shifted_front = 1.0 / (x_values + b)
        shifted_limit = 1.0 / (x_limit + b)
        basis = shifted_front - shifted_limit
        denom = float(np.dot(basis, basis))
        if denom <= 0.0 or not np.isfinite(denom):
            continue
        a = float(np.dot(basis, y_values - y_limit) / denom)
        y_pred = y_limit + a * basis
        loss = float(np.mean((y_pred - y_values) ** 2))
        if np.isfinite(loss) and loss < best_loss:
            best_loss = loss
            best_params = (a, b, y_limit)
    if best_params is None:
        return None
    a, b, c = best_params
    x_fit = np.linspace(x_min, x_limit, 256)
    y_fit = c + a * (1.0 / (x_fit + b) - 1.0 / (x_limit + b))
    return x_fit, y_fit


def plot_tradeoff_fit_curve(
    ax: plt.Axes,
    gnn_df: pd.DataFrame,
    pois_df: pd.DataFrame,
    x: str,
    y: str,
) -> None:
    front = gnn_df[gnn_df["is_pareto_front"] == True].copy()
    fit = build_tradeoff_envelope_curve(front, pois_df, x, y)
    if fit is None:
        return
    x_fit, y_fit = fit
    ax.plot(
        x_fit,
        y_fit,
        color="#111111",
        linewidth=1.8,
        alpha=0.9,
        zorder=2.6,
    )


def plot_poiseuille(ax: plt.Axes, pois_df: pd.DataFrame, x: str, y: str) -> None:
    if pois_df.empty:
        return
    ax.scatter(
        pois_df[x],
        pois_df[y],
        marker="D",
        s=40,
        color="#9a9a9a",
        edgecolors="#5f5f5f",
        linewidths=0.5,
        alpha=0.75,
        zorder=2,
    )


def plot_selected_stars(ax: plt.Axes, rep_df: pd.DataFrame, x: str, y: str) -> None:
    if rep_df.empty:
        return
    ax.scatter(
        rep_df[x],
        rep_df[y],
        marker="*",
        s=95,
        color="#111111",
        edgecolors="white",
        linewidths=0.6,
        alpha=1.0,
        zorder=4,
    )


def annotate_selected_labels(ax: plt.Axes, rep_df: pd.DataFrame, x: str, y: str) -> None:
    rep_df = rep_df[pd.to_numeric(rep_df["selection_rank_within_regime"], errors="coerce") <= 2].copy()
    offset_map = {
        "F1": (8, 8),
        "F2": (8, -10),
        "B1": (8, 10),
        "B2": (-18, 10),
        "K1": (-18, 8),
        "K2": (-18, -10),
        "C1": (8, 12),
        "C2": (-18, 12),
    }
    for _, row in rep_df.iterrows():
        label = str(row.get("plot_label", ""))
        if not label:
            continue
        dx, dy = offset_map.get(label, (8, 8))
        ax.annotate(
            label,
            xy=(row[x], row[y]),
            xytext=(dx, dy),
            textcoords="offset points",
            fontsize=9,
            color="#111111",
            zorder=5,
        )


def base_axes(title: str, xlabel: str, ylabel: str) -> tuple[plt.Figure, plt.Axes]:
    fig, ax = plt.subplots(figsize=(7.2, 5.4))
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    return fig, ax


def add_legend(
    ax: plt.Axes,
    include_pareto: bool,
    include_baseline: bool,
    include_selected_representative: bool,
    include_tradeoff_fit: bool = False,
) -> None:
    handles = weighting_regime_legend_handles(
        include_pareto=include_pareto,
        include_baseline=include_baseline,
    )
    if include_tradeoff_fit:
        handles.append(
            Line2D(
                [0],
                [0],
                color="#111111",
                linewidth=1.8,
                label="Hyperbola-style fit to Poiseuille limit",
            )
        )
    if include_selected_representative:
        handles.append(
            Line2D(
                [0],
                [0],
                marker="*",
                linestyle="None",
                markerfacecolor="#111111",
                markeredgecolor="white",
                markeredgewidth=0.6,
                markersize=10,
                label="Selected representative",
            )
        )
    ax.legend(handles=handles, frameon=False, loc="best")


def plot_pareto(
    gnn_df: pd.DataFrame,
    pois_df: pd.DataFrame,
    rep_df: pd.DataFrame,
    output_dir: Path,
    dpi: int,
    label_selected: bool,
    show_selected_stars: bool,
    filename: str,
) -> None:
    fig, ax = base_axes(
        "Flow-conservation trade-off",
        "Flow RMSE (nL/s)",
        "Kirchhoff RMS per internal node (nL/s)",
    )
    plot_nonselected_gnn(ax, gnn_df, "flow_rmse_nl_s", "kirchhoff_rms_per_internal_node_nl_s")
    overlay_pareto_highlights(
        ax,
        gnn_df,
        "flow_rmse_nl_s",
        "kirchhoff_rms_per_internal_node_nl_s",
    )
    plot_poiseuille(ax, pois_df, "flow_rmse_nl_s", "kirchhoff_rms_per_internal_node_nl_s")
    if show_selected_stars:
        plot_selected_stars(ax, rep_df, "flow_rmse_nl_s", "kirchhoff_rms_per_internal_node_nl_s")
    if label_selected and show_selected_stars:
        annotate_selected_labels(ax, rep_df, "flow_rmse_nl_s", "kirchhoff_rms_per_internal_node_nl_s")
    add_legend(
        ax,
        include_pareto=True,
        include_baseline=True,
        include_selected_representative=show_selected_stars,
    )
    save(fig, output_dir / filename, dpi=dpi)


def plot_pareto_with_tradeoff_fit(
    gnn_df: pd.DataFrame,
    pois_df: pd.DataFrame,
    rep_df: pd.DataFrame,
    output_dir: Path,
    dpi: int,
    filename: str,
) -> None:
    fig, ax = base_axes(
        "Flow-conservation trade-off with frontier fit",
        "Flow RMSE (nL/s)",
        "Kirchhoff RMS per internal node (nL/s)",
    )
    plot_nonselected_gnn(ax, gnn_df, "flow_rmse_nl_s", "kirchhoff_rms_per_internal_node_nl_s")
    plot_tradeoff_fit_curve(
        ax,
        gnn_df,
        pois_df,
        "flow_rmse_nl_s",
        "kirchhoff_rms_per_internal_node_nl_s",
    )
    overlay_pareto_highlights(
        ax,
        gnn_df,
        "flow_rmse_nl_s",
        "kirchhoff_rms_per_internal_node_nl_s",
    )
    plot_poiseuille(ax, pois_df, "flow_rmse_nl_s", "kirchhoff_rms_per_internal_node_nl_s")
    plot_selected_stars(ax, rep_df, "flow_rmse_nl_s", "kirchhoff_rms_per_internal_node_nl_s")
    add_legend(
        ax,
        include_pareto=True,
        include_baseline=True,
        include_selected_representative=True,
        include_tradeoff_fit=True,
    )
    save(fig, output_dir / filename, dpi=dpi)


def plot_delta_metric(
    gnn_df: pd.DataFrame,
    rep_df: pd.DataFrame,
    output_dir: Path,
    dpi: int,
    y_column: str,
    y_label: str,
    title: str,
    filename: str,
) -> None:
    df = filter_valid(gnn_df, ["delta_rms", y_column])
    rep = filter_valid(rep_df, ["delta_rms", y_column])
    fig, ax = base_axes(title, "Correction RMS", y_label)
    plot_nonselected_gnn(ax, df, "delta_rms", y_column)
    plot_selected_stars(ax, rep, "delta_rms", y_column)
    add_legend(ax, include_pareto=False, include_baseline=False, include_selected_representative=True)
    save(fig, output_dir / filename, dpi=dpi)


def plot_lambda_ratio_metric(
    gnn_df: pd.DataFrame,
    rep_df: pd.DataFrame,
    output_dir: Path,
    dpi: int,
    y_column: str,
    y_label: str,
    title: str,
    filename: str,
) -> None:
    df = filter_valid(gnn_df, ["jittered_log_q_over_k", y_column])
    rep = filter_valid(rep_df, ["jittered_log_q_over_k", y_column])
    fig, ax = base_axes(title, "log10(lambda_Q / lambda_K)", y_label)
    plot_nonselected_gnn(ax, df, "jittered_log_q_over_k", y_column)
    plot_selected_stars(ax, rep, "jittered_log_q_over_k", y_column)
    ax.axvline(0.0, color="#6f6f6f", linewidth=1.0, linestyle="--", alpha=0.7, zorder=0)
    add_legend(ax, include_pareto=False, include_baseline=False, include_selected_representative=True)
    save(fig, output_dir / filename, dpi=dpi)


def plot_lambda_ratio_by_delta(
    gnn_df: pd.DataFrame,
    rep_df: pd.DataFrame,
    output_dir: Path,
    dpi: int,
    y_column: str,
    y_label: str,
    title: str,
    filename: str,
) -> None:
    df = filter_valid(gnn_df, ["jittered_log_q_over_k", "lambda_delta", y_column])
    rep = filter_valid(rep_df, ["jittered_log_q_over_k", "lambda_delta", y_column])
    delta_values = sorted(df["lambda_delta"].dropna().unique())
    if not delta_values:
        return
    fig, axes = plt.subplots(
        1,
        len(delta_values),
        figsize=(4.0 * len(delta_values), 4.4),
        sharey=True,
    )
    if len(delta_values) == 1:
        axes = [axes]
    for ax, delta_value in zip(axes, delta_values):
        panel_df = df[df["lambda_delta"] == delta_value]
        panel_rep = rep[rep["lambda_delta"] == delta_value]
        plot_nonselected_gnn(ax, panel_df, "jittered_log_q_over_k", y_column)
        plot_selected_stars(ax, panel_rep, "jittered_log_q_over_k", y_column)
        ax.axvline(0.0, color="#6f6f6f", linewidth=1.0, linestyle="--", alpha=0.7, zorder=0)
        ax.set_title(f"lambda_delta = {delta_value:g}")
        ax.set_xlabel("log10(lambda_Q / lambda_K)")
        ax.set_axisbelow(True)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
    axes[0].set_ylabel(y_label)
    fig.suptitle(title)
    fig.tight_layout()
    save(fig, output_dir / filename, dpi=dpi)


def plot_lambda_delta_metric(
    gnn_df: pd.DataFrame,
    rep_df: pd.DataFrame,
    output_dir: Path,
    dpi: int,
    y_column: str,
    y_label: str,
    title: str,
    filename: str,
) -> None:
    df = filter_valid(gnn_df, ["lambda_delta", y_column])
    rep = filter_valid(rep_df, ["lambda_delta", y_column])
    fig, ax = base_axes(title, "lambda_delta", y_label)
    plot_nonselected_gnn(ax, df, "lambda_delta", y_column)
    plot_selected_stars(ax, rep, "lambda_delta", y_column)
    ax.set_xscale("log")
    add_legend(ax, include_pareto=False, include_baseline=False, include_selected_representative=True)
    save(fig, output_dir / filename, dpi=dpi)


def plot_normalized_kirchhoff_pareto(
    gnn_df: pd.DataFrame,
    pois_df: pd.DataFrame,
    rep_df: pd.DataFrame,
    output_dir: Path,
    dpi: int,
) -> None:
    df = filter_valid(gnn_df, ["flow_rmse_nl_s", "normalized_kirchhoff_violation"])
    pois = filter_valid(pois_df, ["flow_rmse_nl_s", "normalized_kirchhoff_violation"])
    reps = filter_valid(rep_df, ["flow_rmse_nl_s", "normalized_kirchhoff_violation"])
    gnn_mask = mad_inlier_mask(df["normalized_kirchhoff_violation"])
    pois_mask = mad_inlier_mask(pois["normalized_kirchhoff_violation"]) if not pois.empty else pd.Series(dtype=bool)
    df = df.loc[gnn_mask].copy()
    if not pois.empty:
        pois = pois.loc[pois_mask].copy()
    if not reps.empty:
        allowed_names = set(df["run_name"].astype(str))
        reps = reps[reps["run_name"].astype(str).isin(allowed_names)].copy()
    fig, ax = base_axes(
        "Flow-conservation trade-off",
        "Flow RMSE (nL/s)",
        "Kirchhoff violation RMS per internal node: sum(Q_i) / sum(abs(Q_i))",
    )
    plot_nonselected_gnn(ax, df, "flow_rmse_nl_s", "normalized_kirchhoff_violation")
    overlay_pareto_highlights(
        ax,
        df,
        "flow_rmse_nl_s",
        "normalized_kirchhoff_violation",
    )
    plot_poiseuille(ax, pois, "flow_rmse_nl_s", "normalized_kirchhoff_violation")
    plot_selected_stars(ax, reps, "flow_rmse_nl_s", "normalized_kirchhoff_violation")
    add_legend(
        ax,
        include_pareto=True,
        include_baseline=True,
        include_selected_representative=True,
    )
    save(fig, output_dir / "flow_kirchhoff_normalized_violation_pareto.png", dpi=dpi)


def write_representative_plot_labels(output_dir: Path, label_df: pd.DataFrame) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    label_df.to_csv(output_dir / "representative_plot_labels.csv", index=False)


def main() -> None:
    args = parse_args()
    paths = resolve_paths(args)
    configure_matplotlib()
    if args.clean_obsolete:
        maybe_clean_obsolete(paths["output_dir"])

    all_df = load_csv(paths["all_runs_csv"])
    gnn_df = load_csv(paths["gnn_summary_csv"])
    pois_df = load_csv(paths["poiseuille_summary_csv"], allow_empty=True)
    rep_df = load_csv(paths["representatives_csv"])

    label_df = build_label_lookup(
        numeric(
            rep_df,
            [
                "selection_rank_within_regime",
                "selection_score",
                "lambda_q",
                "lambda_k",
                "lambda_delta",
                "flow_rmse_nl_s",
                "kirchhoff_rms_per_internal_node_nl_s",
                "delta_rms",
            ],
        )
    )
    rep_df = merge_representative_labels(rep_df, label_df)
    gnn_all, gnn_summary, pois_summary, rep_prepped = prepare_data(all_df, gnn_df, pois_df, rep_df)
    gnn_all = attach_normalized_kirchhoff_violation(gnn_all)
    gnn_summary = attach_normalized_kirchhoff_violation(gnn_summary)
    pois_summary = attach_normalized_kirchhoff_violation(pois_summary)
    rep_prepped = attach_normalized_kirchhoff_violation(rep_prepped)

    write_representative_plot_labels(paths["output_dir"], label_df)
    plot_pareto(
        gnn_df=gnn_all,
        pois_df=pois_summary,
        rep_df=rep_prepped,
        output_dir=paths["output_dir"],
        dpi=args.dpi,
        label_selected=False,
        show_selected_stars=True,
        filename="flow_kirchhoff_pareto.png",
    )
    plot_pareto(
        gnn_df=gnn_all,
        pois_df=pois_summary,
        rep_df=rep_prepped,
        output_dir=paths["output_dir"],
        dpi=args.dpi,
        label_selected=True,
        show_selected_stars=True,
        filename="flow_kirchhoff_pareto_labeled.png",
    )
    plot_pareto(
        gnn_df=gnn_all,
        pois_df=pois_summary,
        rep_df=rep_prepped,
        output_dir=paths["output_dir"],
        dpi=args.dpi,
        label_selected=False,
        show_selected_stars=False,
        filename="flow_kirchhoff_pareto_no_selected_stars.png",
    )
    plot_pareto_with_tradeoff_fit(
        gnn_df=gnn_all,
        pois_df=pois_summary,
        rep_df=rep_prepped,
        output_dir=paths["output_dir"],
        dpi=args.dpi,
        filename="flow_kirchhoff_pareto_with_fit.png",
    )
    plot_delta_metric(
        gnn_df=gnn_summary,
        rep_df=rep_prepped,
        output_dir=paths["output_dir"],
        dpi=args.dpi,
        y_column="flow_rmse_nl_s",
        y_label="Flow RMSE (nL/s)",
        title="Flow error versus conductance-correction magnitude",
        filename="flow_rmse_vs_delta_rms.png",
    )
    plot_delta_metric(
        gnn_df=gnn_summary,
        rep_df=rep_prepped,
        output_dir=paths["output_dir"],
        dpi=args.dpi,
        y_column="kirchhoff_rms_per_internal_node_nl_s",
        y_label="Kirchhoff RMS per internal node (nL/s)",
        title="Conservation error versus conductance-correction magnitude",
        filename="kirchhoff_rms_vs_delta_rms.png",
    )
    plot_lambda_ratio_metric(
        gnn_df=gnn_summary,
        rep_df=rep_prepped,
        output_dir=paths["output_dir"],
        dpi=args.dpi,
        y_column="flow_rmse_nl_s",
        y_label="Flow RMSE (nL/s)",
        title="Flow error versus relative flow-conservation weighting",
        filename="supp_flow_rmse_vs_log_lambda_q_over_k.png",
    )
    plot_lambda_ratio_metric(
        gnn_df=gnn_summary,
        rep_df=rep_prepped,
        output_dir=paths["output_dir"],
        dpi=args.dpi,
        y_column="kirchhoff_rms_per_internal_node_nl_s",
        y_label="Kirchhoff RMS per internal node (nL/s)",
        title="Conservation error versus relative flow-conservation weighting",
        filename="supp_kirchhoff_rms_vs_log_lambda_q_over_k.png",
    )
    plot_lambda_delta_metric(
        gnn_df=gnn_summary,
        rep_df=rep_prepped,
        output_dir=paths["output_dir"],
        dpi=args.dpi,
        y_column="delta_rms",
        y_label="Correction RMS",
        title="Correction magnitude versus correction weight",
        filename="supp_delta_rms_vs_lambda_delta.png",
    )
    plot_lambda_delta_metric(
        gnn_df=gnn_summary,
        rep_df=rep_prepped,
        output_dir=paths["output_dir"],
        dpi=args.dpi,
        y_column="flow_rmse_nl_s",
        y_label="Flow RMSE (nL/s)",
        title="Flow error versus correction weight",
        filename="supp_flow_rmse_vs_lambda_delta.png",
    )
    plot_normalized_kirchhoff_pareto(
        gnn_df=gnn_all,
        pois_df=pois_summary,
        rep_df=rep_prepped,
        output_dir=paths["output_dir"],
        dpi=args.dpi,
    )
    plot_lambda_ratio_by_delta(
        gnn_df=gnn_summary,
        rep_df=rep_prepped,
        output_dir=paths["output_dir"],
        dpi=args.dpi,
        y_column="flow_rmse_nl_s",
        y_label="Flow RMSE (nL/s)",
        title="Flow error versus relative flow-conservation weighting by correction weight",
        filename="flow_rmse_vs_log_lambda_q_over_k_by_delta.png",
    )
    plot_lambda_ratio_by_delta(
        gnn_df=gnn_summary,
        rep_df=rep_prepped,
        output_dir=paths["output_dir"],
        dpi=args.dpi,
        y_column="kirchhoff_rms_per_internal_node_nl_s",
        y_label="Kirchhoff RMS per internal node (nL/s)",
        title="Conservation error versus relative flow-conservation weighting by correction weight",
        filename="kirchhoff_rms_vs_log_lambda_q_over_k_by_delta.png",
    )


if __name__ == "__main__":
    main()
