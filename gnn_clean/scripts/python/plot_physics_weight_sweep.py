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
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D
from matplotlib.colors import LogNorm, Normalize
import numpy as np
import pandas as pd

from physics_weight_sweep_lib import classify_weighting_regime

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
    "flow_kirchhoff_normalized_violation_pareto.png",
    "flow_kirchhoff_pareto_labeled.png",
    "flow_kirchhoff_pareto_no_selected_stars.png",
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
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def maybe_clean_obsolete(output_dir: Path) -> None:
    for name in OBSOLETE_OUTPUTS:
        path = output_dir / name
        if path.exists():
            path.unlink()


def label_for_regime(regime: str) -> str:
    return REGIME_LABELS.get(regime, regime.replace("_", " ").title())


def weighting_regime_legend_handles(include_baseline: bool) -> list[Line2D]:
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
    if "plot_label" in rep_df.columns:
        result = rep_df.copy()
        existing = result["plot_label"].astype("string")
        lookup = label_df[["run_name", "plot_label"]].drop_duplicates(subset=["run_name"]).copy()
        fill_map = dict(zip(lookup["run_name"].astype(str), lookup["plot_label"].astype(str)))
        result["plot_label"] = existing.fillna(result["run_name"].astype(str).map(fill_map))
        return result
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
    if "weighting_regime" not in pois_summary.columns:
        pois_summary["weighting_regime"] = ""
    pois_summary["weighting_regime"] = [
        classify_weighting_regime(
            safe_float(row.get("lambda_q")),
            safe_float(row.get("lambda_k")),
            0.0,
        )
        for _, row in pois_summary.iterrows()
    ]
    if "is_pareto_front" in pois_summary.columns:
        pois_summary["is_pareto_front"] = pois_summary["is_pareto_front"].fillna(False).astype(bool)
    else:
        pois_summary["is_pareto_front"] = False

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


def build_tradeoff_envelope_curve(
    front: pd.DataFrame,
    pois_df: pd.DataFrame,
    x: str,
    y: str,
    x_domain: tuple[float, float] | None = None,
    *,
    loss_mode: str = "absolute",
) -> tuple[np.ndarray, np.ndarray] | None:
    if front.empty:
        return None
    front_coords = (
        front[[x, y]]
        .apply(pd.to_numeric, errors="coerce")
        .replace([np.inf, -np.inf], np.nan)
        .dropna()
        .sort_values(x)
        .drop_duplicates(subset=[x, y])
    )
    if front_coords.empty:
        return None
    x_values = front_coords[x].to_numpy(dtype=float)
    y_values = front_coords[y].to_numpy(dtype=float)
    fit_x_min = float(np.min(x_values))
    fit_x_max = float(np.max(x_values))
    # Fit y = a / (x + b) + c by grid-searching b and solving least squares
    # for a and c. For some datasets, absolute-error fitting overweights the
    # largest y-values, so we also support log-space scoring for a more balanced
    # visual hyperbola across the full point set.
    b_candidates = np.linspace(0.0, max(2.0 * fit_x_max, 1.0), 1200, dtype=float)
    best_params: tuple[float, float, float] | None = None
    best_score = float("inf")
    for b in b_candidates:
        basis = 1.0 / (x_values + b)
        valid = np.isfinite(basis) & np.isfinite(y_values) & ((x_values + b) > 0.0)
        if int(np.sum(valid)) < 2:
            continue
        design = np.column_stack([basis[valid], np.ones(int(np.sum(valid)), dtype=float)])
        target = y_values[valid]
        try:
            coeffs, _, _, _ = np.linalg.lstsq(design, target, rcond=None)
        except np.linalg.LinAlgError:
            continue
        a = float(coeffs[0])
        c = float(coeffs[1])
        if not (np.isfinite(a) and np.isfinite(c)):
            continue
        y_pred = a * basis[valid] + c
        if not np.all(np.isfinite(y_pred)):
            continue
        if loss_mode == "log":
            positive = (target > 0.0) & (y_pred > 0.0)
            if int(np.sum(positive)) < 2:
                continue
            score = float(
                np.mean(
                    (
                        np.log(y_pred[positive])
                        - np.log(target[positive])
                    )
                    ** 2
                )
            )
        else:
            score = float(np.mean((y_pred - target) ** 2))
        if np.isfinite(score) and score < best_score:
            best_score = score
            best_params = (a, b, c)
    if best_params is None:
        return None
    a, b, c = best_params
    if x_domain is None:
        x_min = fit_x_min
        x_max = fit_x_max
    else:
        x_min = float(min(x_domain))
        x_max = float(max(x_domain))
    x_fit = np.linspace(x_min, x_max, 512)
    y_fit = a / (x_fit + b) + c
    return x_fit, y_fit


def pareto_front_subset(df: pd.DataFrame, x: str, y: str) -> pd.DataFrame:
    coords = (
        df[[x, y]]
        .apply(pd.to_numeric, errors="coerce")
        .replace([np.inf, -np.inf], np.nan)
    )
    valid = coords.notna().all(axis=1)
    if not bool(valid.any()):
        return df.iloc[0:0].copy()
    work = df.loc[valid].copy()
    x_values = pd.to_numeric(work[x], errors="coerce").to_numpy(dtype=float)
    y_values = pd.to_numeric(work[y], errors="coerce").to_numpy(dtype=float)
    keep = np.ones(len(work), dtype=bool)
    for i in range(len(work)):
        for j in range(len(work)):
            if i == j:
                continue
            if (
                x_values[j] <= x_values[i]
                and y_values[j] <= y_values[i]
                and (x_values[j] < x_values[i] or y_values[j] < y_values[i])
            ):
                keep[i] = False
                break
    return work.loc[keep].copy()


def plot_tradeoff_fit_curve(
    ax: plt.Axes,
    df: pd.DataFrame,
    x: str,
    y: str,
    x_domain: tuple[float, float] | None = None,
    *,
    color: str,
    linestyle: str,
    fit_subset: str = "pareto",
    loss_mode: str = "absolute",
) -> None:
    if fit_subset == "all":
        front = df.copy()
    else:
        if "is_pareto_front" in df.columns and bool(df["is_pareto_front"].fillna(False).astype(bool).any()):
            front = df[df["is_pareto_front"] == True].copy()
        else:
            front = pareto_front_subset(df, x, y)
    fit = build_tradeoff_envelope_curve(front, df, x, y, x_domain=x_domain, loss_mode=loss_mode)
    if fit is None:
        return
    x_fit, y_fit = fit
    ax.plot(
        x_fit,
        y_fit,
        color=color,
        linewidth=1.8,
        linestyle=linestyle,
        alpha=0.9,
        zorder=2.6,
    )


def plot_poiseuille(ax: plt.Axes, pois_df: pd.DataFrame, x: str, y: str) -> None:
    if pois_df.empty:
        return
    for regime in REGIME_ORDER:
        subset = pois_df[pois_df["weighting_regime"] == regime]
        if subset.empty:
            continue
        ax.scatter(
            subset[x],
            subset[y],
            marker="D",
            s=40,
            color=REGIME_COLORS[regime],
            edgecolors="#5f5f5f",
            linewidths=0.5,
            alpha=0.55,
            zorder=2,
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
    include_baseline: bool,
    include_tradeoff_fit: bool = False,
    include_poiseuille_fit: bool = False,
) -> None:
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
    handles.append(
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="None",
            markerfacecolor="#7f7f7f",
            markeredgecolor="none",
            markersize=7,
            alpha=0.9,
            label="GNN",
        )
    )
    if include_tradeoff_fit:
        handles.append(
            Line2D(
                [0],
                [0],
                color="#111111",
                linewidth=1.8,
                linestyle="--",
                label="GNN Trade-off Fit",
            )
        )
    if include_baseline:
        handles.append(
            Line2D(
                [0],
                [0],
                marker="D",
                linestyle="None",
                markerfacecolor="#7f7f7f",
                markeredgecolor="#5f5f5f",
                markersize=7,
                alpha=0.8,
                label="Poiseuille baseline",
            )
        )
    if include_poiseuille_fit:
        handles.append(
            Line2D(
                [0],
                [0],
                color="#5f5f5f",
                linewidth=1.8,
                linestyle=":",
                label="Poiseuille Trade-off Fit",
            )
        )
    ax.legend(handles=handles, frameon=False, loc="best")


def plot_pareto(
    gnn_df: pd.DataFrame,
    pois_df: pd.DataFrame,
    output_dir: Path,
    dpi: int,
    filename: str,
) -> None:
    fig, ax = base_axes(
        "Flow-conservation trade-off",
        "Flow RMSE (nL/s)",
        "Kirchhoff RMS per internal node (nL/s)",
    )
    plot_nonselected_gnn(ax, gnn_df, "flow_rmse_nl_s", "kirchhoff_rms_per_internal_node_nl_s")
    plot_poiseuille(ax, pois_df, "flow_rmse_nl_s", "kirchhoff_rms_per_internal_node_nl_s")
    add_legend(ax, include_baseline=True)
    save(fig, output_dir / filename, dpi=dpi)


def plot_pareto_with_tradeoff_fit(
    gnn_df: pd.DataFrame,
    pois_df: pd.DataFrame,
    output_dir: Path,
    dpi: int,
    filename: str,
) -> None:
    fig, ax = base_axes(
        "Flow-conservation trade-off with trade-off fits",
        "Flow RMSE (nL/s)",
        "Kirchhoff RMS per internal node (nL/s)",
    )
    plot_nonselected_gnn(ax, gnn_df, "flow_rmse_nl_s", "kirchhoff_rms_per_internal_node_nl_s")
    ref_fig, ref_ax = base_axes(
        "",
        "Flow RMSE (nL/s)",
        "Kirchhoff RMS per internal node (nL/s)",
    )
    plot_nonselected_gnn(ref_ax, gnn_df, "flow_rmse_nl_s", "kirchhoff_rms_per_internal_node_nl_s")
    plot_poiseuille(ref_ax, pois_df, "flow_rmse_nl_s", "kirchhoff_rms_per_internal_node_nl_s")
    scatter_xlim = ref_ax.get_xlim()
    scatter_ylim = ref_ax.get_ylim()
    plt.close(ref_fig)
    plot_tradeoff_fit_curve(
        ax,
        gnn_df,
        "flow_rmse_nl_s",
        "kirchhoff_rms_per_internal_node_nl_s",
        x_domain=scatter_xlim,
        color="#111111",
        linestyle="--",
        fit_subset="pareto",
        loss_mode="absolute",
    )
    plot_tradeoff_fit_curve(
        ax,
        pois_df,
        "flow_rmse_nl_s",
        "kirchhoff_rms_per_internal_node_nl_s",
        x_domain=scatter_xlim,
        color="#5f5f5f",
        linestyle=":",
        fit_subset="all",
        loss_mode="log",
    )
    plot_poiseuille(ax, pois_df, "flow_rmse_nl_s", "kirchhoff_rms_per_internal_node_nl_s")
    ax.set_xlim(scatter_xlim)
    ax.set_ylim(scatter_ylim)
    add_legend(
        ax,
        include_baseline=True,
        include_tradeoff_fit=True,
        include_poiseuille_fit=True,
    )
    save(fig, output_dir / filename, dpi=dpi)


def plot_delta_metric(
    gnn_df: pd.DataFrame,
    output_dir: Path,
    dpi: int,
    y_column: str,
    y_label: str,
    title: str,
    filename: str,
) -> None:
    df = filter_valid(gnn_df, ["delta_rms", y_column])
    fig, ax = base_axes(title, "Correction RMS", y_label)
    plot_nonselected_gnn(ax, df, "delta_rms", y_column)
    add_legend(ax, include_baseline=False)
    save(fig, output_dir / filename, dpi=dpi)


def plot_lambda_ratio_metric(
    gnn_df: pd.DataFrame,
    output_dir: Path,
    dpi: int,
    y_column: str,
    y_label: str,
    title: str,
    filename: str,
) -> None:
    df = filter_valid(gnn_df, ["jittered_log_q_over_k", y_column])
    fig, ax = base_axes(title, r"$\log_{10}(\lambda_Q / \lambda_K)$", y_label)
    plot_nonselected_gnn(ax, df, "jittered_log_q_over_k", y_column)
    ax.axvline(0.0, color="#6f6f6f", linewidth=1.0, linestyle="--", alpha=0.7, zorder=0)
    add_legend(ax, include_baseline=False)
    save(fig, output_dir / filename, dpi=dpi)


def plot_lambda_ratio_by_delta(
    gnn_df: pd.DataFrame,
    output_dir: Path,
    dpi: int,
    y_column: str,
    y_label: str,
    title: str,
    filename: str,
) -> None:
    df = filter_valid(gnn_df, ["jittered_log_q_over_k", "lambda_delta", y_column])
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
        plot_nonselected_gnn(ax, panel_df, "jittered_log_q_over_k", y_column)
        ax.axvline(0.0, color="#6f6f6f", linewidth=1.0, linestyle="--", alpha=0.7, zorder=0)
        ax.set_title(rf"$\lambda_\delta = {delta_value:g}$")
        ax.set_xlabel(r"$\log_{10}(\lambda_Q / \lambda_K)$")
        ax.set_axisbelow(True)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
    axes[0].set_ylabel(y_label)
    handles = weighting_regime_legend_handles(include_baseline=False)
    fig.legend(
        handles=handles,
        frameon=False,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.94),
        ncol=min(4, len(handles)),
    )
    fig.suptitle(title)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.80))
    save(fig, output_dir / filename, dpi=dpi)


def plot_lambda_delta_metric(
    gnn_df: pd.DataFrame,
    output_dir: Path,
    dpi: int,
    y_column: str,
    y_label: str,
    title: str,
    filename: str,
) -> None:
    df = filter_valid(gnn_df, ["lambda_delta", y_column])
    fig, ax = base_axes(title, r"$\lambda_\delta$", y_label)
    plot_nonselected_gnn(ax, df, "lambda_delta", y_column)
    ax.set_xscale("log")
    add_legend(ax, include_baseline=False)
    save(fig, output_dir / filename, dpi=dpi)


def field_axes(title: str) -> tuple[plt.Figure, plt.Axes]:
    fig, ax = plt.subplots(figsize=(6.8, 6.2), constrained_layout=True)
    ax.set_title(title)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    return fig, ax


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
    arterial = nodes[nodes["is_arterial"].astype(str).str.lower().isin({"true", "1", "yes"})] if "is_arterial" in nodes.columns else nodes.iloc[0:0]
    venous = nodes[nodes["is_venous"].astype(str).str.lower().isin({"true", "1", "yes"})] if "is_venous" in nodes.columns else nodes.iloc[0:0]
    if not arterial.empty:
        ax.scatter(*transform_mosaic_coords(arterial["x_px"], arterial["y_px"], x_bounds, y_bounds), marker="^", color="black", s=18, zorder=4)
    if not venous.empty:
        ax.scatter(*transform_mosaic_coords(venous["x_px"], venous["y_px"], x_bounds, y_bounds), marker="s", color="black", s=16, zorder=4)


def first_populated_numeric_column(df: pd.DataFrame, columns: list[str]) -> np.ndarray:
    for column in columns:
        if column not in df.columns:
            continue
        values = pd.to_numeric(df[column], errors="coerce").to_numpy(dtype=float)
        if np.isfinite(values).any():
            if column == "q_pred_m3_s":
                values = values * 1.0e12
            return values
    return np.full((len(df),), np.nan, dtype=float)


def log_widths(values: np.ndarray) -> np.ndarray:
    return 0.5 + 2.0 * np.clip(np.log10(np.clip(np.abs(values), 1.0e-6, None)) + 3.0, 0.0, 3.0) / 3.0


def load_field_tables(run_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    node_df = numeric(pd.read_csv(run_dir / "node_predictions.csv"), ["node_index", "x_px", "y_px", "pressure_pa"])
    edge_df = numeric(pd.read_csv(run_dir / "edge_predictions.csv"), ["predicted_flow_nl_s", "predicted_flow_physical_nl_s", "q_pred_m3_s", "delta_e"])
    return node_df, edge_df


def plot_field_set(run_dir: Path, output_dir: Path, stem_prefix: str, title_prefix: str, dpi: int) -> None:
    nodes, edges = load_field_tables(run_dir)
    x_bounds, y_bounds = bounds_from_nodes(nodes)
    segments = transform_segments(build_edge_segments(edges, nodes), x_bounds, y_bounds)
    keep = np.asarray([np.isfinite(segment).all() for segment in segments], dtype=bool) if segments else np.asarray([], dtype=bool)
    segments = [segment for segment, ok in zip(segments, keep) if ok]

    flow_values = first_populated_numeric_column(edges, ["predicted_flow_physical_nl_s", "predicted_flow_nl_s", "q_pred_m3_s"])
    if keep.size:
        flow_values = flow_values[keep]
    flow_mag_values = np.abs(flow_values)
    if "delta_e" in edges.columns:
        correction_values = pd.to_numeric(edges["delta_e"], errors="coerce").to_numpy(dtype=float)
    else:
        correction_values = np.zeros(len(edges), dtype=float)
    if keep.size:
        correction_values = correction_values[keep]
    pressure_values = pd.to_numeric(nodes["pressure_pa"], errors="coerce").to_numpy(dtype=float)

    fig, ax = field_axes(f"{title_prefix} Flow Field")
    if segments:
        collection = LineCollection(segments, cmap="coolwarm", linewidths=1.15, zorder=2)
        finite = flow_values[np.isfinite(flow_values)]
        vmax = max(float(np.nanpercentile(np.abs(finite), 95.0)), 1.0e-12) if finite.size else 1.0
        collection.set_array(flow_values)
        collection.set_clim(-vmax, vmax)
        ax.add_collection(collection)
        cbar = fig.colorbar(collection, ax=ax, shrink=0.88, pad=0.02)
        cbar.set_label("Predicted flow (nL/s)")
    draw_boundary_markers_field(ax, nodes, x_bounds, y_bounds)
    decorate_field_axes(ax, x_bounds, y_bounds)
    save(fig, output_dir / f"{stem_prefix}_flow_field.png", dpi=dpi)

    fig, ax = field_axes(f"{title_prefix} Flow Amplitude")
    if segments:
        finite_positive = flow_mag_values[np.isfinite(flow_mag_values) & (flow_mag_values > 0.0)]
        background = LineCollection(segments, colors="#d0cbc4", linewidths=0.5, alpha=0.35, zorder=1)
        ax.add_collection(background)
        collection = LineCollection(
            segments,
            cmap="coolwarm",
            norm=LogNorm(
                vmin=max(float(np.nanpercentile(finite_positive, 1.0)), 1.0e-3) if finite_positive.size else 1.0e-3,
                vmax=max(float(np.nanpercentile(finite_positive, 99.5)), 1.0e-3) if finite_positive.size else 1.0,
            ),
            linewidths=log_widths(flow_mag_values if flow_mag_values.size else np.asarray([1.0])),
            zorder=2,
        )
        collection.set_array(np.clip(flow_mag_values if flow_mag_values.size else np.asarray([1.0]), 1.0e-12, None))
        ax.add_collection(collection)
        cbar = fig.colorbar(collection, ax=ax, shrink=0.88, pad=0.02)
        cbar.set_label("Predicted flow amplitude |Q| (nL/s)")
        node_x, node_y = transform_mosaic_coords(nodes["x_px"], nodes["y_px"], x_bounds, y_bounds)
        ax.scatter(node_x, node_y, s=3, c="#5f5f5f", linewidths=0.0, zorder=3)
    draw_boundary_markers_field(ax, nodes, x_bounds, y_bounds)
    decorate_field_axes(ax, x_bounds, y_bounds)
    save(fig, output_dir / f"{stem_prefix}_flow_amplitude_field.png", dpi=dpi)

    fig, ax = field_axes(f"{title_prefix} Pressure Field")
    if segments:
        ax.add_collection(LineCollection(segments, colors="#d0d0d0", linewidths=0.55, zorder=1))
    node_x, node_y = transform_mosaic_coords(nodes["x_px"], nodes["y_px"], x_bounds, y_bounds)
    scatter = ax.scatter(node_x, node_y, c=pressure_values, cmap="viridis", s=12, zorder=2)
    finite_pressure = pressure_values[np.isfinite(pressure_values)]
    if finite_pressure.size:
        scatter.set_clim(float(np.nanpercentile(finite_pressure, 2.5)), float(np.nanpercentile(finite_pressure, 97.5)))
    cbar = fig.colorbar(scatter, ax=ax, shrink=0.88, pad=0.02)
    cbar.set_label("Pressure [Pa]")
    draw_boundary_markers_field(ax, nodes, x_bounds, y_bounds)
    decorate_field_axes(ax, x_bounds, y_bounds)
    save(fig, output_dir / f"{stem_prefix}_pressure_field.png", dpi=dpi)

    fig, ax = field_axes(f"{title_prefix} Correction Field")
    if segments:
        vmax = max(float(np.nanpercentile(np.abs(correction_values[np.isfinite(correction_values)]), 99.5)), 1.0e-6) if np.isfinite(correction_values).any() else 1.0
        collection = LineCollection(
            segments,
            cmap="coolwarm",
            norm=Normalize(vmin=-vmax, vmax=vmax),
            linewidths=log_widths(np.abs(flow_mag_values) if flow_mag_values.size else np.asarray([1.0])),
            zorder=2,
        )
        collection.set_array(correction_values if correction_values.size else np.asarray([0.0]))
        background = LineCollection(segments, colors="#d0cbc4", linewidths=0.5, alpha=0.35, zorder=1)
        ax.add_collection(background)
        ax.add_collection(collection)
        cbar = fig.colorbar(collection, ax=ax, shrink=0.88, pad=0.02)
        cbar.set_label(r"Correction field $\delta_e$")
        node_x, node_y = transform_mosaic_coords(nodes["x_px"], nodes["y_px"], x_bounds, y_bounds)
        ax.scatter(node_x, node_y, s=3, c="#5f5f5f", linewidths=0.0, zorder=3)
    draw_boundary_markers_field(ax, nodes, x_bounds, y_bounds)
    decorate_field_axes(ax, x_bounds, y_bounds)
    save(fig, output_dir / f"{stem_prefix}_correction_field.png", dpi=dpi)


def select_poiseuille_baseline_row(pois_df: pd.DataFrame) -> pd.Series:
    preferred = pois_df[
        (pd.to_numeric(pois_df["lambda_q"], errors="coerce") == 1.0)
        & (pd.to_numeric(pois_df["lambda_k"], errors="coerce") == 1.0)
    ]
    if not preferred.empty:
        return preferred.iloc[0]
    return pois_df.iloc[0]


def plot_representative_field_sets(rep_df: pd.DataFrame, pois_df: pd.DataFrame, output_dir: Path, dpi: int) -> None:
    field_dir = output_dir / "representative_fields"
    wanted = {"F1", "B1", "K1", "C1"}
    selected = rep_df[rep_df["plot_label"].astype(str).isin(wanted)].copy()
    for _, row in selected.iterrows():
        run_dir = Path(str(row["output_dir"])).expanduser().resolve()
        label = str(row["plot_label"])
        plot_field_set(run_dir, field_dir, label, label, dpi)
    if not pois_df.empty and "output_dir" in pois_df.columns:
        baseline_row = select_poiseuille_baseline_row(pois_df)
        plot_field_set(
            Path(str(baseline_row["output_dir"])).expanduser().resolve(),
            field_dir,
            "poiseuille_baseline",
            "Poiseuille Baseline",
            dpi,
        )


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
    maybe_clean_obsolete(paths["output_dir"])
    plot_pareto(
        gnn_df=gnn_all,
        pois_df=pois_summary,
        output_dir=paths["output_dir"],
        dpi=args.dpi,
        filename="flow_kirchhoff_pareto.png",
    )
    plot_pareto_with_tradeoff_fit(
        gnn_df=gnn_all,
        pois_df=pois_summary,
        output_dir=paths["output_dir"],
        dpi=args.dpi,
        filename="flow_kirchhoff_pareto_with_fit.png",
    )
    plot_delta_metric(
        gnn_df=gnn_summary,
        output_dir=paths["output_dir"],
        dpi=args.dpi,
        y_column="flow_rmse_nl_s",
        y_label="Flow RMSE (nL/s)",
        title="Flow error versus conductance-correction magnitude",
        filename="flow_rmse_vs_delta_rms.png",
    )
    plot_delta_metric(
        gnn_df=gnn_summary,
        output_dir=paths["output_dir"],
        dpi=args.dpi,
        y_column="kirchhoff_rms_per_internal_node_nl_s",
        y_label="Kirchhoff RMS per internal node (nL/s)",
        title="Conservation error versus conductance-correction magnitude",
        filename="kirchhoff_rms_vs_delta_rms.png",
    )
    plot_lambda_ratio_metric(
        gnn_df=gnn_summary,
        output_dir=paths["output_dir"],
        dpi=args.dpi,
        y_column="flow_rmse_nl_s",
        y_label="Flow RMSE (nL/s)",
        title="Flow error versus relative flow-conservation weighting",
        filename="supp_flow_rmse_vs_log_lambda_q_over_k.png",
    )
    plot_lambda_ratio_metric(
        gnn_df=gnn_summary,
        output_dir=paths["output_dir"],
        dpi=args.dpi,
        y_column="kirchhoff_rms_per_internal_node_nl_s",
        y_label="Kirchhoff RMS per internal node (nL/s)",
        title="Conservation error versus relative flow-conservation weighting",
        filename="supp_kirchhoff_rms_vs_log_lambda_q_over_k.png",
    )
    plot_lambda_delta_metric(
        gnn_df=gnn_summary,
        output_dir=paths["output_dir"],
        dpi=args.dpi,
        y_column="delta_rms",
        y_label="Correction RMS",
        title="Correction magnitude versus correction weight",
        filename="supp_delta_rms_vs_lambda_delta.png",
    )
    plot_lambda_delta_metric(
        gnn_df=gnn_summary,
        output_dir=paths["output_dir"],
        dpi=args.dpi,
        y_column="flow_rmse_nl_s",
        y_label="Flow RMSE (nL/s)",
        title="Flow error versus correction weight",
        filename="supp_flow_rmse_vs_lambda_delta.png",
    )
    plot_lambda_ratio_by_delta(
        gnn_df=gnn_summary,
        output_dir=paths["output_dir"],
        dpi=args.dpi,
        y_column="flow_rmse_nl_s",
        y_label="Flow RMSE (nL/s)",
        title="Flow error versus relative flow-conservation weighting by correction weight",
        filename="flow_rmse_vs_log_lambda_q_over_k_by_delta.png",
    )
    plot_lambda_ratio_by_delta(
        gnn_df=gnn_summary,
        output_dir=paths["output_dir"],
        dpi=args.dpi,
        y_column="kirchhoff_rms_per_internal_node_nl_s",
        y_label="Kirchhoff RMS per internal node (nL/s)",
        title="Conservation error versus relative flow-conservation weighting by correction weight",
        filename="kirchhoff_rms_vs_log_lambda_q_over_k_by_delta.png",
    )
    plot_representative_field_sets(rep_prepped, pois_summary, paths["output_dir"], args.dpi)


if __name__ == "__main__":
    main()
