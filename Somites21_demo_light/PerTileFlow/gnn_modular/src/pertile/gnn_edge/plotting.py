"""Plotting utilities for the modular GNN edge-flow workflow.

This module owns matplotlib-based diagnostic plots only.

It should not contain model definitions, training loops, pressure solves,
data loading, or command-line parsing.
"""

from __future__ import annotations

import math
import os
from pathlib import Path
from typing import Sequence, Tuple

os.environ.setdefault("MPLCONFIGDIR", "/tmp/gnn_edge_mpl")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .utils import safe_float, write_csv


def plot_loss(history: Sequence[dict], out_dir: Path) -> None:
    """Plot training loss curves for all trained models."""
    if not history:
        return

    plt.figure(figsize=(7, 4))

    for model_name in sorted({r["model"] for r in history}):
        rows = [r for r in history if r["model"] == model_name]
        plt.plot(
            [r["epoch"] for r in rows],
            [r["loss"] for r in rows],
            label=model_name,
        )

    plt.xlabel("epoch")
    plt.ylabel("training loss")
    plt.yscale("log")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "training_loss.png", dpi=180)
    plt.close()


def plot_validation_history(history: Sequence[dict], out_dir: Path) -> None:
    """Plot train/validation loss history."""
    if not history:
        return

    epochs = [int(r["epoch"]) for r in history]

    plt.figure(figsize=(7, 4))
    plt.plot(epochs, [float(r["train_loss"]) for r in history], label="train")
    plt.plot(epochs, [float(r["val_loss"]) for r in history], label="validation")
    plt.xlabel("epoch")
    plt.ylabel("flow MSE (nL/s)^2")
    plt.yscale("log")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "validation_loss.png", dpi=180)
    plt.close()


def plot_regression_fit(
    rows: Sequence[dict],
    model_name: str,
    y: np.ndarray,
    pred: np.ndarray,
    out_dir: Path,
) -> None:
    """Plot observed-vs-predicted delta for a linear diagnostic regression."""
    del rows

    safe_name = model_name.lower().replace(" ", "_")

    plt.figure(figsize=(5, 5))
    plt.scatter(y, pred, s=12, alpha=0.6)

    lo = float(np.nanmin([np.nanmin(y), np.nanmin(pred)]))
    hi = float(np.nanmax([np.nanmax(y), np.nanmax(pred)]))

    plt.plot([lo, hi], [lo, hi], color="black", lw=1)
    plt.xlabel("observed delta")
    plt.ylabel("regression-predicted delta")
    plt.tight_layout()
    plt.savefig(out_dir / f"regression_{safe_name}_delta_pred_vs_obs.png", dpi=180)
    plt.close()

    plt.figure(figsize=(6, 4))
    plt.hist(pred - y, bins=40)
    plt.xlabel("regression residual in delta")
    plt.ylabel("count")
    plt.tight_layout()
    plt.savefig(out_dir / f"regression_{safe_name}_residual_hist.png", dpi=180)
    plt.close()


def plot_prediction(edge_rows: Sequence[dict], out_dir: Path, prefix: str) -> None:
    """Plot prediction-vs-observation and conductance-correction diagnostics."""
    rows = [r for r in edge_rows if int(r["valid_obs"]) == 1]
    if not rows:
        return

    q_obs = np.asarray([r["Q_obs_nL_s"] for r in rows], dtype=float)
    q_hat = np.asarray([r["Q_hat_nL_s"] for r in rows], dtype=float)
    resid = q_obs - q_hat

    radius = np.asarray([r["radius_m"] for r in rows], dtype=float)
    length = np.asarray([r["length_m"] for r in rows], dtype=float)
    g_pois = np.asarray([r["G_pois"] for r in rows], dtype=float)

    plt.figure(figsize=(5, 5))
    plt.scatter(q_obs, q_hat, s=12, alpha=0.6)

    lo = float(np.nanmin([q_obs.min(), q_hat.min()]))
    hi = float(np.nanmax([q_obs.max(), q_hat.max()]))

    plt.plot([lo, hi], [lo, hi], color="black", lw=1)
    plt.xlabel("observed Q_DC (nL/s)")
    plt.ylabel("predicted Q_DC (nL/s)")
    plt.tight_layout()
    plt.savefig(out_dir / f"{prefix}_q_pred_vs_obs.png", dpi=180)
    plt.close()

    plt.figure(figsize=(6, 4))
    plt.hist(resid, bins=50)
    plt.xlabel("residual Q_DC (nL/s)")
    plt.ylabel("count")
    plt.tight_layout()
    plt.savefig(out_dir / f"{prefix}_residual_hist.png", dpi=180)
    plt.close()

    c_vals = []
    for row in rows:
        try:
            c_vals.append(float(row["C"]))
        except (TypeError, ValueError):
            c_vals.append(float("nan"))

    c = np.asarray(c_vals, dtype=float)

    if np.isfinite(c).any():
        plt.figure(figsize=(6, 4))
        plt.hist(c[np.isfinite(c)], bins=50)
        plt.xlabel("C = G_hat / G_pois")
        plt.ylabel("count")
        plt.tight_layout()
        plt.savefig(out_dir / f"{prefix}_C_hist.png", dpi=180)
        plt.close()

        for x, name, xlabel, logx in (
            (radius, "radius", "radius (m)", True),
            (length, "length", "length (m)", True),
            (g_pois, "G_pois", "Poiseuille conductance", True),
        ):
            plt.figure(figsize=(5.5, 4))
            plt.scatter(x, c, s=12, alpha=0.55)

            if logx:
                plt.xscale("log")

            plt.yscale("log")
            plt.xlabel(xlabel)
            plt.ylabel("C = G_hat / G_pois")
            plt.tight_layout()
            plt.savefig(out_dir / f"{prefix}_C_vs_{name}.png", dpi=180)
            plt.close()


def _edge_midpoints(rows: Sequence[dict]) -> Tuple[np.ndarray, np.ndarray]:
    xs = []
    ys = []

    for r in rows:
        xs.append(safe_float(r.get("x_mid")))
        ys.append(safe_float(r.get("y_mid")))

    return np.asarray(xs, dtype=float), np.asarray(ys, dtype=float)


def _plot_edge_map(
    rows: Sequence[dict],
    key: str,
    out_dir: Path,
    filename: str,
    label: str,
) -> None:
    vals = np.asarray([safe_float(r.get(key)) for r in rows], dtype=float)
    x, y = _edge_midpoints(rows)

    mask = np.isfinite(vals) & np.isfinite(x) & np.isfinite(y)
    if not np.any(mask):
        return

    plt.figure(figsize=(6, 5))
    sc = plt.scatter(x[mask], y[mask], c=vals[mask], s=14)
    plt.gca().invert_yaxis()
    plt.gca().set_aspect("equal", adjustable="box")
    plt.colorbar(sc, label=label)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.tight_layout()
    plt.savefig(out_dir / filename, dpi=180)
    plt.close()


def _plot_node_map(
    rows: Sequence[dict],
    key: str,
    out_dir: Path,
    filename: str,
    label: str,
) -> None:
    vals = np.asarray([safe_float(r.get(key)) for r in rows], dtype=float)
    x = np.asarray([safe_float(r.get("x")) for r in rows], dtype=float)
    y = np.asarray([safe_float(r.get("y")) for r in rows], dtype=float)

    mask = np.isfinite(vals) & np.isfinite(x) & np.isfinite(y)
    if not np.any(mask):
        return

    plt.figure(figsize=(6, 5))
    sc = plt.scatter(x[mask], y[mask], c=vals[mask], s=14)
    plt.gca().invert_yaxis()
    plt.gca().set_aspect("equal", adjustable="box")
    plt.colorbar(sc, label=label)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.tight_layout()
    plt.savefig(out_dir / filename, dpi=180)
    plt.close()


def _scatter(
    rows: Sequence[dict],
    x_key: str,
    y_key: str,
    out_dir: Path,
    filename: str,
    xlabel: str,
    ylabel: str,
    logx: bool = False,
    logy: bool = False,
) -> None:
    x = np.asarray([safe_float(r.get(x_key)) for r in rows], dtype=float)
    y = np.asarray([safe_float(r.get(y_key)) for r in rows], dtype=float)

    mask = np.isfinite(x) & np.isfinite(y)

    if logx:
        mask &= x > 0
    if logy:
        mask &= y > 0

    if not np.any(mask):
        return

    plt.figure(figsize=(5.5, 4))
    plt.scatter(x[mask], y[mask], s=12, alpha=0.55)

    if logx:
        plt.xscale("log")
    if logy:
        plt.yscale("log")

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.tight_layout()
    plt.savefig(out_dir / filename, dpi=180)
    plt.close()


def _hist(
    rows: Sequence[dict],
    key: str,
    out_dir: Path,
    filename: str,
    xlabel: str,
    logx: bool = False,
) -> None:
    vals = np.asarray([safe_float(r.get(key)) for r in rows], dtype=float)
    vals = vals[np.isfinite(vals)]

    if logx:
        vals = vals[vals > 0]

    if vals.size == 0:
        return

    plt.figure(figsize=(6, 4))
    plt.hist(vals, bins=50)

    if logx:
        plt.xscale("log")

    plt.xlabel(xlabel)
    plt.ylabel("count")
    plt.tight_layout()
    plt.savefig(out_dir / filename, dpi=180)
    plt.close()


def _binned_error(
    rows: Sequence[dict],
    by_key: str,
    out_dir: Path,
    filename: str,
    xlabel: str,
) -> None:
    vals = np.asarray([safe_float(r.get(by_key)) for r in rows], dtype=float)
    err = np.asarray([safe_float(r.get("abs_error_nL_s")) for r in rows], dtype=float)

    mask = np.isfinite(vals) & np.isfinite(err)
    vals = vals[mask]
    err = err[mask]

    if vals.size < 5:
        return

    qs = np.unique(np.nanquantile(vals, np.linspace(0.0, 1.0, 6)))
    if qs.size < 3:
        return

    plot_rows = []

    for lo, hi in zip(qs[:-1], qs[1:]):
        m = (vals >= lo) & (vals <= hi if hi == qs[-1] else vals < hi)

        if np.any(m):
            plot_rows.append({
                "bin_center": float(np.nanmean(vals[m])),
                "mean_abs_error": float(np.nanmean(err[m])),
                "n": int(np.sum(m)),
                "lo": float(lo),
                "hi": float(hi),
            })

    write_csv(out_dir / filename.replace(".png", ".csv"), plot_rows)

    if not plot_rows:
        return

    plt.figure(figsize=(6, 4))
    plt.plot(
        [r["bin_center"] for r in plot_rows],
        [r["mean_abs_error"] for r in plot_rows],
        marker="o",
    )
    plt.xlabel(xlabel)
    plt.ylabel("mean |Q_obs - Q_hat| (nL/s)")
    plt.tight_layout()
    plt.savefig(out_dir / filename, dpi=180)
    plt.close()


def plot_group_box(
    rows: Sequence[dict],
    value_key: str,
    group_key: str,
    out_dir: Path,
    filename: str,
    ylabel: str,
) -> None:
    """Make a boxplot grouped by a categorical row field."""
    groups = sorted({
        str(r.get(group_key, ""))
        for r in rows
        if str(r.get(group_key, ""))
    })

    data = []
    labels = []

    for g in groups:
        vals = [
            safe_float(r.get(value_key))
            for r in rows
            if str(r.get(group_key, "")) == g
        ]
        vals = [v for v in vals if math.isfinite(v)]

        if vals:
            data.append(vals)
            labels.append(g)

    if not data:
        return

    plt.figure(figsize=(7, 4))
    plt.boxplot(data, labels=labels, showfliers=False)
    plt.xticks(rotation=25, ha="right")
    plt.ylabel(ylabel)
    plt.tight_layout()
    plt.savefig(out_dir / filename, dpi=180)
    plt.close()


def plot_diagnostic_suite(
    rows: Sequence[dict],
    node_rows: Sequence[dict],
    conservation_rows: Sequence[dict],
    out_dir: Path,
) -> None:
    """Generate the full edge/node diagnostic plot suite."""
    del node_rows

    valid = [r for r in rows if int(r.get("valid_obs", 0)) == 1]
    if not valid:
        return

    _scatter(
        valid,
        "Q_obs_nL_s",
        "residual_nL_s",
        out_dir,
        "residual_vs_Q_obs.png",
        "Q_obs (nL/s)",
        "Q_obs - Q_hat (nL/s)",
    )
    _scatter(
        valid,
        "abs_Q_obs",
        "abs_error_nL_s",
        out_dir,
        "abs_error_vs_abs_Q_obs.png",
        "|Q_obs| (nL/s)",
        "|error| (nL/s)",
    )

    _binned_error(valid, "abs_Q_obs", out_dir, "binned_error_by_abs_Q_obs.png", "|Q_obs|")
    _binned_error(valid, "radius_m", out_dir, "binned_error_by_radius.png", "radius (m)")
    _binned_error(valid, "G_pois", out_dir, "binned_error_by_G_pois.png", "G_pois")

    _plot_edge_map(valid, "residual_nL_s", out_dir, "signed_error_map.png", "Q_obs - Q_hat (nL/s)")
    _plot_edge_map(valid, "abs_error_nL_s", out_dir, "absolute_error_map.png", "|error| (nL/s)")

    _hist(valid, "delta", out_dir, "delta_hist.png", "delta = log C")
    _hist(valid, "C", out_dir, "C_hist_linear.png", "C")
    _hist(valid, "C", out_dir, "C_hist_logx.png", "C", logx=True)

    _scatter(valid, "log_radius", "delta", out_dir, "delta_vs_log_radius.png", "log radius", "delta")
    _scatter(valid, "log_length", "delta", out_dir, "delta_vs_log_length.png", "log length", "delta")
    _scatter(valid, "log_G_pois", "delta", out_dir, "delta_vs_log_G_pois.png", "log G_pois", "delta")

    _scatter(valid, "radius_m", "C", out_dir, "C_vs_radius.png", "radius (m)", "C", logx=True, logy=True)
    _scatter(valid, "length_m", "C", out_dir, "C_vs_length.png", "length (m)", "C", logx=True, logy=True)
    _scatter(valid, "G_pois", "C", out_dir, "C_vs_G_pois.png", "G_pois", "C", logx=True, logy=True)

    _scatter(valid, "abs_Q_obs", "delta", out_dir, "delta_vs_abs_Q_obs.png", "|Q_obs| (nL/s)", "delta")
    _scatter(valid, "pressure_drop_Pa", "delta", out_dir, "delta_vs_pressure_drop.png", "pressure drop (Pa)", "delta")
    _scatter(valid, "residual_nL_s", "delta", out_dir, "delta_vs_residual.png", "Q_obs - Q_hat (nL/s)", "delta")

    _plot_edge_map(valid, "C", out_dir, "C_map.png", "C")
    _plot_edge_map(valid, "delta", out_dir, "delta_map.png", "delta")

    _hist(valid, "pressure_drop_Pa", out_dir, "pressure_drop_hist.png", "pressure drop (Pa)")
    _scatter(valid, "Q_hat_nL_s", "pressure_drop_Pa", out_dir, "pressure_drop_vs_flow.png", "Q_hat (nL/s)", "pressure drop (Pa)")
    _scatter(valid, "radius_m", "pressure_drop_Pa", out_dir, "pressure_drop_vs_radius.png", "radius (m)", "pressure drop (Pa)", logx=True)
    _plot_edge_map(valid, "pressure_drop_Pa", out_dir, "pressure_drop_map.png", "pressure drop (Pa)")

    _scatter(valid, "distance_to_A", "delta", out_dir, "delta_vs_distance_to_A.png", "distance to A/source", "delta")
    _scatter(valid, "distance_to_V", "delta", out_dir, "delta_vs_distance_to_V.png", "distance to V/sink", "delta")
    _scatter(valid, "distance_to_A", "residual_nL_s", out_dir, "residual_vs_distance_to_A.png", "distance to A/source", "residual")
    _scatter(valid, "distance_to_V", "residual_nL_s", out_dir, "residual_vs_distance_to_V.png", "distance to V/sink", "residual")

    plot_group_box(valid, "delta", "topology_class", out_dir, "delta_by_topology_class.png", "delta")
    plot_group_box(valid, "residual_nL_s", "topology_class", out_dir, "residual_by_topology_class.png", "residual (nL/s)")
    plot_group_box(valid, "C", "topology_class", out_dir, "C_by_topology_class.png", "C")
    plot_group_box(
        [r for r in valid if r.get("split") == "val"],
        "abs_error_nL_s",
        "topology_class",
        out_dir,
        "validation_error_by_topology_class.png",
        "|validation error| (nL/s)",
    )

    for key, filename, xlabel in (
        ("abs_H1", "delta_vs_abs_H1.png", "|H1|"),
        ("abs_H2", "delta_vs_abs_H2.png", "|H2|"),
        ("harmonic_ratio", "delta_vs_harmonic_ratio.png", "|H2|/(|H1|+eps)"),
        ("phase_dispersion_H1", "delta_vs_phase_dispersion_H1.png", "H1 phase dispersion"),
    ):
        _scatter(valid, key, "delta", out_dir, filename, xlabel, "delta")

    _scatter(valid, "harmonic_ratio", "residual_nL_s", out_dir, "residual_vs_harmonic_ratio.png", "harmonic ratio", "residual (nL/s)")
    _scatter(valid, "harmonic_ratio", "C", out_dir, "C_vs_harmonic_ratio.png", "harmonic ratio", "C", logy=True)

    _hist(conservation_rows, "conservation_residual_nL_s", out_dir, "conservation_residual_hist.png", "B Q_hat - s (nL/s)")
    _plot_node_map(conservation_rows, "conservation_residual_nL_s", out_dir, "conservation_residual_spatial.png", "B Q_hat - s (nL/s)")


def plot_pressure(
    node_rows: Sequence[dict],
    edge_rows: Sequence[dict],
    out_dir: Path,
    prefix: str,
) -> None:
    """Plot pressure distribution and pressure-vs-distance diagnostics."""
    del edge_rows

    p = np.asarray([r["pressure_Pa"] for r in node_rows], dtype=float)

    plt.figure(figsize=(6, 4))
    plt.hist(p[np.isfinite(p)], bins=50)
    plt.xlabel("pressure (Pa)")
    plt.ylabel("count")
    plt.tight_layout()
    plt.savefig(out_dir / f"{prefix}_pressure_hist.png", dpi=180)
    plt.close()

    spatial = [
        r for r in node_rows
        if math.isfinite(float(r["x"])) and math.isfinite(float(r["y"]))
    ]

    if spatial:
        plt.figure(figsize=(6, 5))
        sc = plt.scatter(
            [r["x"] for r in spatial],
            [r["y"] for r in spatial],
            c=[r["pressure_Pa"] for r in spatial],
            s=12,
        )
        plt.gca().invert_yaxis()
        plt.gca().set_aspect("equal", adjustable="box")
        plt.colorbar(sc, label="pressure (Pa)")
        plt.xlabel("x")
        plt.ylabel("y")
        plt.tight_layout()
        plt.savefig(out_dir / f"{prefix}_pressure_spatial.png", dpi=180)
        plt.close()

    _scatter(
        node_rows,
        "distance_to_A",
        "pressure_Pa",
        out_dir,
        f"{prefix}_pressure_vs_distance_to_A.png",
        "distance to A/source",
        "pressure (Pa)",
    )
    _scatter(
        node_rows,
        "distance_to_V",
        "pressure_Pa",
        out_dir,
        f"{prefix}_pressure_vs_distance_to_V.png",
        "distance to V/sink",
        "pressure (Pa)",
    )


def plot_sweep_summary(rows: Sequence[dict], out_dir: Path) -> None:
    """Plot validation NRMSE summaries across sweep dimensions."""
    if not rows:
        return

    for key, filename, xlabel, logx in (
        ("K", "sweep_val_nrmse_vs_K.png", "message-passing layers K", False),
        ("hidden_dim", "sweep_val_nrmse_vs_hidden_dim.png", "hidden dimension", False),
        ("lambda_delta", "sweep_val_nrmse_vs_lambda_delta.png", "lambda_delta", True),
    ):
        xs = np.asarray([safe_float(r.get(key)) for r in rows], dtype=float)
        ys = np.asarray([safe_float(r.get("val_NRMSE")) for r in rows], dtype=float)

        mask = np.isfinite(xs) & np.isfinite(ys)
        if logx:
            mask &= xs > 0

        if not np.any(mask):
            continue

        plt.figure(figsize=(6, 4))
        plt.scatter(xs[mask], ys[mask], s=40, alpha=0.75)

        groups = sorted(set(float(x) for x in xs[mask]))
        med = [float(np.nanmedian(ys[mask & (xs == g)])) for g in groups]

        plt.plot(groups, med, color="black", lw=1.2, marker="o", label="median")

        if logx:
            plt.xscale("log")

        plt.xlabel(xlabel)
        plt.ylabel("final validation NRMSE")
        plt.legend()
        plt.tight_layout()
        plt.savefig(out_dir / filename, dpi=180)
        plt.close()
