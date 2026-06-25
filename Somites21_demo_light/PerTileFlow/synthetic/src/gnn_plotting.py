"""Diagnostics and standalone dashboards for completed synthetic GNN runs."""

from __future__ import annotations

import json
import os
import textwrap
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection

from distensibility.io import VascularDataset, write_json


CHANNEL_LABELS = ("DC", "H1", "H2")
MODEL_LABELS = {
    "physics_informed_gnn": "Physics-informed GNN",
    "vanilla_gcn": "Vanilla pressure GCN",
    "edge_local_mlp": "K=0 edge-local MLP",
}


def plot_not_applicable(path: Path, title: str, explanation: str) -> None:
    fig, axis = plt.subplots(figsize=(8, 4.5), constrained_layout=True)
    axis.axis("off")
    axis.text(
        0.5,
        0.55,
        title,
        ha="center",
        va="center",
        fontsize=20,
        weight="bold",
    )
    axis.text(
        0.5,
        0.42,
        textwrap.fill(explanation, width=72),
        ha="center",
        va="center",
        fontsize=10,
    )
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _segments(dataset: VascularDataset):
    source = dataset.edge_source_index
    target = dataset.edge_target_index
    xy = dataset.node_xy_px
    valid = np.isfinite(xy[source]).all(axis=1) & np.isfinite(
        xy[target]
    ).all(axis=1)
    edges = np.flatnonzero(valid)
    return edges, np.stack([xy[source[edges]], xy[target[edges]]], axis=1)


def _line_map(path, dataset, values, title, colorbar, cmap="viridis"):
    edges, segments = _segments(dataset)
    values = np.asarray(values, dtype=float)
    finite = np.isfinite(values[edges])
    edges = edges[finite]
    segments = segments[finite]
    displayed = values[edges]
    if not len(displayed):
        return
    low, high = np.nanpercentile(displayed, [2, 98])
    if not np.isfinite(low) or not np.isfinite(high) or low == high:
        low, high = float(np.nanmin(displayed)), float(np.nanmax(displayed))
    if low == high:
        high = low + 1.0
    fig, axis = plt.subplots(figsize=(9, 7), constrained_layout=True)
    collection = LineCollection(
        segments,
        array=np.clip(displayed, low, high),
        cmap=cmap,
        linewidths=1.0,
    )
    collection.set_clim(low, high)
    axis.add_collection(collection)
    axis.autoscale()
    axis.invert_yaxis()
    axis.set_aspect("equal")
    axis.set_xlabel("mosaic x [px]")
    axis.set_ylabel("mosaic y [px]")
    axis.set_title(title)
    fig.colorbar(collection, ax=axis, label=colorbar)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_training_history(path: Path, history: list[dict]) -> None:
    epochs = [row["epoch"] for row in history]
    fig, axis = plt.subplots(figsize=(7, 4.5), constrained_layout=True)
    axis.plot(epochs, [row["train_loss"] for row in history], label="train")
    axis.plot(epochs, [row["val_loss"] for row in history], label="validation")
    axis.set_xlabel("epoch")
    axis.set_ylabel("configured loss")
    axis.set_yscale("log")
    axis.set_title("Training history")
    axis.legend()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_pressure_comparison(
    path: Path,
    dataset: VascularDataset,
    predicted_pressure: np.ndarray,
) -> None:
    xy = dataset.node_xy_px
    valid = np.isfinite(xy).all(axis=1) & np.isfinite(predicted_pressure)
    if not valid.any():
        plot_not_applicable(
            path,
            "Pressure field unavailable",
            "This model predicts velocity directly and does not solve nodal pressure.",
        )
        return
    truth = np.asarray(dataset.pressure_true_pa[:, 0].real)
    prediction = np.asarray(predicted_pressure, dtype=float)
    error = prediction - truth
    p_low, p_high = np.nanpercentile(
        np.concatenate([truth[valid], prediction[valid]]), [2, 98]
    )
    emax = max(float(np.nanpercentile(np.abs(error[valid]), 98)), 1.0e-12)
    fig, axes = plt.subplots(1, 3, figsize=(17, 5.4), constrained_layout=True)
    panels = (
        (truth, "True DC pressure", "viridis", p_low, p_high, "P [Pa]"),
        (
            prediction,
            "Predicted DC pressure",
            "viridis",
            p_low,
            p_high,
            "P [Pa]",
        ),
        (error, "Pressure error", "coolwarm", -emax, emax, "error [Pa]"),
    )
    for axis, (values, title, cmap, low, high, label) in zip(axes, panels):
        scatter = axis.scatter(
            xy[valid, 0],
            xy[valid, 1],
            c=values[valid],
            s=4,
            linewidths=0,
            cmap=cmap,
            vmin=low,
            vmax=high,
        )
        axis.invert_yaxis()
        axis.set_aspect("equal")
        axis.set_xlabel("mosaic x [px]")
        axis.set_ylabel("mosaic y [px]")
        axis.set_title(title)
        fig.colorbar(scatter, ax=axis, label=label)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_velocity_parity(
    path: Path,
    dataset: VascularDataset,
    predicted: np.ndarray,
    harmonics: np.ndarray,
) -> None:
    n_channels = len(harmonics)
    fig, axes = plt.subplots(
        1,
        n_channels,
        figsize=(5.3 * n_channels, 5),
        squeeze=False,
        constrained_layout=True,
    )
    for column, harmonic in enumerate(harmonics):
        axis = axes[0, column]
        truth = dataset.velocity_observed_m_s[:, harmonic]
        pred = predicted[:, column]
        if harmonic == 0:
            x, y = truth.real, pred.real
            xlabel = "observed velocity [m/s]"
            ylabel = "predicted velocity [m/s]"
        else:
            x, y = np.abs(truth), np.abs(pred)
            xlabel = "observed |velocity| [m/s]"
            ylabel = "predicted |velocity| [m/s]"
        valid = np.isfinite(x) & np.isfinite(y)
        axis.scatter(x[valid], y[valid], s=5, alpha=0.25)
        low = float(min(np.nanmin(x[valid]), np.nanmin(y[valid])))
        high = float(max(np.nanmax(x[valid]), np.nanmax(y[valid])))
        axis.plot([low, high], [low, high], color="black", linewidth=1)
        axis.set_xlabel(xlabel)
        axis.set_ylabel(ylabel)
        axis.set_title(CHANNEL_LABELS[int(harmonic)])
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_corrections(path: Path, delta: np.ndarray, multiplier: np.ndarray):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.3), constrained_layout=True)
    axes[0].hist(delta[np.isfinite(delta)], bins=60)
    axes[0].set_xlabel(r"$\delta_{DC} = \log(C)$")
    axes[0].set_ylabel("edge count")
    axes[0].set_title("Learned log-conductance correction")
    positive = multiplier[np.isfinite(multiplier) & (multiplier > 0)]
    if positive.min() == positive.max():
        width = max(abs(float(positive[0])) * 0.02, 1.0e-6)
        bins = np.linspace(positive[0] - width, positive[0] + width, 20)
    else:
        bins = np.geomspace(positive.min(), positive.max(), 60)
    axes[1].hist(positive, bins=bins)
    axes[1].set_xscale("log")
    axes[1].set_xlabel("conductance multiplier C")
    axes[1].set_ylabel("edge count")
    axes[1].set_title("Conductance multiplier")
    fig.savefig(path, dpi=180)
    plt.close(fig)


def create_gnn_report(
    dataset: VascularDataset,
    run_dir: Path,
    output_dir: Path,
    selection_metric: str,
    selection_score: float,
) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    config = json.loads((run_dir / "config.yaml").read_text())
    model_name = config["model_name"]
    metrics = json.loads((run_dir / "metrics.json").read_text())
    history = json.loads((run_dir / "training_history.json").read_text())[
        "history"
    ]
    with np.load(run_dir / "predicted_velocities.npz") as archive:
        predicted = archive["predicted_velocity_m_s"].copy()
        harmonics = archive["harmonic_index"].astype(int)
    with np.load(run_dir / "pressure_field.npz") as archive:
        pressure = archive["predicted_pressure_pa"].copy()
    with np.load(run_dir / "corrections.npz") as archive:
        delta = archive["delta_dc"].copy()
        multiplier = archive["conductance_multiplier"].copy()

    plot_training_history(output_dir / "training_history.png", history)
    plot_pressure_comparison(
        output_dir / "pressure_comparison.png", dataset, pressure
    )
    plot_velocity_parity(
        output_dir / "velocity_parity.png",
        dataset,
        predicted,
        harmonics,
    )
    if model_name == "vanilla_gcn":
        explanation = (
            "The vanilla pressure GCN predicts nodal pressure directly and "
            "does not learn an edge conductance correction."
        )
        for filename, title in (
            ("correction_distributions.png", "Conductance correction not applicable"),
            ("delta_map.png", "Correction map not applicable"),
            (
                "conductance_multiplier_map.png",
                "Conductance multiplier not applicable",
            ),
        ):
            plot_not_applicable(output_dir / filename, title, explanation)
    else:
        plot_corrections(
            output_dir / "correction_distributions.png", delta, multiplier
        )
        _line_map(
            output_dir / "delta_map.png",
            dataset,
            delta,
            "Learned DC conductance correction",
            r"$\delta_{DC}=\log(C)$",
            "coolwarm",
        )
        _line_map(
            output_dir / "conductance_multiplier_map.png",
            dataset,
            np.log10(np.maximum(multiplier, 1.0e-30)),
            "Learned conductance multiplier",
            r"$\log_{10}(C)$",
            "coolwarm",
        )
    for column, harmonic in enumerate(harmonics):
        truth = dataset.velocity_observed_m_s[:, harmonic]
        scale = np.maximum(np.abs(truth), np.nanmedian(np.abs(truth)) * 0.05)
        relative = np.abs(predicted[:, column] - truth) / np.maximum(
            scale, 1.0e-15
        )
        _line_map(
            output_dir / f"velocity_error_{CHANNEL_LABELS[harmonic].lower()}.png",
            dataset,
            relative,
            f"{CHANNEL_LABELS[harmonic]} velocity relative error",
            "relative error (clipped 2nd–98th percentiles)",
            "magma",
        )

    summary = {
        "dataset": dataset.path.name,
        "selected_run": run_dir,
        "selection_metric": selection_metric,
        "selection_score": selection_score,
        "model_name": model_name,
        "model_label": MODEL_LABELS.get(model_name, model_name),
        "K": config["K"],
        "harmonic_mode": config["harmonic_mode"],
        "metrics": metrics,
    }
    write_json(output_dir / "selection_summary.json", summary)
    write_dashboard(output_dir / "dashboard.html", summary, harmonics)
    return summary


def plot_validation_comparison(
    path: Path,
    dataset_names: list[str],
    reports_by_model: dict[str, dict[str, dict]],
) -> None:
    """Compare selected validation errors across model families."""
    fig, axes = plt.subplots(1, 2, figsize=(15, 5), constrained_layout=True)
    positions = np.arange(len(dataset_names))
    box_values = []
    box_labels = []
    for model, reports in reports_by_model.items():
        values = np.asarray(
            [
                reports[name]["selection_score"]
                if name in reports
                else np.nan
                for name in dataset_names
            ],
            dtype=float,
        )
        label = MODEL_LABELS.get(model, model)
        axes[0].plot(
            positions,
            values,
            marker="o",
            markersize=3,
            linewidth=1,
            label=label,
        )
        finite = values[np.isfinite(values)]
        if len(finite):
            box_values.append(finite)
            box_labels.append(label)
    axes[0].set_xlabel("synthetic dataset index")
    axes[0].set_ylabel("validation DC relative RMSE")
    axes[0].set_title("Selected validation error by dataset")
    axes[0].set_xticks(
        np.linspace(0, max(len(dataset_names) - 1, 0), num=min(9, len(dataset_names)))
    )
    axes[0].grid(True, axis="y", alpha=0.25)
    axes[0].legend()
    if box_values:
        axes[1].boxplot(box_values, tick_labels=box_labels, showfliers=False)
        axes[1].tick_params(axis="x", rotation=18)
    axes[1].set_ylabel("validation DC relative RMSE")
    axes[1].set_title("Distribution across datasets")
    axes[1].grid(True, axis="y", alpha=0.25)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def write_dashboard(path: Path, summary: dict, harmonics: np.ndarray) -> None:
    images = [
        ("Training history", "training_history.png"),
        ("DC pressure", "pressure_comparison.png"),
        ("Velocity parity", "velocity_parity.png"),
        ("Correction distributions", "correction_distributions.png"),
        ("Correction map", "delta_map.png"),
        ("Conductance multiplier map", "conductance_multiplier_map.png"),
    ]
    images.extend(
        (
            f"{CHANNEL_LABELS[h]} velocity error",
            f"velocity_error_{CHANNEL_LABELS[h].lower()}.png",
        )
        for h in harmonics
    )
    metrics_json = json.dumps(summary["metrics"]).replace("</", "<\\/")
    cards = [
        ("Selected run", Path(summary["selected_run"]).name),
        ("Model", summary["model_name"]),
        ("K", summary["K"]),
        ("Harmonics", summary["harmonic_mode"]),
        ("Validation DC relative RMSE", summary["selection_score"]),
    ]
    card_html = "".join(
        f"<div class='card'><b>{label}</b><span>{value}</span></div>"
        for label, value in cards
    )
    gallery = "".join(
        f"<figure><img src='{filename}' alt='{label}'><figcaption>{label}</figcaption></figure>"
        for label, filename in images
    )
    path.write_text(
        f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Best {summary['model_label']} — {summary['dataset']}</title>
<style>
body{{font:14px/1.45 system-ui,sans-serif;margin:0;background:#f4f6f8;color:#18212b}}
header{{background:#202b36;color:white;padding:18px 24px}}
main{{padding:18px;max-width:1500px;margin:auto}}
.cards{{display:grid;grid-template-columns:repeat(auto-fit,minmax(180px,1fr));gap:10px;margin-bottom:16px}}
.card,figure,section{{background:white;border:1px solid #d9e1e8;border-radius:8px;padding:12px}}
.card b{{display:block;color:#627180;font-size:11px;margin-bottom:4px}}
.gallery{{display:grid;grid-template-columns:repeat(auto-fit,minmax(430px,1fr));gap:14px}}
figure{{margin:0}} img{{width:100%;height:auto}} figcaption{{margin-top:7px;font-weight:600}}
table{{width:100%;border-collapse:collapse}}td{{padding:6px;border-bottom:1px solid #edf1f5}}
td:last-child{{text-align:right}}code{{font-size:12px}}
</style></head><body>
<header><h1>Best {summary['model_label']}</h1><div>{summary['dataset']}</div></header>
<main><div class="cards">{card_html}</div>
<div class="gallery">{gallery}</div>
<section><h2>Saved metrics</h2><table id="metrics"></table></section>
</main><script id="payload" type="application/json">{metrics_json}</script>
<script>
const m=JSON.parse(document.getElementById("payload").textContent);
const rows=[];
function flatten(x,p=""){{for(const [k,v] of Object.entries(x)){{const q=p?p+"."+k:k;if(v&&typeof v==="object"&&!Array.isArray(v))flatten(v,q);else rows.push([q,v]);}}}}
flatten(m);document.getElementById("metrics").innerHTML=rows.map(([k,v])=>`<tr><td>${{k}}</td><td>${{v}}</td></tr>`).join("");
</script></body></html>"""
    )
