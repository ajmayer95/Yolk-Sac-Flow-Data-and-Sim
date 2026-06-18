"""Experiment orchestration for the modular GNN edge-flow workflow.

This module glues together data, training, physics forward passes, evaluation,
diagnostics, plotting, and checkpoint writing.
"""

from __future__ import annotations

import copy
import math
from pathlib import Path
from typing import List

import numpy as np
import torch

from .data import MosaicData, selected_harmonics
from .diagnostics import (
    add_edge_midpoints,
    add_node_distance_diagnostics,
    collect_conservation_rows,
    collect_edge_rows,
    collect_harmonic_rows,
    collect_node_rows,
    collect_regression_edges,
    enrich_edge_rows,
    interpretation_summary,
    run_delta_regressions,
    write_top_edge_tables,
    write_top_pressure_nodes,
)
from .evaluate import evaluate_arrays, mass_residual_norm, metrics_rows, split_masks
from .physics import physics_forward, poisson_baseline
from .plotting import (
    plot_diagnostic_suite,
    plot_loss,
    plot_prediction,
    plot_pressure,
    plot_sweep_summary,
    plot_validation_history,
)
from .train import train_direct_model, train_physics_model
from .utils import set_seed, write_csv, write_json

def run_experiment(base_data: MosaicData, args, device: torch.device,
                   out_dir: Path, label: str,
                   train_mask: torch.Tensor, val_mask: torch.Tensor,
                   graph=None, train_direct: bool = True) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    data = base_data
    history: List[dict] = []

    has_validation = bool(val_mask.any())
    physics_model, hist, validation_history = train_physics_model(
        data, train_mask, args, device, label,
        val_mask=val_mask if has_validation else None)
    physics_history = list(hist)
    history.extend(hist)
    direct_model = None
    if train_direct:
        direct_model, hist = train_direct_model(data, train_mask, args, device, label)
        history.extend(hist)

    d = data.to(device)
    with torch.no_grad():
        delta = physics_model(d)
        q_hat_h = physics_model.forward_harmonics(d)
        physics_out = physics_forward(d, delta, args.jitter)
        pois_out = poisson_baseline(d, args.jitter)
        if direct_model is not None:
            direct_q = direct_model(d)
            zero_p = torch.full((d.x_node.shape[0],), float("nan"), device=device)
            direct_out = (direct_q, zero_p, torch.full_like(d.g_pois, float("nan")))

    outputs = {
        "poiseuille": pois_out,
        "physics_gnn": physics_out,
    }
    if direct_model is not None:
        outputs["direct_gnn"] = direct_out
    metrics = metrics_rows(data, outputs, train_mask, val_mask)
    write_csv(out_dir / "metrics.csv", metrics)
    write_json(out_dir / "metrics.json", metrics)
    write_csv(out_dir / "training_history.csv", history)
    if validation_history:
        write_csv(out_dir / "validation_history.csv", validation_history)
    write_json(out_dir / "feature_stats.json", data.feature_stats)

    q_hat, p, g_hat = physics_out
    edge_rows = collect_edge_rows(data, q_hat, p, g_hat, train_mask, val_mask)
    harmonic_rows = collect_harmonic_rows(
        data, q_hat_h, train_mask, val_mask,
        selected_harmonics(args.flow_components))
    if harmonic_rows:
        write_csv(out_dir / "harmonic_predictions_physics_gnn.csv", harmonic_rows)
    add_edge_midpoints(data, edge_rows)
    node_rows = collect_node_rows(data, p)
    add_node_distance_diagnostics(data, node_rows)
    write_csv(out_dir / "edge_predictions_physics_gnn.csv", edge_rows)
    write_csv(out_dir / "node_pressures_physics_gnn.csv", node_rows)
    write_top_pressure_nodes(node_rows, out_dir)
    conservation_rows = collect_conservation_rows(data, q_hat)
    write_csv(out_dir / "node_conservation_residuals.csv", conservation_rows)
    max_cons = max((abs(float(r["conservation_residual_nL_s"])) for r in conservation_rows), default=float("nan"))
    print(f"[{label}] max |B Q_hat - s| = {max_cons:.4e} nL/s")
    for r in conservation_rows:
        if r.get("boundary_kind") in ("source", "sink"):
            print(
                f"[{label}] boundary {r['node_id']} {r['boundary_kind']}: "
                f"pred={float(r['predicted_net_flow_nL_s']):+.4g} nL/s "
                f"target={float(r['source_sink_value_nL_s']):+.4g} nL/s"
            )

    enriched_edge_rows = enrich_edge_rows(data, graph, edge_rows) if graph is not None else edge_rows
    write_csv(out_dir / "edge_diagnostics_physics_gnn.csv", enriched_edge_rows)
    write_csv(out_dir / "harmonic_diagnostics.csv", [
        {k: r.get(k, "") for k in [
            "edge_id", "source", "target", "abs_H1", "abs_H2",
            "harmonic_ratio", "phase_H1", "phase_H2", "phase_dispersion_H1",
            "delta", "C", "residual_nL_s",
        ]}
        for r in enriched_edge_rows
    ])
    write_top_edge_tables(enriched_edge_rows, out_dir)

    regression_edge_rows = collect_regression_edges(data, enriched_edge_rows)
    write_csv(out_dir / "delta_regression_edges.csv", regression_edge_rows)
    regression_summary, regression_coefficients = run_delta_regressions(
        regression_edge_rows, out_dir)

    q_pois, p_pois, g_pois = pois_out
    write_csv(
        out_dir / "edge_predictions_poiseuille.csv",
        collect_edge_rows(data, q_pois, p_pois, g_pois, train_mask, val_mask),
    )
    write_csv(out_dir / "node_pressures_poiseuille.csv", collect_node_rows(data, p_pois))

    if direct_model is not None:
        direct_edge_rows = collect_edge_rows(data, direct_q, p, g_hat, train_mask, val_mask)
        for row in direct_edge_rows:
            row["G_hat"] = ""
            row["delta"] = ""
            row["C"] = ""
            row["pressure_drop_Pa"] = ""
        write_csv(out_dir / "edge_predictions_direct_gnn.csv", direct_edge_rows)

    ckpt = {
        "physics_model_state_dict": physics_model.state_dict(),
        "args": vars(args),
        "node_ids": [str(n) for n in data.node_ids],
        "edge_ids": [(str(u), str(v)) for u, v in data.edge_ids],
        "feature_stats": data.feature_stats,
    }
    if direct_model is not None:
        ckpt["direct_model_state_dict"] = direct_model.state_dict()
    torch.save(ckpt, out_dir / "models.pt")

    plot_loss(history, out_dir)
    plot_validation_history(validation_history, out_dir)
    plot_prediction(edge_rows, out_dir, "physics_gnn")
    plot_pressure(node_rows, edge_rows, out_dir, "physics_gnn")
    plot_diagnostic_suite(enriched_edge_rows, node_rows, conservation_rows, out_dir)
    if direct_model is not None:
        plot_prediction(direct_edge_rows, out_dir, "direct_gnn")
    interpretation_summary(
        label, validation_history, regression_summary, regression_coefficients)

    train_metrics = evaluate_arrays(data, q_hat, train_mask)
    val_metrics = evaluate_arrays(data, q_hat, val_mask) if bool(val_mask.any()) else {
        "RMSE_nL_s": float("nan"), "normalized_RMSE": float("nan"),
        "MAE_nL_s": float("nan"), "pearson_corr": float("nan"),
    }
    delta_np = delta.detach().cpu().numpy()
    c_np = np.exp(delta_np)
    p_np = p.detach().cpu().numpy()
    summary = {
        "K": int(args.layers),
        "hidden_dim": int(args.hidden_dim),
        "lambda_delta": float(args.lambda_delta),
        "lambda_h1": float(args.lambda_h1),
        "lambda_h2": float(args.lambda_h2),
        "flow_components": str(args.flow_components),
        "seed": int(args.seed),
        "train_loss_final": float(validation_history[-1]["train_loss"]) if validation_history else float(physics_history[-1]["q_loss"]),
        "val_loss_final": float(validation_history[-1]["val_loss"]) if validation_history else float("nan"),
        "train_RMSE": float(train_metrics["RMSE_nL_s"]),
        "val_RMSE": float(val_metrics["RMSE_nL_s"]),
        "train_NRMSE": float(train_metrics["normalized_RMSE"]),
        "val_NRMSE": float(val_metrics["normalized_RMSE"]),
        "train_MAE": float(train_metrics["MAE_nL_s"]),
        "val_MAE": float(val_metrics["MAE_nL_s"]),
        "train_Pearson": float(train_metrics["pearson_corr"]),
        "val_Pearson": float(val_metrics["pearson_corr"]),
        "mass_residual_norm": mass_residual_norm(data, q_hat),
        "mean_abs_delta": float(np.nanmean(np.abs(delta_np))),
        "median_abs_delta": float(np.nanmedian(np.abs(delta_np))),
        "mean_C": float(np.nanmean(c_np)),
        "median_C": float(np.nanmedian(c_np)),
        "p95_C": float(np.nanpercentile(c_np, 95)),
        "max_C": float(np.nanmax(c_np)),
        "pressure_min": float(np.nanmin(p_np)),
        "pressure_median": float(np.nanmedian(p_np)),
        "pressure_max": float(np.nanmax(p_np)),
        "pressure_range": float(np.nanmax(p_np) - np.nanmin(p_np)),
        "number_edges": int(len(data.edge_ids)),
        "number_nodes": int(len(data.node_ids)),
        "out_dir": str(out_dir),
    }
    write_csv(out_dir / "run_summary.csv", [summary])
    return summary


def _lambda_label(value: float) -> str:
    return f"{float(value):g}".replace(".", "p").replace("-", "m")

def run_sweep(data: MosaicData, graph, args, device: torch.device,
              out_root: Path) -> List[dict]:
    train_mask, val_mask = split_masks(data, args.val_fraction, args.seed)
    summaries: List[dict] = []
    total = (
        len(args.K_values) * len(args.hidden_dim_values)
        * len(args.lambda_delta_values) * len(args.seeds)
    )
    idx = 0
    for seed in args.seeds:
        for k in args.K_values:
            for hidden_dim in args.hidden_dim_values:
                for lambda_delta in args.lambda_delta_values:
                    idx += 1
                    run_args = copy.copy(args)
                    run_args.seed = int(seed)
                    run_args.layers = int(k)
                    run_args.hidden_dim = int(hidden_dim)
                    run_args.lambda_delta = float(lambda_delta)
                    set_seed(run_args.seed)
                    run_dir = (
                        out_root
                        / f"K{int(k)}_hidden{int(hidden_dim)}_lambda{_lambda_label(float(lambda_delta))}_seed{int(seed)}"
                    )
                    print(
                        f"\n=== Sweep {idx}/{total}: K={k}, hidden={hidden_dim}, "
                        f"lambda_delta={lambda_delta:g}, seed={seed} ==="
                    )
                    summary = run_experiment(
                        data, run_args, device, run_dir,
                        label=f"K{k}_h{hidden_dim}_l{lambda_delta:g}_s{seed}",
                        train_mask=train_mask,
                        val_mask=val_mask,
                        graph=graph,
                        train_direct=False,
                    )
                    summaries.append(summary)
                    write_csv(out_root / "sweep_summary.csv", summaries)
                    plot_sweep_summary(summaries, out_root)
    return summaries