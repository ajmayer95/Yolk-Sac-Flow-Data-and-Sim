#!/usr/bin/env python
"""Run the Step 5 radius-refinement and GNN feedback experiment."""

from __future__ import annotations

import argparse
import copy
import math
import pickle
import shutil
import subprocess
import sys
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from experiment import build_model
from gnn_flow import (
    DifferentiablePressureSolver,
    edge_rows as gnn_edge_rows,
    forward_model as gnn_forward_model,
    node_rows as gnn_node_rows,
    pressure_sanity_checks,
    write_csv as gnn_write_csv,
    write_yaml as gnn_write_yaml,
)
from poiseuille_only_baseline import NL_PER_M3
from radius_correction_experiment_lib import (
    CONDITION_DISPLAY,
    DEFAULT_OUTPUT_ROOT,
    DEFAULT_REPRESENTATIVE_CSV,
    DEFAULT_REPRESENTATIVE_LABELS_CSV,
    DEFAULT_STEP2_ROOT,
    DEFAULT_TARGETED_EDGE_CSV,
    SHARED_CONDITIONS,
    STRATEGY_DISPLAY,
    STRATEGY_ORDER,
    condition_dir,
    expected_condition_files,
    normalize_bool,
    read_summary_csv,
    safe_float,
    select_step2_run,
    shared_condition_dir,
    step2_run_dir,
    strategy_dir,
    write_rows,
)
from real_data import MU, PX_SIZE_M, _measurement_snr, build_real_gnn_data, load_graph
from utils import load_yaml, resolve_device, set_random_seed, write_yaml


POISEUILLE_SCRIPT = PROJECT_ROOT / "scripts" / "python" / "poiseuille_only_baseline.py"
GNN_SCRIPT = PROJECT_ROOT / "scripts" / "python" / "gnn_flow.py"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--graph", type=Path, required=True)
    parser.add_argument("--step2-root", type=Path, default=DEFAULT_STEP2_ROOT)
    parser.add_argument("--representative-csv", type=Path, default=DEFAULT_REPRESENTATIVE_CSV)
    parser.add_argument(
        "--representative-labels-csv",
        type=Path,
        default=DEFAULT_REPRESENTATIVE_LABELS_CSV,
    )
    parser.add_argument("--selected-run-name", default=None)
    parser.add_argument("--representative-label", default=None)
    parser.add_argument("--targeted-edge-csv", type=Path, default=DEFAULT_TARGETED_EDGE_CSV)
    parser.add_argument("--low-snr-fraction", type=float, default=0.20)
    parser.add_argument("--allow-target-count-mismatch", action="store_true")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--python-bin", default=sys.executable)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--viscosity-pa-s", type=float, default=float(MU))
    parser.add_argument("--metric-tolerance-fraction", type=float, default=0.02)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--aggregate-after", action="store_true")
    parser.add_argument("--plot-after", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def load_step2_config(run_dir: Path) -> dict:
    config_path = run_dir / "config_used.yaml"
    if not config_path.exists():
        raise FileNotFoundError(f"Missing Step 2 config snapshot: {config_path}")
    return load_yaml(config_path)


def normalize_edge_df(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path).copy()
    if "source_node" not in df.columns and "source" in df.columns:
        df["source_node"] = df["source"]
    if "target_node" not in df.columns and "target" in df.columns:
        df["target_node"] = df["target"]
    for column in (
        "edge_id",
        "q_obs_m3_s",
        "q_pred_m3_s",
        "observed_flow_nl_s",
        "predicted_flow_nl_s",
        "flow_residual_nl_s",
        "pressure_drop_pa",
        "radius_m",
        "length_m",
        "delta_e",
        "Gcorr_over_G0",
    ):
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors="coerce")
    if "observed_flow_nl_s" not in df.columns and "q_obs_m3_s" in df.columns:
        df["observed_flow_nl_s"] = pd.to_numeric(df["q_obs_m3_s"], errors="coerce") * NL_PER_M3
    if "predicted_flow_nl_s" not in df.columns and "q_pred_m3_s" in df.columns:
        df["predicted_flow_nl_s"] = pd.to_numeric(df["q_pred_m3_s"], errors="coerce") * NL_PER_M3
    if "flow_residual_nl_s" not in df.columns and {
        "observed_flow_nl_s",
        "predicted_flow_nl_s",
    } <= set(df.columns):
        df["flow_residual_nl_s"] = (
            pd.to_numeric(df["predicted_flow_nl_s"], errors="coerce")
            - pd.to_numeric(df["observed_flow_nl_s"], errors="coerce")
        )
    if "valid_observed_flow" in df.columns:
        df["valid_observed_flow"] = df["valid_observed_flow"].map(
            lambda value: normalize_bool(value, False)
        )
    elif "observed_flow_valid" in df.columns:
        df["valid_observed_flow"] = df["observed_flow_valid"].map(
            lambda value: normalize_bool(value, False)
        )
    else:
        df["valid_observed_flow"] = False
    if "sign_flip" in df.columns:
        df["sign_flip"] = df["sign_flip"].map(lambda value: normalize_bool(value, False))
    else:
        df["sign_flip"] = False
    return df


def normalize_node_df(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path).copy()
    if "node_type" not in df.columns and "boundary_role" in df.columns:
        df["node_type"] = df["boundary_role"]
    if "boundary_injection_nl_s" not in df.columns and "boundary_injection_m3_s" in df.columns:
        df["boundary_injection_nl_s"] = pd.to_numeric(
            df["boundary_injection_m3_s"], errors="coerce"
        ) * NL_PER_M3
    if "kirchhoff_residual_nl_s" not in df.columns and "kirchhoff_residual_m3_s" in df.columns:
        df["kirchhoff_residual_nl_s"] = pd.to_numeric(
            df["kirchhoff_residual_m3_s"], errors="coerce"
        ) * NL_PER_M3
    if (
        "predicted_net_flow_nl_s" not in df.columns
        and {"boundary_injection_nl_s", "kirchhoff_residual_nl_s"} <= set(df.columns)
    ):
        df["predicted_net_flow_nl_s"] = (
            pd.to_numeric(df["boundary_injection_nl_s"], errors="coerce")
            + pd.to_numeric(df["kirchhoff_residual_nl_s"], errors="coerce")
        )
    for column in (
        "node_index",
        "pressure_pa",
        "boundary_injection_nl_s",
        "predicted_net_flow_nl_s",
        "kirchhoff_residual_nl_s",
        "x_px",
        "y_px",
    ):
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors="coerce")
    if "boundary_role" in df.columns:
        role = df["boundary_role"].astype(str).str.lower()
        if "is_arterial" not in df.columns:
            df["is_arterial"] = role.eq("arterial")
        if "is_venous" not in df.columns:
            df["is_venous"] = role.eq("venous")
        if "is_boundary" not in df.columns:
            df["is_boundary"] = role.isin(["arterial", "venous"])
        if "is_internal" not in df.columns:
            df["is_internal"] = ~role.isin(["arterial", "venous"])
    for column in ("is_arterial", "is_venous", "is_boundary", "is_internal"):
        if column in df.columns:
            df[column] = df[column].map(lambda value: normalize_bool(value, False))
    return df


def compute_base_edge_table(graph_path: Path, config: dict, viscosity_pa_s: float) -> pd.DataFrame:
    graph = load_graph(graph_path)
    data = build_real_gnn_data(graph_path, config)
    base = []
    for edge_idx, (u, v) in enumerate(data.edge_ids):
        edge_data = graph.edges[u, v]
        radius_m = float(data.radius_m[edge_idx].detach().cpu())
        length_m = float(data.length_m[edge_idx].detach().cpu())
        conductance = float(
            np.pi * radius_m**4 / (8.0 * viscosity_pa_s * max(length_m, 1.0e-30))
        )
        base.append(
            {
                "edge_id": int(edge_idx),
                "graph_source_node": u,
                "graph_target_node": v,
                "source_node": str(u),
                "target_node": str(v),
                "source_index": int(data.edge_index[0][edge_idx].detach().cpu()),
                "target_index": int(data.edge_index[1][edge_idx].detach().cpu()),
                "snr": float(_measurement_snr(None, edge_data)),
                "tile_id": int(data.edge_tile_id[edge_idx]) if hasattr(data, "edge_tile_id") else -1,
                "original_radius_m": radius_m,
                "length_m": length_m,
                "original_poiseuille_conductance": conductance,
            }
        )
    return pd.DataFrame(base).sort_values("edge_id").reset_index(drop=True)


def targeted_selection_mask(
    edge_df: pd.DataFrame,
    targeted_csv: Path,
    allow_mismatch: bool,
) -> tuple[np.ndarray, list[str]]:
    df = pd.read_csv(targeted_csv).copy()
    if "selected_for_radius_correction" in df.columns:
        df = df[df["selected_for_radius_correction"].map(lambda value: normalize_bool(value, False))].copy()
    mask = np.zeros(len(edge_df), dtype=bool)
    unmatched: list[str] = []
    if "edge_id" in df.columns:
        selected_ids = {
            int(value)
            for value in pd.to_numeric(df["edge_id"], errors="coerce").dropna().tolist()
        }
        mask[np.isin(edge_df["edge_id"].to_numpy(dtype=np.int64), list(selected_ids))] = True
        missing_ids = sorted(selected_ids - set(edge_df["edge_id"].to_numpy(dtype=np.int64)))
        unmatched.extend([f"edge_id={value}" for value in missing_ids])
    source_col = None
    target_col = None
    for left, right in (("source_node", "target_node"), ("u", "v"), ("source", "target")):
        if left in df.columns and right in df.columns:
            source_col, target_col = left, right
            break
    if source_col is not None and target_col is not None:
        pair_to_index: dict[tuple[str, str], int] = {}
        for row in edge_df.itertuples(index=False):
            pair_to_index[(str(row.source_node), str(row.target_node))] = int(row.edge_id)
            pair_to_index[(str(row.target_node), str(row.source_node))] = int(row.edge_id)
        for row in df.itertuples(index=False):
            key = (str(getattr(row, source_col)), str(getattr(row, target_col)))
            edge_id = pair_to_index.get(key)
            if edge_id is None:
                unmatched.append(f"{source_col}/{target_col}={key[0]}->{key[1]}")
                continue
            mask[edge_id] = True
    count = int(np.sum(mask))
    if count != 166 and not allow_mismatch:
        raise ValueError(
            f"Targeted selection resolved to {count} unique edges, expected exactly 166. "
            "Pass --allow-target-count-mismatch to override."
        )
    return mask, unmatched


def low_snr_selection_mask(edge_df: pd.DataFrame, fraction: float) -> tuple[np.ndarray, dict[str, object]]:
    if not math.isfinite(fraction) or not (0.0 < fraction <= 1.0):
        raise ValueError("--low-snr-fraction must be in (0, 1].")
    eligible = edge_df[np.isfinite(pd.to_numeric(edge_df["snr"], errors="coerce"))].copy()
    eligible["snr"] = pd.to_numeric(eligible["snr"], errors="coerce")
    eligible = eligible.sort_values(["snr", "edge_id"]).reset_index(drop=True)
    n_select = max(1, int(math.ceil(len(eligible) * fraction)))
    selected = eligible.iloc[:n_select].copy()
    mask = np.zeros(len(edge_df), dtype=bool)
    mask[selected["edge_id"].to_numpy(dtype=np.int64)] = True
    cutoff = float(selected["snr"].iloc[-1]) if not selected.empty else float("nan")
    return mask, {
        "eligible_edge_count": int(len(eligible)),
        "selected_edge_count": int(n_select),
        "snr_cutoff": cutoff,
        "selected_edge_ids": selected["edge_id"].astype(int).tolist(),
    }


def overlap_rows(targeted_mask: np.ndarray, low_snr_mask: np.ndarray) -> list[dict[str, object]]:
    rows = []
    for edge_id in range(len(targeted_mask)):
        in_targeted = bool(targeted_mask[edge_id])
        in_low = bool(low_snr_mask[edge_id])
        if in_targeted and in_low:
            category = "overlap"
        elif in_targeted:
            category = "targeted_only"
        elif in_low:
            category = "low_snr_only"
        else:
            category = "neither"
        rows.append(
            {
                "edge_id": int(edge_id),
                "in_targeted_166": in_targeted,
                "in_low_snr_20pct": in_low,
                "category": category,
            }
        )
    return rows


def run_complete(run_dir: Path) -> bool:
    return all((run_dir / name).exists() for name in expected_condition_files())


def write_pickle(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        pickle.dump(payload, handle)


def build_corrected_graph(
    original_graph_path: Path,
    output_path: Path,
    edge_df: pd.DataFrame,
    selected_mask: np.ndarray,
    delta0: np.ndarray,
    viscosity_pa_s: float,
) -> pd.DataFrame:
    graph = load_graph(original_graph_path)
    rows: list[dict[str, object]] = []
    for row in edge_df.itertuples(index=False):
        edge_id = int(row.edge_id)
        u = row.graph_source_node if hasattr(row, "graph_source_node") else row.source_node
        v = row.graph_target_node if hasattr(row, "graph_target_node") else row.target_node
        original_radius_m = float(row.original_radius_m)
        selected = bool(selected_mask[edge_id])
        applied_delta = float(delta0[edge_id]) if selected else 0.0
        radius_ratio = float(math.exp(applied_delta / 4.0))
        corrected_radius_m = original_radius_m * radius_ratio
        corrected_radius_px = corrected_radius_m / PX_SIZE_M
        corrected_conductance = float(
            np.pi * corrected_radius_m**4 / (8.0 * viscosity_pa_s * max(float(row.length_m), 1.0e-30))
        )
        original_conductance = float(row.original_poiseuille_conductance)
        expected_conductance = original_conductance * math.exp(applied_delta)
        attrs = graph.edges[u, v]
        attrs["radius_px_true"] = float(corrected_radius_px)
        attrs["radius"] = float(corrected_radius_px)
        attrs["radius_refined_m"] = float(corrected_radius_m)
        attrs["radius_refined_px_true"] = float(corrected_radius_px)
        attrs["diameter"] = float(2.0 * corrected_radius_px)
        attrs["diameter_px_true"] = float(2.0 * corrected_radius_px)
        attrs["radius_refinement_selected"] = bool(selected)
        attrs["radius_refinement_original_radius_m"] = float(original_radius_m)
        attrs["radius_refinement_corrected_radius_m"] = float(corrected_radius_m)
        attrs["radius_refinement_radius_ratio"] = float(radius_ratio)
        attrs["radius_refinement_delta0"] = float(delta0[edge_id])
        rows.append(
            {
                "edge_id": edge_id,
                "source_node": u,
                "target_node": v,
                "selected_for_radius_correction": selected,
                "original_radius_m": original_radius_m,
                "corrected_radius_m": corrected_radius_m,
                "radius_ratio": radius_ratio,
                "radius_percent_change": (radius_ratio - 1.0) * 100.0,
                "original_poiseuille_conductance": original_conductance,
                "corrected_poiseuille_conductance": corrected_conductance,
                "expected_corrected_conductance": expected_conductance,
                "conductance_equivalence_abs_error": abs(corrected_conductance - expected_conductance),
                "snr": float(row.snr),
                "tile_id": int(row.tile_id),
            }
        )
    write_pickle(output_path, graph)
    return pd.DataFrame(rows).sort_values("edge_id").reset_index(drop=True)


def run_command(cmd: list[str], cwd: Path) -> None:
    subprocess.run(cmd, cwd=str(cwd), check=True)


def poiseuille_command(
    python_bin: str,
    graph_path: Path,
    output_parent: Path,
    run_name: str,
    viscosity_pa_s: float,
    lambda_q: float,
    lambda_k: float,
) -> list[str]:
    return [
        str(python_bin),
        str(POISEUILLE_SCRIPT),
        str(graph_path),
        "--output-dir",
        str(output_parent),
        "--run-name",
        run_name,
        "--device",
        "cpu",
        "--viscosity-pa-s",
        str(viscosity_pa_s),
        "--dc-solve-mode",
        "reduced-soft-constrained-lstsq",
        "--arterial-flow-mode",
        "dataset",
        "--pressure-constraint",
        "equal-a-equal-v",
        "--lambda-kirchhoff",
        str(lambda_k),
        "--lambda-pressure-constraints",
        "100.0",
        "--lambda-flow-residual",
        str(lambda_q),
    ]


def gnn_retrain_command(
    python_bin: str,
    graph_path: Path,
    output_parent: Path,
    run_name: str,
    device: str,
    seed: int,
    viscosity_pa_s: float,
    config_path: Path,
    epochs: int,
) -> list[str]:
    return [
        str(python_bin),
        str(GNN_SCRIPT),
        str(graph_path),
        "--output-dir",
        str(output_parent),
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
        "--viscosity-pa-s",
        str(viscosity_pa_s),
        "--config",
        str(config_path),
        "--no-pressure-detach",
    ]


def copy_g_original(step2_run_dir_path: Path, output_root: Path, overwrite: bool) -> None:
    dst = shared_condition_dir(output_root, "g_original")
    if dst.exists() and overwrite:
        shutil.rmtree(dst)
    if dst.exists():
        return
    dst.mkdir(parents=True, exist_ok=True)
    for name in (
        "summary.csv",
        "summary.yaml",
        "edge_predictions.csv",
        "node_predictions.csv",
        "config_used.yaml",
        "resolved_config_snapshot.yaml",
        "training_history.csv",
        "exploration_diagnostics.csv",
        "model_checkpoint.pt",
    ):
        src = step2_run_dir_path / name
        if src.exists():
            shutil.copy2(src, dst / name)


def run_g_fixed(
    corrected_graph_path: Path,
    output_dir: Path,
    config: dict,
    checkpoint_path: Path,
    device_name: str,
    viscosity_pa_s: float,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    device = resolve_device(device_name)
    data = build_real_gnn_data(corrected_graph_path, config)
    conductance = (
        math.pi
        * data.radius_m.detach().cpu().numpy().astype(np.float64) ** 4
        / (8.0 * viscosity_pa_s * np.maximum(data.length_m.detach().cpu().numpy().astype(np.float64), 1.0e-30))
    )
    data.base_conductance = torch.tensor(conductance, dtype=torch.float32)
    model = build_model(data, config).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint["model_state_dict"] if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint else checkpoint
    model.load_state_dict(state_dict)
    model.eval()
    solver = DifferentiablePressureSolver(config).to(device)
    data = data.to(device)
    with torch.no_grad():
        outputs = gnn_forward_model(model, solver, data, config)
        sanity = pressure_sanity_checks(model, solver, data, config)
    edge_payload = gnn_edge_rows(outputs, data, config)
    node_payload = gnn_node_rows(outputs, data)
    summary = {
        "script_name": Path(__file__).name,
        "graph_path": str(corrected_graph_path),
        "output_dir": str(output_dir),
        "run_name": output_dir.name,
        "preset": "fixed_checkpoint_forward",
        "n_nodes": int(len(data.node_id)),
        "n_edges": int(data.n_edges),
        "n_observed_edges": int(sum(bool(row.get("valid_observed_flow", False)) for row in edge_payload)),
        "device": str(device),
        "viscosity_pa_s": float(viscosity_pa_s),
        "pressure_constraints": "equal-a-equal-v",
        "arterial_flow_mode": "dataset",
        "lambda_q": float(config["gnn_outer_losses"]["flow"]),
        "lambda_k": float(config["gnn_outer_losses"]["kirchhoff"]),
        "lambda_b": float(config["gnn_outer_losses"]["boundary"]),
        "lambda_delta": float(config["gnn_outer_losses"]["delta_l2"]),
        "solver_success": True,
        "checkpoint_source": str(checkpoint_path),
    }
    for key, value in outputs["raw_losses"].items():
        summary[key] = float(value.detach().cpu())
    for key, value in outputs["solver_diagnostics"].items():
        if torch.is_tensor(value):
            summary[key] = float(value.detach().cpu())
        elif isinstance(value, list):
            summary[key] = "|".join(str(item) for item in value)
        else:
            summary[key] = value
    for key, value in outputs["global_metrics"].items():
        summary[key] = float(value.detach().cpu()) if torch.is_tensor(value) else value
    for key, value in sanity.items():
        summary[key] = value
    gnn_write_csv(output_dir / "edge_predictions.csv", edge_payload)
    gnn_write_csv(output_dir / "node_predictions.csv", node_payload)
    gnn_write_csv(output_dir / "summary.csv", [summary])
    gnn_write_yaml(output_dir / "summary.yaml", summary)
    gnn_write_yaml(output_dir / "resolved_config.yaml", config)


def enrich_condition_outputs(
    run_dir: Path,
    strategy_name: str,
    condition_name: str,
    geometry_df: pd.DataFrame,
    delta0: np.ndarray,
    snr_cutoff: float | None,
    metric_tolerance_fraction: float,
) -> None:
    edge_df = normalize_edge_df(run_dir / "edge_predictions.csv")
    node_df = normalize_node_df(run_dir / "node_predictions.csv")
    summary = read_summary_csv(run_dir / "summary.csv")
    geometry_cols = geometry_df[
        [
            "edge_id",
            "selected_for_radius_correction",
            "original_radius_m",
            "corrected_radius_m",
            "radius_ratio",
            "radius_percent_change",
            "original_poiseuille_conductance",
            "corrected_poiseuille_conductance",
            "snr",
            "tile_id",
        ]
    ].copy()
    edge_df = edge_df.merge(geometry_cols, on="edge_id", how="left")
    if "delta_e" not in edge_df.columns:
        edge_df["delta_e"] = 0.0
    edge_df["delta_e"] = pd.to_numeric(edge_df["delta_e"], errors="coerce").fillna(0.0)
    edge_df["original_delta"] = np.asarray(delta0, dtype=np.float64)
    edge_df["selection_strategy"] = strategy_name
    edge_df["condition_name"] = condition_name
    edge_df["condition_display_name"] = CONDITION_DISPLAY[condition_name]
    edge_df["sign_flip"] = edge_df["sign_flip"].map(lambda value: normalize_bool(value, False))
    edge_df["selected_for_radius_correction"] = edge_df["selected_for_radius_correction"].map(
        lambda value: normalize_bool(value, False)
    )

    flow_valid = edge_df["valid_observed_flow"].to_numpy(dtype=bool)
    flow_residual = pd.to_numeric(edge_df.get("flow_residual_nl_s"), errors="coerce").to_numpy(
        dtype=np.float64
    )
    valid_flow_residual = flow_residual[flow_valid & np.isfinite(flow_residual)]
    computed_flow_rmse_nl_s = (
        float(np.sqrt(np.mean(valid_flow_residual**2)))
        if valid_flow_residual.size
        else float("nan")
    )
    existing_flow_rmse_nl_s = safe_float(summary.get("flow_rmse_nl_s"))
    if not math.isfinite(existing_flow_rmse_nl_s):
        summary["flow_rmse_nl_s"] = computed_flow_rmse_nl_s

    internal = node_df[node_df["is_internal"].astype(bool)] if "is_internal" in node_df.columns else node_df
    internal_resid = pd.to_numeric(internal.get("kirchhoff_residual_nl_s"), errors="coerce")
    valid_internal = internal_resid[np.isfinite(internal_resid)]
    boundary_nodes = node_df[node_df["is_boundary"].astype(bool)] if "is_boundary" in node_df.columns else node_df.iloc[0:0]
    boundary_net = pd.to_numeric(boundary_nodes.get("predicted_net_flow_nl_s"), errors="coerce")
    inflow = float(boundary_net[boundary_net > 0.0].sum()) if not boundary_net.empty else float("nan")
    outflow = float((-boundary_net[boundary_net < 0.0]).sum()) if not boundary_net.empty else float("nan")
    balance_residual = inflow - outflow if math.isfinite(inflow) and math.isfinite(outflow) else float("nan")
    balance_relative = (
        balance_residual / max(abs(inflow), abs(outflow), 1.0e-30)
        if math.isfinite(balance_residual)
        else float("nan")
    )

    delta = pd.to_numeric(edge_df["delta_e"], errors="coerce").fillna(0.0).to_numpy(dtype=np.float64)
    selected_mask = edge_df["selected_for_radius_correction"].to_numpy(dtype=bool)
    unselected_mask = ~selected_mask

    def rms(values: np.ndarray) -> float:
        finite = values[np.isfinite(values)]
        return float(np.sqrt(np.mean(finite**2))) if finite.size else float("nan")

    def mean_abs(values: np.ndarray) -> float:
        finite = values[np.isfinite(values)]
        return float(np.mean(np.abs(finite))) if finite.size else float("nan")

    correction_min = float(summary.get("resolved_correction_bounds_correction_min", -0.5))
    correction_max = float(summary.get("resolved_correction_bounds_correction_max", 0.5))
    sat_tol = 5.0e-3
    delta_sat = float(
        np.mean((delta <= correction_min + sat_tol) | (delta >= correction_max - sat_tol))
    ) if len(delta) else float("nan")

    sign_flip_fraction = (
        float(edge_df.loc[flow_valid, "sign_flip"].mean()) if np.any(flow_valid) else float("nan")
    )
    geometry_success = False
    if condition_name == "g_retrained":
        geometry_success = True

    summary.update(
        {
            "strategy_name": strategy_name,
            "strategy_display_name": STRATEGY_DISPLAY.get(strategy_name, strategy_name),
            "condition_name": condition_name,
            "condition_display_name": CONDITION_DISPLAY[condition_name],
            "metric_tolerance_fraction": float(metric_tolerance_fraction),
            "number_selected_edges": int(np.sum(selected_mask)),
            "fraction_selected_edges": float(np.mean(selected_mask)) if len(selected_mask) else float("nan"),
            "snr_cutoff": float(snr_cutoff) if snr_cutoff is not None else float("nan"),
            "sign_flip_fraction": sign_flip_fraction,
            "total_inflow_nl_s": inflow,
            "total_outflow_nl_s": outflow,
            "global_flow_balance_residual_nl_s": balance_residual,
            "global_flow_balance_relative_error": balance_relative,
            "delta_rms_all_edges": rms(delta),
            "delta_mean_abs_all_edges": mean_abs(delta),
            "delta_saturation_fraction": delta_sat,
            "delta_rms_selected_edges": rms(delta[selected_mask]),
            "delta_mean_abs_selected_edges": mean_abs(delta[selected_mask]),
            "delta_rms_unselected_edges": rms(delta[unselected_mask]),
            "delta_mean_abs_unselected_edges": mean_abs(delta[unselected_mask]),
            "radius_ratio_mean_selected": float(edge_df.loc[selected_mask, "radius_ratio"].mean()) if np.any(selected_mask) else float("nan"),
            "radius_ratio_median_selected": float(edge_df.loc[selected_mask, "radius_ratio"].median()) if np.any(selected_mask) else float("nan"),
            "radius_ratio_min_selected": float(edge_df.loc[selected_mask, "radius_ratio"].min()) if np.any(selected_mask) else float("nan"),
            "radius_ratio_max_selected": float(edge_df.loc[selected_mask, "radius_ratio"].max()) if np.any(selected_mask) else float("nan"),
            "radius_percent_change_mean_abs_selected": float(edge_df.loc[selected_mask, "radius_percent_change"].abs().mean()) if np.any(selected_mask) else float("nan"),
            "radius_percent_change_p95_abs_selected": float(edge_df.loc[selected_mask, "radius_percent_change"].abs().quantile(0.95)) if np.any(selected_mask) else float("nan"),
            "geometry_refinement_success": bool(geometry_success),
            "resolved_config_path": str(run_dir / "resolved_config.yaml"),
            "kirchhoff_rms_per_internal_node_nl_s": rms(valid_internal.to_numpy(dtype=np.float64)),
            "kirchhoff_mae_per_internal_node_nl_s": mean_abs(valid_internal.to_numpy(dtype=np.float64)),
            "kirchhoff_p95_abs_nl_s": float(valid_internal.abs().quantile(0.95)) if not valid_internal.empty else float("nan"),
            "kirchhoff_max_abs_nl_s": float(valid_internal.abs().max()) if not valid_internal.empty else float("nan"),
        }
    )

    gnn_write_csv(run_dir / "edge_predictions.csv", edge_df.to_dict(orient="records"))
    gnn_write_csv(run_dir / "node_predictions.csv", node_df.to_dict(orient="records"))
    gnn_write_csv(run_dir / "summary.csv", [summary])
    write_yaml(run_dir / "summary.yaml", summary)


def write_resolved_config(run_dir: Path, config: dict) -> None:
    write_yaml(run_dir / "resolved_config.yaml", config)


def status_row(strategy_name: str, condition_name: str, status: str, message: str = "") -> dict[str, object]:
    return {
        "strategy_name": strategy_name,
        "condition_name": condition_name,
        "status": status,
        "message": message,
    }


def main() -> None:
    args = parse_args()
    graph_path = args.graph.expanduser().resolve()
    step2_root = args.step2_root.expanduser().resolve()
    output_root = args.output_root.expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    selected = select_step2_run(
        representative_csv=args.representative_csv.expanduser().resolve(),
        representative_labels_csv=args.representative_labels_csv.expanduser().resolve(),
        selected_run_name=args.selected_run_name,
        requested_representative_label=args.representative_label,
    )
    selected_run_dir = step2_run_dir(step2_root, selected["run_name"])
    checkpoint_path = selected_run_dir / "model_checkpoint.pt"
    selected["run_dir"] = str(selected_run_dir)
    selected["checkpoint_path"] = str(checkpoint_path)
    config = load_step2_config(selected_run_dir)
    config["K"] = 2
    config["training"]["seed"] = int(args.seed)
    config["physics"]["pressure_constraints"] = ["equal-a-equal-v"]
    config["physics"]["pressure_solver_lambda_pressure_constraints"] = 100.0
    config["gnn_outer_losses"]["boundary"] = 100.0
    config["gnn_outer_losses"]["delta_smooth"] = 0.0
    config["gnn_outer_losses"]["pressure_shape"] = 0.0
    config["model"]["correction_min"] = -0.5
    config["model"]["correction_max"] = 0.5
    config["model"]["correction_bound"] = 0.5

    base_edge_df = compute_base_edge_table(graph_path, config, float(args.viscosity_pa_s))
    original_edge_df = normalize_edge_df(selected_run_dir / "edge_predictions.csv")
    delta0 = pd.to_numeric(original_edge_df["delta_e"], errors="coerce").fillna(0.0).to_numpy(dtype=np.float64)

    targeted_mask, targeted_unmatched = targeted_selection_mask(
        base_edge_df,
        args.targeted_edge_csv.expanduser().resolve(),
        args.allow_target_count_mismatch,
    )
    low_snr_mask, low_snr_info = low_snr_selection_mask(
        base_edge_df,
        float(args.low_snr_fraction),
    )

    targeted_rows = base_edge_df[targeted_mask].copy()
    targeted_rows["selection_strategy"] = "targeted_166"
    targeted_rows["selected_for_radius_correction"] = True
    targeted_rows.to_csv(output_root / "selected_edges_targeted_166.csv", index=False)

    low_rows = base_edge_df[low_snr_mask].copy()
    low_rows["selection_strategy"] = "low_snr_20pct"
    low_rows["selected_for_radius_correction"] = True
    low_rows.to_csv(output_root / "selected_edges_low_snr_20pct.csv", index=False)

    overlap = pd.DataFrame(overlap_rows(targeted_mask, low_snr_mask))
    overlap.to_csv(output_root / "edge_selection_overlap.csv", index=False)

    manifest = {
        "graph_path": str(graph_path),
        "output_root": str(output_root),
        "selected_step2_model": selected,
        "resolved_lambda_q": float(selected["lambda_q"]),
        "resolved_lambda_k": float(selected["lambda_k"]),
        "resolved_lambda_delta": float(selected["lambda_delta"]),
        "lambda_b": 100.0,
        "message_passing_depth": 2,
        "targeted_edge_count": int(np.sum(targeted_mask)),
        "low_snr_edge_count": int(np.sum(low_snr_mask)),
        "low_snr_fraction": float(args.low_snr_fraction),
        "low_snr_cutoff": float(low_snr_info["snr_cutoff"]),
        "targeted_unmatched_entries": targeted_unmatched,
        "overlap_count": int(np.sum(targeted_mask & low_snr_mask)),
        "overlap_fraction_of_targeted": float(
            np.sum(targeted_mask & low_snr_mask) / max(np.sum(targeted_mask), 1)
        ),
        "overlap_fraction_of_low_snr": float(
            np.sum(targeted_mask & low_snr_mask) / max(np.sum(low_snr_mask), 1)
        ),
    }
    write_yaml(output_root / "experiment_manifest.yaml", manifest)

    if args.dry_run:
        print(f"Selected Step 2 model: {selected['run_name']} ({selected['plot_label']})")
        print(f"lambda_q={selected['lambda_q']}, lambda_k={selected['lambda_k']}, lambda_delta={selected['lambda_delta']}")
        print(f"targeted_edge_count={int(np.sum(targeted_mask))}")
        print(f"low_snr_edge_count={int(np.sum(low_snr_mask))}")
        print(f"overlap_count={int(np.sum(targeted_mask & low_snr_mask))}")
        return

    statuses: list[dict[str, object]] = []

    shared_parent = output_root / "shared"
    shared_parent.mkdir(parents=True, exist_ok=True)
    p_original_dir = shared_condition_dir(output_root, "p_original")
    if args.overwrite and p_original_dir.exists():
        shutil.rmtree(p_original_dir)
    if not run_complete(p_original_dir):
        run_command(
            poiseuille_command(
                python_bin=args.python_bin,
                graph_path=graph_path,
                output_parent=shared_parent,
                run_name="p_original",
                viscosity_pa_s=float(args.viscosity_pa_s),
                lambda_q=float(selected["lambda_q"]),
                lambda_k=float(selected["lambda_k"]),
            ),
            cwd=PROJECT_ROOT,
        )
    write_resolved_config(p_original_dir, config)
    original_geometry = base_edge_df.copy()
    original_geometry["selected_for_radius_correction"] = False
    original_geometry["corrected_radius_m"] = original_geometry["original_radius_m"]
    original_geometry["radius_ratio"] = 1.0
    original_geometry["radius_percent_change"] = 0.0
    original_geometry["corrected_poiseuille_conductance"] = original_geometry["original_poiseuille_conductance"]
    enrich_condition_outputs(
        p_original_dir,
        "shared",
        "p_original",
        original_geometry,
        delta0=np.zeros_like(delta0),
        snr_cutoff=None,
        metric_tolerance_fraction=float(args.metric_tolerance_fraction),
    )
    statuses.append(status_row("shared", "p_original", "success"))

    copy_g_original(selected_run_dir, output_root, overwrite=args.overwrite)
    g_original_dir = shared_condition_dir(output_root, "g_original")
    write_resolved_config(g_original_dir, config)
    enrich_condition_outputs(
        g_original_dir,
        "shared",
        "g_original",
        original_geometry,
        delta0=delta0,
        snr_cutoff=None,
        metric_tolerance_fraction=float(args.metric_tolerance_fraction),
    )
    statuses.append(status_row("shared", "g_original", "success"))

    strategy_specs = {
        "targeted_166": {
            "mask": targeted_mask,
            "snr_cutoff": None,
        },
        "low_snr_20pct": {
            "mask": low_snr_mask,
            "snr_cutoff": float(low_snr_info["snr_cutoff"]),
        },
    }

    for strategy_name in STRATEGY_ORDER:
        mask = np.asarray(strategy_specs[strategy_name]["mask"], dtype=bool)
        strategy_path = strategy_dir(output_root, strategy_name)
        strategy_path.mkdir(parents=True, exist_ok=True)
        corrected_graph_path = strategy_path / "corrected_graph.gpickle"
        geometry_df = build_corrected_graph(
            original_graph_path=graph_path,
            output_path=corrected_graph_path,
            edge_df=base_edge_df,
            selected_mask=mask,
            delta0=delta0,
            viscosity_pa_s=float(args.viscosity_pa_s),
        )
        geometry_df.to_csv(strategy_path / "resolved_edge_geometry.csv", index=False)

        for condition_name in ("p_corrected", "g_fixed", "g_retrained"):
            run_dir = condition_dir(output_root, strategy_name, condition_name)
            if args.overwrite and run_dir.exists():
                shutil.rmtree(run_dir)
            try:
                if condition_name == "p_corrected":
                    if not run_complete(run_dir):
                        run_command(
                            poiseuille_command(
                                python_bin=args.python_bin,
                                graph_path=corrected_graph_path,
                                output_parent=strategy_path,
                                run_name="p_corrected",
                                viscosity_pa_s=float(args.viscosity_pa_s),
                                lambda_q=float(selected["lambda_q"]),
                                lambda_k=float(selected["lambda_k"]),
                            ),
                            cwd=PROJECT_ROOT,
                        )
                    write_resolved_config(run_dir, config)
                elif condition_name == "g_fixed":
                    if not run_complete(run_dir):
                        run_g_fixed(
                            corrected_graph_path=corrected_graph_path,
                            output_dir=run_dir,
                            config=config,
                            checkpoint_path=checkpoint_path,
                            device_name=args.device,
                            viscosity_pa_s=float(args.viscosity_pa_s),
                        )
                elif condition_name == "g_retrained":
                    if not run_complete(run_dir):
                        config_path = strategy_path / "g_retrained_config.yaml"
                        write_yaml(config_path, config)
                        epochs = int(config["training"].get("epochs", 100))
                        run_command(
                            gnn_retrain_command(
                                python_bin=args.python_bin,
                                graph_path=corrected_graph_path,
                                output_parent=strategy_path,
                                run_name="g_retrained",
                                device=args.device,
                                seed=int(args.seed),
                                viscosity_pa_s=float(args.viscosity_pa_s),
                                config_path=config_path,
                                epochs=epochs,
                            ),
                            cwd=PROJECT_ROOT,
                        )
                    write_resolved_config(run_dir, config)

                enrich_condition_outputs(
                    run_dir,
                    strategy_name,
                    condition_name,
                    geometry_df,
                    delta0=delta0,
                    snr_cutoff=strategy_specs[strategy_name]["snr_cutoff"],
                    metric_tolerance_fraction=float(args.metric_tolerance_fraction),
                )
                statuses.append(status_row(strategy_name, condition_name, "success"))
            except Exception as exc:  # noqa: BLE001
                trace = "".join(traceback.format_exception_only(type(exc), exc)).strip()
                statuses.append(status_row(strategy_name, condition_name, "failed", trace))

    write_rows(output_root / "run_status.csv", statuses)

    failed_rows = [row for row in statuses if row["status"] != "success"]
    if failed_rows:
        failure_messages = "; ".join(
            f"{row['strategy_name']}/{row['condition_name']}: {row['message']}"
            for row in failed_rows
        )
        raise RuntimeError(
            "Radius-refinement run completed with failed conditions. "
            "Skipping aggregation/plotting until they are rerun successfully. "
            f"Failures: {failure_messages}"
        )

    if args.aggregate_after:
        run_command(
            [
                str(args.python_bin),
                str(PROJECT_ROOT / "scripts" / "python" / "analyze_radius_correction_experiment.py"),
                "--input-root",
                str(output_root),
            ],
            cwd=PROJECT_ROOT,
        )
    if args.plot_after:
        run_command(
            [
                str(args.python_bin),
                str(PROJECT_ROOT / "scripts" / "python" / "plot_radius_correction_experiment.py"),
                "--input-root",
                str(output_root),
                "--output-dir",
                str(output_root / "figures"),
            ],
            cwd=PROJECT_ROOT,
        )


if __name__ == "__main__":
    main()
