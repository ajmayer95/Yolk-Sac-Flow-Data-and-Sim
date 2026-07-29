#!/usr/bin/env python
"""Analyze completed Step 3 pressure-constraint sensitivity runs."""

from __future__ import annotations

import argparse
import csv
import math
import sys
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from analyze_physics_weight_sweep import compute_from_gnn_run, compute_from_poiseuille_run, safe_float
from pressure_constraint_sensitivity_lib import CONSTRAINT_DISPLAY, CONSTRAINT_ORDER, DEFAULT_OUTPUT_ROOT, launcher_metadata_path
from utils import load_yaml


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    return parser.parse_args()


def read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)


def load_metadata(path: Path) -> dict[str, object]:
    payload = load_yaml(path)
    if not isinstance(payload, dict):
        raise RuntimeError(f"Unexpected launcher metadata format: {path}")
    return payload


def normalized_metadata(metadata: dict[str, object]) -> dict[str, object]:
    result = dict(metadata)
    model_family = str(result.get("model_family", ""))
    if "message_passing_layers" not in result:
        result["message_passing_layers"] = 2 if model_family == "gnn" else 0
    if "lambda_b" not in result:
        result["lambda_b"] = 100.0
    if "run_name" not in result:
        result["run_name"] = ""
    if "graph_path" not in result:
        result["graph_path"] = ""
    if "lambda_delta" not in result and model_family != "gnn":
        result["lambda_delta"] = float("nan")
    return result


def discover_run_dirs(input_root: Path) -> list[Path]:
    return sorted(path.parent for path in input_root.rglob("launcher_run_config.yaml"))


def parse_node_df(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    for column in ("pressure_pa", "x_px", "y_px", "selection_rank_within_regime"):
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors="coerce")
    return df


def parse_edge_df(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    for column in ("edge_id", "delta_e", "Gcorr_over_G0", "source_index", "target_index"):
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors="coerce")
    return df


def align_pressure(node_df: pd.DataFrame, gauge_node_id: str | None) -> pd.Series:
    if "boundary_role" in node_df.columns:
        venous = node_df[node_df["boundary_role"].astype(str) == "venous"]
    else:
        venous = node_df.iloc[0:0].copy()
    if not venous.empty and node_df["pressure_pa"].notna().any():
        ref = float(pd.to_numeric(venous["pressure_pa"], errors="coerce").mean())
    else:
        gauge_mask = node_df["node_id"].astype(str) == str(gauge_node_id or "")
        ref = float(pd.to_numeric(node_df.loc[gauge_mask, "pressure_pa"], errors="coerce").mean())
    return pd.to_numeric(node_df["pressure_pa"], errors="coerce") - ref


def pressure_metrics_for_pair(left_nodes: pd.DataFrame, right_nodes: pd.DataFrame, left_gauge: str, right_gauge: str) -> dict[str, float]:
    left = left_nodes[["node_id", "pressure_pa"]].copy()
    right = right_nodes[["node_id", "pressure_pa"]].copy()
    left["aligned_pressure_pa"] = align_pressure(left_nodes, left_gauge)
    right["aligned_pressure_pa"] = align_pressure(right_nodes, right_gauge)
    merged = left.merge(right[["node_id", "aligned_pressure_pa"]], on="node_id", suffixes=("_left", "_right"))
    a = pd.to_numeric(merged["aligned_pressure_pa_left"], errors="coerce")
    b = pd.to_numeric(merged["aligned_pressure_pa_right"], errors="coerce")
    finite = a.notna() & b.notna()
    a = a[finite]
    b = b[finite]
    if len(a) < 2:
        return {
            "pressure_pearson_aligned": float("nan"),
            "pressure_spearman_aligned": float("nan"),
            "pressure_rmse_aligned_pa": float("nan"),
            "pressure_mae_aligned_pa": float("nan"),
        }
    residual = a - b
    return {
        "pressure_pearson_aligned": float(a.corr(b, method="pearson")),
        "pressure_spearman_aligned": float(a.corr(b, method="spearman")),
        "pressure_rmse_aligned_pa": float(np.sqrt(np.mean(np.square(residual)))),
        "pressure_mae_aligned_pa": float(np.mean(np.abs(residual))),
    }


def jaccard_top_fraction(a: pd.Series, b: pd.Series, fraction: float) -> float:
    n = len(a)
    if n == 0:
        return float("nan")
    k = max(1, int(math.ceil(fraction * n)))
    a_idx = set(a.abs().nlargest(k).index.tolist())
    b_idx = set(b.abs().nlargest(k).index.tolist())
    union = a_idx | b_idx
    if not union:
        return float("nan")
    return float(len(a_idx & b_idx) / len(union))


def correction_metrics_for_pair(left_edges: pd.DataFrame, right_edges: pd.DataFrame) -> dict[str, float]:
    merged = left_edges.merge(
        right_edges,
        on="edge_id",
        suffixes=("_left", "_right"),
    )
    delta_left = pd.to_numeric(merged["delta_e_left"], errors="coerce")
    delta_right = pd.to_numeric(merged["delta_e_right"], errors="coerce")
    ratio_left = pd.to_numeric(merged["Gcorr_over_G0_left"], errors="coerce")
    ratio_right = pd.to_numeric(merged["Gcorr_over_G0_right"], errors="coerce")
    finite = delta_left.notna() & delta_right.notna()
    delta_left = delta_left[finite]
    delta_right = delta_right[finite]
    if len(delta_left) < 2:
        return {
            "delta_pearson": float("nan"),
            "delta_spearman": float("nan"),
            "delta_rmse": float("nan"),
            "delta_mae": float("nan"),
            "conductance_ratio_pearson": float("nan"),
            "hotspot_jaccard_top5": float("nan"),
            "hotspot_jaccard_top10": float("nan"),
        }
    residual = delta_left - delta_right
    return {
        "delta_pearson": float(delta_left.corr(delta_right, method="pearson")),
        "delta_spearman": float(delta_left.corr(delta_right, method="spearman")),
        "delta_rmse": float(np.sqrt(np.mean(np.square(residual)))),
        "delta_mae": float(np.mean(np.abs(residual))),
        "conductance_ratio_pearson": float(ratio_left.corr(ratio_right, method="pearson")),
        "hotspot_jaccard_top5": jaccard_top_fraction(delta_left, delta_right, 0.05),
        "hotspot_jaccard_top10": jaccard_top_fraction(delta_left, delta_right, 0.10),
    }


def compute_constraint_specifics(node_df: pd.DataFrame, metadata: dict[str, object], summary_row: dict[str, object]) -> dict[str, float]:
    if "boundary_role" in node_df.columns:
        arterial = node_df[node_df["boundary_role"].astype(str) == "arterial"]["pressure_pa"].astype(float).tolist()
        venous = node_df[node_df["boundary_role"].astype(str) == "venous"]["pressure_pa"].astype(float).tolist()
    else:
        arterial = []
        venous = []
    result = {
        "gauge_residual_pa": float("nan"),
        "arterial_equality_residual_pa": float("nan"),
        "venous_equality_residual_pa": float("nan"),
        "equal_drop_residual_pa": float("nan"),
        "fixed_drop_residual_pa": float("nan"),
        "av_drop_1_pa": float("nan"),
        "av_drop_2_pa": float("nan"),
        "mean_av_drop_pa": float("nan"),
    }
    gauge_node = str(summary_row.get("gauge_node_id", metadata.get("gauge_node_id", "")))
    if gauge_node:
        mask = node_df["node_id"].astype(str) == gauge_node
        if bool(mask.any()):
            result["gauge_residual_pa"] = abs(float(pd.to_numeric(node_df.loc[mask, "pressure_pa"], errors="coerce").iloc[0]))
    if len(arterial) >= 2:
        result["arterial_equality_residual_pa"] = abs(float(arterial[0] - arterial[1]))
    if len(venous) >= 2:
        result["venous_equality_residual_pa"] = abs(float(venous[0] - venous[1]))
    if len(arterial) >= 2 and len(venous) >= 2:
        result["av_drop_1_pa"] = float(arterial[0] - venous[0])
        result["av_drop_2_pa"] = float(arterial[1] - venous[1])
        result["mean_av_drop_pa"] = 0.5 * (result["av_drop_1_pa"] + result["av_drop_2_pa"])
        result["equal_drop_residual_pa"] = abs(float((arterial[0] - venous[0]) - (arterial[1] - venous[1])))
    alpha_pa = safe_float(metadata.get("alpha_pa"))
    if len(arterial) >= 2 and len(venous) >= 1 and math.isfinite(alpha_pa):
        result["fixed_drop_residual_pa"] = abs(float(((arterial[0] + arterial[1]) * 0.5) - venous[0] - alpha_pa))
    return result


def build_run_rows(input_root: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, object]] = []
    cache: list[dict[str, object]] = []
    for run_dir in discover_run_dirs(input_root):
        metadata = normalized_metadata(load_metadata(launcher_metadata_path(run_dir)))
        summary_path = run_dir / "summary.csv"
        if not summary_path.exists():
            rows.append({**metadata, "output_dir": str(run_dir), "solver_success": False, "convergence_status": False})
            continue
        if str(metadata["model_family"]) == "gnn":
            base_row = compute_from_gnn_run(run_dir, metadata)
        else:
            base_row = compute_from_poiseuille_run(run_dir, metadata)
        summary_row = next(csv.DictReader(summary_path.open("r", newline="", encoding="utf-8")))
        node_df = parse_node_df(run_dir / "node_predictions.csv")
        base_row.update(
            {
                "parent_step2_run_name": metadata.get("parent_step2_run_name", ""),
                "representative_label": metadata.get("representative_label", ""),
                "selection_category": metadata.get("selection_category", ""),
                "selection_rank_within_regime": metadata.get("selection_rank_within_regime", float("nan")),
                "pressure_constraint_type": metadata.get("pressure_constraint_type", ""),
                "alpha_pa": safe_float(metadata.get("alpha_pa")),
                "constraint_description": metadata.get("constraint_description", ""),
                "pressure_constraint_display": CONSTRAINT_DISPLAY.get(str(metadata.get("pressure_constraint_type", "")), str(metadata.get("pressure_constraint_type", ""))),
                "gauge_node_id": summary_row.get("gauge_node_id", ""),
                "arterial_node_ids": summary_row.get("arterial_node_ids", ""),
                "venous_node_ids": summary_row.get("venous_node_ids", ""),
                "number_of_constraint_equations": safe_float(summary_row.get("reduced_constraint_count")),
            }
        )
        base_row.update(compute_constraint_specifics(node_df, metadata, summary_row))
        rows.append(base_row)
        cache.append({"run_name": base_row["run_name"], "run_dir": run_dir, "metadata": metadata})
    return pd.DataFrame(rows), pd.DataFrame(cache)


def pairwise_pressure_metrics(all_runs: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    gnn_groups = all_runs[all_runs["model_family"] == "gnn"].groupby("parent_step2_run_name")
    pois_groups = all_runs[all_runs["model_family"] == "poiseuille_baseline"].groupby(["lambda_q", "lambda_k"])
    for group_name, group_df in list(gnn_groups) + list(pois_groups):
        run_rows = group_df.dropna(subset=["output_dir"])
        for (_, left), (_, right) in combinations(run_rows.iterrows(), 2):
            left_nodes = parse_node_df(Path(left["output_dir"]) / "node_predictions.csv")
            right_nodes = parse_node_df(Path(right["output_dir"]) / "node_predictions.csv")
            metrics = pressure_metrics_for_pair(left_nodes, right_nodes, str(left.get("gauge_node_id", "")), str(right.get("gauge_node_id", "")))
            rows.append(
                {
                    "model_family": left["model_family"],
                    "model_key": group_name if isinstance(group_name, str) else "|".join(str(x) for x in group_name),
                    "representative_label": left.get("representative_label", ""),
                    "constraint_left": left["pressure_constraint_type"],
                    "constraint_right": right["pressure_constraint_type"],
                    **metrics,
                }
            )
    return pd.DataFrame(rows)


def pairwise_correction_metrics(all_runs: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for group_name, group_df in all_runs[all_runs["model_family"] == "gnn"].groupby("parent_step2_run_name"):
        run_rows = group_df.dropna(subset=["output_dir"])
        for (_, left), (_, right) in combinations(run_rows.iterrows(), 2):
            left_edges = parse_edge_df(Path(left["output_dir"]) / "edge_predictions.csv")
            right_edges = parse_edge_df(Path(right["output_dir"]) / "edge_predictions.csv")
            metrics = correction_metrics_for_pair(left_edges, right_edges)
            rows.append(
                {
                    "parent_step2_run_name": group_name,
                    "representative_label": left.get("representative_label", ""),
                    "constraint_left": left["pressure_constraint_type"],
                    "constraint_right": right["pressure_constraint_type"],
                    **metrics,
                }
            )
    return pd.DataFrame(rows)


def write_df(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def main() -> None:
    args = parse_args()
    input_root = args.input_root.expanduser().resolve()
    all_runs, _ = build_run_rows(input_root)
    gnn_runs = all_runs[all_runs["model_family"] == "gnn"].copy()
    poiseuille_runs = all_runs[all_runs["model_family"] == "poiseuille_baseline"].copy()
    pressure_pairwise = pairwise_pressure_metrics(all_runs)
    correction_pairwise = pairwise_correction_metrics(all_runs)

    write_df(all_runs, input_root / "pressure_constraint_all_runs.csv")
    write_df(gnn_runs, input_root / "pressure_constraint_gnn_summary.csv")
    write_df(poiseuille_runs, input_root / "pressure_constraint_poiseuille_summary.csv")
    write_df(pressure_pairwise, input_root / "pressure_field_pairwise_metrics.csv")
    write_df(correction_pairwise, input_root / "correction_field_pairwise_metrics.csv")
    write_df(
        pressure_pairwise[
            [
                "representative_label",
                "model_family",
                "constraint_left",
                "constraint_right",
                "pressure_pearson_aligned",
            ]
        ],
        input_root / "pressure_correlation_matrix.csv",
    )
    write_df(
        correction_pairwise[
            [
                "representative_label",
                "constraint_left",
                "constraint_right",
                "delta_pearson",
            ]
        ],
        input_root / "correction_correlation_matrix.csv",
    )


if __name__ == "__main__":
    main()
