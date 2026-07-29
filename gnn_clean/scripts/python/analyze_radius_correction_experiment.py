#!/usr/bin/env python
"""Aggregate Step 5 radius-refinement outputs."""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from radius_correction_experiment_lib import (
    CONDITION_DISPLAY,
    CONDITION_ORDER,
    DEFAULT_OUTPUT_ROOT,
    SHARED_CONDITIONS,
    STRATEGY_DISPLAY,
    STRATEGY_ORDER,
    condition_dir,
    safe_float,
    shared_condition_dir,
)
from utils import load_yaml, write_yaml


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    return parser.parse_args()


def condition_summary(input_root: Path, strategy_name: str, condition_name: str) -> dict[str, object]:
    path = (
        shared_condition_dir(input_root, condition_name) / "summary.csv"
        if condition_name in SHARED_CONDITIONS
        else condition_dir(input_root, strategy_name, condition_name) / "summary.csv"
    )
    df = pd.read_csv(path)
    if df.empty:
        raise RuntimeError(f"Empty summary: {path}")
    row = df.iloc[0].to_dict()
    row["strategy_name"] = strategy_name
    row["strategy_display_name"] = STRATEGY_DISPLAY[strategy_name]
    row["condition_name"] = condition_name
    row["condition_display_name"] = CONDITION_DISPLAY[condition_name]
    return row


def edge_df(input_root: Path, strategy_name: str, condition_name: str) -> pd.DataFrame:
    path = (
        shared_condition_dir(input_root, condition_name) / "edge_predictions.csv"
        if condition_name in SHARED_CONDITIONS
        else condition_dir(input_root, strategy_name, condition_name) / "edge_predictions.csv"
    )
    df = pd.read_csv(path)
    for column in (
        "edge_id",
        "snr",
        "original_radius_m",
        "corrected_radius_m",
        "radius_ratio",
        "radius_percent_change",
        "original_poiseuille_conductance",
        "corrected_poiseuille_conductance",
        "original_delta",
        "delta_e",
        "observed_flow_nl_s",
        "predicted_flow_nl_s",
        "flow_residual_nl_s",
    ):
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors="coerce")
    if "source_node" not in df.columns and "source" in df.columns:
        df["source_node"] = df["source"]
    if "target_node" not in df.columns and "target" in df.columns:
        df["target_node"] = df["target"]
    if "tile_id" not in df.columns:
        if "tile_id_x" in df.columns:
            df["tile_id"] = df["tile_id_x"]
        elif "tile_id_y" in df.columns:
            df["tile_id"] = df["tile_id_y"]
    if "tile_id" in df.columns:
        df["tile_id"] = pd.to_numeric(df["tile_id"], errors="coerce")
    return df


def _require_columns(df: pd.DataFrame, required: list[str], label: str) -> pd.DataFrame:
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise KeyError(f"{label} is missing required columns: {missing}")
    return df[required].copy()


def compare_change(current: float, baseline: float) -> tuple[float, float]:
    if not math.isfinite(current) or not math.isfinite(baseline):
        return float("nan"), float("nan")
    change = current - baseline
    percent = change / max(abs(baseline), 1.0e-30) * 100.0
    return change, percent


def selected_delta_metrics(df: pd.DataFrame, column: str) -> tuple[float, float]:
    selected = df[df["selected_for_radius_correction"].astype(bool)].copy()
    values = pd.to_numeric(selected[column], errors="coerce").abs()
    if values.empty or not values.notna().any():
        return float("nan"), float("nan")
    return float(np.sqrt(np.mean(values.dropna().to_numpy(dtype=np.float64) ** 2))), float(values.median())


def fraction_delta_reduced(reference_abs: pd.Series, candidate_abs: pd.Series) -> float:
    mask = reference_abs.notna() & candidate_abs.notna()
    if not mask.any():
        return float("nan")
    return float((candidate_abs[mask] < reference_abs[mask]).mean())


def build_edge_comparison(input_root: Path, strategy_name: str) -> pd.DataFrame:
    p_original = edge_df(input_root, strategy_name, "p_original")
    p_corrected = edge_df(input_root, strategy_name, "p_corrected")
    g_original = edge_df(input_root, strategy_name, "g_original")
    g_fixed = edge_df(input_root, strategy_name, "g_fixed")
    g_retrained = edge_df(input_root, strategy_name, "g_retrained")

    merged = _require_columns(
        p_corrected,
        [
            "edge_id",
            "source_node",
            "target_node",
            "selected_for_radius_correction",
            "snr",
            "tile_id",
            "original_radius_m",
            "corrected_radius_m",
            "radius_ratio",
            "radius_percent_change",
            "original_poiseuille_conductance",
            "corrected_poiseuille_conductance",
            "original_delta",
        ],
        label=f"{strategy_name}/p_corrected edge table",
    )
    merged["selection_strategy"] = strategy_name
    merged = merged.merge(
        p_original[["edge_id", "predicted_flow_nl_s", "flow_residual_nl_s"]].rename(
            columns={
                "predicted_flow_nl_s": "p_original_flow_nl_s",
                "flow_residual_nl_s": "p_original_flow_residual_nl_s",
            }
        ),
        on="edge_id",
        how="left",
    )
    merged = merged.merge(
        p_corrected[["edge_id", "observed_flow_nl_s", "predicted_flow_nl_s", "flow_residual_nl_s"]].rename(
            columns={
                "predicted_flow_nl_s": "p_corrected_flow_nl_s",
                "flow_residual_nl_s": "p_corrected_flow_residual_nl_s",
            }
        ),
        on="edge_id",
        how="left",
    )
    merged = merged.merge(
        g_original[["edge_id", "delta_e", "predicted_flow_nl_s", "flow_residual_nl_s"]].rename(
            columns={
                "delta_e": "original_delta",
                "predicted_flow_nl_s": "g_original_flow_nl_s",
                "flow_residual_nl_s": "g_original_flow_residual_nl_s",
            }
        ),
        on="edge_id",
        how="left",
        suffixes=("", "_dup"),
    )
    if "original_delta_dup" in merged.columns:
        merged = merged.drop(columns=["original_delta_dup"])
    merged = merged.merge(
        g_fixed[["edge_id", "delta_e", "predicted_flow_nl_s", "flow_residual_nl_s"]].rename(
            columns={
                "delta_e": "fixed_weight_delta",
                "predicted_flow_nl_s": "g_fixed_flow_nl_s",
                "flow_residual_nl_s": "g_fixed_flow_residual_nl_s",
            }
        ),
        on="edge_id",
        how="left",
    )
    merged = merged.merge(
        g_retrained[["edge_id", "delta_e", "predicted_flow_nl_s", "flow_residual_nl_s"]].rename(
            columns={
                "delta_e": "retrained_delta",
                "predicted_flow_nl_s": "g_retrained_flow_nl_s",
                "flow_residual_nl_s": "g_retrained_flow_residual_nl_s",
            }
        ),
        on="edge_id",
        how="left",
    )
    return merged.sort_values("edge_id").reset_index(drop=True)


def build_delta_comparison(edge_tables: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for strategy_name, table in edge_tables.items():
        table = table.copy()
        selected = table[table["selected_for_radius_correction"].astype(bool)].copy()
        unselected = table[~table["selected_for_radius_correction"].astype(bool)].copy()
        ref_sel = pd.to_numeric(selected["original_delta"], errors="coerce").abs()
        fix_sel = pd.to_numeric(selected["fixed_weight_delta"], errors="coerce").abs()
        retr_sel = pd.to_numeric(selected["retrained_delta"], errors="coerce").abs()
        ref_unsel = pd.to_numeric(unselected["original_delta"], errors="coerce").abs()
        fix_unsel = pd.to_numeric(unselected["fixed_weight_delta"], errors="coerce").abs()
        retr_unsel = pd.to_numeric(unselected["retrained_delta"], errors="coerce").abs()
        rows.append(
            {
                "selection_strategy": strategy_name,
                "selected_mean_abs_original_delta": float(ref_sel.mean()),
                "selected_median_abs_original_delta": float(ref_sel.median()),
                "selected_mean_abs_fixed_delta": float(fix_sel.mean()),
                "selected_median_abs_fixed_delta": float(fix_sel.median()),
                "selected_mean_abs_retrained_delta": float(retr_sel.mean()),
                "selected_median_abs_retrained_delta": float(retr_sel.median()),
                "fraction_selected_edges_abs_delta_reduced_fixed": fraction_delta_reduced(ref_sel, fix_sel),
                "fraction_selected_edges_abs_delta_reduced_retrained": fraction_delta_reduced(ref_sel, retr_sel),
                "unselected_mean_abs_original_delta": float(ref_unsel.mean()),
                "unselected_mean_abs_fixed_delta": float(fix_unsel.mean()),
                "unselected_mean_abs_retrained_delta": float(retr_unsel.mean()),
                "fraction_unselected_edges_abs_delta_reduced_fixed": fraction_delta_reduced(ref_unsel, fix_unsel),
                "fraction_unselected_edges_abs_delta_reduced_retrained": fraction_delta_reduced(ref_unsel, retr_unsel),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    input_root = args.input_root.expanduser().resolve()
    manifest = load_yaml(input_root / "experiment_manifest.yaml")
    if not isinstance(manifest, dict):
        manifest = {}

    rows: list[dict[str, object]] = []
    edge_tables: dict[str, pd.DataFrame] = {}
    for strategy_name in STRATEGY_ORDER:
        p_original = condition_summary(input_root, strategy_name, "p_original")
        p_corrected = condition_summary(input_root, strategy_name, "p_corrected")
        g_original = condition_summary(input_root, strategy_name, "g_original")
        g_fixed = condition_summary(input_root, strategy_name, "g_fixed")
        g_retrained = condition_summary(input_root, strategy_name, "g_retrained")
        comparison_table = build_edge_comparison(input_root, strategy_name)
        edge_tables[strategy_name] = comparison_table

        poiseuille_flow_change, poiseuille_flow_pct = compare_change(
            safe_float(p_corrected.get("flow_rmse_nl_s")),
            safe_float(p_original.get("flow_rmse_nl_s")),
        )
        poiseuille_k_change, poiseuille_k_pct = compare_change(
            safe_float(p_corrected.get("kirchhoff_rms_per_internal_node_nl_s")),
            safe_float(p_original.get("kirchhoff_rms_per_internal_node_nl_s")),
        )
        poiseuille_pressure_change, _ = compare_change(
            safe_float(p_corrected.get("pressure_range_pa")),
            safe_float(p_original.get("pressure_range_pa")),
        )
        poiseuille_sign_change, _ = compare_change(
            safe_float(p_corrected.get("sign_flip_fraction")),
            safe_float(p_original.get("sign_flip_fraction")),
        )

        g_fixed_flow_change, g_fixed_flow_pct = compare_change(
            safe_float(g_fixed.get("flow_rmse_nl_s")),
            safe_float(g_original.get("flow_rmse_nl_s")),
        )
        g_fixed_k_change, g_fixed_k_pct = compare_change(
            safe_float(g_fixed.get("kirchhoff_rms_per_internal_node_nl_s")),
            safe_float(g_original.get("kirchhoff_rms_per_internal_node_nl_s")),
        )
        g_fixed_pressure_change, _ = compare_change(
            safe_float(g_fixed.get("pressure_range_pa")),
            safe_float(g_original.get("pressure_range_pa")),
        )
        g_fixed_sign_change, _ = compare_change(
            safe_float(g_fixed.get("sign_flip_fraction")),
            safe_float(g_original.get("sign_flip_fraction")),
        )

        g_retrained_flow_change, g_retrained_flow_pct = compare_change(
            safe_float(g_retrained.get("flow_rmse_nl_s")),
            safe_float(g_original.get("flow_rmse_nl_s")),
        )
        g_retrained_k_change, g_retrained_k_pct = compare_change(
            safe_float(g_retrained.get("kirchhoff_rms_per_internal_node_nl_s")),
            safe_float(g_original.get("kirchhoff_rms_per_internal_node_nl_s")),
        )
        g_retrained_pressure_change, _ = compare_change(
            safe_float(g_retrained.get("pressure_range_pa")),
            safe_float(g_original.get("pressure_range_pa")),
        )
        g_retrained_sign_change, _ = compare_change(
            safe_float(g_retrained.get("sign_flip_fraction")),
            safe_float(g_original.get("sign_flip_fraction")),
        )

        selected_original_rms, selected_original_median = selected_delta_metrics(
            comparison_table, "original_delta"
        )
        selected_fixed_rms, _ = selected_delta_metrics(comparison_table, "fixed_weight_delta")
        selected_retrained_rms, _ = selected_delta_metrics(comparison_table, "retrained_delta")
        selected_delta_rms_change_fixed, selected_delta_rms_pct_fixed = compare_change(
            selected_fixed_rms, selected_original_rms
        )
        selected_delta_rms_change_retrained, selected_delta_rms_pct_retrained = compare_change(
            selected_retrained_rms, selected_original_rms
        )
        selected_abs_original = pd.to_numeric(comparison_table["original_delta"], errors="coerce").abs()
        selected_abs_fixed = pd.to_numeric(comparison_table["fixed_weight_delta"], errors="coerce").abs()
        selected_abs_retrained = pd.to_numeric(comparison_table["retrained_delta"], errors="coerce").abs()
        selected_mask = comparison_table["selected_for_radius_correction"].astype(bool)
        reduced_fixed = fraction_delta_reduced(
            selected_abs_original[selected_mask],
            selected_abs_fixed[selected_mask],
        )
        reduced_retrained = fraction_delta_reduced(
            selected_abs_original[selected_mask],
            selected_abs_retrained[selected_mask],
        )

        tolerance = float(manifest.get("metric_tolerance_fraction", 0.02))
        g_retrained_flow_ok = (
            math.isfinite(g_retrained_flow_pct) and g_retrained_flow_pct <= tolerance * 100.0
        )
        g_retrained_k_ok = (
            math.isfinite(g_retrained_k_pct) and g_retrained_k_pct <= tolerance * 100.0
        )
        residual_delta_ok = (
            math.isfinite(selected_delta_rms_change_retrained)
            and selected_delta_rms_change_retrained < 0.0
        )
        geometry_success = bool(g_retrained_flow_ok and g_retrained_k_ok and residual_delta_ok)

        per_condition = {
            "p_original": p_original,
            "p_corrected": p_corrected,
            "g_original": g_original,
            "g_fixed": g_fixed,
            "g_retrained": g_retrained,
        }
        for condition_name, row in per_condition.items():
            row = dict(row)
            row["geometry_refinement_success"] = geometry_success if condition_name == "g_retrained" else False
            row["poiseuille_flow_rmse_change_nl_s"] = poiseuille_flow_change
            row["poiseuille_flow_rmse_percent_change"] = poiseuille_flow_pct
            row["poiseuille_kirchhoff_rms_change_nl_s"] = poiseuille_k_change
            row["poiseuille_kirchhoff_rms_percent_change"] = poiseuille_k_pct
            row["poiseuille_pressure_range_change_pa"] = poiseuille_pressure_change
            row["poiseuille_sign_flip_fraction_change"] = poiseuille_sign_change
            row["g_fixed_flow_rmse_change_nl_s"] = g_fixed_flow_change
            row["g_fixed_flow_rmse_percent_change"] = g_fixed_flow_pct
            row["g_fixed_kirchhoff_rms_change_nl_s"] = g_fixed_k_change
            row["g_fixed_kirchhoff_rms_percent_change"] = g_fixed_k_pct
            row["g_fixed_pressure_range_change_pa"] = g_fixed_pressure_change
            row["g_fixed_sign_flip_fraction_change"] = g_fixed_sign_change
            row["g_retrained_flow_rmse_change_nl_s"] = g_retrained_flow_change
            row["g_retrained_flow_rmse_percent_change"] = g_retrained_flow_pct
            row["g_retrained_kirchhoff_rms_change_nl_s"] = g_retrained_k_change
            row["g_retrained_kirchhoff_rms_percent_change"] = g_retrained_k_pct
            row["g_retrained_pressure_range_change_pa"] = g_retrained_pressure_change
            row["g_retrained_sign_flip_fraction_change"] = g_retrained_sign_change
            row["selected_delta_rms_original"] = selected_original_rms
            row["selected_delta_rms_fixed"] = selected_fixed_rms
            row["selected_delta_rms_retrained"] = selected_retrained_rms
            row["selected_delta_median_abs_original"] = selected_original_median
            row["selected_delta_rms_change_fixed"] = selected_delta_rms_change_fixed
            row["selected_delta_rms_percent_change_fixed"] = selected_delta_rms_pct_fixed
            row["selected_delta_rms_change_retrained"] = selected_delta_rms_change_retrained
            row["selected_delta_rms_percent_change_retrained"] = selected_delta_rms_pct_retrained
            row["fraction_selected_edges_abs_delta_reduced_fixed"] = reduced_fixed
            row["fraction_selected_edges_abs_delta_reduced_retrained"] = reduced_retrained
            rows.append(row)

        comparison_table.to_csv(
            input_root / f"radius_refinement_edge_comparison_{strategy_name}.csv",
            index=False,
        )
        pd.DataFrame(per_condition.values()).to_csv(
            input_root / f"radius_refinement_{strategy_name}_summary.csv",
            index=False,
        )

    summary_df = pd.DataFrame(rows)
    summary_df["condition_name"] = pd.Categorical(
        summary_df["condition_name"], categories=list(CONDITION_ORDER), ordered=True
    )
    summary_df = summary_df.sort_values(["strategy_name", "condition_name"]).reset_index(drop=True)
    summary_df.to_csv(input_root / "radius_refinement_summary.csv", index=False)

    delta_comparison_df = build_delta_comparison(edge_tables)
    delta_comparison_df.to_csv(input_root / "radius_refinement_delta_comparison.csv", index=False)

    payload = {
        "experiment_manifest": manifest,
        "summary_rows": summary_df.to_dict(orient="records"),
        "delta_comparison_rows": delta_comparison_df.to_dict(orient="records"),
    }
    write_yaml(input_root / "radius_refinement_summary.yaml", payload)


if __name__ == "__main__":
    main()
