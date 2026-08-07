#!/usr/bin/env python
"""Analyze AC Step 3 distensibility-alpha profile sweeps."""

from __future__ import annotations

import argparse
import csv
import functools
import math
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT_ROOT = PROJECT_ROOT / "outputs" / "ac" / "03_distensibility_alpha_profiles" / "H1"
LABEL_ORDER = ("F1", "B1", "K1")
DISPLAY_LABEL_BY_CODE = {
    "F1": "Flow-Prioritized",
    "B1": "Balanced",
    "K1": "Kirchhoff-Prioritized",
}
REGIME_BY_LABEL = {
    "F1": "flow_prioritized",
    "B1": "balanced",
    "K1": "conservation_prioritized",
}
SCORE_FORMULA_BY_REGIME = {
    "flow_prioritized": "0.75*complex_flow_rmse_nl_s + 0.25*kirchhoff_rms_per_internal_node_nl_s",
    "balanced": "0.5*complex_flow_rmse_nl_s + 0.5*kirchhoff_rms_per_internal_node_nl_s",
    "conservation_prioritized": "0.25*complex_flow_rmse_nl_s + 0.75*kirchhoff_rms_per_internal_node_nl_s",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, default=DEFAULT_INPUT_ROOT)
    return parser.parse_args()


def safe_float(value) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return result if math.isfinite(result) else float("nan")


def fold_phase_difference_deg(value: float) -> float:
    phase = safe_float(value)
    if not math.isfinite(phase):
        return float("nan")
    while phase > 180.0:
        phase -= 360.0
    while phase <= -180.0:
        phase += 360.0
    if phase > 90.0:
        phase -= 180.0
    elif phase < -90.0:
        phase += 180.0
    return phase


def write_rows(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    fieldnames: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def physical_selection_score(regime: str, flow_rmse_nl_s: float, kirchhoff_rms_nl_s: float) -> float:
    if not (math.isfinite(flow_rmse_nl_s) and math.isfinite(kirchhoff_rms_nl_s)):
        return float("nan")
    if regime == "flow_prioritized":
        return 0.75 * flow_rmse_nl_s + 0.25 * kirchhoff_rms_nl_s
    if regime == "conservation_prioritized":
        return 0.25 * flow_rmse_nl_s + 0.75 * kirchhoff_rms_nl_s
    if regime == "balanced":
        return 0.5 * flow_rmse_nl_s + 0.5 * kirchhoff_rms_nl_s
    return float("nan")


def representative_compare(a: dict[str, object], b: dict[str, object]) -> int:
    score_a = safe_float(a.get("selection_score"))
    score_b = safe_float(b.get("selection_score"))
    phase_a = safe_float(a.get("folded_arterial_phase_difference_deg"))
    phase_b = safe_float(b.get("folded_arterial_phase_difference_deg"))
    d0_a = safe_float(a.get("D0"))
    d0_b = safe_float(b.get("D0"))

    if math.isfinite(score_a) and math.isfinite(score_b):
        baseline = max(min(score_a, score_b), 1.0e-12)
        relative_gap = abs(score_a - score_b) / baseline
        if relative_gap <= 0.01:
            if math.isfinite(phase_a) and math.isfinite(phase_b) and not math.isclose(phase_a, phase_b):
                return -1 if phase_a < phase_b else 1
        if not math.isclose(score_a, score_b):
            return -1 if score_a < score_b else 1
    elif math.isfinite(score_a) != math.isfinite(score_b):
        return -1 if math.isfinite(score_a) else 1

    if math.isfinite(phase_a) and math.isfinite(phase_b) and not math.isclose(phase_a, phase_b):
        return -1 if phase_a < phase_b else 1
    if math.isfinite(phase_a) != math.isfinite(phase_b):
        return -1 if math.isfinite(phase_a) else 1

    if math.isfinite(d0_a) and math.isfinite(d0_b) and not math.isclose(d0_a, d0_b):
        return -1 if d0_a < d0_b else 1
    return 0


def select_pareto_representative(regime: str, candidates: list[dict[str, object]]) -> dict[str, object]:
    if not candidates:
        raise ValueError("No candidates provided for pareto representative selection.")

    def kirchhoff_value(row: dict[str, object]) -> float:
        return safe_float(row.get("kirchhoff_rms_per_internal_node_nl_s"))

    def flow_value(row: dict[str, object]) -> float:
        return safe_float(row.get("complex_flow_rmse_nl_s"))

    def phase_value(row: dict[str, object]) -> float:
        return safe_float(row.get("arterial_pressure_phase_difference_deg"))

    def d0_value(row: dict[str, object]) -> float:
        return safe_float(row.get("D0"))

    if regime == "flow_prioritized":
        ranked = sorted(
            candidates,
            key=lambda row: (
                kirchhoff_value(row),
                phase_value(row),
                d0_value(row),
            ),
        )
        return ranked[0]
    if regime == "conservation_prioritized":
        ranked = sorted(
            candidates,
            key=lambda row: (
                flow_value(row),
                phase_value(row),
                d0_value(row),
            ),
        )
        return ranked[0]
    if regime == "balanced":
        ranked = sorted(
            candidates,
            key=lambda row: (
                math.hypot(flow_value(row), kirchhoff_value(row)),
                phase_value(row),
                d0_value(row),
            ),
        )
        return ranked[0]
    ranked = list(candidates)
    ranked.sort(key=functools.cmp_to_key(representative_compare))
    return ranked[0]


def main() -> None:
    args = parse_args()
    input_root = args.input_root.expanduser().resolve()
    rows: list[dict[str, object]] = []

    for summary_path in sorted(input_root.glob("*/alpha_*/D0_*/summary.csv")):
        label = summary_path.parents[2].name
        alpha_dir = summary_path.parents[1].name
        d0_dir = summary_path.parent.name
        if label not in LABEL_ORDER:
            continue
        regime = REGIME_BY_LABEL[label]
        table = pd.read_csv(summary_path)
        for _, series in table.iterrows():
            row = {key: series[key] for key in table.columns}
            row["representative_label"] = label
            row["selection_category"] = regime
            row["selection_display_label"] = DISPLAY_LABEL_BY_CODE[label]
            row["alpha_dir"] = alpha_dir
            row["D0_dir"] = d0_dir
            row["profile_run_dir"] = str(summary_path.parent.resolve())
            row["summary_path"] = str(summary_path.resolve())
            row["run_name"] = f"{label}__{alpha_dir}__{d0_dir}__{row.get('model_name', '')}"
            row["D0"] = safe_float(row.get("D0"))
            row["alpha"] = safe_float(row.get("alpha"))
            row["lambda_q"] = safe_float(row.get("lambda_q"))
            row["lambda_k"] = safe_float(row.get("lambda_k"))
            row["lambda_b"] = safe_float(row.get("lambda_b"))
            row["complex_flow_rmse_nl_s"] = safe_float(row.get("complex_flow_rmse_nl_s"))
            row["kirchhoff_rms_per_internal_node_nl_s"] = safe_float(row.get("kirchhoff_rms_per_internal_node_nl_s"))
            row["arterial_pressure_phase_difference_deg"] = safe_float(row.get("arterial_pressure_phase_difference_deg"))
            row["folded_arterial_phase_difference_deg"] = fold_phase_difference_deg(
                row["arterial_pressure_phase_difference_deg"]
            )
            row["selection_score"] = physical_selection_score(
                regime,
                row["complex_flow_rmse_nl_s"],
                row["kirchhoff_rms_per_internal_node_nl_s"],
            )
            row["selection_score_formula"] = SCORE_FORMULA_BY_REGIME[regime]
            rows.append(row)

    if not rows:
        raise FileNotFoundError(f"No summary.csv files found under {input_root}.")

    combined = pd.DataFrame(rows)
    combined = combined.sort_values(
        ["model_name", "representative_label", "alpha", "D0"],
        kind="stable",
    )
    combined.to_csv(input_root / "combined_results.csv", index=False)

    representatives: list[dict[str, object]] = []
    metric_minima: list[dict[str, object]] = []
    for (model_name, label, alpha), group in combined.groupby(
        ["model_name", "representative_label", "alpha"],
        dropna=False,
        sort=False,
    ):
        candidates = group.to_dict("records")
        regime = REGIME_BY_LABEL[str(label)]
        best = select_pareto_representative(regime, candidates)
        for metric_name in ("complex_flow_rmse_nl_s", "kirchhoff_rms_per_internal_node_nl_s"):
            valid = [
                row
                for row in candidates
                if math.isfinite(safe_float(row.get(metric_name)))
            ]
            if not valid:
                continue
            best_metric = min(
                valid,
                key=lambda row: (
                    safe_float(row.get(metric_name)),
                    safe_float(row.get("folded_arterial_phase_difference_deg")),
                    safe_float(row.get("D0")),
                ),
            )
            metric_minima.append(
                {
                    "model_name": str(model_name),
                    "model_label": str(best_metric.get("model_label", "")),
                    "representative_label": str(label),
                    "selection_display_label": DISPLAY_LABEL_BY_CODE[str(label)],
                    "selection_category": str(best_metric.get("selection_category", "")),
                    "harmonic_number": int(float(best_metric.get("harmonic_number", 0))),
                    "alpha": float(best_metric["alpha"]),
                    "metric_name": metric_name,
                    "metric_value": float(best_metric[metric_name]),
                    "representative_D0": float(best_metric["D0"]),
                    "arterial_pressure_phase_difference_deg": float(best_metric["arterial_pressure_phase_difference_deg"]),
                    "folded_arterial_phase_difference_deg": float(best_metric["folded_arterial_phase_difference_deg"]),
                    "profile_run_dir": str(best_metric["profile_run_dir"]),
                }
            )
        representatives.append(
            {
                "model_name": str(model_name),
                "model_label": str(best.get("model_label", "")),
                "representative_label": str(label),
                "selection_display_label": DISPLAY_LABEL_BY_CODE[str(label)],
                "selection_category": str(best.get("selection_category", "")),
                "harmonic_number": int(float(best.get("harmonic_number", 0))),
                "alpha": float(best["alpha"]),
                "representative_D0": float(best["D0"]),
                "lambda_q": float(best["lambda_q"]),
                "lambda_k": float(best["lambda_k"]),
                "lambda_b": float(best["lambda_b"]),
                "complex_flow_rmse_nl_s": float(best["complex_flow_rmse_nl_s"]),
                "kirchhoff_rms_per_internal_node_nl_s": float(best["kirchhoff_rms_per_internal_node_nl_s"]),
                "arterial_pressure_phase_difference_deg": float(best["arterial_pressure_phase_difference_deg"]),
                "folded_arterial_phase_difference_deg": float(best["folded_arterial_phase_difference_deg"]),
                "selection_score": float(best["selection_score"]),
                "selection_score_formula": str(best["selection_score_formula"]),
                "pareto_selection_strategy": (
                    "min_kirchhoff_rms"
                    if regime == "flow_prioritized"
                    else "min_flow_rmse"
                    if regime == "conservation_prioritized"
                    else "min_distance_to_origin"
                ),
                "profile_run_dir": str(best["profile_run_dir"]),
            }
        )
    representatives = sorted(
        representatives,
        key=lambda row: (
            str(row["model_name"]),
            LABEL_ORDER.index(str(row["representative_label"])),
            float(row["alpha"]),
        ),
    )
    write_rows(input_root / "representative_configurations.csv", representatives)
    write_rows(input_root / "metric_minima.csv", metric_minima)
    print(f"[ok] Wrote {input_root / 'combined_results.csv'}")
    print(f"[ok] Wrote {input_root / 'representative_configurations.csv'}")
    print(f"[ok] Wrote {input_root / 'metric_minima.csv'}")


if __name__ == "__main__":
    main()
