#!/usr/bin/env python
"""Analyze AC Step 2 physics-weight sweep results."""

from __future__ import annotations

import argparse
import csv
import functools
import math
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from physics_weight_sweep_lib import category_scores, nondominated_sort, percentile_scale  # noqa: E402
from utils import write_yaml  # noqa: E402


DEFAULT_INPUT_ROOT = PROJECT_ROOT / "outputs" / "ac" / "02_physics_weight_sweep" / "H1"
REGIME_ORDER = ("flow_prioritized", "balanced", "conservation_prioritized")
REGIME_PREFIX = {
    "flow_prioritized": "F",
    "balanced": "B",
    "conservation_prioritized": "K",
}
REGIME_SCORE_FORMULAS = {
    "flow_prioritized": "0.75*complex_flow_rmse_nl_s + 0.25*kirchhoff_rms_per_internal_node_nl_s",
    "balanced": "0.5*complex_flow_rmse_nl_s + 0.5*kirchhoff_rms_per_internal_node_nl_s",
    "conservation_prioritized": "0.25*complex_flow_rmse_nl_s + 0.75*kirchhoff_rms_per_internal_node_nl_s",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, default=DEFAULT_INPUT_ROOT)
    parser.add_argument("--top-k", type=int, default=4)
    return parser.parse_args()


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


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


def safe_float(value) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return result if math.isfinite(result) else float("nan")


def parse_scalar(value: str) -> object:
    text = str(value).strip()
    if text == "":
        return ""
    lowered = text.lower()
    if lowered == "true":
        return True
    if lowered == "false":
        return False
    try:
        number = float(text)
    except ValueError:
        return text
    if number.is_integer() and lowered not in {"nan", "inf", "-inf"}:
        return int(number)
    return number


def classify_weighting_regime(lambda_q: float, lambda_k: float) -> str:
    if lambda_q >= 10.0 * lambda_k:
        return "flow_prioritized"
    if lambda_k >= 10.0 * lambda_q:
        return "conservation_prioritized"
    return "balanced"


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
    flow_a = safe_float(a.get("complex_flow_rmse_nl_s"))
    flow_b = safe_float(b.get("complex_flow_rmse_nl_s"))
    kirchhoff_a = safe_float(a.get("kirchhoff_rms_per_internal_node_nl_s"))
    kirchhoff_b = safe_float(b.get("kirchhoff_rms_per_internal_node_nl_s"))
    dist_a = math.hypot(flow_a, kirchhoff_a) if math.isfinite(flow_a) and math.isfinite(kirchhoff_a) else float("inf")
    dist_b = math.hypot(flow_b, kirchhoff_b) if math.isfinite(flow_b) and math.isfinite(kirchhoff_b) else float("inf")
    phase_a = safe_float(a.get("arterial_pressure_phase_difference_deg"))
    phase_b = safe_float(b.get("arterial_pressure_phase_difference_deg"))
    run_a = str(a.get("run_name", ""))
    run_b = str(b.get("run_name", ""))

    if math.isfinite(dist_a) and math.isfinite(dist_b) and not math.isclose(dist_a, dist_b):
        return -1 if dist_a < dist_b else 1
    if math.isfinite(dist_a) != math.isfinite(dist_b):
        return -1 if math.isfinite(dist_a) else 1

    if math.isfinite(phase_a) and math.isfinite(phase_b) and not math.isclose(phase_a, phase_b):
        return -1 if phase_a < phase_b else 1
    if math.isfinite(phase_a) != math.isfinite(phase_b):
        return -1 if math.isfinite(phase_a) else 1

    if run_a < run_b:
        return -1
    if run_a > run_b:
        return 1
    return 0


def percentile_or_nan(values: list[float], percentile: float) -> float:
    finite = [value for value in values if math.isfinite(value)]
    if not finite:
        return float("nan")
    return float(np.percentile(np.asarray(finite, dtype=np.float64), percentile))


def discover_summary_paths(input_root: Path) -> list[Path]:
    return sorted(input_root.glob("*/summary.csv"))


def load_rows(input_root: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for summary_path in discover_summary_paths(input_root):
        run_name = summary_path.parent.name
        summary_rows = read_csv_rows(summary_path)
        for summary_row in summary_rows:
            row = {key: parse_scalar(value) for key, value in summary_row.items()}
            row["run_name"] = run_name
            row["run_dir"] = str(summary_path.parent.resolve())
            row["summary_path"] = str(summary_path.resolve())
            row["lambda_q"] = safe_float(row.get("lambda_q"))
            row["lambda_k"] = safe_float(row.get("lambda_k"))
            row["lambda_b"] = safe_float(row.get("lambda_b"))
            row["complex_flow_rmse_nl_s"] = safe_float(row.get("complex_flow_rmse_nl_s"))
            row["kirchhoff_rms_per_internal_node_nl_s"] = safe_float(
                row.get("kirchhoff_rms_per_internal_node_nl_s")
            )
            row["arterial_pressure_phase_difference_deg"] = safe_float(
                row.get("arterial_pressure_phase_difference_deg")
            )
            row["solver_success"] = bool(row.get("solver_success", False))
            row["weighting_regime"] = classify_weighting_regime(
                float(row["lambda_q"]),
                float(row["lambda_k"]),
            )
            rows.append(row)
    return rows


def analyze_rows(rows: list[dict[str, object]], top_k: int) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    representatives: list[dict[str, object]] = []
    model_names = sorted({str(row.get("model_name", "")) for row in rows})

    for row in rows:
        row["flow_rmse_rank_scaled"] = float("nan")
        row["kirchhoff_rms_rank_scaled"] = float("nan")
        row["flow_rmse_p05"] = float("nan")
        row["flow_rmse_p95"] = float("nan")
        row["kirchhoff_rms_p05"] = float("nan")
        row["kirchhoff_rms_p95"] = float("nan")
        row["score_flow_prioritized"] = float("nan")
        row["score_balanced"] = float("nan")
        row["score_conservation_prioritized"] = float("nan")
        row["selection_score"] = float("nan")
        row["selection_score_formula"] = ""
        row["selection_score_regime"] = ""
        row["pareto_rank"] = float("nan")
        row["is_pareto_front"] = False
        row["selected_representative"] = False
        row["selection_rank_within_regime"] = float("nan")
        row["plot_label"] = ""

    for model_name in model_names:
        model_rows = [row for row in rows if str(row.get("model_name", "")) == model_name]
        valid_rows = [
            row
            for row in model_rows
            if math.isfinite(safe_float(row.get("complex_flow_rmse_nl_s")))
            and math.isfinite(safe_float(row.get("kirchhoff_rms_per_internal_node_nl_s")))
        ]
        flow_values = [safe_float(row["complex_flow_rmse_nl_s"]) for row in valid_rows]
        kirchhoff_values = [safe_float(row["kirchhoff_rms_per_internal_node_nl_s"]) for row in valid_rows]
        flow_p05 = percentile_or_nan(flow_values, 5.0)
        flow_p95 = percentile_or_nan(flow_values, 95.0)
        kirchhoff_p05 = percentile_or_nan(kirchhoff_values, 5.0)
        kirchhoff_p95 = percentile_or_nan(kirchhoff_values, 95.0)

        for row in model_rows:
            flow_scaled = percentile_scale(
                safe_float(row.get("complex_flow_rmse_nl_s")),
                flow_p05,
                flow_p95,
            )
            kirchhoff_scaled = percentile_scale(
                safe_float(row.get("kirchhoff_rms_per_internal_node_nl_s")),
                kirchhoff_p05,
                kirchhoff_p95,
            )
            row["flow_rmse_p05"] = flow_p05
            row["flow_rmse_p95"] = flow_p95
            row["kirchhoff_rms_p05"] = kirchhoff_p05
            row["kirchhoff_rms_p95"] = kirchhoff_p95
            row["flow_rmse_rank_scaled"] = flow_scaled
            row["kirchhoff_rms_rank_scaled"] = kirchhoff_scaled
            scores = category_scores(flow_scaled, kirchhoff_scaled)
            row["score_flow_prioritized"] = scores["score_flow_prioritized"]
            row["score_balanced"] = scores["score_balanced"]
            row["score_conservation_prioritized"] = scores["score_conservation_prioritized"]

        ranks = nondominated_sort(valid_rows, ("complex_flow_rmse_nl_s", "kirchhoff_rms_per_internal_node_nl_s"))
        for row, rank in zip(valid_rows, ranks):
            row["pareto_rank"] = rank if rank is not None else float("nan")
            row["is_pareto_front"] = rank == 1

        for regime in REGIME_ORDER:
            candidates = [row for row in valid_rows if str(row.get("weighting_regime", "")) == regime]
            for row in candidates:
                row["selection_score"] = physical_selection_score(
                    regime,
                    safe_float(row.get("complex_flow_rmse_nl_s")),
                    safe_float(row.get("kirchhoff_rms_per_internal_node_nl_s")),
                )
                row["selection_score_formula"] = REGIME_SCORE_FORMULAS[regime]
                row["selection_score_regime"] = regime
            candidates.sort(key=functools.cmp_to_key(representative_compare))
            for rank_index, row in enumerate(candidates[:top_k], start=1):
                row["selected_representative"] = True
                row["selection_rank_within_regime"] = rank_index
                row["plot_label"] = f"{REGIME_PREFIX[regime]}{rank_index}"
                representatives.append(row)

    representatives.sort(
        key=lambda row: (
            str(row.get("model_name", "")),
            REGIME_ORDER.index(str(row.get("weighting_regime", "balanced"))),
            safe_float(row.get("selection_rank_within_regime")),
        )
    )
    return rows, representatives


def main() -> None:
    args = parse_args()
    input_root = args.input_root.expanduser().resolve()
    rows = load_rows(input_root)
    all_rows, representatives = analyze_rows(rows, top_k=int(args.top_k))

    write_rows(input_root / "ac_physics_weight_all_runs.csv", all_rows)
    write_rows(input_root / "ac_physics_weight_representatives.csv", representatives)
    write_yaml(
        input_root / "ac_physics_weight_analysis.yaml",
        {
            "input_root": str(input_root),
            "n_all_rows": len(all_rows),
            "n_representatives": len(representatives),
            "representative_count_per_regime": int(args.top_k),
            "all_runs_csv": str(input_root / "ac_physics_weight_all_runs.csv"),
            "representatives_csv": str(input_root / "ac_physics_weight_representatives.csv"),
        },
    )


if __name__ == "__main__":
    main()
