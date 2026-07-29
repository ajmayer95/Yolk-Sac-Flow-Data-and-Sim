#!/usr/bin/env python
"""Compute message-passing field similarity against the K=2 reference within each dataset."""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT = (
    PROJECT_ROOT
    / "outputs"
    / "dc"
    / "04_message_passing_sensitivity"
    / "combined_gnn_message_passing_depth_summary.csv"
)
DEFAULT_OUTPUT = (
    PROJECT_ROOT
    / "outputs"
    / "dc"
    / "04_message_passing_sensitivity"
    / "message_passing_field_similarity.csv"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-summary", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-csv", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def read_rows(path: Path) -> list[dict[str, str]]:
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
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def correlation(values_a: list[float], values_b: list[float]) -> float:
    pairs = [
        (a, b)
        for a, b in zip(values_a, values_b, strict=False)
        if math.isfinite(a) and math.isfinite(b)
    ]
    if len(pairs) < 2:
        return float("nan")
    xs = [pair[0] for pair in pairs]
    ys = [pair[1] for pair in pairs]
    mean_x = sum(xs) / len(xs)
    mean_y = sum(ys) / len(ys)
    centered_x = [x - mean_x for x in xs]
    centered_y = [y - mean_y for y in ys]
    denom_x = math.sqrt(sum(value * value for value in centered_x))
    denom_y = math.sqrt(sum(value * value for value in centered_y))
    if denom_x <= 1.0e-30 or denom_y <= 1.0e-30:
        return float("nan")
    return sum(x * y for x, y in zip(centered_x, centered_y, strict=False)) / (denom_x * denom_y)


def read_edge_field(path: Path, column: str) -> list[float] | None:
    if not path.exists():
        return None
    rows = read_rows(path)
    values = [safe_float(row.get(column)) for row in rows]
    return values if values else None


def read_node_field(path: Path, column: str) -> list[float] | None:
    if not path.exists():
        return None
    rows = read_rows(path)
    values = [safe_float(row.get(column)) for row in rows]
    return values if values else None


def main() -> None:
    args = parse_args()
    input_summary = args.input_summary.expanduser().resolve()
    output_csv = args.output_csv.expanduser().resolve()

    summary_rows = read_rows(input_summary)
    rows_by_dataset: dict[str, list[dict[str, str]]] = {}
    for row in summary_rows:
        rows_by_dataset.setdefault(row.get("dataset_name", ""), []).append(row)

    output_rows: list[dict[str, object]] = []
    warnings: list[str] = []

    for dataset_name, dataset_rows in rows_by_dataset.items():
        reference = None
        for row in dataset_rows:
            if int(safe_float(row.get("message_passing_layers"))) == 2:
                reference = row
                break
        if reference is None:
            warnings.append(f"Missing K=2 reference for dataset {dataset_name}.")
            for row in dataset_rows:
                output_rows.append(
                    {
                        "dataset_name": dataset_name,
                        "run_name": row.get("run_name", ""),
                        "message_passing_layers": safe_float(row.get("message_passing_layers")),
                        "delta_corr_vs_K2": float("nan"),
                        "pressure_corr_vs_K2": float("nan"),
                        "flow_corr_vs_K2": float("nan"),
                        "warning": "missing_K2_reference",
                    }
                )
            continue

        ref_dir = Path(reference.get("output_dir", "")).expanduser()
        ref_edge_path = ref_dir / "edge_predictions.csv"
        ref_node_path = ref_dir / "node_predictions.csv"
        ref_delta = read_edge_field(ref_edge_path, "delta_e")
        ref_flow = read_edge_field(ref_edge_path, "q_pred_m3_s")
        ref_pressure = read_node_field(ref_node_path, "pressure_pa")

        for row in dataset_rows:
            run_dir = Path(row.get("output_dir", "")).expanduser()
            edge_path = run_dir / "edge_predictions.csv"
            node_path = run_dir / "node_predictions.csv"
            delta_values = read_edge_field(edge_path, "delta_e")
            flow_values = read_edge_field(edge_path, "q_pred_m3_s")
            pressure_values = read_node_field(node_path, "pressure_pa")
            row_warnings: list[str] = []

            if ref_delta is None or delta_values is None:
                delta_corr = float("nan")
                row_warnings.append("missing_delta_field")
            else:
                delta_corr = correlation(delta_values, ref_delta)

            if ref_flow is None or flow_values is None:
                flow_corr = float("nan")
                row_warnings.append("missing_flow_field")
            else:
                flow_corr = correlation(flow_values, ref_flow)

            if ref_pressure is None or pressure_values is None:
                pressure_corr = float("nan")
                row_warnings.append("missing_pressure_field")
            else:
                pressure_corr = correlation(pressure_values, ref_pressure)

            output_rows.append(
                {
                    "dataset_name": dataset_name,
                    "run_name": row.get("run_name", ""),
                    "message_passing_layers": safe_float(row.get("message_passing_layers")),
                    "delta_corr_vs_K2": delta_corr,
                    "pressure_corr_vs_K2": pressure_corr,
                    "flow_corr_vs_K2": flow_corr,
                    "warning": "|".join(row_warnings),
                }
            )

    write_rows(output_csv, output_rows)
    print(output_csv)
    for warning in warnings:
        print(f"WARNING: {warning}")


if __name__ == "__main__":
    main()
