#!/usr/bin/env python
"""Create plots and dashboards for the best GNN run of each dataset."""

from __future__ import annotations

import argparse
import csv
import html
import json
import math
import sys
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from distensibility.io import load_dataset, write_json
from gnn_plotting import (
    MODEL_LABELS,
    create_gnn_report,
    plot_validation_comparison,
)


DEFAULT_MODELS = [
    "physics_informed_gnn",
    "vanilla_gcn",
    "edge_local_mlp",
]


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=PROJECT_ROOT / "data" / "synthetic" / "manifest.csv",
    )
    parser.add_argument("--datasets", type=Path, nargs="*", default=None)
    parser.add_argument(
        "--model",
        choices=DEFAULT_MODELS,
        default=None,
        help="Generate one model family only (backward-compatible shortcut).",
    )
    parser.add_argument(
        "--models",
        choices=DEFAULT_MODELS,
        nargs="+",
        default=None,
        help="Model families to compare; defaults to all three.",
    )
    parser.add_argument(
        "--selection-metric",
        default="splits.validation.dc_relative_rmse",
        help="Dot-separated metrics.json field minimized for selection.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "figures" / "gnn_comparison",
    )
    parser.add_argument(
        "--require-complete",
        action="store_true",
        help="Fail if any requested model has no complete run for a dataset.",
    )
    return parser.parse_args()


def nested_value(payload: dict, dotted: str) -> float:
    value = payload
    for part in dotted.split("."):
        value = value[part]
    return float(value)


def select_best(dataset_name: str, model: str, metric: str):
    root = PROJECT_ROOT / "outputs" / "runs" / "gnn" / dataset_name
    candidates = []
    for run_dir in sorted(root.glob(f"{model}__*/")):
        try:
            config = json.loads((run_dir / "config.yaml").read_text())
            metrics = json.loads((run_dir / "metrics.json").read_text())
            score = nested_value(metrics, metric)
        except (OSError, KeyError, ValueError, json.JSONDecodeError):
            continue
        required = (
            "predicted_velocities.npz",
            "pressure_field.npz",
            "corrections.npz",
            "training_history.json",
        )
        has_pressure = True
        if model == "vanilla_gcn":
            try:
                with np.load(run_dir / "pressure_field.npz") as archive:
                    has_pressure = bool(
                        np.isfinite(archive["predicted_pressure_pa"]).any()
                    )
            except (OSError, KeyError, ValueError):
                has_pressure = False
        if (
            math.isfinite(score)
            and has_pressure
            and all((run_dir / name).is_file() for name in required)
        ):
            candidates.append((score, run_dir, config))
    if not candidates:
        raise RuntimeError(
            f"No complete {model} candidates for {dataset_name}"
        )
    return min(candidates, key=lambda item: (item[0], item[1].name))


def manifest_datasets(path: Path):
    with path.open(newline="") as handle:
        return [
            PROJECT_ROOT / "data" / "synthetic" / row["file"]
            for row in csv.DictReader(handle)
        ]


def write_model_index(output_root, model, summaries):
    label = MODEL_LABELS.get(model, model)
    rows = []
    for summary in summaries:
        dataset_name = Path(summary["dataset"]).stem
        rows.append(
            "<tr>"
            f"<td><a href='{dataset_name}/dashboard.html'>{html.escape(dataset_name)}</a></td>"
            f"<td>{html.escape(Path(summary['selected_run']).name)}</td>"
            f"<td>{summary['K']}</td>"
            f"<td>{html.escape(summary['harmonic_mode'])}</td>"
            f"<td>{summary['selection_score']:.6g}</td>"
            "</tr>"
        )
    (output_root / "index.html").write_text(
        f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Best synthetic {html.escape(label)} reports</title>
<style>
body{{font:14px/1.45 system-ui,sans-serif;margin:24px;color:#18212b}}
table{{width:100%;border-collapse:collapse}}th,td{{padding:8px;border-bottom:1px solid #e2e8ee;text-align:right}}
th:first-child,td:first-child{{text-align:left}}a{{color:#1769aa}}
code{{background:#f2f4f7;padding:2px 4px;border-radius:4px}}
</style></head><body><h1>Best {html.escape(label)} by synthetic dataset</h1>
<p>Selection minimizes validation DC relative RMSE within the selected model family.</p>
<table><thead><tr><th>dataset</th><th>selected run</th><th>K</th><th>harmonics</th><th>validation DC relative RMSE</th></tr></thead>
<tbody>"""
        + "".join(rows)
        + "</tbody></table></body></html>"
    )


def write_comparison_index(output_root: Path, datasets, by_model, missing):
    model_headers = "".join(
        f"<th>{html.escape(MODEL_LABELS.get(model, model))}</th>"
        for model in by_model
    )
    rows = []
    for dataset_path in datasets:
        dataset_name = dataset_path.stem
        cells = []
        for model, reports in by_model.items():
            summary = reports.get(dataset_name)
            if summary is None:
                cells.append("<td class='missing'>pending</td>")
            else:
                cells.append(
                    "<td>"
                    f"<a href='{model}/{dataset_name}/dashboard.html'>"
                    f"K={summary['K']}, {html.escape(summary['harmonic_mode'])}</a>"
                    f"<br><small>val DC RMSE={summary['selection_score']:.4f}</small>"
                    "</td>"
                )
        rows.append(
            f"<tr><td>{html.escape(dataset_name)}</td>{''.join(cells)}</tr>"
        )
    payload = {
        "models": list(by_model),
        "n_datasets": len(datasets),
        "missing": missing,
    }
    write_json(output_root / "comparison_manifest.json", payload)
    plot_validation_comparison(
        output_root / "validation_comparison.png",
        [path.stem for path in datasets],
        by_model,
    )
    (output_root / "index.html").write_text(
        """<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Synthetic GNN comparison</title><style>
body{font:14px/1.45 system-ui,sans-serif;margin:24px;color:#18212b}
table{width:100%;border-collapse:collapse}th,td{padding:9px;border-bottom:1px solid #e2e8ee;text-align:left}
th{position:sticky;top:0;background:white}a{color:#1769aa}.missing{color:#9a6700}
small{color:#637282}
</style></head><body><h1>Best neural model configurations by dataset</h1>
<p>Every model family is selected independently using validation DC relative RMSE. Test metrics are not used.</p>
<p><img src="validation_comparison.png" alt="Validation comparison" style="max-width:100%;border:1px solid #e2e8ee"></p>
<table><thead><tr><th>dataset</th>"""
        + model_headers
        + "</tr></thead><tbody>"
        + "".join(rows)
        + "</tbody></table></body></html>"
    )


def main():
    args = parse_args()
    datasets = (
        [path.expanduser().resolve() for path in args.datasets]
        if args.datasets
        else manifest_datasets(args.manifest.expanduser().resolve())
    )
    models = [args.model] if args.model else (args.models or DEFAULT_MODELS)
    output_root = args.output_root.expanduser().resolve()
    single_model = args.model is not None
    by_model = {}
    missing = []
    for model in models:
        model_root = output_root if single_model else output_root / model
        summaries = []
        reports = {}
        for index, dataset_path in enumerate(datasets, 1):
            dataset = load_dataset(dataset_path)
            try:
                score, run_dir, _config = select_best(
                    dataset.path.stem, model, args.selection_metric
                )
            except RuntimeError as error:
                missing.append(
                    {"model": model, "dataset": dataset.path.stem, "error": str(error)}
                )
                print(f"[{model} {index}/{len(datasets)}] pending {dataset.path.stem}")
                if args.require_complete:
                    raise
                continue
            print(
                f"[{model} {index}/{len(datasets)}] {dataset.path.stem}: "
                f"{run_dir.name} {args.selection_metric}={score:.6g}"
            )
            summary = create_gnn_report(
                dataset,
                run_dir,
                model_root / dataset.path.stem,
                args.selection_metric,
                score,
            )
            summaries.append(summary)
            reports[dataset.path.stem] = summary
        write_json(
            model_root / "best_gnn_manifest.json",
            {
                "model": model,
                "selection_metric": args.selection_metric,
                "n_datasets": len(summaries),
                "reports": summaries,
            },
        )
        write_model_index(model_root, model, summaries)
        by_model[model] = reports
    if not single_model:
        write_comparison_index(output_root, datasets, by_model, missing)


if __name__ == "__main__":
    main()
