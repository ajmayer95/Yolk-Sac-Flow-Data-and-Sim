"""Create a static dashboard for synthetic gnn_edge DC outputs.

Example:
    python gnn_edge_dashboard.py

By default this local copy reads ``gnn_edge_dc`` in the same folder and
selects the validated run, ``masked_edge_validation_15pct``.
"""
from __future__ import annotations

import argparse
import csv
import html
import json
import math
import random
import statistics
from pathlib import Path
from typing import Iterable, List, Sequence


SCRIPT_DIR = Path(__file__).resolve().parent
RESULTS_DIR = SCRIPT_DIR / "d0p001_results"
DEFAULT_RESULTS_DIR = RESULTS_DIR / "gnn_edge_dc"
DEFAULT_PRIMARY_RUN = "masked_edge_validation_15pct"

NUMERIC_COLUMNS = {
    "K", "hidden_dim", "lambda_delta", "seed",
    "train_loss_final", "val_loss_final",
    "train_RMSE", "val_RMSE", "train_NRMSE", "val_NRMSE",
    "train_MAE", "val_MAE", "train_Pearson", "val_Pearson",
    "mass_residual_norm", "mean_abs_delta", "median_abs_delta",
    "mean_C", "median_C", "p95_C", "max_C",
    "pressure_min", "pressure_median", "pressure_max", "pressure_range",
    "number_edges", "number_nodes",
}

GALLERY_IMAGES = [
    "validation_loss.png",
    "training_loss.png",
    "physics_gnn_q_pred_vs_obs.png",
    "residual_vs_Q_obs.png",
    "abs_error_vs_abs_Q_obs.png",
    "signed_error_map.png",
    "absolute_error_map.png",
    "delta_hist.png",
    "C_hist_logx.png",
    "delta_vs_log_radius.png",
    "delta_vs_log_G_pois.png",
    "delta_vs_distance_to_A.png",
    "C_map.png",
    "delta_map.png",
    "physics_gnn_pressure_spatial.png",
    "conservation_residual_spatial.png",
    "regression_model_d_delta_pred_vs_obs.png",
    "regression_model_d_standardized_coefficients.png",
]


def safe_float(value, default=float("nan")) -> float:
    try:
        x = float(value)
    except (TypeError, ValueError):
        return default
    return x if math.isfinite(x) else default


def clean_json_value(value):
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {str(k): clean_json_value(v) for k, v in value.items()}
    if isinstance(value, list):
        return [clean_json_value(v) for v in value]
    return value


def coerce_numeric(row: dict) -> dict:
    for col in NUMERIC_COLUMNS:
        if col in row:
            row[col] = safe_float(row[col])
    if "out_dir" in row:
        row["run"] = Path(str(row["out_dir"])).name
    elif "run" not in row:
        row["run"] = ""
    return row


def load_csv(path: Path) -> List[dict]:
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    for row in rows:
        coerce_numeric(row)
    return rows


def load_rows(results_dir: Path) -> tuple[List[dict], str]:
    sweep_path = results_dir / "sweep_summary.csv"
    if sweep_path.exists():
        return load_csv(sweep_path), "sweep_summary"

    run_dirs = []
    if (results_dir / "run_summary.csv").exists():
        run_dirs.append(results_dir)
    run_dirs.extend(
        p for p in sorted(results_dir.iterdir())
        if p.is_dir() and (p / "run_summary.csv").exists()
    )
    rows: List[dict] = []
    for run_dir in run_dirs:
        run_rows = load_csv(run_dir / "run_summary.csv")
        if not run_rows:
            continue
        row = run_rows[0]
        row["run"] = run_dir.name
        row["out_dir"] = str(run_dir)
        row["source"] = "run_summary"
        rows.append(coerce_numeric(row))
    return rows, "run_summary"


def write_csv(path: Path, rows: Sequence[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def mean(xs: Sequence[float]) -> float:
    vals = [x for x in xs if math.isfinite(x)]
    return float(statistics.mean(vals)) if vals else float("nan")


def median(xs: Sequence[float]) -> float:
    vals = [x for x in xs if math.isfinite(x)]
    return float(statistics.median(vals)) if vals else float("nan")


def group_rows(rows: Sequence[dict], factor: str) -> List[dict]:
    out = []
    for level in sorted({row[factor] for row in rows}):
        vals = [float(row["val_NRMSE"]) for row in rows if row[factor] == level]
        vals = [v for v in vals if math.isfinite(v)]
        if not vals:
            continue
        out.append({
            "factor": factor,
            "level": level,
            "n": len(vals),
            "mean_val_NRMSE": mean(vals),
            "median_val_NRMSE": median(vals),
            "min_val_NRMSE": min(vals),
            "max_val_NRMSE": max(vals),
        })
    return out


def factor_statistic(rows: Sequence[dict], factor: str):
    y = [float(row["val_NRMSE"]) for row in rows]
    levels = sorted({row[factor] for row in rows})
    groups = [[i for i, row in enumerate(rows) if row[factor] == level] for level in levels]
    n = len(y)

    def stat(vals: Sequence[float]):
        grand = sum(vals) / len(vals)
        ss_between = 0.0
        ss_within = 0.0
        for group in groups:
            mu = sum(vals[i] for i in group) / len(group)
            ss_between += len(group) * (mu - grand) ** 2
            ss_within += sum((vals[i] - mu) ** 2 for i in group)
        df_between = len(groups) - 1
        df_within = n - len(groups)
        f_val = (
            (ss_between / df_between) / (ss_within / df_within)
            if df_between > 0 and df_within > 0 and ss_within > 0
            else float("inf")
        )
        eta2 = ss_between / (ss_between + ss_within) if (ss_between + ss_within) > 0 else float("nan")
        return f_val, eta2

    return y, groups, stat


def permutation_pvalue(rows: Sequence[dict], factor: str,
                       n_perm: int, seed: int = 0) -> dict:
    y, _groups, stat = factor_statistic(rows, factor)
    observed_f, eta2 = stat(y)
    rng = random.Random(seed)
    shuffled = list(y)
    ge = 0
    for _ in range(int(n_perm)):
        rng.shuffle(shuffled)
        f_perm, _ = stat(shuffled)
        if f_perm >= observed_f - 1e-15:
            ge += 1
    return {
        "factor": factor,
        "n": len(rows),
        "F_observed": observed_f,
        "eta_squared": eta2,
        "permutation_p": (ge + 1) / (int(n_perm) + 1),
        "n_permutations": int(n_perm),
    }


def can_compute_factor(rows: Sequence[dict], factor: str) -> bool:
    vals = [safe_float(r.get("val_NRMSE")) for r in rows]
    finite_rows = [r for r, v in zip(rows, vals) if math.isfinite(v)]
    levels = {r.get(factor) for r in finite_rows}
    return len(finite_rows) > len(levels) and len(levels) > 1


def image_exists(results_dir: Path, run: str, image: str) -> bool:
    return (results_dir / run / image).exists()


def best_summary(rows: Sequence[dict]) -> dict:
    finite = [r for r in rows if math.isfinite(float(r.get("val_NRMSE", float("nan"))))]
    if not finite:
        return {}
    best = min(finite, key=lambda r: float(r["val_NRMSE"]))
    worst = max(finite, key=lambda r: float(r["val_NRMSE"]))
    return {
        "n": len(finite),
        "mean_val_NRMSE": mean([r["val_NRMSE"] for r in finite]),
        "median_val_NRMSE": median([r["val_NRMSE"] for r in finite]),
        "best_run": best["run"],
        "best_val_NRMSE": best["val_NRMSE"],
        "worst_run": worst["run"],
        "worst_val_NRMSE": worst["val_NRMSE"],
    }


def build_html(results_dir: Path, rows: Sequence[dict], outlier: str,
               stats_rows: Sequence[dict], group_stats: Sequence[dict],
               output_path: Path, primary_run: str, source_kind: str) -> None:
    rel_gallery = {
        row["run"]: [img for img in GALLERY_IMAGES if image_exists(results_dir, row["run"], img)]
        for row in rows
    }
    payload = {
        "rows": rows,
        "outlier": outlier,
        "primaryRun": primary_run,
        "sourceKind": source_kind,
        "stats": list(stats_rows),
        "groups": list(group_stats),
        "gallery": rel_gallery,
    }
    best_no = best_summary([r for r in rows if r["run"] != outlier])
    best_all = best_summary(rows)
    doc = f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>Synthetic GNN Edge DC Dashboard</title>
  <style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, sans-serif; margin: 24px; color: #1f2933; }}
    .controls, .cards, .grid {{ display: grid; gap: 12px; }}
    .controls {{ grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); align-items: end; }}
    .cards {{ grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); margin: 18px 0; }}
    .card {{ border: 1px solid #d8dee4; border-radius: 6px; padding: 12px; background: #fbfbfc; }}
    .card b {{ display: block; font-size: 12px; color: #667085; margin-bottom: 6px; }}
    table {{ border-collapse: collapse; width: 100%; margin: 14px 0; font-size: 13px; }}
    th, td {{ border-bottom: 1px solid #e5e7eb; padding: 6px 8px; text-align: right; }}
    th:first-child, td:first-child {{ text-align: left; }}
    img {{ max-width: 100%; border: 1px solid #e5e7eb; border-radius: 6px; background: white; }}
    .gallery {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(360px, 1fr)); gap: 14px; }}
    .muted {{ color: #667085; }}
    select, input {{ padding: 6px; }}
    h1, h2 {{ margin-bottom: 6px; }}
    code {{ background: #f2f4f7; padding: 2px 4px; border-radius: 4px; }}
  </style>
</head>
<body>
  <h1>Synthetic GNN Edge DC Dashboard</h1>
  <p class="muted">Results: <code>{html.escape(str(results_dir))}</code></p>
  <p class="muted">Source: <code>{html.escape(source_kind)}</code>; primary validated run: <code>{html.escape(primary_run)}</code></p>
  <div class="controls">
    <label>Selected run<br><select id="runSelect"></select></label>
    <label><input id="excludeOutlier" type="checkbox" {"checked" if outlier else ""} {"disabled" if not outlier else ""}> Exclude outlier <code>{html.escape(outlier or "none")}</code></label>
    <label>Sort runs by<br>
      <select id="sortKey">
        <option value="val_NRMSE">val_NRMSE</option>
        <option value="val_RMSE">val_RMSE</option>
        <option value="val_Pearson">val_Pearson</option>
        <option value="mean_abs_delta">mean_abs_delta</option>
        <option value="median_C">median_C</option>
      </select>
    </label>
  </div>
  <div class="cards" id="cards"></div>
  <h2>Factor Statistics</h2>
  <div id="stats"></div>
  <h2>Runs</h2>
  <div id="runs"></div>
  <h2>Selected Run Gallery</h2>
  <div id="selected"></div>
  <script id="payload" type="application/json">{json.dumps(clean_json_value(payload), allow_nan=False)}</script>
  <script>
    const data = JSON.parse(document.getElementById('payload').textContent);
    const fmt = x => Number.isFinite(+x) ? (+x).toFixed(4) : 'nan';
    const runSelect = document.getElementById('runSelect');
    const exclude = document.getElementById('excludeOutlier');
    const sortKey = document.getElementById('sortKey');
    function activeRows() {{
      let rows = data.rows.slice();
      if (data.outlier && exclude.checked) rows = rows.filter(r => r.run !== data.outlier);
      const key = sortKey.value;
      rows.sort((a,b) => {{
        const av = Number.isFinite(+a[key]) ? +a[key] : Number.POSITIVE_INFINITY;
        const bv = Number.isFinite(+b[key]) ? +b[key] : Number.POSITIVE_INFINITY;
        return av - bv;
      }});
      return rows;
    }}
    function renderSelect() {{
      const previous = runSelect.value;
      runSelect.innerHTML = '';
      for (const r of activeRows()) {{
        const opt = document.createElement('option');
        opt.value = r.run;
        opt.textContent = `${{r.run}}  val_NRMSE=${{fmt(r.val_NRMSE)}}`;
        runSelect.appendChild(opt);
      }}
      if ([...runSelect.options].some(o => o.value === previous)) runSelect.value = previous;
      else if ([...runSelect.options].some(o => o.value === data.primaryRun)) runSelect.value = data.primaryRun;
    }}
    function renderCards() {{
      const rows = activeRows();
      const vals = rows.map(r => +r.val_NRMSE).filter(Number.isFinite);
      const best = rows.reduce((a,b) => (+b.val_NRMSE < +a.val_NRMSE ? b : a), rows[0]);
      const median = vals.slice().sort((a,b)=>a-b)[Math.floor(vals.length/2)];
      document.getElementById('cards').innerHTML = `
        <div class="card"><b>Runs shown</b>${{rows.length}}</div>
        <div class="card"><b>Best run</b>${{best?.run || ''}}</div>
        <div class="card"><b>Best val NRMSE</b>${{fmt(best?.val_NRMSE)}}</div>
        <div class="card"><b>Median val NRMSE</b>${{fmt(median)}}</div>
      `;
    }}
    function renderStats() {{
      const rows = data.stats.filter(r => exclude.checked ? r.outlier_mode === 'excluded' : r.outlier_mode === 'included');
      if (!rows.length) {{
        document.getElementById('stats').innerHTML = '<p class="muted">No sweep factor statistics for this run-summary layout.</p>';
        return;
      }}
      let html = '<table><tr><th>factor</th><th>F</th><th>eta²</th><th>permutation p</th><th>n</th></tr>';
      for (const r of rows) html += `<tr><td>${{r.factor}}</td><td>${{fmt(r.F_observed)}}</td><td>${{fmt(r.eta_squared)}}</td><td>${{fmt(r.permutation_p)}}</td><td>${{r.n}}</td></tr>`;
      html += '</table>';
      document.getElementById('stats').innerHTML = html;
    }}
    function renderRuns() {{
      let html = '<table><tr><th>run</th><th>K</th><th>hidden</th><th>lambda</th><th>val NRMSE</th><th>val Pearson</th><th>median C</th><th>max C</th></tr>';
      for (const r of activeRows()) {{
        html += `<tr><td style="text-align:left">${{r.run}}</td><td>${{r.K}}</td><td>${{r.hidden_dim}}</td><td>${{r.lambda_delta}}</td><td>${{fmt(r.val_NRMSE)}}</td><td>${{fmt(r.val_Pearson)}}</td><td>${{fmt(r.median_C)}}</td><td>${{fmt(r.max_C)}}</td></tr>`;
      }}
      html += '</table>';
      document.getElementById('runs').innerHTML = html;
    }}
    function renderSelected() {{
      const run = runSelect.value;
      const row = data.rows.find(r => r.run === run);
      const imgs = data.gallery[run] || [];
      if (!row) {{
        document.getElementById('selected').innerHTML = '<p class="muted">No run selected.</p>';
        return;
      }}
      let html = `<h3>${{run}}</h3><p class="muted">K=${{row.K}}, hidden=${{row.hidden_dim}}, lambda=${{row.lambda_delta}}, val_NRMSE=${{fmt(row.val_NRMSE)}}, val_Pearson=${{fmt(row.val_Pearson)}}</p>`;
      html += '<div class="gallery">';
      for (const img of imgs) html += `<figure><img src="${{run}}/${{img}}"><figcaption>${{img}}</figcaption></figure>`;
      html += '</div>';
      document.getElementById('selected').innerHTML = html;
    }}
    function render() {{ renderSelect(); renderCards(); renderStats(); renderRuns(); renderSelected(); }}
    exclude.addEventListener('change', render);
    sortKey.addEventListener('change', render);
    runSelect.addEventListener('change', renderSelected);
    render();
  </script>
</body>
</html>
"""
    output_path.write_text(doc)
    print(f"Wrote dashboard: {output_path}")
    print(f"All runs: {best_all}")
    print(f"Excluding outlier: {best_no}")


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results-dir", default=str(DEFAULT_RESULTS_DIR))
    ap.add_argument("--primary-run", default=DEFAULT_PRIMARY_RUN,
                    help="Run selected by default in the dashboard.")
    ap.add_argument("--outlier", default="",
                    help="Optional run to exclude from sweep statistics.")
    ap.add_argument("--n-permutations", type=int, default=10000)
    ap.add_argument("--output", default="dashboard.html")
    args = ap.parse_args(argv)

    results_dir = Path(args.results_dir).resolve()
    rows, source_kind = load_rows(results_dir)
    if not rows:
        raise SystemExit(
            f"No sweep_summary.csv or run_summary.csv outputs found under {results_dir}"
        )

    stats_rows = []
    group_stats = []
    for mode, active in (
        ("included", rows),
        ("excluded", [r for r in rows if args.outlier and r["run"] != args.outlier]),
    ):
        if mode == "excluded" and not args.outlier:
            continue
        for factor in ("K", "hidden_dim", "lambda_delta"):
            if not can_compute_factor(active, factor):
                continue
            stat = permutation_pvalue(active, factor, args.n_permutations)
            stat["outlier_mode"] = mode
            stats_rows.append(stat)
            for row in group_rows(active, factor):
                row["outlier_mode"] = mode
                group_stats.append(row)

    write_csv(results_dir / "dashboard_factor_stats.csv", stats_rows)
    write_csv(results_dir / "dashboard_group_stats.csv", group_stats)
    build_html(results_dir, rows, args.outlier, stats_rows, group_stats,
               results_dir / args.output, args.primary_run, source_kind)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
