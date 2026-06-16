"""Create a static dashboard for a gnn_edge DC sweep directory.

Example:
    python gnn_edge/gnn_edge_dashboard.py \
      --results-dir renders/gnn_edge_dc_YYYYMMDD_HHMMSS
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


PROJECT_ROOT = Path(__file__).resolve().parents[1]

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


def load_csv(path: Path) -> List[dict]:
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    for row in rows:
        for col in NUMERIC_COLUMNS:
            if col in row:
                row[col] = safe_float(row[col])
        if "out_dir" in row:
            row["run"] = Path(str(row["out_dir"])).name
        elif "run" not in row:
            row["run"] = ""
    return rows


def resolve_results_dir(path_str: str | None) -> Path:
    if path_str:
        return Path(path_str).expanduser().resolve()
    candidates = sorted(
        p for p in (PROJECT_ROOT / "renders").glob("gnn_edge_dc_*")
        if (p / "sweep_summary.csv").exists()
    )
    if not candidates:
        raise SystemExit(
            "No gnn_edge sweep directory found. Pass --results-dir or run the sweep first."
        )
    return candidates[-1].resolve()


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


def json_safe(obj):
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    if isinstance(obj, dict):
        return {key: json_safe(value) for key, value in obj.items()}
    if isinstance(obj, list):
        return [json_safe(value) for value in obj]
    return obj


def build_html(results_dir: Path, rows: Sequence[dict], outlier: str,
               stats_rows: Sequence[dict], group_stats: Sequence[dict],
               output_path: Path) -> None:
    rel_gallery = {
        row["run"]: [img for img in GALLERY_IMAGES if image_exists(results_dir, row["run"], img)]
        for row in rows
    }
    payload = json_safe({
        "rows": rows,
        "outlier": outlier,
        "stats": list(stats_rows),
        "groups": list(group_stats),
        "gallery": rel_gallery,
    })
    rows_no_outlier = [r for r in rows if outlier and r["run"] != outlier]
    best_no = best_summary(rows_no_outlier) if outlier else best_summary(rows)
    best_all = best_summary(rows)
    outlier_label = html.escape(outlier) if outlier else "none"
    outlier_checked = " checked" if outlier else ""
    doc = f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>GNN Edge DC Dashboard</title>
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
  <h1>GNN Edge DC Sweep Dashboard</h1>
  <p class="muted">Results: <code>{html.escape(str(results_dir))}</code></p>
  <div class="controls">
    <label>Selected run<br><select id="runSelect"></select></label>
    <label><input id="excludeOutlier" type="checkbox"{outlier_checked}> Exclude outlier <code>{outlier_label}</code></label>
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
  <script id="payload" type="application/json">{json.dumps(payload, allow_nan=False)}</script>
  <script>
    const data = JSON.parse(document.getElementById('payload').textContent);
    const fmt = x => x !== null && Number.isFinite(+x) ? (+x).toFixed(4) : 'nan';
    const runSelect = document.getElementById('runSelect');
    const exclude = document.getElementById('excludeOutlier');
    const sortKey = document.getElementById('sortKey');
    function activeRows() {{
      let rows = data.rows.slice();
      if (exclude.checked && data.outlier) rows = rows.filter(r => r.run !== data.outlier);
      const key = sortKey.value;
      rows.sort((a,b) => (+a[key]) - (+b[key]));
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
      const rows = data.stats.filter(r => exclude.checked && data.outlier ? r.outlier_mode === 'excluded' : r.outlier_mode === 'included');
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
      if (!row) {{
        document.getElementById('selected').innerHTML = '<p class="muted">No run selected.</p>';
        return;
      }}
      const imgs = data.gallery[run] || [];
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
    ap.add_argument("--results-dir", default=None)
    ap.add_argument("--outlier", default="")
    ap.add_argument("--n-permutations", type=int, default=10000)
    ap.add_argument("--output", default="dashboard.html")
    args = ap.parse_args(argv)

    results_dir = resolve_results_dir(args.results_dir)
    rows = load_csv(results_dir / "sweep_summary.csv")
    if not rows:
        raise SystemExit(f"No rows found in {results_dir / 'sweep_summary.csv'}")

    stats_rows = []
    group_stats = []
    rows_excluding_outlier = [r for r in rows if args.outlier and r["run"] != args.outlier]
    if not args.outlier:
        rows_excluding_outlier = rows
    for mode, active in (
        ("included", rows),
        ("excluded", rows_excluding_outlier),
    ):
        for factor in ("K", "hidden_dim", "lambda_delta"):
            stat = permutation_pvalue(active, factor, args.n_permutations)
            stat["outlier_mode"] = mode
            stats_rows.append(stat)
            for row in group_rows(active, factor):
                row["outlier_mode"] = mode
                group_stats.append(row)

    write_csv(results_dir / "dashboard_factor_stats.csv", stats_rows)
    write_csv(results_dir / "dashboard_group_stats.csv", group_stats)
    build_html(results_dir, rows, args.outlier, stats_rows, group_stats,
               results_dir / args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
