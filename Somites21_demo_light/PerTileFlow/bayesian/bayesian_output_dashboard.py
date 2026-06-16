"""Build an interactive dashboard from Bayesian tile-distensibility outputs.

This script is intentionally separate from ``bayesian_tile_distensibility.py``.
It reads an existing output directory containing:

  * tile_D_posterior_curves.csv
  * tile_D_posterior_summary.csv

and writes a standalone HTML file with:

  * all-tile marginal likelihood profiles,
  * horizontal likelihood-reference lines at 1 and 3.84,
  * highlighted selected-tile likelihood profile,
  * selected-tile prior and posterior densities over log(D0),
  * a compact per-tile summary table.
"""
from __future__ import annotations

import argparse
import csv
import html
import json
import math
from pathlib import Path


HTML_TEMPLATE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width, initial-scale=1" />
<title>Bayesian Tile Distensibility Dashboard</title>
<style>
:root {
  --bg: #f6f7f9;
  --panel: #ffffff;
  --ink: #17212b;
  --muted: #637282;
  --line: #d6e0e8;
  --blue: #2868b7;
  --orange: #c27a22;
  --red: #c74e45;
  --green: #2c8a68;
}
* { box-sizing: border-box; }
body {
  margin: 0;
  background: var(--bg);
  color: var(--ink);
  font: 14px/1.4 -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
}
header {
  background: #202b36;
  color: #fff;
  padding: 14px 18px;
  display: flex;
  justify-content: space-between;
  gap: 16px;
  align-items: center;
}
h1 { font-size: 18px; margin: 0; }
main { padding: 14px; display: grid; gap: 14px; }
section {
  background: var(--panel);
  border: 1px solid var(--line);
  border-radius: 8px;
  padding: 12px;
  min-width: 0;
}
h2 { font-size: 15px; margin: 0 0 10px; }
.controls {
  display: flex;
  flex-wrap: wrap;
  gap: 10px 14px;
  align-items: end;
}
.control { display: grid; gap: 4px; }
.control label { color: var(--muted); font-size: 12px; }
select, input {
  border: 1px solid #bdc9d5;
  border-radius: 6px;
  background: #fff;
  padding: 6px 8px;
  min-width: 150px;
}
.grid2 {
  display: grid;
  grid-template-columns: minmax(0, 1.1fr) minmax(0, .9fr);
  gap: 14px;
}
.cards {
  display: grid;
  grid-template-columns: repeat(6, minmax(120px, 1fr));
  gap: 10px;
}
.card {
  border: 1px solid var(--line);
  border-radius: 8px;
  padding: 9px;
  background: #fbfcfd;
}
.label { color: var(--muted); font-size: 12px; }
.value { font-size: 18px; font-weight: 650; margin-top: 2px; }
canvas {
  width: 100%;
  height: 420px;
  border: 1px solid var(--line);
  border-radius: 6px;
  background: #fff;
}
.tableWrap {
  max-height: 360px;
  overflow: auto;
  border: 1px solid var(--line);
  border-radius: 6px;
}
table { width: 100%; border-collapse: collapse; font-size: 12px; }
th, td { padding: 6px 8px; border-bottom: 1px solid #edf1f5; text-align: right; white-space: nowrap; }
th { position: sticky; top: 0; background: #f8fafc; color: var(--muted); font-weight: 600; }
th:first-child, td:first-child { text-align: left; }
.note { color: var(--muted); font-size: 12px; margin-top: 8px; }
@media (max-width: 1000px) {
  .grid2, .cards { grid-template-columns: 1fr; }
}
</style>
</head>
<body>
<header>
  <h1>Bayesian Tile Distensibility Dashboard</h1>
  <div id="meta"></div>
</header>
<main>
  <section>
    <h2>Controls</h2>
    <div class="controls">
      <div class="control">
        <label for="tileSelect">tile</label>
        <select id="tileSelect"></select>
      </div>
      <div class="control">
        <label for="yMaxInput">likelihood y max</label>
        <input id="yMaxInput" type="number" min="1" step="1" value="20" />
      </div>
      <div class="control">
        <label for="identifiedSelect">table filter</label>
        <select id="identifiedSelect">
          <option value="all">all</option>
          <option value="true">identified</option>
          <option value="false">not identified</option>
        </select>
      </div>
    </div>
    <div class="note">Likelihood curves use -2 delta log marginal likelihood. Lower is better; selected tile is highlighted.</div>
  </section>

  <section>
    <h2>Selected Tile Summary</h2>
    <div class="cards" id="cards"></div>
  </section>

  <div class="grid2">
    <section>
      <h2>All-Tile Marginal Likelihood Profiles</h2>
      <canvas id="likeCanvas"></canvas>
      <div class="note">Reference lines: 1.0 and 3.84. These are likelihood-profile diagnostics, not exact Bayesian credible intervals.</div>
    </section>
    <section>
      <h2>Selected Prior and Posterior</h2>
      <canvas id="postCanvas"></canvas>
      <div class="note">Densities are over log(D0), so posterior mass is comparable across log-spaced D values.</div>
    </section>
  </div>

  <section>
    <h2>Tile Summary Table</h2>
    <div class="tableWrap"><table id="summaryTable"></table></div>
  </section>
</main>
<script id="payload" type="application/json">__DATA__</script>
<script>
const data = JSON.parse(document.getElementById("payload").textContent);

function byId(id) { return document.getElementById(id); }
function num(v, digits=3) {
  const x = Number(v);
  if (!Number.isFinite(x)) return "n/a";
  if (Math.abs(x) >= 1000 || (Math.abs(x) > 0 && Math.abs(x) < 0.001)) return x.toExponential(2);
  return x.toFixed(digits).replace(/\.?0+$/, "");
}
function ctxOf(canvas) {
  const dpr = window.devicePixelRatio || 1;
  const r = canvas.getBoundingClientRect();
  canvas.width = Math.max(420, r.width * dpr);
  canvas.height = Math.max(300, r.height * dpr);
  const ctx = canvas.getContext("2d");
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  return {ctx, w: canvas.width / dpr, h: canvas.height / dpr};
}
function groupBy(rows, key) {
  const m = new Map();
  for (const r of rows) {
    const k = String(r[key]);
    if (!m.has(k)) m.set(k, []);
    m.get(k).push(r);
  }
  return m;
}
const curvesByTile = groupBy(data.curves, "tile_id");
const summaryByTile = new Map(data.summary.map(r => [String(r.tile_id), r]));
const tileIds = [...curvesByTile.keys()].sort((a,b) => Number(a) - Number(b));

function logPriorDensity(D) {
  const x = Math.log(Number(D));
  const mu = Math.log(Number(data.meta.logD_prior_median || 1.5e-3));
  const tau = Number(data.meta.logD_prior_tau || Math.log(10));
  return Math.exp(-0.5 * Math.pow((x - mu) / tau, 2)) / (tau * Math.sqrt(2 * Math.PI));
}
function sortedRows(tile) {
  return (curvesByTile.get(String(tile)) || []).slice().sort((a,b) => Number(a.D0) - Number(b.D0));
}
function allDRange() {
  const vals = data.curves.map(r => Number(r.D0)).filter(v => Number.isFinite(v) && v > 0);
  return [Math.min(...vals), Math.max(...vals)];
}
function axes(ctx,w,h,xlab,ylab) {
  ctx.clearRect(0,0,w,h);
  ctx.fillStyle = "white";
  ctx.fillRect(0,0,w,h);
  ctx.strokeStyle = "#d6e0e8";
  ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.moveTo(58, 18);
  ctx.lineTo(58, h - 44);
  ctx.lineTo(w - 18, h - 44);
  ctx.stroke();
  ctx.fillStyle = "#637282";
  ctx.font = "12px sans-serif";
  ctx.textAlign = "center";
  ctx.fillText(xlab, w / 2, h - 13);
  ctx.save();
  ctx.translate(17, h / 2);
  ctx.rotate(-Math.PI / 2);
  ctx.fillText(ylab, 0, 0);
  ctx.restore();
}
function drawXTicks(ctx,w,h,xmin,xmax,sx) {
  const minExp = Math.ceil(Math.log10(xmin));
  const maxExp = Math.floor(Math.log10(xmax));
  ctx.fillStyle = "#637282";
  ctx.strokeStyle = "#9aa8b5";
  ctx.font = "11px sans-serif";
  ctx.textAlign = "center";
  ctx.textBaseline = "top";
  for (let e = minExp; e <= maxExp; e++) {
    const xval = Math.pow(10, e);
    const x = sx(xval);
    ctx.beginPath();
    ctx.moveTo(x, h - 44);
    ctx.lineTo(x, h - 38);
    ctx.stroke();
    ctx.fillText(`1e${e}`, x, h - 34);
  }
}
function drawYTicks(ctx,w,h,ymin,ymax,sy) {
  ctx.fillStyle = "#637282";
  ctx.strokeStyle = "#9aa8b5";
  ctx.font = "11px sans-serif";
  ctx.textAlign = "right";
  ctx.textBaseline = "middle";
  const n = 5;
  for (let i = 0; i <= n; i++) {
    const yv = ymin + (ymax - ymin) * i / n;
    const y = sy(yv);
    ctx.beginPath();
    ctx.moveTo(52, y);
    ctx.lineTo(58, y);
    ctx.stroke();
    ctx.fillText(num(yv, 1), 48, y);
  }
}
function renderCards(tile) {
  const s = summaryByTile.get(String(tile)) || {};
  const items = [
    ["tile", tile],
    ["D mode", s.D_mode],
    ["D median", s.D_median],
    ["95% CI", `${num(s.D_ci95_low)} - ${num(s.D_ci95_high)}`],
    ["W95 decades", s.W95_decades],
    ["identified", String(s.identified)],
    ["edges", s.n_observed_edges],
    ["boundary", s.n_boundary_nodes],
    ["lower mass", s.posterior_mass_at_lower_boundary],
    ["upper mass", s.posterior_mass_at_upper_boundary],
  ];
  byId("cards").innerHTML = items.map(([k,v]) => (
    `<div class="card"><div class="label">${k}</div><div class="value">${typeof v === "number" ? num(v) : v}</div></div>`
  )).join("");
}
function renderLikelihood() {
  const tile = byId("tileSelect").value;
  const yMax = Math.max(1, Number(byId("yMaxInput").value) || 20);
  const {ctx,w,h} = ctxOf(byId("likeCanvas"));
  axes(ctx,w,h,"D0 (1/Pa, log scale)","-2 delta log marginal likelihood");
  const [xmin, xmax] = allDRange();
  const lx0 = Math.log10(xmin), lx1 = Math.log10(xmax);
  const sx = x => 58 + (Math.log10(x) - lx0) / ((lx1 - lx0) || 1) * (w - 82);
  const sy = y => h - 44 - (Math.min(Math.max(y, 0), yMax) / yMax) * (h - 68);
  drawXTicks(ctx,w,h,xmin,xmax,sx);
  drawYTicks(ctx,w,h,0,yMax,sy);

  for (const ref of [1, 3.84]) {
    if (ref > yMax) continue;
    ctx.strokeStyle = ref === 1 ? "rgba(44,138,104,.7)" : "rgba(199,78,69,.7)";
    ctx.setLineDash([5, 4]);
    ctx.beginPath();
    ctx.moveTo(58, sy(ref));
    ctx.lineTo(w - 18, sy(ref));
    ctx.stroke();
    ctx.setLineDash([]);
    ctx.fillStyle = ref === 1 ? "#2c8a68" : "#c74e45";
    ctx.fillText(String(ref), w - 22, sy(ref) - 7);
  }

  for (const id of tileIds) {
    const rows = sortedRows(id);
    if (!rows.length) continue;
    const maxLL = Math.max(...rows.map(r => Number(r.log_likelihood_marginal)).filter(Number.isFinite));
    ctx.strokeStyle = id === String(tile) ? "#2868b7" : "rgba(101,116,135,.20)";
    ctx.lineWidth = id === String(tile) ? 2.6 : 1.0;
    ctx.beginPath();
    rows.forEach((r, i) => {
      const yv = -2 * (Number(r.log_likelihood_marginal) - maxLL);
      const x = sx(Number(r.D0));
      const y = sy(yv);
      if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
    });
    ctx.stroke();
  }
}
function renderPosterior() {
  const tile = byId("tileSelect").value;
  const rows = sortedRows(tile);
  const {ctx,w,h} = ctxOf(byId("postCanvas"));
  axes(ctx,w,h,"D0 (1/Pa, log scale)","density over log(D0)");
  if (!rows.length) return;
  const xs = rows.map(r => Number(r.D0)).filter(v => Number.isFinite(v) && v > 0);
  const xmin = Math.min(...xs), xmax = Math.max(...xs);
  const lx0 = Math.log10(xmin), lx1 = Math.log10(xmax);
  const sx = x => 58 + (Math.log10(x) - lx0) / ((lx1 - lx0) || 1) * (w - 82);
  const logDs = rows.map(r => Math.log(Number(r.D0)));
  const dx = logDs.length > 1 ? Math.abs(logDs[1] - logDs[0]) : 1;
  const post = rows.map(r => Number(r.posterior_prob) / dx);
  const priorRaw = rows.map(r => logPriorDensity(Number(r.D0)));
  const priorMass = priorRaw.reduce((acc, v) => acc + v * dx, 0) || 1;
  const prior = priorRaw.map(v => v / priorMass);
  const ymax = Math.max(...post.filter(Number.isFinite), ...prior.filter(Number.isFinite), 1e-12);
  const sy = y => h - 44 - y / ymax * (h - 68);
  drawXTicks(ctx,w,h,xmin,xmax,sx);
  drawYTicks(ctx,w,h,0,ymax,sy);

  function line(vals, color, width, dash=[]) {
    ctx.strokeStyle = color;
    ctx.lineWidth = width;
    ctx.setLineDash(dash);
    ctx.beginPath();
    rows.forEach((r, i) => {
      const x = sx(Number(r.D0));
      const y = sy(vals[i]);
      if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
    });
    ctx.stroke();
    ctx.setLineDash([]);
  }
  line(prior, "#637282", 1.8, [5, 4]);
  line(post, "#2868b7", 2.6);
  ctx.fillStyle = "#637282";
  ctx.textAlign = "left";
  ctx.fillText("prior", 68, 30);
  ctx.fillStyle = "#2868b7";
  ctx.fillText("posterior", 68, 48);
}
function renderTable() {
  const filter = byId("identifiedSelect").value;
  let rows = data.summary.slice().sort((a,b) => Number(a.tile_id) - Number(b.tile_id));
  if (filter !== "all") {
    rows = rows.filter(r => String(r.identified).toLowerCase() === filter);
  }
  const cols = [
    ["tile_id", "tile"],
    ["D_mode", "D mode"],
    ["D_median", "D median"],
    ["D_ci95_low", "CI low"],
    ["D_ci95_high", "CI high"],
    ["W95_decades", "W95"],
    ["identified", "identified"],
    ["n_observed_edges", "edges"],
    ["posterior_mass_at_lower_boundary", "lower mass"],
    ["posterior_mass_at_upper_boundary", "upper mass"],
  ];
  const head = `<tr>${cols.map(([,l]) => `<th>${l}</th>`).join("")}</tr>`;
  const body = rows.map(r => `<tr data-tile="${r.tile_id}">${
    cols.map(([k]) => `<td>${typeof r[k] === "number" ? num(r[k]) : r[k]}</td>`).join("")
  }</tr>`).join("");
  byId("summaryTable").innerHTML = head + body;
  for (const tr of byId("summaryTable").querySelectorAll("tr[data-tile]")) {
    tr.addEventListener("click", () => {
      byId("tileSelect").value = tr.getAttribute("data-tile");
      renderAll();
    });
  }
}
function renderAll() {
  const tile = byId("tileSelect").value || tileIds[0];
  byId("tileSelect").value = tile;
  renderCards(tile);
  renderLikelihood();
  renderPosterior();
  renderTable();
}
function init() {
  byId("meta").textContent = `${data.summary.length} tiles; ${data.meta.run_dir}`;
  byId("tileSelect").innerHTML = tileIds.map(t => `<option value="${t}">${t}</option>`).join("");
  byId("tileSelect").value = tileIds[0];
  for (const id of ["tileSelect", "yMaxInput", "identifiedSelect"]) {
    byId(id).addEventListener("change", renderAll);
    byId(id).addEventListener("input", renderAll);
  }
  renderAll();
}
init();
</script>
</body>
</html>
"""


def _json_clean(value):
    if isinstance(value, dict):
        return {str(k): _json_clean(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_json_clean(v) for v in value]
    if hasattr(value, "item"):
        try:
            return _json_clean(value.item())
        except Exception:
            pass
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    return value


def _coerce_csv_value(value: str):
    if value == "":
        return ""
    low = str(value).strip().lower()
    if low == "true":
        return True
    if low == "false":
        return False
    try:
        number = float(value)
    except (TypeError, ValueError):
        return value
    if math.isfinite(number):
        return number
    return None


def _read_csv_rows(path: Path) -> list[dict]:
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        return [
            {key: _coerce_csv_value(value) for key, value in row.items()}
            for row in reader
        ]


def build_dashboard(run_dir: Path, out_path: Path) -> None:
    curves_path = run_dir / "tile_D_posterior_curves.csv"
    summary_path = run_dir / "tile_D_posterior_summary.csv"
    manifest_path = run_dir / "run_manifest.json"
    if not curves_path.exists():
        raise SystemExit(f"Missing {curves_path}")
    if not summary_path.exists():
        raise SystemExit(f"Missing {summary_path}")

    curves = _read_csv_rows(curves_path)
    summary = _read_csv_rows(summary_path)
    manifest = {}
    if manifest_path.exists():
        with open(manifest_path) as f:
            manifest = json.load(f)

    args = manifest.get("args", {}) if isinstance(manifest, dict) else {}
    payload = {
        "curves": curves,
        "summary": summary,
        "meta": {
            "run_dir": str(run_dir),
            "logD_prior_median": args.get("logD_prior_median", 1.5e-3),
            "logD_prior_tau": args.get("logD_prior_tau", math.log(10.0)),
        },
    }
    data_text = html.escape(json.dumps(_json_clean(payload), separators=(",", ":")),
                            quote=False)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(HTML_TEMPLATE.replace("__DATA__", data_text))


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Create an interactive HTML dashboard from Bayesian "
                    "tile-distensibility CSV outputs.")
    ap.add_argument("--run-dir", required=True,
                    help="Directory containing tile_D_posterior_curves.csv.")
    ap.add_argument("--out", default=None,
                    help="Output HTML path. Defaults to "
                         "<run-dir>/bayesian_tile_dashboard.html.")
    args = ap.parse_args()

    run_dir = Path(args.run_dir).expanduser().resolve()
    out_path = (Path(args.out).expanduser().resolve()
                if args.out else run_dir / "bayesian_tile_dashboard.html")
    build_dashboard(run_dir, out_path)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
