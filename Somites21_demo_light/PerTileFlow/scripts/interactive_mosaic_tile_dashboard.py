"""Build an interactive HTML dashboard for mosaic/tile comparison results.

The dashboard is static after generation: all CSV/profile/flow data are
embedded into one HTML file with a tile selector, profile plot, pressure plot,
flow comparison plots, and metric tables.  It is meant for browsing the output
from ``tile_distensibility_mosaic_comparison.py`` without rerunning any fits.

Example
-------
  python scripts/interactive_mosaic_tile_dashboard.py \
    --comparison-dir renders/meeting/tile_distensibility_mosaic_comparison \
    --mosaic-dir renders/meeting/mosaic_tile_solve \
    --config ../emb1/config.json
"""
from __future__ import annotations

import argparse
import csv
import html
import json
import math
import pickle
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))


def _read_csv(path: Path) -> List[dict]:
    if not path.exists():
        return []
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def _to_number(value):
    if value is None:
        return None
    if isinstance(value, (int, float)):
        if isinstance(value, float) and not math.isfinite(value):
            return None
        return value
    text = str(value).strip()
    if text == "":
        return None
    lowered = text.lower()
    if lowered == "true":
        return True
    if lowered == "false":
        return False
    try:
        val = float(text)
    except ValueError:
        return value
    if not math.isfinite(val):
        return None
    return val


def _clean_row(row: dict) -> dict:
    return {k: _to_number(v) for k, v in row.items()}


def _tile_id_from_dir(path: Path) -> Optional[int]:
    try:
        return int(path.name.split("_", 1)[1])
    except Exception:
        return None


def _load_profiles(comparison_dir: Path) -> Dict[str, List[dict]]:
    profiles = {}
    for csv_path in sorted(comparison_dir.glob("tile_*/profiles.csv")):
        tid = _tile_id_from_dir(csv_path.parent)
        if tid is None:
            continue
        profiles[str(tid)] = [_clean_row(r) for r in _read_csv(csv_path)]
    return profiles


def _group_by_tile(rows: Iterable[dict]) -> Dict[str, List[dict]]:
    grouped: Dict[str, List[dict]] = {}
    for row in rows:
        clean = _clean_row(row)
        tid = clean.get("tile_id")
        if tid is None:
            continue
        grouped.setdefault(str(int(tid)), []).append(clean)
    return grouped


def _load_flow_rows(args, tiles: List[int]) -> Dict[str, List[dict]]:
    """Compute tile-local measured vs mosaic-flow rows when inputs exist."""
    graph_path = None
    result_path = Path(args.mosaic_result) if args.mosaic_result else (
        Path(args.mosaic_dir) / "mosaic_solve_result.pkl")
    try:
        import numpy as np
        from distensibility_ablation import _observations
        from inspect_tile import build_tile_problem
        from synthetic_validation_neumann_bc import nL_per_m3
        from tile_mosaic_simulation import (
            load_graph_from_args,
            observations_from_mosaic_result,
        )
    except Exception as e:
        print(f"Flow rows skipped: could not import analysis modules ({e})")
        return {}

    if not result_path.exists():
        print(f"Flow rows skipped: no mosaic result at {result_path}")
        return {}

    try:
        graph, graph_path = load_graph_from_args(args)
        with open(result_path, "rb") as f:
            result = pickle.load(f)
    except Exception as e:
        print(f"Flow rows skipped: could not load graph/result ({e})")
        return {}

    print(f"Computing tile flow rows from graph {graph_path}")
    grouped: Dict[str, List[dict]] = {}
    for tid in tiles:
        try:
            prob = build_tile_problem(graph, int(tid))
            measured = _observations(graph, prob, (1, 2))
            mosaic = observations_from_mosaic_result(
                prob, result, (1, 2), nL_per_m3)
        except Exception as e:
            print(f"  tile {tid}: flow rows skipped ({type(e).__name__}: {e})")
            continue

        rows = []
        for i, (u, v) in enumerate(prob["edges_in"]):
            m_dc = (measured["q_dc"][i] * nL_per_m3
                    if measured["valid"]["dc"][i] else None)
            s_dc = (mosaic["q_dc"][i] * nL_per_m3
                    if mosaic["valid"]["dc"][i] else None)
            m_h1 = (abs(measured["q_h"][1][i]) * nL_per_m3
                    if measured["valid"][1][i] else None)
            s_h1 = (abs(mosaic["q_h"][1][i]) * nL_per_m3
                    if mosaic["valid"][1][i] else None)
            m_h2 = (abs(measured["q_h"][2][i]) * nL_per_m3
                    if measured["valid"][2][i] else None)
            s_h2 = (abs(mosaic["q_h"][2][i]) * nL_per_m3
                    if mosaic["valid"][2][i] else None)
            rows.append({
                "tile_id": int(tid),
                "u": u,
                "v": v,
                "measured_Q_dc_nL_s": _finite_or_none(m_dc),
                "mosaic_Q_dc_nL_s": _finite_or_none(s_dc),
                "delta_Q_dc_nL_s": _finite_or_none(
                    None if m_dc is None or s_dc is None else m_dc - s_dc),
                "measured_amp_h1_nL_s": _finite_or_none(m_h1),
                "mosaic_amp_h1_nL_s": _finite_or_none(s_h1),
                "measured_amp_h2_nL_s": _finite_or_none(m_h2),
                "mosaic_amp_h2_nL_s": _finite_or_none(s_h2),
            })
        grouped[str(int(tid))] = rows
    return grouped


def _finite_or_none(value):
    if value is None:
        return None
    try:
        value = float(value)
    except Exception:
        return None
    return value if math.isfinite(value) else None


def _load_mosaic_d(args, summary: List[dict]):
    result_path = Path(args.mosaic_result) if args.mosaic_result else (
        Path(args.mosaic_dir) / "mosaic_solve_result.pkl")
    if result_path.exists():
        try:
            with open(result_path, "rb") as f:
                result = pickle.load(f)
            value = getattr(result, "D", None)
            if value is not None:
                return _finite_or_none(value)
        except Exception:
            pass
    for row in summary:
        if row.get("ablation") == "mosaic_flow_fixed_mosaic_pressure":
            return _finite_or_none(row.get("D_hat"))
    return None


HTML_TEMPLATE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width, initial-scale=1" />
<title>Mosaic Tile Distensibility Dashboard</title>
<style>
:root {
  --bg: #f7f8fa;
  --panel: #ffffff;
  --ink: #16202a;
  --muted: #607080;
  --line: #d9e0e7;
  --blue: #2f6fbb;
  --red: #c75146;
  --green: #2f8f6b;
  --orange: #c77b25;
}
* { box-sizing: border-box; }
body {
  margin: 0;
  font: 14px/1.4 -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
  color: var(--ink);
  background: var(--bg);
}
header {
  display: flex;
  gap: 16px;
  align-items: center;
  padding: 14px 18px;
  background: #1f2933;
  color: white;
  position: sticky;
  top: 0;
  z-index: 10;
}
h1 { font-size: 17px; margin: 0; font-weight: 650; }
select, input {
  height: 32px;
  border: 1px solid var(--line);
  border-radius: 6px;
  padding: 0 9px;
  background: white;
  color: var(--ink);
}
main {
  display: grid;
  grid-template-columns: 280px minmax(0, 1fr);
  gap: 14px;
  padding: 14px;
}
aside, section {
  background: var(--panel);
  border: 1px solid var(--line);
  border-radius: 8px;
}
aside { padding: 12px; align-self: start; position: sticky; top: 64px; }
section { padding: 12px; min-width: 0; }
.stack { display: grid; gap: 14px; }
.grid2 { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 14px; }
.label { display:block; font-size: 12px; color: var(--muted); margin: 10px 0 4px; }
.metric {
  display: grid;
  grid-template-columns: 1fr auto;
  gap: 6px;
  padding: 7px 0;
  border-bottom: 1px solid var(--line);
}
.metric:last-child { border-bottom: 0; }
.metric span:first-child { color: var(--muted); }
.metric strong { font-variant-numeric: tabular-nums; }
h2 { font-size: 15px; margin: 0 0 10px; }
canvas {
  width: 100%;
  height: 320px;
  border: 1px solid var(--line);
  border-radius: 6px;
  background: white;
}
table {
  width: 100%;
  border-collapse: collapse;
  font-size: 12px;
}
th, td {
  text-align: right;
  border-bottom: 1px solid var(--line);
  padding: 5px 6px;
  white-space: nowrap;
}
th:first-child, td:first-child,
th:nth-child(2), td:nth-child(2) { text-align: left; }
thead th {
  position: sticky;
  top: 0;
  background: #eef2f6;
  z-index: 1;
}
.scroll { max-height: 320px; overflow: auto; border: 1px solid var(--line); border-radius: 6px; }
.checks { display: grid; gap: 5px; margin-top: 8px; }
.checks label { display:flex; align-items:center; gap:7px; color: var(--muted); }
.checks input { height: auto; }
.hint { color: var(--muted); font-size: 12px; margin-top: 8px; }
@media (max-width: 980px) {
  main, .grid2 { grid-template-columns: 1fr; }
  aside { position: static; }
}
</style>
</head>
<body>
<header>
  <h1>Mosaic Tile Distensibility Dashboard</h1>
  <label>Tile <select id="tileSelect"></select></label>
  <label>Flow rows <input id="flowLimit" type="number" min="10" step="10" value="80"></label>
</header>
<main>
  <aside>
    <h2>Selected Tile</h2>
    <div id="tileMetrics"></div>
    <div class="label">Profile Lines</div>
    <div id="profileChecks" class="checks"></div>
    <div class="hint">D profiles use Delta chi2. Flow plots compare measured edge values to whole-mosaic simulated edge values.</div>
  </aside>
  <div class="stack">
    <section>
      <h2>Distensibility Profiles</h2>
      <canvas id="profileCanvas" height="320"></canvas>
    </section>
    <section>
      <h2>Summary Rows</h2>
      <div class="scroll"><table id="summaryTable"></table></div>
    </section>
    <div class="grid2">
      <section>
        <h2>Mosaic Boundary Pressures</h2>
        <canvas id="pressureCanvas" height="320"></canvas>
      </section>
      <section>
        <h2>Boundary Pressure Table</h2>
        <div class="scroll"><table id="pressureTable"></table></div>
      </section>
    </div>
    <div class="grid2">
      <section>
        <h2>Flow Comparison</h2>
        <canvas id="flowCanvas" height="320"></canvas>
      </section>
      <section>
        <h2>Tile Edge Flows</h2>
        <div class="scroll"><table id="flowTable"></table></div>
      </section>
    </div>
  </div>
</main>
<script id="dashboard-data" type="application/json">__DATA__</script>
<script>
const data = JSON.parse(document.getElementById("dashboard-data").textContent);
const colors = {
  measured_free_boundary: "#2f6fbb",
  measured_fixed_mosaic_pressure: "#c75146",
  mosaic_flow_free_boundary: "#2f8f6b",
  mosaic_flow_fixed_mosaic_pressure: "#c77b25"
};
let activeProfiles = new Set(Object.keys(colors));

function num(v, digits=3) {
  if (v === null || v === undefined || Number.isNaN(Number(v))) return "";
  const x = Number(v);
  if (x === 0) return "0";
  if (Math.abs(x) >= 1000 || Math.abs(x) < 0.01) return x.toExponential(2);
  return x.toFixed(digits).replace(/\.?0+$/, "");
}
function byId(id) { return document.getElementById(id); }
function tileIds() {
  const ids = new Set([
    ...Object.keys(data.summaryByTile),
    ...Object.keys(data.profilesByTile),
    ...Object.keys(data.pressuresByTile),
    ...Object.keys(data.flowsByTile)
  ]);
  return [...ids].map(Number).sort((a,b)=>a-b).map(String);
}
function setup() {
  const select = byId("tileSelect");
  for (const tid of tileIds()) {
    const opt = document.createElement("option");
    opt.value = tid;
    opt.textContent = tid.padStart(3, "0");
    select.appendChild(opt);
  }
  select.addEventListener("change", render);
  byId("flowLimit").addEventListener("input", render);
  setupChecks();
  render();
}
function setupChecks() {
  const box = byId("profileChecks");
  box.innerHTML = "";
  for (const name of Object.keys(colors)) {
    const label = document.createElement("label");
    const cb = document.createElement("input");
    cb.type = "checkbox";
    cb.checked = true;
    cb.addEventListener("change", () => {
      if (cb.checked) activeProfiles.add(name); else activeProfiles.delete(name);
      render();
    });
    const swatch = document.createElement("span");
    swatch.style.cssText = `display:inline-block;width:12px;height:12px;background:${colors[name]};border-radius:2px`;
    label.append(cb, swatch, document.createTextNode(name));
    box.appendChild(label);
  }
}
function render() {
  const tid = byId("tileSelect").value || tileIds()[0];
  if (!tid) return;
  renderMetrics(tid);
  renderSummary(tid);
  renderProfiles(tid);
  renderPressures(tid);
  renderFlows(tid);
}
function renderMetrics(tid) {
  const rows = data.summaryByTile[tid] || [];
  const measured = rows.find(r => r.ablation === "measured_free_boundary") || {};
  const fixed = rows.find(r => r.ablation === "measured_fixed_mosaic_pressure") || {};
  const mosaic = rows.find(r => r.ablation === "mosaic_flow_fixed_mosaic_pressure") || {};
  const html = [
    ["Measured free D", num(measured.D_hat)],
    ["Measured fixed D", num(fixed.D_hat)],
    ["Mosaic fixed D", num(mosaic.D_hat)],
    ["Mosaic D input", num(data.mosaicD)],
    ["Measured free chi2 red", num(measured.chi2_red)],
    ["Measured fixed chi2 red", num(fixed.chi2_red)],
    ["Flow rows", (data.flowsByTile[tid] || []).length],
    ["Boundary nodes", (data.pressuresByTile[tid] || []).length]
  ].map(([k,v]) => `<div class="metric"><span>${k}</span><strong>${v}</strong></div>`).join("");
  byId("tileMetrics").innerHTML = html;
}
function table(el, rows, cols) {
  if (!rows.length) {
    el.innerHTML = "<tbody><tr><td>No rows found</td></tr></tbody>";
    return;
  }
  const head = `<thead><tr>${cols.map(c=>`<th>${c.label}</th>`).join("")}</tr></thead>`;
  const body = rows.map(r => `<tr>${cols.map(c=>`<td>${c.f ? c.f(r[c.key], r) : (r[c.key] ?? "")}</td>`).join("")}</tr>`).join("");
  el.innerHTML = head + `<tbody>${body}</tbody>`;
}
function renderSummary(tid) {
  table(byId("summaryTable"), data.summaryByTile[tid] || [], [
    {key:"ablation", label:"profile"},
    {key:"D_hat", label:"D_hat", f:num},
    {key:"chi2_red", label:"chi2 red", f:num},
    {key:"width_decades_dchi1", label:"width dchi1", f:num},
    {key:"n_dc", label:"n dc"},
    {key:"n_h1", label:"n h1"},
    {key:"n_h2", label:"n h2"},
    {key:"max_delta_chi2", label:"max dchi2", f:num}
  ]);
}
function renderPressures(tid) {
  const rows = data.pressuresByTile[tid] || [];
  table(byId("pressureTable"), rows, [
    {key:"boundary_node", label:"node"},
    {key:"is_pin", label:"pin"},
    {key:"P_dc_Pa", label:"P dc Pa", f:num},
    {key:"amp_P_h1_Pa", label:"amp H1 Pa", f:num},
    {key:"phase_P_h1_rad", label:"phase H1", f:num},
    {key:"amp_P_h2_Pa", label:"amp H2 Pa", f:num}
  ]);
  drawBars(byId("pressureCanvas"), rows, "boundary_node", [
    ["P_dc_Pa", "#2f6fbb"],
    ["amp_P_h1_Pa", "#c75146"],
    ["amp_P_h2_Pa", "#2f8f6b"]
  ], "Boundary nodes", "Pressure / amplitude (Pa)");
}
function renderFlows(tid) {
  const limit = Math.max(10, Number(byId("flowLimit").value || 80));
  const rows = (data.flowsByTile[tid] || []).slice(0, limit);
  table(byId("flowTable"), rows, [
    {key:"u", label:"u"},
    {key:"v", label:"v"},
    {key:"measured_Q_dc_nL_s", label:"meas DC", f:num},
    {key:"mosaic_Q_dc_nL_s", label:"mosaic DC", f:num},
    {key:"delta_Q_dc_nL_s", label:"delta DC", f:num},
    {key:"measured_amp_h1_nL_s", label:"meas H1", f:num},
    {key:"mosaic_amp_h1_nL_s", label:"mosaic H1", f:num}
  ]);
  drawScatter(byId("flowCanvas"), rows, [
    ["measured_Q_dc_nL_s", "mosaic_Q_dc_nL_s", "#2f6fbb", "DC"],
    ["measured_amp_h1_nL_s", "mosaic_amp_h1_nL_s", "#c75146", "H1 amp"],
    ["measured_amp_h2_nL_s", "mosaic_amp_h2_nL_s", "#2f8f6b", "H2 amp"]
  ], "Measured flow (nL/s)", "Mosaic flow (nL/s)");
}
function canvasCtx(canvas) {
  const dpr = window.devicePixelRatio || 1;
  const rect = canvas.getBoundingClientRect();
  canvas.width = Math.max(320, rect.width * dpr);
  canvas.height = Math.max(240, rect.height * dpr);
  const ctx = canvas.getContext("2d");
  ctx.setTransform(dpr,0,0,dpr,0,0);
  return {ctx, w: canvas.width/dpr, h: canvas.height/dpr};
}
function clear(ctx,w,h) {
  ctx.clearRect(0,0,w,h);
  ctx.fillStyle = "white";
  ctx.fillRect(0,0,w,h);
}
function axes(ctx,w,h,xlab,ylab) {
  ctx.strokeStyle = "#d9e0e7"; ctx.lineWidth = 1;
  ctx.beginPath(); ctx.moveTo(52,16); ctx.lineTo(52,h-42); ctx.lineTo(w-16,h-42); ctx.stroke();
  ctx.fillStyle = "#607080"; ctx.font = "12px sans-serif";
  ctx.fillText(xlab, Math.max(58, w/2-60), h-14);
  ctx.save(); ctx.translate(14,h/2+55); ctx.rotate(-Math.PI/2); ctx.fillText(ylab,0,0); ctx.restore();
}
function renderProfiles(tid) {
  const rows = data.profilesByTile[tid] || [];
  const {ctx,w,h} = canvasCtx(byId("profileCanvas"));
  clear(ctx,w,h); axes(ctx,w,h,"D (log scale)","Delta chi2");
  const active = rows.filter(r => activeProfiles.has(r.ablation) && r.D > 0 && r.delta_chi2 !== null);
  if (!active.length) return;
  const xs = active.map(r => Math.log10(r.D));
  const ys = active.map(r => Number(r.delta_chi2));
  const xmin = Math.min(...xs), xmax = Math.max(...xs);
  const ymax = Math.max(4, Math.min(250, Math.max(...ys)));
  const sx = x => 52 + (Math.log10(x)-xmin) / (xmax-xmin || 1) * (w-74);
  const sy = y => h-42 - Math.min(y, ymax) / ymax * (h-64);
  for (const y of [1, 3.84]) {
    ctx.strokeStyle = y === 1 ? "#16202a" : "#607080";
    ctx.setLineDash([4,4]); ctx.beginPath(); ctx.moveTo(52, sy(y)); ctx.lineTo(w-16, sy(y)); ctx.stroke(); ctx.setLineDash([]);
    ctx.fillStyle = "#607080"; ctx.fillText(String(y), 56, sy(y)-3);
  }
  if (data.mosaicD) {
    ctx.strokeStyle = "#c75146"; ctx.beginPath(); ctx.moveTo(sx(data.mosaicD),16); ctx.lineTo(sx(data.mosaicD),h-42); ctx.stroke();
  }
  for (const name of Object.keys(colors)) {
    if (!activeProfiles.has(name)) continue;
    const prof = rows.filter(r => r.ablation === name && r.D > 0 && r.delta_chi2 !== null);
    if (!prof.length) continue;
    ctx.strokeStyle = colors[name]; ctx.fillStyle = colors[name]; ctx.lineWidth = 1.6;
    ctx.beginPath();
    prof.forEach((r,i) => { const x=sx(r.D), y=sy(Number(r.delta_chi2)); if (i) ctx.lineTo(x,y); else ctx.moveTo(x,y); });
    ctx.stroke();
    for (const r of prof) { ctx.beginPath(); ctx.arc(sx(r.D), sy(Number(r.delta_chi2)), 2.4, 0, Math.PI*2); ctx.fill(); }
  }
}
function drawBars(canvas, rows, labelKey, series, xlab, ylab) {
  const {ctx,w,h} = canvasCtx(canvas); clear(ctx,w,h); axes(ctx,w,h,xlab,ylab);
  if (!rows.length) return;
  const vals = [];
  for (const r of rows) for (const [key] of series) if (r[key] !== null && r[key] !== undefined) vals.push(Number(r[key]));
  if (!vals.length) return;
  const min = Math.min(0, ...vals), max = Math.max(...vals);
  const n = rows.length, band = (w-74) / Math.max(n,1), bw = Math.max(1, band / (series.length + 1));
  const sy = y => h-42 - (y-min) / ((max-min) || 1) * (h-64);
  ctx.strokeStyle = "#9aa8b5"; ctx.beginPath(); ctx.moveTo(52, sy(0)); ctx.lineTo(w-16, sy(0)); ctx.stroke();
  rows.forEach((r,i) => {
    series.forEach(([key,color],j) => {
      const v = r[key]; if (v === null || v === undefined) return;
      const x = 52 + i*band + j*bw;
      const y0 = sy(0), y1 = sy(Number(v));
      ctx.fillStyle = color; ctx.fillRect(x, Math.min(y0,y1), bw, Math.max(1, Math.abs(y1-y0)));
    });
  });
  ctx.fillStyle = "#607080"; ctx.fillText(num(max), 56, 24); ctx.fillText(num(min), 56, h-48);
}
function drawScatter(canvas, rows, series, xlab, ylab) {
  const {ctx,w,h} = canvasCtx(canvas); clear(ctx,w,h); axes(ctx,w,h,xlab,ylab);
  const pts = [];
  for (const r of rows) for (const [xk,yk,color,label] of series) {
    const x = r[xk], y = r[yk];
    if (x !== null && y !== null && x !== undefined && y !== undefined) pts.push([Number(x),Number(y),color,label]);
  }
  if (!pts.length) return;
  const min = Math.min(...pts.flatMap(p=>[p[0],p[1]])), max = Math.max(...pts.flatMap(p=>[p[0],p[1]]));
  const pad = (max-min)*0.05 || 1;
  const lo = min-pad, hi = max+pad;
  const sx = x => 52 + (x-lo)/(hi-lo)*(w-74);
  const sy = y => h-42 - (y-lo)/(hi-lo)*(h-64);
  ctx.strokeStyle = "#9aa8b5"; ctx.setLineDash([4,4]); ctx.beginPath(); ctx.moveTo(sx(lo), sy(lo)); ctx.lineTo(sx(hi), sy(hi)); ctx.stroke(); ctx.setLineDash([]);
  for (const [x,y,color] of pts) { ctx.fillStyle = color; ctx.globalAlpha = 0.72; ctx.beginPath(); ctx.arc(sx(x), sy(y), 2.8, 0, Math.PI*2); ctx.fill(); }
  ctx.globalAlpha = 1; ctx.fillStyle = "#607080"; ctx.fillText(num(lo), 56, h-48); ctx.fillText(num(hi), w-62, 24);
}
window.addEventListener("resize", render);
setup();
</script>
</body>
</html>
"""


def build_dashboard(args) -> Path:
    comparison_dir = Path(args.comparison_dir).resolve()
    mosaic_dir = Path(args.mosaic_dir).resolve()
    summary_csv = comparison_dir / "mosaic_comparison_summary.csv"
    pressure_csv = mosaic_dir / "tile_boundary_pressures_from_mosaic.csv"

    summary = [_clean_row(r) for r in _read_csv(summary_csv)]
    profiles_by_tile = _load_profiles(comparison_dir)
    summary_by_tile = _group_by_tile(summary)
    pressures_by_tile = _group_by_tile(_read_csv(pressure_csv))
    tiles = sorted({int(t) for t in (
        set(summary_by_tile) | set(profiles_by_tile) | set(pressures_by_tile)
    )})
    flows_by_tile = _load_flow_rows(args, tiles) if not args.skip_flows else {}

    mosaic_d = _load_mosaic_d(args, summary)

    payload = {
        "summaryByTile": summary_by_tile,
        "profilesByTile": profiles_by_tile,
        "pressuresByTile": pressures_by_tile,
        "flowsByTile": flows_by_tile,
        "mosaicD": mosaic_d,
        "sourceFiles": {
            "summary": str(summary_csv),
            "pressures": str(pressure_csv),
        },
    }

    out_path = Path(args.out).resolve() if args.out else (
        comparison_dir / "interactive_mosaic_tile_dashboard.html")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    data_json = json.dumps(payload, allow_nan=False, separators=(",", ":"))
    out_path.write_text(
        HTML_TEMPLATE.replace("__DATA__", html.escape(data_json, quote=False)),
        encoding="utf-8")
    return out_path


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Create an interactive HTML dashboard for tile results.")
    ap.add_argument("--comparison-dir",
                    default="renders/meeting/"
                            "tile_distensibility_mosaic_comparison")
    ap.add_argument("--mosaic-dir",
                    default="renders/meeting/mosaic_tile_solve")
    ap.add_argument("--config", default="../emb1/config.json",
                    help="Config JSON used to find the graph for flow rows.")
    ap.add_argument("--graph", default=None,
                    help="Optional graph path override for flow rows.")
    ap.add_argument("--mosaic-result", default=None,
                    help="Optional mosaic_solve_result.pkl override.")
    ap.add_argument("--skip-flows", action="store_true",
                    help="Build dashboard without per-edge flow rows.")
    ap.add_argument("--out", default=None,
                    help="Output HTML path.")
    args = ap.parse_args()

    out_path = build_dashboard(args)
    print(f"Wrote {out_path}")
    print("Open that HTML file in a browser to browse tiles interactively.")


if __name__ == "__main__":
    main()
