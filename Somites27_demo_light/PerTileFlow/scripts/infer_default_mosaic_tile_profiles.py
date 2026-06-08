"""Measured-data tile distensibility profile-likelihood dashboard.

This script profiles distensibility directly against the measured tile flow
observations in the analyzed mosaic graph.  Unlike
``default_mosaic_tile_profiles.py``, it does not run a whole-mosaic simulation
with a chosen input D first.

Default assumptions:
  * measured vessel geometry and PIV-derived tile edge flow observations
  * H1+H2 in tile profile scans
  * free tile-boundary pressures are refit independently at each D

Outputs:
  * measured_global_profile_constant_D[...].csv
  * measured_tile_profiles[...].csv
  * measured_tile_profile_summary[...].csv
  * infer_default_mosaic_tile_profiles[...].html
"""
from __future__ import annotations

import argparse
import csv
import html
import json
import math
import sys
import time
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from distensibility_ablation import (
    _metric_row,
    _n_params,
    _observations,
    _profile_free,
    _sigma_vectors,
)
from inspect_tile import build_tile_problem
from tile_mosaic_simulation import choose_tiles, load_graph_from_args
from synthetic_validation_neumann_bc import nL_per_m3


DEFAULT_TILE_PROFILE_HARMONICS = (1, 2)
PERIPHERY_TILES = {
    1, 2, 3, 4, 5, 6, 7, 8, 23, 24, 41, 42, 60, 61, 75, 76, 87, 86,
    84, 83, 81, 69, 68, 52, 51, 33, 32, 15, 14,
}


def _profile_tile_from_measured_flow(graph, tile_id: int, D_grid: np.ndarray,
                                     args):
    harmonics = tuple(int(h) for h in args.tile_harmonics)
    prob = build_tile_problem(graph, int(tile_id))
    obs = _observations(graph, prob, harmonics)
    weighted = _is_weighted_objective(args)
    if weighted:
        sig_dc, sig_h = _sigma_vectors(obs, args)
        weight_mode = "sigma"
    else:
        sig_dc, sig_h = _constant_average_sigma_vectors(obs, args)
        weight_mode = "constant_average_sigma"
    objective = _objective_name(args)
    profile, best = _profile_free(prob, obs, sig_dc, sig_h, D_grid,
                                  harmonics)
    metrics = _metric_row(
        int(tile_id), "measured_flow_free_boundary", harmonics, profile, best,
        prob, obs, _n_params(prob, harmonics, "free"))
    metrics["objective"] = objective
    metrics["observation_source"] = "measured_graph"
    metrics.update(_weight_diagnostics(obs, sig_dc, sig_h, weight_mode))
    return profile, metrics


def _constant_average_sigma_vectors(
        obs: dict, args) -> tuple[np.ndarray, Dict[int, np.ndarray]]:
    weighted_dc, weighted_h = _sigma_vectors(obs, args)
    vals = []
    valid_dc = np.asarray(obs["valid"]["dc"], dtype=bool)
    if valid_dc.any():
        vals.append(np.asarray(weighted_dc, dtype=float)[valid_dc])
    for h, sig in weighted_h.items():
        valid = np.asarray(obs["valid"].get(int(h), []), dtype=bool)
        if valid.any():
            # Complex harmonic residuals contribute real and imaginary parts.
            vals.extend([np.asarray(sig, dtype=float)[valid]] * 2)
    sigma0 = float(np.nanmean(np.concatenate(vals))) if vals else 1.0
    sigma0 = max(sigma0, 1e-30)

    sig_dc = np.full_like(np.asarray(obs["q_dc"], dtype=float), sigma0,
                          dtype=float)
    sig_h = {}
    for h, q in obs["q_h"].items():
        sig_h[int(h)] = np.full_like(np.asarray(q, dtype=complex), sigma0,
                                     dtype=float)
    return sig_dc, sig_h


def _objective_name(args) -> str:
    return ("ordinary_least_squares" if args.ordinary_least_squares
            else "weighted_least_squares")


def _is_weighted_objective(args) -> bool:
    return _objective_name(args) == "weighted_least_squares"


def _weight_diagnostics(obs: dict, sig_dc: np.ndarray,
                        sig_h: Dict[int, np.ndarray],
                        weight_mode: str) -> dict:
    rows = {
        "weight_mode": weight_mode,
    }

    def _range(prefix: str, sig: np.ndarray, valid: np.ndarray) -> None:
        vals = np.asarray(sig)[np.asarray(valid, dtype=bool)]
        if vals.size:
            rows[f"{prefix}_sigma_min"] = float(np.nanmin(vals))
            rows[f"{prefix}_sigma_max"] = float(np.nanmax(vals))
            rows[f"{prefix}_sigma_min_nL_s"] = float(np.nanmin(vals)
                                                     * nL_per_m3)
            rows[f"{prefix}_sigma_max_nL_s"] = float(np.nanmax(vals)
                                                     * nL_per_m3)
        else:
            rows[f"{prefix}_sigma_min"] = float("nan")
            rows[f"{prefix}_sigma_max"] = float("nan")
            rows[f"{prefix}_sigma_min_nL_s"] = float("nan")
            rows[f"{prefix}_sigma_max_nL_s"] = float("nan")

    _range("dc", sig_dc, obs["valid"]["dc"])
    for h in (1, 2):
        if h in sig_h and h in obs["valid"]:
            _range(f"h{h}", sig_h[h], obs["valid"][h])
    return rows


def _run_suffix(args) -> str:
    if args.ordinary_least_squares:
        return "_ordinary"
    return "_weighted" if args.weighted_least_squares else ""


def _out_name(base: str, args) -> str:
    path = Path(base)
    return f"{path.stem}{_run_suffix(args)}{path.suffix}"


def _summed_profile(tile_profiles: Dict[int, List[dict]]) -> List[dict]:
    by_d = {}
    for tile_id, profile in tile_profiles.items():
        for row in profile:
            d = float(row["D"])
            by_d.setdefault(d, 0.0)
            by_d[d] += float(row["chi2"])
    rows = [{"D": d, "chi2": chi2} for d, chi2 in sorted(by_d.items())]
    if not rows:
        return rows
    chi_min = min(r["chi2"] for r in rows)
    for r in rows:
        r["delta_chi2"] = r["chi2"] - chi_min
    return rows


def _best_D(profile: List[dict]):
    if not profile:
        return None
    best = min(profile, key=lambda r: float(r["chi2"]))
    return float(best["D"])


def _tile_location(tile_id: int) -> str:
    return "periphery" if int(tile_id) in PERIPHERY_TILES else "interior"


def _write_csv(path: Path, rows: List[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({k for r in rows for k in r.keys()})
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _clean_json_value(value):
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        value = float(value)
        return value if math.isfinite(value) else None
    if isinstance(value, (np.ndarray,)):
        return [_clean_json_value(v) for v in value.tolist()]
    if isinstance(value, dict):
        return {str(k): _clean_json_value(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_clean_json_value(v) for v in value]
    return value


HTML_TEMPLATE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width, initial-scale=1" />
<title>Measured Tile Distensibility Profile Likelihoods</title>
<style>
:root {
  --bg: #f6f7f9;
  --panel: #ffffff;
  --ink: #18222d;
  --muted: #647486;
  --line: #d8e0e8;
  --blue: #2868b7;
  --red: #c74e45;
  --green: #2c8a68;
  --orange: #c27a22;
}
* { box-sizing: border-box; }
body {
  margin: 0;
  background: var(--bg);
  color: var(--ink);
  font: 14px/1.4 -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
}
header {
  padding: 14px 18px;
  background: #202b36;
  color: white;
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 16px;
  position: sticky;
  top: 0;
  z-index: 5;
}
h1 { font-size: 17px; margin: 0; }
main { padding: 14px; display: grid; gap: 14px; }
section {
  background: var(--panel);
  border: 1px solid var(--line);
  border-radius: 8px;
  padding: 12px;
  min-width: 0;
}
.grid2 { display: grid; grid-template-columns: minmax(0, 1fr) minmax(360px, 0.55fr); gap: 14px; }
.gridTop { display: grid; grid-template-columns: minmax(0, 0.9fr) minmax(0, 1.1fr); gap: 14px; }
h2 { font-size: 15px; margin: 0 0 10px; }
canvas {
  width: 100%;
  height: 380px;
  border: 1px solid var(--line);
  border-radius: 6px;
  background: white;
}
.controls {
  display: grid;
  grid-template-columns: repeat(4, minmax(0, 1fr));
  gap: 8px;
}
label { color: var(--muted); font-size: 12px; display: grid; gap: 3px; }
select {
  height: 34px;
  border: 1px solid var(--line);
  border-radius: 6px;
  background: white;
  color: var(--ink);
  padding: 0 8px;
}
.inlineControls {
  display: flex;
  align-items: center;
  gap: 10px;
  margin: -2px 0 10px;
}
.inlineControls label {
  display: flex;
  align-items: center;
  gap: 6px;
}
.inlineControls select { min-width: 130px; }
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
th:first-child, td:first-child { text-align: left; }
thead th { position: sticky; top: 0; background: #edf2f6; }
.scroll { max-height: 380px; overflow: auto; border: 1px solid var(--line); border-radius: 6px; }
.meta { color: #dce6ef; font-size: 12px; }
.note { color: var(--muted); font-size: 12px; margin-top: 8px; }
.swatches { display: flex; gap: 12px; flex-wrap: wrap; color: var(--muted); font-size: 12px; margin-top: 8px; }
.swatches span::before { content: ""; display: inline-block; width: 10px; height: 10px; margin-right: 5px; border-radius: 2px; background: var(--c); }
@media (max-width: 1050px) {
  .grid2, .gridTop, .controls { grid-template-columns: 1fr; }
}
</style>
</head>
<body>
<header>
  <h1>Measured Tile Distensibility Profile Likelihoods</h1>
  <div class="meta" id="meta"></div>
</header>
<main>
  <div class="gridTop">
    <section>
      <h2>Constant Distensibility Across All Tiles</h2>
      <canvas id="globalCanvas"></canvas>
      <div class="note">The blue curve sums measured-data tile chi2 values at each D. Best D values are the minima of summed profiles: all tiles, interior-only tiles, and periphery-only tiles.</div>
    </section>
    <section>
      <h2>All Tile Profiles Overlayed</h2>
      <div class="inlineControls">
        <label>Show
          <select id="allFilter">
            <option value="all">All tiles</option>
            <option value="interior">Interior</option>
            <option value="periphery">Periphery</option>
          </select>
        </label>
      </div>
      <canvas id="allCanvas"></canvas>
      <div class="note">Each line is one tile's measured-data profile likelihood, normalized to that tile's own minimum. Reference lines show summed-profile best D values for all, interior, and periphery tiles.</div>
    </section>
  </div>
  <div class="grid2">
    <section>
      <h2>Selected Tile Profile Overlay</h2>
      <div class="controls">
        <label>Tile 1<select id="sel0"></select></label>
        <label>Tile 2<select id="sel1"></select></label>
        <label>Tile 3<select id="sel2"></select></label>
        <label>Tile 4<select id="sel3"></select></label>
      </div>
      <canvas id="selectedCanvas"></canvas>
      <div class="swatches">
        <span style="--c:#2868b7">Tile 1</span>
        <span style="--c:#c74e45">Tile 2</span>
        <span style="--c:#2c8a68">Tile 3</span>
        <span style="--c:#c27a22">Tile 4</span>
      </div>
    </section>
    <section>
      <h2>Tile Summary</h2>
      <div class="scroll"><table id="summaryTable"></table></div>
    </section>
  </div>
</main>
<script id="payload" type="application/json">__DATA__</script>
<script>
const data = JSON.parse(document.getElementById("payload").textContent);
const colors = ["#2868b7", "#c74e45", "#2c8a68", "#c27a22"];
const allColors = ["#8aa7c8", "#c8a18a", "#8bc0a9", "#b6a2ce", "#d8ba74", "#9cb5ba"];
const tileIds = Object.keys(data.tileProfiles).map(Number).sort((a,b)=>a-b).map(String);

function num(v) {
  if (v === null || v === undefined || Number.isNaN(Number(v))) return "";
  const x = Number(v);
  if (x === 0) return "0";
  if (Math.abs(x) >= 1000 || Math.abs(x) < 0.01) return x.toExponential(2);
  return x.toFixed(3).replace(/\.?0+$/, "");
}
function byId(id) { return document.getElementById(id); }
function setupSelectors() {
  for (let i = 0; i < 4; i++) {
    const sel = byId(`sel${i}`);
    sel.innerHTML = "";
    const none = document.createElement("option");
    none.value = "";
    none.textContent = "None";
    sel.appendChild(none);
    for (const tid of tileIds) {
      const opt = document.createElement("option");
      opt.value = tid;
      opt.textContent = tid.padStart(3, "0");
      sel.appendChild(opt);
    }
    if (tileIds[i]) sel.value = tileIds[i];
    sel.addEventListener("change", render);
  }
  byId("allFilter").addEventListener("change", render);
}
function canvasCtx(canvas) {
  const dpr = window.devicePixelRatio || 1;
  const rect = canvas.getBoundingClientRect();
  canvas.width = Math.max(360, rect.width * dpr);
  canvas.height = Math.max(260, rect.height * dpr);
  const ctx = canvas.getContext("2d");
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  return {ctx, w: canvas.width / dpr, h: canvas.height / dpr};
}
function clear(ctx,w,h) {
  ctx.clearRect(0,0,w,h);
  ctx.fillStyle = "white";
  ctx.fillRect(0,0,w,h);
}
function axes(ctx,w,h,xlab,ylab) {
  ctx.strokeStyle = "#d8e0e8";
  ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.moveTo(58, 18);
  ctx.lineTo(58, h - 44);
  ctx.lineTo(w - 18, h - 44);
  ctx.stroke();
  ctx.fillStyle = "#647486";
  ctx.font = "12px sans-serif";
  ctx.fillText(xlab, Math.max(64, w / 2 - 45), h - 15);
  ctx.save();
  ctx.translate(16, h / 2 + 52);
  ctx.rotate(-Math.PI / 2);
  ctx.fillText(ylab, 0, 0);
  ctx.restore();
}
function logTickValues(xmin, xmax) {
  const ticks = [];
  const lo = Math.ceil(xmin);
  const hi = Math.floor(xmax);
  for (let e = lo; e <= hi; e++) ticks.push(Math.pow(10, e));
  if (!ticks.length) {
    ticks.push(Math.pow(10, xmin), Math.pow(10, xmax));
  }
  return ticks;
}
function linearTickValues(ymax) {
  const rawStep = ymax / 4;
  const pow = Math.pow(10, Math.floor(Math.log10(rawStep || 1)));
  const candidates = [1, 2, 5, 10].map(v => v * pow);
  let step = candidates[candidates.length - 1];
  for (const c of candidates) {
    if (rawStep <= c) { step = c; break; }
  }
  const ticks = [];
  for (let y = 0; y <= ymax + step * 0.5; y += step) {
    ticks.push(y);
  }
  return ticks;
}
function drawTicks(ctx,w,h,xTicks,yTicks,sx,sy) {
  ctx.font = "11px sans-serif";
  ctx.textAlign = "center";
  ctx.textBaseline = "top";
  ctx.strokeStyle = "#9aa8b5";
  ctx.fillStyle = "#647486";
  for (const xVal of xTicks) {
    const x = sx(xVal);
    if (x < 58 || x > w - 18) continue;
    ctx.beginPath();
    ctx.moveTo(x, h - 44);
    ctx.lineTo(x, h - 38);
    ctx.stroke();
    ctx.fillText(num(xVal), x, h - 34);
  }
  ctx.textAlign = "right";
  ctx.textBaseline = "middle";
  for (const yVal of yTicks) {
    const y = sy(yVal);
    if (y < 18 || y > h - 44) continue;
    ctx.beginPath();
    ctx.moveTo(52, y);
    ctx.lineTo(58, y);
    ctx.stroke();
    ctx.fillText(num(yVal), 48, y);
  }
  ctx.textAlign = "left";
  ctx.textBaseline = "alphabetic";
}
function drawProfiles(canvas, series, opts={}) {
  const {ctx,w,h} = canvasCtx(canvas);
  clear(ctx,w,h);
  axes(ctx,w,h,"D (1/Pa, log scale)","Delta chi2");
  const points = [];
  for (const s of series) {
    for (const p of s.rows) {
      if (p.D > 0 && p.delta_chi2 !== null && p.delta_chi2 !== undefined) {
        points.push([Math.log10(p.D), Number(p.delta_chi2)]);
      }
    }
  }
  if (!points.length) return;
  const xmin = Math.min(...points.map(p => p[0]));
  const xmax = Math.max(...points.map(p => p[0]));
  const rawYmax = Math.max(...points.map(p => p[1]));
  const cap = opts.yCap === undefined ? 12 : opts.yCap;
  const ymax = Math.max(4, Math.min(cap, rawYmax));
  const sx = D => 58 + (Math.log10(D) - xmin) / ((xmax - xmin) || 1) * (w - 82);
  const sy = y => h - 44 - Math.min(y, ymax) / ymax * (h - 68);
  drawTicks(ctx, w, h, logTickValues(xmin, xmax), linearTickValues(ymax),
            sx, sy);
  for (const y of [1, 3.84]) {
    ctx.strokeStyle = y === 1 ? "#18222d" : "#647486";
    ctx.setLineDash([4,4]);
    ctx.beginPath();
    ctx.moveTo(58, sy(y));
    ctx.lineTo(w - 18, sy(y));
    ctx.stroke();
    ctx.setLineDash([]);
    ctx.fillStyle = "#647486";
    ctx.fillText(String(y), 62, sy(y) - 4);
  }
  const refs = opts.referenceLines || [];
  for (const ref of refs) {
    if (!ref || !ref.D) continue;
    ctx.strokeStyle = ref.color;
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(sx(ref.D), 18);
    ctx.lineTo(sx(ref.D), h - 44);
    ctx.stroke();
    if (ref.annotate) {
      ctx.fillStyle = ref.color;
      ctx.font = "12px sans-serif";
      ctx.fillText(`${ref.label} = ${num(ref.D)}`,
                   Math.min(w - 170, sx(ref.D) + 6), ref.y || 31);
    }
  }
  for (const s of series) {
    if (!s.rows.length) continue;
    ctx.strokeStyle = s.color;
    ctx.fillStyle = s.color;
    ctx.globalAlpha = s.alpha || 1;
    ctx.lineWidth = s.width || 1.4;
    ctx.beginPath();
    s.rows.forEach((p, i) => {
      const x = sx(p.D);
      const y = sy(Number(p.delta_chi2));
      if (i) ctx.lineTo(x, y); else ctx.moveTo(x, y);
    });
    ctx.stroke();
    if (s.markers) {
      for (const p of s.rows) {
        ctx.beginPath();
        ctx.arc(sx(p.D), sy(Number(p.delta_chi2)), 2.5, 0, Math.PI * 2);
        ctx.fill();
      }
    }
    ctx.globalAlpha = 1;
  }
  if (opts.legend || opts.referenceLegend) {
    const legendPos = opts.legendPosition || "topRight";
    const legendItemCount = (opts.legend ? series.length : 0) +
                            (opts.referenceLegend ? refs.filter(ref => ref && ref.D).length : 0);
    let y = legendPos === "bottomRight" ? h - Math.max(14, legendItemCount * 17) : 28;
    ctx.font = "12px sans-serif";
    if (opts.legend) {
      for (const s of series) {
        ctx.fillStyle = s.color;
        ctx.fillRect(w - 156, y - 9, 10, 10);
        ctx.fillStyle = "#18222d";
        ctx.fillText(s.label, w - 140, y);
        y += 17;
      }
    }
    if (opts.referenceLegend) {
      for (const ref of refs) {
        if (!ref || !ref.D) continue;
        ctx.strokeStyle = ref.color;
        ctx.beginPath();
        ctx.moveTo(w - 156, y - 5);
        ctx.lineTo(w - 146, y - 5);
        ctx.stroke();
        ctx.fillStyle = "#18222d";
        ctx.fillText(ref.label, w - 140, y);
        y += 17;
      }
    }
  }
}
function referenceLines(opts={}) {
  const lines = [];
  if (opts.global && data.bestD && data.bestD.all) {
    lines.push({D: data.bestD.all, color: "#2c8a68", label: "all-tiles best D", annotate: opts.annotate, y: 31});
  }
  if (opts.interior && data.bestD && data.bestD.interior) {
    lines.push({D: data.bestD.interior, color: "#6f63bd", label: "interior best D", annotate: opts.annotate, y: 47});
  }
  if (opts.periphery && data.bestD && data.bestD.periphery) {
    lines.push({D: data.bestD.periphery, color: "#c27a22", label: "periphery best D", annotate: opts.annotate, y: 63});
  }
  return lines;
}
function renderGlobal() {
  drawProfiles(byId("globalCanvas"), [{
    label: "all tiles constant D",
    rows: data.globalProfile,
    color: "#2868b7",
    width: 2,
    markers: true
  }], {
    legend: true,
    legendPosition: "bottomRight",
    yCap: 12,
    referenceLines: referenceLines({
      global: true, interior: true, periphery: true,
      annotate: true
    })
  });
}
function renderAll() {
  const filter = byId("allFilter").value || "all";
  const shown = tileIds.filter(tid => {
    if (filter === "all") return true;
    return data.tileLocation[tid] === filter;
  });
  const series = shown.map((tid, i) => ({
    label: `tile ${tid}`,
    rows: data.tileProfiles[tid] || [],
    color: allColors[i % allColors.length],
    alpha: 0.42,
    width: 1
  }));
  drawProfiles(byId("allCanvas"), series, {
    yCap: 12,
    referenceLines: referenceLines({
      global: true, interior: true, periphery: true,
      annotate: true
    })
  });
}
function renderSelected() {
  const selected = [];
  for (let i = 0; i < 4; i++) {
    const tid = byId(`sel${i}`).value;
    if (!tid) continue;
    selected.push({
      label: `tile ${tid}`,
      rows: data.tileProfiles[tid] || [],
      color: colors[i],
      width: 2,
      markers: true
    });
  }
  drawProfiles(byId("selectedCanvas"), selected, {legend: true, yCap: 12});
  renderTable(selected.map(s => s.label.replace("tile ", "")));
}
function renderTable(selectedIds) {
  const rows = data.summary
    .filter(r => !selectedIds.length || selectedIds.includes(String(r.tile_id)))
    .sort((a,b) => Number(a.tile_id) - Number(b.tile_id));
  const cols = [
    ["tile_id", "tile"],
    ["location", "location"],
    ["D_hat", "D_hat"],
    ["chi2_red", "chi2 red"],
    ["width_decades_dchi1", "width dchi1"],
    ["max_delta_chi2", "max dchi2"],
    ["n_dc", "n dc"],
    ["n_h1", "n h1"],
    ["n_h2", "n h2"]
  ];
  const cell = (r, key) => {
    const value = r[key];
    if (key === "tile_id" || key === "location") return value ?? "";
    return num(value);
  };
  const head = `<thead><tr>${cols.map(c => `<th>${c[1]}</th>`).join("")}</tr></thead>`;
  const body = rows.map(r => `<tr>${cols.map(c => `<td>${cell(r, c[0])}</td>`).join("")}</tr>`).join("");
  byId("summaryTable").innerHTML = head + `<tbody>${body}</tbody>`;
}
function render() {
  renderGlobal();
  renderAll();
  renderSelected();
}
function setup() {
  const objective = data.defaults.objective === "ordinary_least_squares" ? "ordinary LS" : "weighted LS";
  byId("meta").textContent = `objective=${objective}; source=measured graph; all best D=${num(data.bestD.all)}; interior best D=${num(data.bestD.interior)}; periphery best D=${num(data.bestD.periphery)}; tiles=${tileIds.length}`;
  setupSelectors();
  window.addEventListener("resize", render);
  render();
}
setup();
</script>
</body>
</html>
"""


def _write_dashboard(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data_json = json.dumps(
        _clean_json_value(payload), allow_nan=False, separators=(",", ":"))
    path.write_text(
        HTML_TEMPLATE.replace("__DATA__", html.escape(data_json, quote=False)),
        encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Run measured-data tile distensibility inference dashboard.")
    ap.add_argument("--config", default="../emb1/config.json")
    ap.add_argument("--graph", default=None)
    ap.add_argument("--tiles", nargs="*", type=int, default=None)
    ap.add_argument("--all-tiles", action="store_true",
                    help="Run all measured tiles. This is also the default "
                         "when --tiles is omitted.")
    ap.add_argument("--out-dir", default=None,
                    help="Default: renders/meeting/"
                         "infer_default_mosaic_tile_profiles")
    ap.add_argument("--source-nodes", nargs="*", type=int, default=None,
                    help="Optional source node IDs recorded for provenance. "
                         "Measured tile inference does not use them in the "
                         "fit because no whole-mosaic solve is run.")
    ap.add_argument("--sink-nodes", nargs="*", type=int, default=None,
                    help="Optional sink node IDs recorded for provenance. "
                         "Measured tile inference does not use them in the "
                         "fit because no whole-mosaic solve is run.")
    ap.add_argument("--tile-harmonics", nargs="+", type=int,
                    default=list(DEFAULT_TILE_PROFILE_HARMONICS),
                    choices=[1, 2])
    ap.add_argument("--D-min", type=float, default=1e-6)
    ap.add_argument("--D-max", type=float, default=1e-1)
    ap.add_argument("--D-count", type=int, default=41)
    ap.add_argument("--a-dc", type=float, default=0.061,
                    help="DC additive noise floor in nL/s.")
    ap.add_argument("--a-h1", type=float, default=0.012,
                    help="H1 additive noise floor in nL/s.")
    ap.add_argument("--a-h2", type=float, default=0.030,
                    help="H2 additive noise floor in nL/s.")
    ap.add_argument("--b-dc", type=float, default=0.29)
    ap.add_argument("--b-h1", type=float, default=0.0)
    ap.add_argument("--b-h2", type=float, default=0.0)
    ap.add_argument("--weighted-least-squares", action="store_true",
                    help="Use sigma-weighted least squares and write outputs "
                         "with _weighted in their filenames. This is also "
                         "the default objective when neither LS flag is set.")
    ap.add_argument("--ordinary-least-squares", action="store_true",
                    help="Use one constant sigma equal to the average "
                         "weighted-noise sigma. This gives ordinary LS "
                         "relative weighting without SI-unit chi2 collapse.")
    args = ap.parse_args()
    if args.weighted_least_squares and args.ordinary_least_squares:
        raise SystemExit("Choose only one of --weighted-least-squares or "
                         "--ordinary-least-squares.")

    out_dir = (Path(args.out_dir).resolve() if args.out_dir else
               PROJECT_ROOT / "renders" / "meeting"
               / "infer_default_mosaic_tile_profiles")
    out_dir.mkdir(parents=True, exist_ok=True)

    graph, graph_path = load_graph_from_args(args)
    tiles = choose_tiles(graph, args.tiles, all_tiles=(args.all_tiles
                         or not args.tiles))
    D_grid = np.logspace(np.log10(args.D_min), np.log10(args.D_max),
                         int(args.D_count))

    print(f"Graph: {graph_path}")
    print(f"Tiles: {tiles}")
    print(f"Output: {out_dir}")
    print("Inference config:")
    print("  observation_source=measured_graph")
    print(f"  tile_harmonics={tuple(int(h) for h in args.tile_harmonics)}")
    print(f"  D_grid=[{args.D_min:.3e}, {args.D_max:.3e}], "
          f"count={args.D_count}")
    if args.source_nodes or args.sink_nodes:
        print("  boundary_nodes=recorded for provenance only "
              "(not used by measured tile inference)")
        print(f"  source_nodes={[int(n) for n in (args.source_nodes or [])]}")
        print(f"  sink_nodes={[int(n) for n in (args.sink_nodes or [])]}")
    objective_label = _objective_name(args).replace("_", " ")
    print(f"Tile profile objective: {objective_label}")

    t0 = time.time()
    tile_profiles: Dict[int, List[dict]] = {}
    summary_rows: List[dict] = []
    profile_rows: List[dict] = []
    for i, tile_id in enumerate(tiles, start=1):
        elapsed = time.time() - t0
        eta = elapsed / max(i - 1, 1) * (len(tiles) - i + 1) if i > 1 else 0
        eta_txt = f", eta={eta / 60.0:.1f} min" if i > 1 else ""
        print(f"[{i}/{len(tiles)}] tile {tile_id}{eta_txt}", flush=True)
        try:
            profile, metrics = _profile_tile_from_measured_flow(
                graph, int(tile_id), D_grid, args)
        except Exception as e:
            import traceback
            traceback.print_exc()
            summary_rows.append({
                "tile_id": int(tile_id), "error": f"{type(e).__name__}: {e}"})
            continue
        tile_profiles[int(tile_id)] = profile
        metrics["location"] = _tile_location(int(tile_id))
        summary_rows.append(metrics)
        chi_min = min(p["chi2"] for p in profile)
        for p in profile:
            profile_rows.append({
                "tile_id": int(tile_id),
                "observation_source": "measured_graph",
                "objective": metrics["objective"],
                "weight_mode": metrics["weight_mode"],
                "D": float(p["D"]),
                "chi2": float(p["chi2"]),
                "delta_chi2": float(p["chi2"] - chi_min),
            })
        print(f"  D_hat={metrics['D_hat']:.3e}  "
              f"chi2_red={metrics['chi2_red']:.3g}  "
              f"width1={metrics['width_decades_dchi1']:.3g} decades")

    global_rows = _summed_profile(tile_profiles)
    interior_profiles = {
        tid: prof for tid, prof in tile_profiles.items()
        if _tile_location(tid) == "interior"
    }
    periphery_profiles = {
        tid: prof for tid, prof in tile_profiles.items()
        if _tile_location(tid) == "periphery"
    }
    interior_rows = _summed_profile(interior_profiles)
    periphery_rows = _summed_profile(periphery_profiles)
    best_D = {
        "all": _best_D(global_rows),
        "interior": _best_D(interior_rows),
        "periphery": _best_D(periphery_rows),
    }
    for rows, group in ((global_rows, "all"), (interior_rows, "interior"),
                        (periphery_rows, "periphery")):
        for row in rows:
            row["observation_source"] = "measured_graph"
            row["objective"] = _objective_name(args)
            row["weight_mode"] = ("sigma" if _is_weighted_objective(args)
                                  else "constant_average_sigma")
            row["tile_group"] = group
    csv_paths = [
        out_dir / _out_name("measured_global_profile_constant_D.csv", args),
        out_dir / _out_name("measured_interior_profile_constant_D.csv", args),
        out_dir / _out_name("measured_periphery_profile_constant_D.csv", args),
        out_dir / _out_name("measured_tile_profiles.csv", args),
        out_dir / _out_name("measured_tile_profile_summary.csv", args),
    ]
    for path, rows in zip(csv_paths, [
            global_rows, interior_rows, periphery_rows, profile_rows,
            summary_rows]):
        _write_csv(path, rows)
        print(f"Wrote {path}")

    tile_payload = {}
    for tile_id, profile in tile_profiles.items():
        chi_min = min(p["chi2"] for p in profile)
        tile_payload[str(tile_id)] = [
            {"D": float(p["D"]),
             "chi2": float(p["chi2"]),
             "delta_chi2": float(p["chi2"] - chi_min)}
            for p in profile
        ]
    payload = {
        "defaults": {
            "observation_source": "measured_graph",
            "tile_harmonics": [int(h) for h in args.tile_harmonics],
            "D_min": float(args.D_min),
            "D_max": float(args.D_max),
            "D_count": int(args.D_count),
            "objective": _objective_name(args),
            "weighted_least_squares": bool(_is_weighted_objective(args)),
            "weighted_output_names": bool(args.weighted_least_squares),
            "ordinary_least_squares": bool(args.ordinary_least_squares),
            "source_nodes": [int(n) for n in (args.source_nodes or [])],
            "sink_nodes": [int(n) for n in (args.sink_nodes or [])],
            "boundary_nodes_used_in_fit": False,
            "periphery_tiles": sorted(int(t) for t in PERIPHERY_TILES),
        },
        "globalProfile": global_rows,
        "groupProfiles": {
            "interior": interior_rows,
            "periphery": periphery_rows,
        },
        "bestD": best_D,
        "tileProfiles": tile_payload,
        "tileLocation": {str(t): _tile_location(t) for t in tile_profiles},
        "summary": summary_rows,
    }
    html_path = out_dir / _out_name("infer_default_mosaic_tile_profiles.html",
                                    args)
    _write_dashboard(html_path, payload)
    print(f"Wrote {html_path}")
    print(f"Done in {(time.time() - t0) / 60.0:.1f} min")


if __name__ == "__main__":
    main()
