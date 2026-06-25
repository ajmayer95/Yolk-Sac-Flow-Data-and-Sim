#!/usr/bin/env python
"""Aggregate classical solver runs into cross-dataset tables and a dashboard."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from collections import defaultdict
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASET_RE = re.compile(
    r"^pl_d(?P<D0>[^_]+)_a(?P<alpha>[^_]+)_n(?P<noise>\d+)_s(?P<seed>\d+)$"
)
CONFIG_RE = re.compile(
    r"^alpha_(?P<mode>solved|prescribed_(?P<prescribed>[^_]+))"
    r"__(?P<harmonics>h1(?:_h2)?)"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--metrics-root",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "metrics",
    )
    parser.add_argument(
        "--runs-root",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "runs",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "figures" / "solver_comparison",
    )
    parser.add_argument(
        "--include-engines",
        action="store_true",
        help="Keep duplicate CPU/GPU tile runs instead of preferring GPU.",
    )
    return parser.parse_args()


def parse_dataset(name: str) -> dict:
    match = DATASET_RE.match(name)
    if not match:
        return {"D0": math.nan, "alpha": math.nan, "noise": math.nan, "seed": 0}
    raw = match.groupdict()
    return {
        "D0": float(raw["D0"].replace("m", "-")),
        "alpha": float(raw["alpha"].replace("m", "-").replace("p", ".")),
        "noise": int(raw["noise"]) / 100.0,
        "seed": int(raw["seed"]),
    }


def parse_configuration(name: str) -> dict:
    match = CONFIG_RE.match(name)
    if not match:
        return {
            "alpha_mode": "unknown",
            "prescribed_alpha": math.nan,
            "harmonics": "unknown",
            "base_configuration": name,
        }
    raw = match.groupdict()
    prescribed = raw["prescribed"]
    return {
        "alpha_mode": "solved" if raw["mode"] == "solved" else "prescribed",
        "prescribed_alpha": (
            math.nan if prescribed is None else float(prescribed.replace("p", "."))
        ),
        "harmonics": "H1 + H2" if raw["harmonics"] == "h1_h2" else "H1",
        "base_configuration": match.group(0),
    }


def pressure_label(summary: dict) -> tuple[str, str, str]:
    conditioning = summary.get("pressure_conditioning")
    if not conditioning:
        return "none", "No pressure prior", ""
    source = Path(conditioning.get("source", "")).parent.name
    if "edge_local_mlp" in source and "__K0__" in source:
        family = "k0"
        label = "K=0 pressure prior"
    elif "physics_informed_gnn" in source:
        family = "physics_gnn"
        label = "Physics-GNN pressure prior"
    else:
        family = "other"
        label = "Other pressure prior"
    mode = conditioning.get("mode", "unknown")
    return family, f"{label} ({mode})", source


def finite_median(values) -> float:
    array = np.asarray(list(values), dtype=float)
    array = array[np.isfinite(array)]
    return float(np.median(array)) if array.size else math.nan


def load_profile(path: Path, bayesian: bool) -> tuple[list[float], list[float]]:
    if not path.is_file():
        return [], []
    profiles = []
    grid = None
    with np.load(path, allow_pickle=False) as archive:
        for key in archive.files:
            if not key.endswith("__surface"):
                continue
            prefix = key[: -len("__surface")]
            log_key = f"{prefix}__log10_D0_grid"
            if log_key not in archive:
                continue
            surface = np.asarray(archive[key], dtype=float)
            candidate_grid = np.asarray(archive[log_key], dtype=float)
            if surface.ndim != 2 or surface.shape[0] != candidate_grid.size:
                continue
            if bayesian:
                profile_mass = np.nanmax(surface, axis=1)
                peak = np.nanmax(profile_mass)
                profile = -2.0 * np.log(
                    np.maximum(profile_mass, 1.0e-300) / max(peak, 1.0e-300)
                )
            else:
                profile = np.nanmin(surface, axis=1)
                profile = profile - np.nanmin(profile)
            if grid is None:
                grid = candidate_grid
            if np.array_equal(candidate_grid, grid):
                profiles.append(profile)
        if grid is None or not profiles:
            return [], []
    aggregate = np.nanmedian(np.stack(profiles), axis=0)
    return grid.tolist(), aggregate.tolist()


def load_row(metrics_path: Path, metrics_root: Path, runs_root: Path) -> dict:
    summary = json.loads(metrics_path.read_text())
    raw_method = summary["method"]
    engine = "gpu" if raw_method.endswith("_gpu") else "cpu"
    method = raw_method.removesuffix("_gpu")
    family, prior_label, pressure_source = pressure_label(summary)
    configuration = summary["configuration"]
    parsed_config = parse_configuration(configuration)
    dataset = Path(summary["dataset"]).stem
    parsed_dataset = parse_dataset(dataset)
    spatial = summary.get("spatial_results", [])
    aggregate = summary.get("aggregate_metrics", {})
    D0_hats = [item.get("D0_hat", math.nan) for item in spatial]
    alpha_hats = [item.get("alpha_hat", math.nan) for item in spatial]
    run_path = (
        runs_root / raw_method / dataset / configuration / "parameter_surfaces.npz"
    )
    log_grid, profile = load_profile(run_path, method.startswith("bayesian"))
    relative_dashboard = (
        Path("..")
        / raw_method
        / dataset
        / configuration
        / "distensibility_dashboard.html"
    )
    return {
        "dataset": dataset,
        "method": method,
        "method_label": method.replace("_", " ").title(),
        "raw_method": raw_method,
        "engine": engine,
        "configuration": configuration,
        **parsed_config,
        **parsed_dataset,
        "pressure_prior": family,
        "pressure_prior_label": prior_label,
        "pressure_source": pressure_source,
        "n_spatial_problems": int(summary.get("n_spatial_problems", len(spatial))),
        "D0_hat": finite_median(D0_hats),
        "alpha_hat": finite_median(alpha_hats),
        "D0_interval_low": finite_median(
            item.get("D0_interval_low", math.nan) for item in spatial
        ),
        "D0_interval_high": finite_median(
            item.get("D0_interval_high", math.nan) for item in spatial
        ),
        "alpha_interval_low": finite_median(
            item.get("alpha_interval_low", math.nan) for item in spatial
        ),
        "alpha_interval_high": finite_median(
            item.get("alpha_interval_high", math.nan) for item in spatial
        ),
        "median_relative_D0_error": aggregate.get(
            "median_relative_D0_error", math.nan
        ),
        "median_alpha_absolute_error": aggregate.get(
            "median_alpha_absolute_error", math.nan
        ),
        "median_velocity_relative_rmse": aggregate.get(
            "median_held_out_velocity_relative_rmse", math.nan
        ),
        "D0_coverage_rate": aggregate.get("D0_interval_coverage_rate", math.nan),
        "alpha_coverage_rate": aggregate.get(
            "alpha_interval_coverage_rate", math.nan
        ),
        "boundary_hit_rate": aggregate.get("boundary_hit_rate", math.nan),
        "median_D0_interval_width_decades": aggregate.get(
            "median_D0_interval_width_decades", math.nan
        ),
        "median_alpha_interval_width": aggregate.get(
            "median_alpha_interval_width", math.nan
        ),
        "profile_log10_D0": log_grid,
        "profile_delta": profile,
        "dashboard": relative_dashboard.as_posix(),
        "_metrics_path": str(metrics_path.relative_to(metrics_root.parent.parent)),
    }


def deduplicate(rows: list[dict], include_engines: bool) -> list[dict]:
    if include_engines:
        return rows
    selected = {}
    for row in rows:
        key = (
            row["method"],
            row["dataset"],
            row["base_configuration"],
            row["pressure_prior"],
            row["pressure_source"],
        )
        previous = selected.get(key)
        if previous is None or (row["engine"] == "gpu" and previous["engine"] != "gpu"):
            selected[key] = row
    return list(selected.values())


def write_csv(path: Path, rows: list[dict]) -> None:
    fields = [key for key in rows[0] if not key.startswith("_") and not key.startswith("profile_")]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def grouped_summary(rows: list[dict]) -> list[dict]:
    groups = defaultdict(list)
    for row in rows:
        key = (
            row["method"],
            row["pressure_prior_label"],
            row["alpha_mode"],
            row["prescribed_alpha"],
            row["harmonics"],
        )
        groups[key].append(row)
    output = []
    for key, members in sorted(groups.items(), key=lambda item: str(item[0])):
        output.append(
            {
                "method": key[0],
                "pressure_prior": key[1],
                "alpha_mode": key[2],
                "prescribed_alpha": key[3],
                "harmonics": key[4],
                "n_datasets": len({row["dataset"] for row in members}),
                "n_runs": len(members),
                "median_relative_D0_error": finite_median(
                    row["median_relative_D0_error"] for row in members
                ),
                "median_alpha_absolute_error": finite_median(
                    row["median_alpha_absolute_error"] for row in members
                ),
                "median_velocity_relative_rmse": finite_median(
                    row["median_velocity_relative_rmse"] for row in members
                ),
                "mean_D0_coverage_rate": float(
                    np.nanmean([row["D0_coverage_rate"] for row in members])
                ),
                "mean_alpha_coverage_rate": float(
                    np.nanmean([row["alpha_coverage_rate"] for row in members])
                ),
                "mean_boundary_hit_rate": float(
                    np.nanmean([row["boundary_hit_rate"] for row in members])
                ),
            }
        )
    return output


DASHBOARD = r"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Distensibility solver comparison</title>
<style>
:root{--bg:#f3f6f8;--panel:#fff;--ink:#18232d;--muted:#637282;--line:#d7e0e8}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:14px/1.4 system-ui,sans-serif}
header{padding:18px 22px;background:#202b36;color:#fff}main{padding:14px;display:grid;gap:14px}
section{background:var(--panel);border:1px solid var(--line);border-radius:8px;padding:12px}
.filters{display:flex;gap:10px;flex-wrap:wrap}.filters label,.selector{display:grid;gap:4px;color:var(--muted);font-size:12px}
select{min-width:145px;padding:7px}.cards{display:grid;grid-template-columns:repeat(6,minmax(120px,1fr));gap:8px}
.card{border:1px solid var(--line);border-radius:7px;padding:9px}.card b{display:block;color:var(--muted);font-size:11px}.card span{font-size:18px;font-weight:650}
.grid{display:grid;grid-template-columns:1fr 1fr;gap:14px}canvas{width:100%;height:410px;border:1px solid var(--line);border-radius:6px}
.scroll{max-height:650px;overflow:auto}table{width:100%;border-collapse:collapse;font-size:12px}
th,td{padding:6px;border-bottom:1px solid #edf1f5;text-align:right;white-space:nowrap}th{position:sticky;top:0;background:#fff}
th:first-child,td:first-child{text-align:left}a{color:#1769aa}.note{color:var(--muted)}
@media(max-width:1000px){.grid,.cards{grid-template-columns:1fr}}
</style></head><body><header><h1>Classical distensibility solver comparison</h1>
<div>Linear/Bayesian × tile/mosaic × pressure-prior condition across the synthetic dataset grid.</div></header><main>
<section><div class="filters">
<label>Method<select id="method"></select></label><label>Pressure prior<select id="prior"></select></label>
<label>Alpha mode<select id="alphaMode"></select></label><label>Prescribed alpha<select id="prescribed"></select></label>
<label>Harmonics<select id="harmonics"></select></label><label>True D0<select id="D0"></select></label>
<label>True alpha<select id="alpha"></select></label><label>Noise<select id="noise"></select></label>
</div></section>
<section><div class="cards" id="cards"></div></section>
<div class="grid"><section><h2>Predicted D0 parity</h2><canvas id="D0Parity"></canvas></section>
<section><h2>Predicted alpha parity</h2><canvas id="alphaParity"></canvas></section></div>
<div class="grid"><section><h2>Distensibility profiles</h2><canvas id="profiles"></canvas>
<p class="note">Curves are median tile profiles for tile methods and the whole-mosaic profile otherwise.</p></section>
<section><h2>Selected profile</h2><label class="selector">Run<select id="selected"></select></label>
<canvas id="selectedProfile"></canvas></section></div>
<section><h2>Available results</h2><div class="scroll"><table id="results"></table></div></section>
<section><h2>Across-dataset summary</h2><div class="scroll"><table id="summary"></table></div></section>
</main><script id="payload" type="application/json">__PAYLOAD__</script>
<script>
const payload=JSON.parse(document.getElementById("payload").textContent),all=payload.rows,groups=payload.groups,$=id=>document.getElementById(id);
const colors={linear_tile:"#1f77b4",linear_mosaic:"#17a2b8",bayesian_tile:"#d95f02",bayesian_mosaic:"#8c564b"};
const fmt=x=>x!==null&&x!==""&&Number.isFinite(+x)?((Math.abs(+x)>=1000||(Math.abs(+x)>0&&Math.abs(+x)<.001))?(+x).toExponential(2):(+x).toFixed(3).replace(/\.?0+$/,"")):"n/a";
function opts(id,values,label=x=>x){$(id).innerHTML="<option value='all'>all</option>"+values.map(x=>`<option value="${x}">${label(x)}</option>`).join("")}
opts("method",[...new Set(all.map(r=>r.method))],x=>all.find(r=>r.method===x).method_label);
opts("prior",[...new Set(all.map(r=>r.pressure_prior))],x=>all.find(r=>r.pressure_prior===x).pressure_prior_label);
opts("alphaMode",[...new Set(all.map(r=>r.alpha_mode))]);opts("prescribed",[...new Set(all.map(r=>r.prescribed_alpha).filter(Number.isFinite))].sort((a,b)=>a-b));
opts("harmonics",[...new Set(all.map(r=>r.harmonics))]);opts("D0",[...new Set(all.map(r=>r.D0))].sort((a,b)=>a-b),x=>(+x).toExponential(0));
opts("alpha",[...new Set(all.map(r=>r.alpha))].sort((a,b)=>a-b));opts("noise",[...new Set(all.map(r=>r.noise))].sort((a,b)=>a-b),x=>`${Math.round(100*x)}%`);
function filtered(){return all.filter(r=>
 ($("method").value==="all"||r.method===$("method").value)&&($("prior").value==="all"||r.pressure_prior===$("prior").value)&&
 ($("alphaMode").value==="all"||r.alpha_mode===$("alphaMode").value)&&($("prescribed").value==="all"||r.prescribed_alpha==+$("prescribed").value)&&
 ($("harmonics").value==="all"||r.harmonics===$("harmonics").value)&&($("D0").value==="all"||r.D0==+$("D0").value)&&
 ($("alpha").value==="all"||r.alpha==+$("alpha").value)&&($("noise").value==="all"||r.noise==+$("noise").value));}
function canvas(id){const e=$(id),d=devicePixelRatio||1,r=e.getBoundingClientRect();e.width=Math.max(400,r.width*d);e.height=410*d;const c=e.getContext("2d");c.setTransform(d,0,0,d,0,0);return{c,w:e.width/d,h:e.height/d}}
function formatTick(x,log=false){if(!Number.isFinite(x))return"";if(log)return`10^${fmt(x)}`;return fmt(x)}
function drawTicks(c,xs,ys,scaleX,scaleY,L,T,W,H,logX=false,logY=false){
 c.strokeStyle="#b7c1cb";c.fillStyle="#637282";c.textAlign="center";c.textBaseline="top";
 xs.forEach(v=>{const x=scaleX(v);c.beginPath();c.moveTo(x,T+H);c.lineTo(x,T+H+4);c.stroke();c.fillText(formatTick(v,logX),x,T+H+6)});
 c.save();c.textAlign="right";c.textBaseline="middle";
 ys.forEach(v=>{const y=scaleY(v);c.beginPath();c.moveTo(L-4,y);c.lineTo(L,y);c.stroke();c.fillText(formatTick(v,logY),L-6,y)});
 c.restore();
}
function axes(c,L,T,W,H,xlabel,ylabel){c.strokeStyle="#687684";c.strokeRect(L,T,W,H);c.fillStyle="#263442";c.textAlign="center";c.fillText(xlabel,L+W/2,T+H+38);c.save();c.translate(17,T+H/2);c.rotate(-Math.PI/2);c.fillText(ylabel,0,0);c.restore();c.textAlign="left"}
function parity(id,rows,xKey,yKey,label,log=false){const {c,w,h}=canvas(id);c.clearRect(0,0,w,h);const good=rows.filter(r=>Number.isFinite(r[xKey])&&Number.isFinite(r[yKey])&&(!log||(r[xKey]>0&&r[yKey]>0)));if(!good.length)return;
 const values=good.flatMap(r=>[log?Math.log10(r[xKey]):r[xKey],log?Math.log10(r[yKey]):r[yKey]]),lo=Math.min(...values),hi=Math.max(...values),L=65,T=20,R=20,B=58,W=w-L-R,H=h-T-B,s=v=>L+(v-lo)/(hi-lo||1)*W,sy=v=>T+(hi-v)/(hi-lo||1)*H;axes(c,L,T,W,H,`true ${label}`,`predicted ${label}`);
 c.strokeStyle="#555";c.setLineDash([5,4]);c.beginPath();c.moveTo(s(lo),sy(lo));c.lineTo(s(hi),sy(hi));c.stroke();c.setLineDash([]);
 const xticks=[0,.25,.5,.75,1].map(t=>lo+t*(hi-lo));const yticks=[0,.25,.5,.75,1].map(t=>lo+t*(hi-lo));
 drawTicks(c,xticks,yticks,s,sy,L,T,W,H,log,log);
 good.forEach(r=>{const x=log?Math.log10(r[xKey]):r[xKey],y=log?Math.log10(r[yKey]):r[yKey];c.fillStyle=colors[r.method]||"#444";c.globalAlpha=.65;c.beginPath();c.arc(s(x),sy(y),4,0,2*Math.PI);c.fill()});c.globalAlpha=1}
function profilePlot(id,rows,single=false){const {c,w,h}=canvas(id);c.clearRect(0,0,w,h);const good=rows.filter(r=>r.profile_log10_D0.length&&r.profile_delta.length);if(!good.length)return;const L=68,T=20,R=20,B=58,W=w-L-R,H=h-T-B,xlo=Math.min(...good.flatMap(r=>r.profile_log10_D0)),xhi=Math.max(...good.flatMap(r=>r.profile_log10_D0)),yhi=10,sx=x=>L+(x-xlo)/(xhi-xlo||1)*W,sy=y=>T+(yhi-Math.min(y,yhi))/yhi*H;axes(c,L,T,W,H,"log10(D0 [1/Pa])","profile Δχ² / -2 log ratio");
 const xTicks=[0,.25,.5,.75,1].map(t=>xlo+t*(xhi-xlo));const yTicks=[0,2,4,6,8,10];
 drawTicks(c,xTicks,yTicks,sx,sy,L,T,W,H,false,false);
 good.forEach(r=>{c.strokeStyle=colors[r.method]||"#444";c.globalAlpha=single?1:.15;c.lineWidth=single?2.5:1;c.beginPath();r.profile_delta.forEach((y,i)=>i?c.lineTo(sx(r.profile_log10_D0[i]),sy(y)):c.moveTo(sx(r.profile_log10_D0[i]),sy(y)));c.stroke()});c.globalAlpha=1;
 for(const [v,color,dash,text] of [[3.8414588,"#d62728",[6,4],"χ²₁ 95% = 3.84"],[5.9914645,"#111",[3,4],"χ²₂ 95% = 5.99"]]){c.strokeStyle=color;c.setLineDash(dash);c.beginPath();c.moveTo(L,sy(v));c.lineTo(L+W,sy(v));c.stroke();c.setLineDash([]);c.fillStyle=color;c.fillText(text,L+8,sy(v)-5)}
 if(single&&good[0].D0>0){const x=sx(Math.log10(good[0].D0));c.strokeStyle="#d62728";c.setLineDash([6,4]);c.beginPath();c.moveTo(x,T);c.lineTo(x,T+H);c.stroke();c.setLineDash([])}}
function updateSelected(rows){const s=$("selected"),old=s.value;s.innerHTML=rows.map((r,i)=>`<option value="${i}">${r.dataset} · ${r.method_label} · ${r.pressure_prior_label} · ${r.alpha_mode} · ${r.harmonics}</option>`).join("");if([...s.options].some(o=>o.value===old))s.value=old;selected(rows)}
function selected(rows){const r=rows[+$("selected").value]||rows[0];profilePlot("selectedProfile",r?[r]:[],true)}
function table(rows){$("results").innerHTML="<thead><tr><th>dataset</th><th>method</th><th>prior</th><th>alpha mode</th><th>harmonics</th><th>true D0</th><th>D0 hat</th><th>true alpha</th><th>alpha hat</th><th>D0 rel. error</th><th>alpha error</th><th>velocity RMSE</th><th>D0 coverage</th><th>alpha coverage</th></tr></thead><tbody>"+rows.map(r=>`<tr><td><a href="${r.dashboard}">${r.dataset}</a></td><td>${r.method_label}</td><td>${r.pressure_prior_label}</td><td>${r.alpha_mode}${Number.isFinite(r.prescribed_alpha)?` (${r.prescribed_alpha})`:""}</td><td>${r.harmonics}</td><td>${(+r.D0).toExponential(1)}</td><td>${fmt(r.D0_hat)}</td><td>${r.alpha}</td><td>${fmt(r.alpha_hat)}</td><td>${fmt(r.median_relative_D0_error)}</td><td>${fmt(r.median_alpha_absolute_error)}</td><td>${fmt(r.median_velocity_relative_rmse)}</td><td>${fmt(r.D0_coverage_rate)}</td><td>${fmt(r.alpha_coverage_rate)}</td></tr>`).join("")+"</tbody>"}
function summaryTable(rows){const buckets=new Map(),median=x=>{x=x.filter(Number.isFinite).sort((a,b)=>a-b);return x.length?x[Math.floor(x.length/2)]:NaN},mean=x=>{x=x.filter(Number.isFinite);return x.length?x.reduce((a,b)=>a+b,0)/x.length:NaN};for(const r of rows){const key=[r.method,r.pressure_prior_label,r.alpha_mode,r.prescribed_alpha,r.harmonics].join("|");if(!buckets.has(key))buckets.set(key,[]);buckets.get(key).push(r)}const g=[...buckets.values()].map(x=>({method:x[0].method,prior:x[0].pressure_prior_label,alpha_mode:x[0].alpha_mode,prescribed_alpha:x[0].prescribed_alpha,harmonics:x[0].harmonics,n_datasets:new Set(x.map(r=>r.dataset)).size,D0_error:median(x.map(r=>r.median_relative_D0_error)),alpha_error:median(x.map(r=>r.median_alpha_absolute_error)),velocity:median(x.map(r=>r.median_velocity_relative_rmse)),D0_coverage:mean(x.map(r=>r.D0_coverage_rate)),alpha_coverage:mean(x.map(r=>r.alpha_coverage_rate)),boundary:mean(x.map(r=>r.boundary_hit_rate))}));$("summary").innerHTML="<thead><tr><th>method</th><th>prior</th><th>alpha mode</th><th>alpha</th><th>harmonics</th><th>datasets</th><th>D0 rel. error</th><th>alpha error</th><th>velocity RMSE</th><th>D0 coverage</th><th>alpha coverage</th><th>boundary hits</th></tr></thead><tbody>"+g.map(r=>`<tr><td>${r.method.replaceAll("_"," ")}</td><td>${r.prior}</td><td>${r.alpha_mode}</td><td>${fmt(r.prescribed_alpha)}</td><td>${r.harmonics}</td><td>${r.n_datasets}</td><td>${fmt(r.D0_error)}</td><td>${fmt(r.alpha_error)}</td><td>${fmt(r.velocity)}</td><td>${fmt(r.D0_coverage)}</td><td>${fmt(r.alpha_coverage)}</td><td>${fmt(r.boundary)}</td></tr>`).join("")+"</tbody>"}
function render(){const rows=filtered(),med=k=>{const x=rows.map(r=>r[k]).filter(Number.isFinite).sort((a,b)=>a-b);return x.length?x[Math.floor(x.length/2)]:NaN};$("cards").innerHTML=[["Runs",rows.length],["Datasets",new Set(rows.map(r=>r.dataset)).size],["Median D0 error",fmt(med("median_relative_D0_error"))],["Median alpha error",fmt(med("median_alpha_absolute_error"))],["Median velocity RMSE",fmt(med("median_velocity_relative_rmse"))],["Mean boundary hits",fmt(rows.reduce((a,r)=>a+(Number.isFinite(r.boundary_hit_rate)?r.boundary_hit_rate:0),0)/(rows.length||1))]].map(([k,v])=>`<div class=card><b>${k}</b><span>${v}</span></div>`).join("");parity("D0Parity",rows,"D0","D0_hat","D0",true);parity("alphaParity",rows,"alpha","alpha_hat","alpha");profilePlot("profiles",rows);updateSelected(rows);table(rows);summaryTable(rows)}
for(const id of ["method","prior","alphaMode","prescribed","harmonics","D0","alpha","noise"])$(id).addEventListener("change",render);$("selected").addEventListener("change",()=>selected(filtered()));window.addEventListener("resize",render);render();
</script></body></html>"""


def sanitize_json(value):
    if isinstance(value, dict):
        return {key: sanitize_json(item) for key, item in value.items()}
    if isinstance(value, list):
        return [sanitize_json(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def main() -> None:
    args = parse_args()
    metrics_root = args.metrics_root.expanduser().resolve()
    runs_root = args.runs_root.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    rows = []
    for path in sorted(metrics_root.glob("*/*/*/summary_metrics.json")):
        try:
            row = load_row(path, metrics_root, runs_root)
        except Exception as error:
            print(f"Skipping {path}: {error}")
            continue
        if row["method"] in {
            "linear_tile",
            "linear_mosaic",
            "bayesian_tile",
            "bayesian_mosaic",
        }:
            rows.append(row)
    rows = deduplicate(rows, args.include_engines)
    rows.sort(
        key=lambda row: (
            row["dataset"],
            row["method"],
            row["pressure_prior"],
            row["configuration"],
        )
    )
    if not rows:
        raise RuntimeError(f"No solver summaries found below {metrics_root}")
    groups = grouped_summary(rows)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(output_dir / "all_solver_results.csv", rows)
    write_csv(output_dir / "configuration_summary.csv", groups)
    payload = json.dumps(
        sanitize_json({"rows": rows, "groups": groups}), separators=(",", ":")
    ).replace("</", "<\\/")
    (output_dir / "dashboard.html").write_text(
        DASHBOARD.replace("__PAYLOAD__", payload)
    )
    manifest = {
        "n_runs": len(rows),
        "n_datasets": len({row["dataset"] for row in rows}),
        "methods": sorted({row["method"] for row in rows}),
        "pressure_priors": sorted({row["pressure_prior"] for row in rows}),
        "missing_dataset_warning": len({row["dataset"] for row in rows}) < 36,
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Wrote {output_dir / 'dashboard.html'}")
    print(f"Wrote {output_dir / 'all_solver_results.csv'}")
    print(f"Wrote {output_dir / 'configuration_summary.csv'}")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
