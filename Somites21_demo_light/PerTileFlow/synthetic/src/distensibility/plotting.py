"""Static maps and standalone interactive dashboards for solver results."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Sequence

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection

from .io import VascularDataset
from models._shared import SolverResult


def _segments(dataset: VascularDataset, edge_mask: np.ndarray):
    edges = np.flatnonzero(edge_mask)
    xy = dataset.node_xy_px
    source = dataset.edge_source_index[edges]
    target = dataset.edge_target_index[edges]
    finite = np.isfinite(xy[source]).all(axis=1) & np.isfinite(
        xy[target]
    ).all(axis=1)
    edges = edges[finite]
    segments = np.stack(
        [xy[dataset.edge_source_index[edges]], xy[dataset.edge_target_index[edges]]],
        axis=1,
    )
    return edges, segments


def plot_pressure_map(
    path: Path,
    dataset: VascularDataset,
    predicted_pressure: np.ndarray,
    harmonic: int,
    title: str,
) -> None:
    """Plot true and inferred nodal pressure amplitudes."""
    valid = np.isfinite(predicted_pressure[:, harmonic])
    xy = dataset.node_xy_px
    valid &= np.isfinite(xy).all(axis=1)
    truth = np.abs(dataset.pressure_true_pa[:, harmonic])
    prediction = np.abs(predicted_pressure[:, harmonic])
    vmax = float(
        np.nanpercentile(
            np.concatenate([truth[valid], prediction[valid]]), 98
        )
    )
    vmax = max(vmax, 1.0e-12)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5), constrained_layout=True)
    for axis, values, label in (
        (axes[0], truth, "Ground-truth pressure amplitude"),
        (axes[1], prediction, "Inferred pressure amplitude"),
    ):
        scatter = axis.scatter(
            xy[valid, 0],
            xy[valid, 1],
            c=values[valid],
            s=4,
            cmap="viridis",
            vmin=0,
            vmax=vmax,
            linewidths=0,
        )
        axis.invert_yaxis()
        axis.set_aspect("equal")
        axis.set_title(label)
        axis.set_xlabel("mosaic x [px]")
        axis.set_ylabel("mosaic y [px]")
        fig.colorbar(scatter, ax=axis, label="|P| [Pa]")
    fig.suptitle(f"{title} — H{harmonic}")
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_flow_error_map(
    path: Path,
    dataset: VascularDataset,
    predicted_velocity: np.ndarray,
    harmonics: Sequence[int],
    title: str,
) -> None:
    """Plot edge-wise RMS relative velocity error across fitted harmonics."""
    finite = np.ones(dataset.n_edges, dtype=bool)
    squared_error = np.zeros(dataset.n_edges, dtype=float)
    squared_truth = np.zeros(dataset.n_edges, dtype=float)
    for harmonic in harmonics:
        finite &= np.isfinite(predicted_velocity[:, harmonic])
        squared_error += np.abs(
            predicted_velocity[:, harmonic]
            - dataset.velocity_true_m_s[:, harmonic]
        ) ** 2
        squared_truth += np.abs(dataset.velocity_true_m_s[:, harmonic]) ** 2
    edges, segments = _segments(dataset, finite)
    relative = np.sqrt(squared_error[edges]) / np.maximum(
        np.sqrt(squared_truth[edges]), 1.0e-15
    )
    vmax = max(float(np.nanpercentile(relative, 95)), 1.0e-6)
    fig, axis = plt.subplots(figsize=(9, 7), constrained_layout=True)
    collection = LineCollection(
        segments,
        array=np.clip(relative, 0, vmax),
        cmap="magma",
        linewidths=1.0,
    )
    collection.set_clim(0, vmax)
    axis.add_collection(collection)
    axis.autoscale()
    axis.invert_yaxis()
    axis.set_aspect("equal")
    axis.set_title(title)
    axis.set_xlabel("mosaic x [px]")
    axis.set_ylabel("mosaic y [px]")
    fig.colorbar(
        collection,
        ax=axis,
        label="RMS relative velocity error (clipped at 95th percentile)",
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180)
    plt.close(fig)


DASHBOARD_TEMPLATE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Distensibility Solver Dashboard</title>
<style>
:root { --bg:#f4f6f8; --panel:#fff; --ink:#19232d; --muted:#637282; --line:#d7e0e8; }
* { box-sizing:border-box; }
body { margin:0; background:var(--bg); color:var(--ink); font:14px/1.4 system-ui,sans-serif; }
header { padding:14px 18px; background:#202b36; color:#fff; display:flex; justify-content:space-between; }
main { display:grid; gap:14px; padding:14px; }
section { background:var(--panel); border:1px solid var(--line); border-radius:8px; padding:12px; }
.controls { display:flex; gap:14px; align-items:end; flex-wrap:wrap; }
label { display:grid; gap:4px; color:var(--muted); font-size:12px; }
select { min-width:220px; padding:7px; border:1px solid #b9c6d2; border-radius:6px; background:#fff; }
.cards { display:grid; grid-template-columns:repeat(6,minmax(120px,1fr)); gap:8px; }
.card { border:1px solid var(--line); border-radius:7px; padding:9px; }
.value { font-size:18px; font-weight:650; }
.label { color:var(--muted); font-size:11px; }
.grid { display:grid; grid-template-columns:1fr 1fr; gap:14px; }
canvas { width:100%; height:430px; border:1px solid var(--line); border-radius:6px; }
table { width:100%; border-collapse:collapse; font-size:12px; }
th,td { padding:6px; border-bottom:1px solid #edf1f5; text-align:right; }
th:first-child,td:first-child { text-align:left; }
@media(max-width:900px){ .grid,.cards{grid-template-columns:1fr;} }
</style>
</head>
<body>
<header><strong>Distensibility Solver Dashboard</strong><span id="meta"></span></header>
<main>
<section><div class="controls"><label>Spatial problem<select id="problem"></select></label></div></section>
<section><div class="cards" id="cards"></div></section>
<div class="grid">
<section><h3>D0–alpha surface</h3><canvas id="surface"></canvas></section>
<section><h3>Profile over D0</h3><canvas id="profile"></canvas></section>
</div>
<section><h3>Profile over D0 — y-axis limited to 10</h3><canvas id="profile10"></canvas></section>
<section><h3>Problem summary</h3><table id="summary"></table></section>
</main>
<script id="payload" type="application/json">__DATA__</script>
<script>
const data=JSON.parse(document.getElementById("payload").textContent);
const byId=id=>document.getElementById(id);
const fmt=x=>{x=Number(x);if(!Number.isFinite(x))return"n/a";return (Math.abs(x)>=1000||(Math.abs(x)>0&&Math.abs(x)<.001))?x.toExponential(2):x.toFixed(3).replace(/\.?0+$/,"");};
const selector=byId("problem");
for(const row of data.results){const o=document.createElement("option");o.value=row.problem_name;o.textContent=row.problem_name;selector.appendChild(o);}
byId("meta").textContent=`${data.method} · ${data.dataset}`;
function canvas(id){const el=byId(id),d=window.devicePixelRatio||1,r=el.getBoundingClientRect();el.width=Math.max(420,r.width*d);el.height=Math.max(300,r.height*d);const c=el.getContext("2d");c.setTransform(d,0,0,d,0,0);return{c,w:el.width/d,h:el.height/d};}
function ticks(c,values,scale,horizontal,L,T,W,H,formatter){
  c.font="11px system-ui";c.fillStyle="#52606d";c.strokeStyle="#8996a3";
  values.forEach(v=>{const p=scale(v);c.beginPath();
    if(horizontal){c.moveTo(p,T+H);c.lineTo(p,T+H+5);c.stroke();c.textAlign="center";c.fillText(formatter(v),p,T+H+18);}
    else{c.moveTo(L-5,p);c.lineTo(L,p);c.stroke();c.textAlign="right";c.fillText(formatter(v),L-8,p+4);}
  });c.textAlign="left";
}
function contour(c,z,xs,ys,level,sx,sy){
  if(xs.length<2||ys.length<2)return;
  const cross=(x0,y0,v0,x1,y1,v1)=>{
    const d=v1-v0,t=Math.abs(d)<1e-15?.5:(level-v0)/d;
    return[sx(x0+t*(x1-x0)),sy(y0+t*(y1-y0))];
  };
  c.beginPath();
  for(let iy=0;iy<ys.length-1;iy++)for(let ix=0;ix<xs.length-1;ix++){
    const x0=xs[ix],x1=xs[ix+1],y0=ys[iy],y1=ys[iy+1];
    const v=[z[iy][ix],z[iy][ix+1],z[iy+1][ix+1],z[iy+1][ix]];
    const p=[],edges=[
      [x0,y0,v[0],x1,y0,v[1]],[x1,y0,v[1],x1,y1,v[2]],
      [x1,y1,v[2],x0,y1,v[3]],[x0,y1,v[3],x0,y0,v[0]]
    ];
    for(const e of edges)if((e[2]<level&&e[5]>=level)||(e[2]>=level&&e[5]<level))p.push(cross(...e));
    if(p.length===2){c.moveTo(...p[0]);c.lineTo(...p[1]);}
    else if(p.length===4){c.moveTo(...p[0]);c.lineTo(...p[1]);c.moveTo(...p[2]);c.lineTo(...p[3]);}
  }
  c.stroke();
}
function surface(row){
  const {c,w,h}=canvas("surface");c.clearRect(0,0,w,h);
  const z=row.surface,xs=row.alpha_grid,ys=row.log10_D0_grid,nx=xs.length,ny=ys.length;
  let vals=z.flat().filter(Number.isFinite),lo=Math.min(...vals),hi=Math.max(...vals);if(hi===lo)hi=lo+1;
  const L=70,T=24,R=24,B=62,W=w-L-R,H=h-T-B,cw=W/nx,ch=H/ny;
  const sx=x=>L+(x-xs[0])/(xs.at(-1)-xs[0]||1)*W;
  const sy=y=>T+(ys.at(-1)-y)/(ys.at(-1)-ys[0]||1)*H;
  for(let iy=0;iy<ny;iy++)for(let ix=0;ix<nx;ix++){const q=(z[iy][ix]-lo)/(hi-lo),hue=data.method.includes("bayesian")?220:25;c.fillStyle=`hsl(${hue},70%,${92-65*q}%)`;c.fillRect(L+ix*cw,T+(ny-1-iy)*ch,cw+1,ch+1);}
  c.strokeStyle="#556";c.strokeRect(L,T,W,H);
  ticks(c,[xs[0],xs[Math.floor((nx-1)/2)],xs.at(-1)],sx,true,L,T,W,H,v=>fmt(v));
  ticks(c,[ys[0],ys[Math.floor((ny-1)/2)],ys.at(-1)],sy,false,L,T,W,H,v=>fmt(v));
  c.fillStyle="#263442";c.textAlign="center";c.fillText("alpha",L+W/2,h-10);
  c.save();c.translate(16,T+H/2);c.rotate(-Math.PI/2);c.fillText("log10(D0 [1/Pa])",0,0);c.restore();c.textAlign="left";
  const jointCutoff=5.991464547107979,bayes=data.method.includes("bayesian");
  const delta=bayes?z.map(r=>r.map(v=>-2*Math.log(Math.max(v,1e-300)/Math.max(hi,1e-300)))):z;
  c.save();c.beginPath();c.rect(L,T,W,H);c.clip();
  c.strokeStyle="#111";c.lineWidth=2;c.setLineDash([8,4]);contour(c,delta,xs,ys,jointCutoff,sx,sy);c.setLineDash([]);c.restore();
  c.fillStyle="#111";c.fillText("joint 95% χ²₂ contour = 5.99",L+8,T+14);
  const tx=sx(row.metrics.alpha_true),ty=sy(Math.log10(row.metrics.D0_true));
  c.strokeStyle="#d62728";c.lineWidth=1.5;c.setLineDash([5,4]);c.beginPath();c.moveTo(tx,T);c.lineTo(tx,T+H);c.moveTo(L,ty);c.lineTo(L+W,ty);c.stroke();c.setLineDash([]);
  c.fillStyle="#d62728";c.beginPath();c.arc(tx,ty,5,0,2*Math.PI);c.fill();c.strokeStyle="#fff";c.lineWidth=1.5;c.stroke();
  c.fillStyle="#d62728";c.fillText("true (D0, alpha)",Math.min(tx+8,L+W-100),Math.max(ty-8,T+12));
}
function profile(row,id,yLimit=null){
  const {c,w,h}=canvas(id);c.clearRect(0,0,w,h);
  const xs=row.log10_D0_grid,bayes=data.method.includes("bayesian");
  let ys;
  if(bayes){
    const p=row.surface.map(r=>Math.max(...r)),peak=Math.max(...p);
    ys=p.map(v=>-2*Math.log(Math.max(v,1e-300)/Math.max(peak,1e-300)));
  }else{
    ys=row.surface.map(r=>Math.min(...r));
  }
  let ymin=0,ymax=Math.max(...ys.filter(Number.isFinite));if(yLimit!==null)ymax=Math.min(ymax,yLimit);if(ymax<=ymin)ymax=yLimit||1;
  const L=76,T=24,R=22,B=62,W=w-L-R,H=h-T-B;
  const sx=x=>L+(x-xs[0])/(xs.at(-1)-xs[0]||1)*W,sy=y=>T+(ymax-y)/(ymax-ymin)*H;
  c.save();c.beginPath();c.rect(L,T,W,H);c.clip();
  c.strokeStyle="#2868b7";c.lineWidth=2;c.beginPath();ys.forEach((y,i)=>{const yy=Math.min(Math.max(y,ymin),ymax);i?c.lineTo(sx(xs[i]),sy(yy)):c.moveTo(sx(xs[i]),sy(yy));});c.stroke();
  const cutoff=3.841458820694124,jointCutoff=5.991464547107979;
  if(cutoff<=ymax){const cy=sy(cutoff);c.strokeStyle="#d62728";c.lineWidth=1.5;c.setLineDash([6,4]);c.beginPath();c.moveTo(L,cy);c.lineTo(L+W,cy);c.stroke();c.setLineDash([]);}
  if(jointCutoff<=ymax){const cy=sy(jointCutoff);c.strokeStyle="#111";c.lineWidth=1.5;c.setLineDash([3,4]);c.beginPath();c.moveTo(L,cy);c.lineTo(L+W,cy);c.stroke();c.setLineDash([]);}
  c.restore();
  c.strokeStyle="#667";c.lineWidth=1;c.strokeRect(L,T,W,H);
  ticks(c,[xs[0],xs[Math.floor((xs.length-1)/2)],xs.at(-1)],sx,true,L,T,W,H,v=>fmt(v));
  ticks(c,[ymin,(ymin+ymax)/2,ymax],sy,false,L,T,W,H,v=>fmt(v));
  c.fillStyle="#263442";c.textAlign="center";c.fillText("log10(D0 [1/Pa])",L+W/2,h-10);
  c.save();c.translate(17,T+H/2);c.rotate(-Math.PI/2);c.fillText(bayes?"-2 log profile posterior ratio":"profile delta chi-square",0,0);c.restore();c.textAlign="left";
  if(cutoff<=ymax){c.fillStyle="#d62728";c.fillText("95% χ²₁ cutoff = 3.84",L+8,sy(cutoff)-7);}
  if(jointCutoff<=ymax){c.fillStyle="#111";c.fillText("joint 95% χ²₂ reference = 5.99",L+8,sy(jointCutoff)-7);}
  const trueX=sx(Math.log10(row.metrics.D0_true));
  c.strokeStyle="#d62728";c.lineWidth=1.5;c.setLineDash([6,4]);c.beginPath();c.moveTo(trueX,T);c.lineTo(trueX,T+H);c.stroke();c.setLineDash([]);
  c.fillStyle="#d62728";c.fillText("true D0",Math.min(trueX+6,L+W-52),T+14);
  const low=row.metrics.D0_interval_low,high=row.metrics.D0_interval_high;
  if(Number.isFinite(low)&&Number.isFinite(high)&&low>0&&high>0){
    c.fillStyle="rgba(31,119,180,.12)";const x0=sx(Math.log10(low)),x1=sx(Math.log10(high));c.fillRect(Math.min(x0,x1),T,Math.abs(x1-x0),H);
    c.fillStyle="#2868b7";c.fillText(bayes?"95% credible interval":"95% profile interval",Math.max(L+8,Math.min(x0,x1)+5),T+32);
  }
}
function render(){const row=data.results.find(x=>x.problem_name===selector.value);const m=row.metrics;const cards=[["D0 true",m.D0_true],["D0 estimate",m.D0_hat],["alpha true",m.alpha_true],["alpha estimate",m.alpha_hat],["D0 error",m.relative_D0_error],["held-out RMSE",m.held_out_velocity_relative_rmse]];byId("cards").innerHTML=cards.map(([k,v])=>`<div class=card><div class=label>${k}</div><div class=value>${fmt(v)}</div></div>`).join("");surface(row);profile(row,"profile");profile(row,"profile10",10);byId("summary").innerHTML="<tbody>"+Object.entries(m).map(([k,v])=>`<tr><td>${k}</td><td>${Array.isArray(v)?v.join(", "):fmt(v)}</td></tr>`).join("")+"</tbody>";}
selector.addEventListener("change",render);window.addEventListener("resize",render);render();
</script></body></html>"""


def write_dashboard(
    path: Path,
    dataset: VascularDataset,
    method: str,
    results: Sequence[SolverResult],
) -> None:
    payload = {
        "dataset": dataset.path.stem,
        "method": method,
        "results": [
            {
                "problem_name": result.problem_name,
                "log10_D0_grid": result.log10_D0_grid.tolist(),
                "alpha_grid": result.alpha_grid.tolist(),
                "surface": result.surface.tolist(),
                "metrics": result.metrics,
            }
            for result in results
        ],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        DASHBOARD_TEMPLATE.replace(
            "__DATA__", json.dumps(payload).replace("</", "<\\/")
        )
    )
