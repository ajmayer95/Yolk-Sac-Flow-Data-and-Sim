#!/usr/bin/env python
"""Build an interactive dashboard over all 108 selected GNN configurations."""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_ROOT = PROJECT_ROOT / "outputs" / "figures" / "gnn_comparison"
MODELS = ("physics_informed_gnn", "vanilla_gcn", "edge_local_mlp")
DATASET_RE = re.compile(
    r"^pl_d(?P<D0>[^_]+)_a(?P<alpha>[^_]+)_n(?P<noise>\d+)_s(?P<seed>\d+)$"
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--comparison-root", type=Path, default=DEFAULT_ROOT
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Default: <comparison-root>/all_108_dashboard.html",
    )
    return parser.parse_args()


def parse_dataset(name: str) -> dict:
    match = DATASET_RE.match(name)
    if not match:
        raise ValueError(f"Unrecognized synthetic dataset name: {name}")
    raw = match.groupdict()
    return {
        "D0": float(raw["D0"].replace("m", "-")),
        "alpha": float(raw["alpha"].replace("m", "-").replace("p", ".")),
        "noise": int(raw["noise"]) / 100.0,
        "seed": int(raw["seed"]),
    }


def load_rows(root: Path) -> list[dict]:
    rows = []
    for model in MODELS:
        manifest_path = root / model / "best_gnn_manifest.json"
        if not manifest_path.is_file():
            raise FileNotFoundError(manifest_path)
        manifest = json.loads(manifest_path.read_text())
        for report in manifest["reports"]:
            dataset = Path(report["dataset"]).stem
            parameters = parse_dataset(dataset)
            metrics = report["metrics"]
            split = metrics["splits"]
            rows.append(
                {
                    "dataset": dataset,
                    "model": model,
                    "model_label": report.get("model_label", model),
                    "K": int(report["K"]),
                    "harmonic_mode": report["harmonic_mode"],
                    **parameters,
                    "validation_relative_rmse": float(
                        split["validation"]["dc_relative_rmse"]
                    ),
                    "test_relative_rmse": float(
                        split["test"]["dc_relative_rmse"]
                    ),
                    "train_relative_rmse": float(
                        split["train"]["dc_relative_rmse"]
                    ),
                    "validation_rmse_m_s": float(
                        split["validation"]["dc_rmse_m_s"]
                    ),
                    "test_rmse_m_s": float(split["test"]["dc_rmse_m_s"]),
                    "epochs": int(metrics["epochs_completed"]),
                    "best_validation_loss": float(
                        metrics["best_validation_loss"]
                    ),
                    "pressure_variation_penalty": float(
                        metrics["pressure_variation_penalty"]
                    ),
                    "correction_mean": float(
                        metrics["corrections"]["mean"]
                    ),
                    "correction_std": float(metrics["corrections"]["std"]),
                    "correction_near_bound_pct": float(
                        metrics["corrections"]["percent_near_bound"]
                    ),
                    "dashboard": (
                        f"{model}/{dataset}/dashboard.html"
                    ),
                    "run": Path(report["selected_run"]).name,
                }
            )
    if len(rows) != 108:
        raise RuntimeError(f"Expected 108 selected results, found {len(rows)}")
    return rows


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_dashboard(path: Path, rows: list[dict]) -> None:
    payload = json.dumps(rows).replace("</", "<\\/")
    path.write_text(
        """<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>All 108 selected GNN configurations</title>
<style>
:root{--bg:#f4f6f8;--panel:#fff;--ink:#17212b;--muted:#637282;--line:#d9e1e8}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:14px/1.4 system-ui,sans-serif}
header{padding:18px 22px;background:#202b36;color:white}main{padding:14px;display:grid;gap:14px}
section{background:var(--panel);border:1px solid var(--line);border-radius:8px;padding:12px}
.filters{display:flex;gap:12px;flex-wrap:wrap}.filters label{display:grid;gap:4px;color:var(--muted);font-size:12px}
select{min-width:145px;padding:7px}.cards{display:grid;grid-template-columns:repeat(6,minmax(130px,1fr));gap:8px}
.card{border:1px solid var(--line);border-radius:7px;padding:9px}.card b{display:block;color:var(--muted);font-size:11px}
.card span{font-size:19px;font-weight:650}.grid{display:grid;grid-template-columns:1fr 1fr;gap:14px}
canvas{width:100%;height:410px;border:1px solid var(--line);border-radius:6px}
.gallery{display:grid;grid-template-columns:repeat(auto-fit,minmax(420px,1fr));gap:14px}
figure{margin:0;border:1px solid var(--line);border-radius:7px;padding:9px}figure img{width:100%;height:auto}
figcaption{margin-top:6px;font-weight:600}.selected-meta{color:var(--muted);margin:6px 0 12px}
.scroll{max-height:560px;overflow:auto}table{width:100%;border-collapse:collapse;font-size:12px}
th,td{padding:6px;border-bottom:1px solid #edf1f5;text-align:right;white-space:nowrap}
th{position:sticky;top:0;background:white}th:first-child,td:first-child{text-align:left}a{color:#1769aa}
@media(max-width:1000px){.grid,.cards{grid-template-columns:1fr}}
</style></head><body>
<header><h1>All 108 selected neural configurations</h1>
<div>36 synthetic datasets × 3 model families; selection uses validation DC relative RMSE.</div></header>
<main>
<section><div class="filters">
<label>Model<select id="model"></select></label>
<label>D0 [1/Pa]<select id="D0"></select></label>
<label>Alpha<select id="alpha"></select></label>
<label>Noise<select id="noise"></select></label>
<label>Metric<select id="metric">
<option value="validation_relative_rmse">validation relative RMSE</option>
<option value="test_relative_rmse">test relative RMSE</option>
<option value="train_relative_rmse">train relative RMSE</option>
<option value="pressure_variation_penalty">pressure variation penalty</option>
</select></label></div></section>
<section><div class="cards" id="cards"></div></section>
<div class="grid">
<section><h2>Error by dataset</h2><canvas id="scatter"></canvas></section>
<section><h2>Grouped mean by noise</h2><canvas id="grouped"></canvas></section>
</div>
<section><h2>Configuration gallery</h2>
<label>Selected configuration<select id="selected"></select></label>
<div class="selected-meta" id="selectedMeta"></div>
<div class="gallery" id="gallery"></div></section>
<section><h2>Selected configurations</h2><div class="scroll"><table id="table"></table></div></section>
</main>
<script id="payload" type="application/json">__PAYLOAD__</script>
<script>
const all=JSON.parse(document.getElementById("payload").textContent);
const $=id=>document.getElementById(id);
const colors={"physics_informed_gnn":"#1f77b4","vanilla_gcn":"#ff7f0e","edge_local_mlp":"#2ca02c"};
const fmt=x=>Number.isFinite(+x)?((Math.abs(+x)<.001&&+x!==0)?(+x).toExponential(3):(+x).toFixed(4).replace(/0+$/,"").replace(/\.$/,"")):"n/a";
function options(id,values,formatter=x=>x){const s=$(id);s.innerHTML="<option value='all'>all</option>"+values.map(x=>`<option value="${x}">${formatter(x)}</option>`).join("");}
options("model",[...new Set(all.map(r=>r.model))],x=>all.find(r=>r.model===x).model_label);
options("D0",[...new Set(all.map(r=>r.D0))].sort((a,b)=>a-b),x=>Number(x).toExponential(0));
options("alpha",[...new Set(all.map(r=>r.alpha))].sort((a,b)=>a-b));
options("noise",[...new Set(all.map(r=>r.noise))].sort((a,b)=>a-b),x=>`${Math.round(100*x)}%`);
function filtered(){return all.filter(r=>( $("model").value==="all"||r.model===$("model").value)&&($("D0").value==="all"||r.D0==+$("D0").value)&&($("alpha").value==="all"||r.alpha==+$("alpha").value)&&($("noise").value==="all"||r.noise==+$("noise").value));}
function canvas(id){const el=$(id),d=devicePixelRatio||1,r=el.getBoundingClientRect();el.width=r.width*d;el.height=r.height*d;const c=el.getContext("2d");c.setTransform(d,0,0,d,0,0);return{c,w:r.width,h:r.height};}
function formatTick(x){return Number.isFinite(x)?((Math.abs(x)>=1000||(Math.abs(x)>0&&Math.abs(x)<.001))?x.toExponential(2):x.toFixed(3).replace(/\.?0+$/,"")):"";}
function axes(c,L,T,W,H,xlabel,ylabel){c.strokeStyle="#687684";c.strokeRect(L,T,W,H);c.fillStyle="#263442";c.textAlign="center";c.fillText(xlabel,L+W/2,T+H+34);c.save();c.translate(16,T+H/2);c.rotate(-Math.PI/2);c.fillText(ylabel,0,0);c.restore();c.textAlign="left";}
function drawTicks(c,xTicks,yTicks,scaleX,scaleY,L,T,W,H){c.strokeStyle="#b7c1cb";c.fillStyle="#637282";c.textAlign="center";c.textBaseline="top";xTicks.forEach(v=>{const x=scaleX(v);c.beginPath();c.moveTo(x,T+H);c.lineTo(x,T+H+4);c.stroke();c.fillText(formatTick(v),x,T+H+6)});c.save();c.textAlign="right";c.textBaseline="middle";yTicks.forEach(v=>{const y=scaleY(v);c.beginPath();c.moveTo(L-4,y);c.lineTo(L,y);c.stroke();c.fillText(formatTick(v),L-6,y)});c.restore();}
function scatter(rows,metric){const {c,w,h}=canvas("scatter");c.clearRect(0,0,w,h);const L=62,T=20,R=18,B=52,W=w-L-R,H=h-T-B;const vals=rows.map(r=>r[metric]).filter(Number.isFinite);if(!vals.length)return;let lo=Math.min(...vals),hi=Math.max(...vals);if(lo===hi){lo-=.01;hi+=.01}const xScale=i=>L+(i+.5)/Math.max(rows.length,1)*W,yScale=v=>T+(hi-v)/(hi-lo)*H;axes(c,L,T,W,H,"filtered configuration index",$("metric").selectedOptions[0].text);drawTicks(c,[0,Math.floor(rows.length/4),Math.floor(rows.length/2),Math.floor(3*rows.length/4),Math.max(rows.length-1,0)],[lo,(lo+hi)/2,hi],xScale,yScale,L,T,W,H);rows.forEach((r,i)=>{const x=xScale(i),y=yScale(r[metric]);c.fillStyle=colors[r.model];c.beginPath();c.arc(x,y,4,0,2*Math.PI);c.fill();});}
function grouped(rows,metric){const {c,w,h}=canvas("grouped");c.clearRect(0,0,w,h);const L=62,T=20,R=18,B=52,W=w-L-R,H=h-T-B;const groups=[];for(const model of Object.keys(colors))for(const noise of [0,.1,.25,.5]){const x=rows.filter(r=>r.model===model&&r.noise===noise).map(r=>r[metric]).filter(Number.isFinite);if(x.length)groups.push({model,noise,value:x.reduce((a,b)=>a+b,0)/x.length});}if(!groups.length)return;let lo=Math.min(...groups.map(x=>x.value)),hi=Math.max(...groups.map(x=>x.value));if(lo===hi){lo-=.01;hi+=.01}const xScale=noise=>L+(noise/.5)*W,yScale=v=>T+(hi-v)/(hi-lo)*H;axes(c,L,T,W,H,"noise level",$("metric").selectedOptions[0].text);drawTicks(c,[0,.1,.25,.5],[lo,(lo+hi)/2,hi],xScale,yScale,L,T,W,H);groups.forEach(g=>{const modelIndex=Object.keys(colors).indexOf(g.model),x=xScale(g.noise)+(modelIndex-1)*7,y=yScale(g.value);c.fillStyle=colors[g.model];c.beginPath();c.arc(x,y,5,0,2*Math.PI);c.fill();});}
const galleryImages=[
["Conductance multiplier map","conductance_multiplier_map.png"],
["Correction distributions","correction_distributions.png"],
["Delta map","delta_map.png"],
["Pressure comparison","pressure_comparison.png"],
["Velocity parity","velocity_parity.png"],
["Training history","training_history.png"]
];
function renderGallery(){const rows=filtered(),selected=$("selected"),key=selected.value,row=rows.find(r=>`${r.model}|${r.dataset}`===key)||rows[0];if(!row){$("gallery").innerHTML="";$("selectedMeta").textContent="No matching configurations.";return;}const base=`${row.model}/${row.dataset}`;$("selectedMeta").innerHTML=`${row.model_label} · K=${row.K} · ${row.harmonic_mode} · validation relative RMSE=${fmt(row.validation_relative_rmse)} · <a href="${row.dashboard}">open full dashboard</a>`;$("gallery").innerHTML=galleryImages.map(([label,file])=>`<figure><img src="${base}/${file}" alt="${label}"><figcaption>${label}</figcaption></figure>`).join("");}
function updateSelected(rows){const selected=$("selected"),previous=selected.value;selected.innerHTML=rows.map(r=>`<option value="${r.model}|${r.dataset}">${r.dataset} · ${r.model_label} · K=${r.K}</option>`).join("");if([...selected.options].some(o=>o.value===previous))selected.value=previous;renderGallery();}
function render(){const rows=filtered(),metric=$("metric").value,vals=rows.map(r=>r[metric]).filter(Number.isFinite),mean=vals.length?vals.reduce((a,b)=>a+b,0)/vals.length:NaN,best=rows.reduce((a,b)=>!a||b[metric]<a[metric]?b:a,null);$("cards").innerHTML=[["Rows",rows.length],["Datasets",new Set(rows.map(r=>r.dataset)).size],["Models",new Set(rows.map(r=>r.model)).size],["Mean metric",fmt(mean)],["Best metric",best?fmt(best[metric]):"n/a"],["Best model",best?best.model_label:"n/a"]].map(([k,v])=>`<div class="card"><b>${k}</b><span>${v}</span></div>`).join("");scatter(rows,metric);grouped(rows,metric);updateSelected(rows);$("table").innerHTML="<thead><tr><th>dataset</th><th>model</th><th>K</th><th>channels</th><th>D0</th><th>alpha</th><th>noise</th><th>val rel RMSE</th><th>test rel RMSE</th><th>epochs</th></tr></thead><tbody>"+rows.sort((a,b)=>a.dataset.localeCompare(b.dataset)||a.model.localeCompare(b.model)).map(r=>`<tr><td><a href="${r.dashboard}">${r.dataset}</a></td><td>${r.model_label}</td><td>${r.K}</td><td>${r.harmonic_mode}</td><td>${r.D0.toExponential(0)}</td><td>${r.alpha}</td><td>${Math.round(100*r.noise)}%</td><td>${fmt(r.validation_relative_rmse)}</td><td>${fmt(r.test_relative_rmse)}</td><td>${r.epochs}</td></tr>`).join("")+"</tbody>";}
for(const id of ["model","D0","alpha","noise","metric"])$(id).addEventListener("change",render);$("selected").addEventListener("change",renderGallery);window.addEventListener("resize",()=>{scatter(filtered(),$("metric").value);grouped(filtered(),$("metric").value)});render();
</script></body></html>""".replace("__PAYLOAD__", payload)
    )


def main():
    args = parse_args()
    root = args.comparison_root.expanduser().resolve()
    output = (
        args.output.expanduser().resolve()
        if args.output
        else root / "all_108_dashboard.html"
    )
    rows = load_rows(root)
    write_csv(root / "all_108_results.csv", rows)
    write_dashboard(output, rows)
    print(f"Wrote {output}")
    print(f"Wrote {root / 'all_108_results.csv'}")


if __name__ == "__main__":
    main()
