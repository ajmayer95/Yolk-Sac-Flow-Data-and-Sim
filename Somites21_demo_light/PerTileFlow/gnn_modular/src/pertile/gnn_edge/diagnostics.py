"""Diagnostics and table-building utilities for the GNN edge-flow workflow."""

from __future__ import annotations

import math
import os
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

os.environ.setdefault("MPLCONFIGDIR", "/tmp/gnn_edge_mpl")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

try:
    import networkx as nx
except Exception:  # pragma: no cover
    nx = None

from .constants import nL_per_m3
from .data import MosaicData
from .utils import safe_float, write_csv


def collect_edge_rows(
    data: MosaicData,
    q_hat: torch.Tensor,
    p: torch.Tensor,
    g_hat: torch.Tensor,
    train_mask: torch.Tensor,
    val_mask: torch.Tensor,
) -> List[dict]:
    q_obs = data.q_obs.cpu().numpy() * nL_per_m3
    q_pred = q_hat.detach().cpu().numpy() * nL_per_m3
    p_np = p.detach().cpu().numpy()
    g_hat_np = g_hat.detach().cpu().numpy()
    src, dst = data.edge_index.cpu().numpy()

    rows = []
    for ei, (u, v) in enumerate(data.edge_ids):
        split = "unobserved"
        if bool(train_mask[ei]):
            split = "train"
        elif bool(val_mask[ei]):
            split = "val"

        rows.append({
            "edge_id": ei,
            "source": str(u),
            "target": str(v),
            "radius_m": float(data.radius_m[ei]),
            "length_m": float(data.length_m[ei]),
            "G_pois": float(data.g_pois[ei]),
            "G_hat": float(g_hat_np[ei]),
            "delta": float(math.log(max(g_hat_np[ei] / float(data.g_pois[ei]), 1e-30))),
            "C": float(g_hat_np[ei] / max(float(data.g_pois[ei]), 1e-30)),
            "Q_obs_nL_s": float(q_obs[ei]),
            "Q_hat_nL_s": float(q_pred[ei]),
            "residual_nL_s": float(q_obs[ei] - q_pred[ei]),
            "pressure_drop_Pa": float(p_np[int(src[ei])] - p_np[int(dst[ei])]),
            "split": split,
            "valid_obs": int(bool(data.valid_mask[ei])),
        })

    return rows


@torch.no_grad()
def collect_harmonic_rows(
    data: MosaicData,
    q_hat_h: torch.Tensor,
    train_mask: torch.Tensor,
    val_mask: torch.Tensor,
    harmonics: Sequence[int],
) -> List[dict]:
    if q_hat_h.shape[1] == 0:
        return []

    pred = q_hat_h.detach().cpu().numpy() * nL_per_m3
    obs = data.q_harmonic_obs.detach().cpu().numpy() * nL_per_m3
    valid = data.harmonic_valid_mask.detach().cpu().numpy().astype(bool)
    weights = data.harmonic_loss_weight.detach().cpu().numpy()

    rows = []
    for ei, (u, v) in enumerate(data.edge_ids):
        split = "unobserved"
        if bool(train_mask[ei]):
            split = "train"
        elif bool(val_mask[ei]):
            split = "val"

        for hi, h in enumerate(harmonics):
            if hi >= pred.shape[1]:
                continue

            resid_re = float(obs[ei, hi, 0] - pred[ei, hi, 0])
            resid_im = float(obs[ei, hi, 1] - pred[ei, hi, 1])

            rows.append({
                "edge_id": ei,
                "source": str(u),
                "target": str(v),
                "harmonic": int(h),
                "split": split,
                "valid_obs": int(valid[ei, hi]),
                "Q_obs_real_nL_s": float(obs[ei, hi, 0]),
                "Q_obs_imag_nL_s": float(obs[ei, hi, 1]),
                "Q_hat_real_nL_s": float(pred[ei, hi, 0]),
                "Q_hat_imag_nL_s": float(pred[ei, hi, 1]),
                "residual_real_nL_s": resid_re,
                "residual_imag_nL_s": resid_im,
                "residual_abs_nL_s": float(math.hypot(resid_re, resid_im)),
                "snr_loss_weight_inv_nL_s": float(weights[ei, hi]),
            })

    return rows


@torch.no_grad()
def collect_node_rows(data: MosaicData, p: torch.Tensor) -> List[dict]:
    p_np = p.detach().cpu().numpy()
    xy = data.node_xy.cpu().numpy()
    deg = data.x_node[:, 0].cpu().numpy()

    rows = []
    for i, node_id in enumerate(data.node_ids):
        rows.append({
            "node_index": i,
            "node_id": str(node_id),
            "pressure_Pa": float(p_np[i]),
            "source_sink_nL_s": float(data.source_sink[i] * nL_per_m3),
            "boundary_kind": data.boundary_kind[i],
            "x": float(xy[i, 0]),
            "y": float(xy[i, 1]),
            "standardized_degree_feature": float(deg[i]),
        })

    return rows


def add_node_distance_diagnostics(data: MosaicData, rows: Sequence[dict]) -> None:
    source_nodes = [i for i, kind in enumerate(data.boundary_kind) if kind == "source"]
    sink_nodes = [i for i, kind in enumerate(data.boundary_kind) if kind == "sink"]

    dist_a = _shortest_distances_from_sources(data, source_nodes)
    dist_v = _shortest_distances_from_sources(data, sink_nodes)

    for i, row in enumerate(rows):
        row["distance_to_A"] = (
            float(dist_a[i]) if math.isfinite(float(dist_a[i])) else float("nan")
        )
        row["distance_to_V"] = (
            float(dist_v[i]) if math.isfinite(float(dist_v[i])) else float("nan")
        )


@torch.no_grad()
def collect_conservation_rows(data: MosaicData, q_hat: torch.Tensor) -> List[dict]:
    src, dst = data.edge_index.cpu()

    net = torch.zeros(data.x_node.shape[0], dtype=torch.float32)
    q_cpu = q_hat.detach().cpu() * nL_per_m3

    net.index_add_(0, src, q_cpu)
    net.index_add_(0, dst, -q_cpu)

    target = data.source_sink.cpu() * nL_per_m3
    residual = net - target
    xy = data.node_xy.cpu().numpy()

    rows = []
    for i, node_id in enumerate(data.node_ids):
        rows.append({
            "node_index": i,
            "node_id": str(node_id),
            "predicted_net_flow_nL_s": float(net[i]),
            "source_sink_value_nL_s": float(target[i]),
            "conservation_residual_nL_s": float(residual[i]),
            "abs_conservation_residual_nL_s": float(abs(residual[i])),
            "boundary_kind": data.boundary_kind[i],
            "x": float(xy[i, 0]),
            "y": float(xy[i, 1]),
        })

    return rows


def _edge_endpoint_degrees(data: MosaicData) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    src, dst = data.edge_index.cpu().numpy()
    deg = np.bincount(
        np.concatenate([src, dst]),
        minlength=data.x_node.shape[0],
    ).astype(float)
    return deg, deg[src], deg[dst]


def _shortest_distances_from_sources(
    data: MosaicData,
    source_indices: Sequence[int],
) -> np.ndarray:
    n = data.x_node.shape[0]
    dist = np.full(n, np.nan, dtype=float)

    if not source_indices:
        return dist

    src, dst = data.edge_index.cpu().numpy()
    adj: List[List[int]] = [[] for _ in range(n)]

    for a, b in zip(src, dst):
        adj[int(a)].append(int(b))
        adj[int(b)].append(int(a))

    queue = [int(i) for i in source_indices]
    for i in queue:
        dist[i] = 0.0

    head = 0
    while head < len(queue):
        cur = queue[head]
        head += 1

        for nb in adj[cur]:
            if not math.isfinite(float(dist[nb])):
                dist[nb] = dist[cur] + 1.0
                queue.append(nb)

    return dist


def _topology_class(deg_src: float, deg_dst: float) -> str:
    max_deg = max(deg_src, deg_dst)
    min_deg = min(deg_src, deg_dst)

    if min_deg <= 1:
        return "terminal"
    if max_deg >= 4:
        return "high-degree-adjacent"
    if max_deg >= 3:
        return "bifurcation-adjacent"

    return "chain"


def topology_diagnostics(data: MosaicData) -> Dict[int, dict]:
    _deg, deg_src, deg_dst = _edge_endpoint_degrees(data)
    src, dst = data.edge_index.cpu().numpy()

    source_nodes = [i for i, kind in enumerate(data.boundary_kind) if kind == "source"]
    sink_nodes = [i for i, kind in enumerate(data.boundary_kind) if kind == "sink"]

    dist_a = _shortest_distances_from_sources(data, source_nodes)
    dist_v = _shortest_distances_from_sources(data, sink_nodes)

    out: Dict[int, dict] = {}

    for ei, (s_i, d_i) in enumerate(zip(src, dst)):
        da = np.nanmin([dist_a[int(s_i)], dist_a[int(d_i)]])
        dv = np.nanmin([dist_v[int(s_i)], dist_v[int(d_i)]])

        out[int(ei)] = {
            "degree_src": float(deg_src[ei]),
            "degree_dst": float(deg_dst[ei]),
            "distance_to_A": float(da) if math.isfinite(float(da)) else float("nan"),
            "distance_to_V": float(dv) if math.isfinite(float(dv)) else float("nan"),
            "topology_class": _topology_class(float(deg_src[ei]), float(deg_dst[ei])),
            "node_betweenness_src": float("nan"),
            "node_betweenness_dst": float("nan"),
            "edge_betweenness": float("nan"),
        }

    if nx is not None and data.x_node.shape[0] <= 2000:
        try:
            graph = nx.Graph()
            graph.add_nodes_from(range(data.x_node.shape[0]))
            graph.add_edges_from((int(a), int(b)) for a, b in zip(src, dst))

            node_bc = nx.betweenness_centrality(graph, normalized=True)
            edge_bc = nx.edge_betweenness_centrality(graph, normalized=True)

            for ei, (s_i, d_i) in enumerate(zip(src, dst)):
                key = (int(s_i), int(d_i))
                rev = (int(d_i), int(s_i))

                out[int(ei)]["node_betweenness_src"] = float(
                    node_bc.get(int(s_i), float("nan"))
                )
                out[int(ei)]["node_betweenness_dst"] = float(
                    node_bc.get(int(d_i), float("nan"))
                )
                out[int(ei)]["edge_betweenness"] = float(
                    edge_bc.get(key, edge_bc.get(rev, float("nan")))
                )
        except Exception as exc:
            print(f"WARNING: centrality diagnostics skipped: {exc}")

    elif nx is None:
        print("WARNING: networkx unavailable; centrality diagnostics skipped.")

    return out


def harmonic_diagnostics(graph, data: MosaicData) -> Dict[int, dict]:
    out: Dict[int, dict] = {}
    phases_h1 = np.full(len(data.edge_ids), np.nan, dtype=float)

    for ei, (u, v) in enumerate(data.edge_ids):
        ed = graph.edges[u, v]

        sign = 1.0
        ff = ed.get("flow_from")
        ft = ed.get("flow_to")
        if ff is not None and ft is not None and not (ff == u and ft == v):
            sign = -1.0

        amp1 = safe_float(ed.get("Q_H1_amp", ed.get("amp_Q_h1_piv", ed.get("amp_Q"))))
        phi1 = safe_float(ed.get("Q_H1_phi", ed.get("phase_h1_piv", ed.get("phase"))))
        amp2 = safe_float(ed.get("Q_H2_amp", ed.get("amp_Q_h2_piv")))
        phi2 = safe_float(ed.get("Q_H2_phi", ed.get("phase_h2_piv")))

        q1 = (
            sign * amp1 * complex(math.cos(phi1), math.sin(phi1))
            if math.isfinite(amp1) and math.isfinite(phi1)
            else complex(float("nan"), float("nan"))
        )
        q2 = (
            sign * amp2 * complex(math.cos(phi2), math.sin(phi2))
            if math.isfinite(amp2) and math.isfinite(phi2)
            else complex(float("nan"), float("nan"))
        )

        phases_h1[ei] = (
            math.atan2(q1.imag, q1.real)
            if math.isfinite(q1.real) and math.isfinite(q1.imag)
            else float("nan")
        )

        out[ei] = {
            "abs_H1": float(abs(q1)) if math.isfinite(q1.real) else float("nan"),
            "abs_H2": float(abs(q2)) if math.isfinite(q2.real) else float("nan"),
            "harmonic_ratio": (
                float(abs(q2) / (abs(q1) + 1e-30))
                if math.isfinite(q1.real) and math.isfinite(q2.real)
                else float("nan")
            ),
            "phase_H1": phases_h1[ei],
            "phase_H2": (
                math.atan2(q2.imag, q2.real)
                if math.isfinite(q2.real) and math.isfinite(q2.imag)
                else float("nan")
            ),
            "phase_dispersion_H1": float("nan"),
        }

    src, dst = data.edge_index.cpu().numpy()
    incident: Dict[int, List[int]] = {}

    for ei, (s_i, d_i) in enumerate(zip(src, dst)):
        incident.setdefault(int(s_i), []).append(ei)
        incident.setdefault(int(d_i), []).append(ei)

    for ei, (s_i, d_i) in enumerate(zip(src, dst)):
        ph = phases_h1[ei]
        if not math.isfinite(float(ph)):
            continue

        neigh = set(incident.get(int(s_i), [])) | set(incident.get(int(d_i), []))
        neigh.discard(ei)

        diffs = [
            abs(math.atan2(math.sin(ph - phases_h1[j]), math.cos(ph - phases_h1[j])))
            for j in neigh
            if math.isfinite(float(phases_h1[j]))
        ]

        if diffs:
            out[ei]["phase_dispersion_H1"] = float(np.mean(diffs))

    return out


def enrich_edge_rows(data: MosaicData, graph, edge_rows: Sequence[dict]) -> List[dict]:
    topo = topology_diagnostics(data)
    harm = harmonic_diagnostics(graph, data)

    enriched = []

    for row in edge_rows:
        ei = int(row["edge_id"])
        new = dict(row)

        c_val = safe_float(new.get("C"))
        g_hat = safe_float(new.get("G_hat"))
        g_pois = safe_float(new.get("G_pois"))

        new.update({
            "log_radius": math.log(max(safe_float(new.get("radius_m")), 1e-30)),
            "log_length": math.log(max(safe_float(new.get("length_m")), 1e-30)),
            "log_G_pois": math.log(max(g_pois, 1e-60)),
            "abs_Q_obs": abs(safe_float(new.get("Q_obs_nL_s"))),
            "abs_error_nL_s": abs(safe_float(new.get("residual_nL_s"))),
            "effective_resistance": 1.0 / max(g_hat, 1e-60),
            "poiseuille_resistance": 1.0 / max(g_pois, 1e-60),
            "resistance_ratio": 1.0 / max(c_val, 1e-30),
        })

        new.update(topo.get(ei, {}))
        new.update(harm.get(ei, {}))
        enriched.append(new)

    return enriched


def _linear_fit(X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, float, np.ndarray, float]:
    X_aug = np.column_stack([np.ones(X.shape[0]), X])
    xt = torch.as_tensor(X_aug, dtype=torch.float64)
    yt = torch.as_tensor(y, dtype=torch.float64)

    gram = xt.T @ xt
    rhs = xt.T @ yt
    ridge = torch.eye(gram.shape[0], dtype=gram.dtype) * 1e-12
    ridge[0, 0] = 0.0

    beta = torch.linalg.solve(gram + ridge, rhs).cpu().numpy()
    intercept = float(beta[0])
    coef = np.asarray(beta[1:], dtype=float)
    pred = X_aug @ beta

    ss_res = float(np.sum((y - pred) ** 2))
    ss_tot = float(np.sum((y - float(np.mean(y))) ** 2))
    r2 = float(1.0 - ss_res / ss_tot) if ss_tot > 1e-30 else float("nan")

    return coef, intercept, pred, r2


def _standardize(a: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean = np.nanmean(a, axis=0)
    std = np.nanstd(a, axis=0)
    std = np.where(std > 1e-30, std, 1.0)
    return (a - mean) / std, mean, std


def collect_regression_edges(
    data: MosaicData,
    edge_rows: Sequence[dict],
) -> List[dict]:
    src, dst = data.edge_index.cpu().numpy()

    deg = np.bincount(
        np.concatenate([src, dst]),
        minlength=data.x_node.shape[0],
    ).astype(float)

    rows = []

    for row in edge_rows:
        try:
            delta = float(row["delta"])
            c_val = float(row["C"])
        except (TypeError, ValueError):
            continue

        radius = max(float(row["radius_m"]), 1e-30)
        length = max(float(row["length_m"]), 1e-30)
        g_pois = max(float(row["G_pois"]), 1e-60)
        q_obs = float(row["Q_obs_nL_s"])

        if not all(math.isfinite(x) for x in (delta, c_val, radius, length, g_pois, q_obs)):
            continue

        ei = int(row["edge_id"])

        rows.append({
            "edge_id": ei,
            "source": row["source"],
            "target": row["target"],
            "delta": delta,
            "C": c_val,
            "log_radius": math.log(radius),
            "log_length": math.log(length),
            "log_G_pois": math.log(g_pois),
            "abs_Q_obs": abs(q_obs),
            "degree_src": float(deg[int(src[ei])]),
            "degree_dst": float(deg[int(dst[ei])]),
            "residual": float(row["residual_nL_s"]),
            "split": row.get("split", ""),
            "valid_obs": int(row.get("valid_obs", 0)),
            "distance_to_A": safe_float(row.get("distance_to_A")),
            "distance_to_V": safe_float(row.get("distance_to_V")),
            "topology_class": row.get("topology_class", ""),
            "abs_H1": safe_float(row.get("abs_H1")),
            "abs_H2": safe_float(row.get("abs_H2")),
            "harmonic_ratio": safe_float(row.get("harmonic_ratio")),
            "phase_H1": safe_float(row.get("phase_H1")),
            "phase_H2": safe_float(row.get("phase_H2")),
            "phase_dispersion_H1": safe_float(row.get("phase_dispersion_H1")),
        })

    return rows


def plot_regression_fit(
    rows: Sequence[dict],
    model_name: str,
    y: np.ndarray,
    pred: np.ndarray,
    out_dir: Path,
) -> None:
    del rows

    safe_name = model_name.lower().replace(" ", "_")

    plt.figure(figsize=(5, 5))
    plt.scatter(y, pred, s=12, alpha=0.6)

    lo = float(np.nanmin([np.nanmin(y), np.nanmin(pred)]))
    hi = float(np.nanmax([np.nanmax(y), np.nanmax(pred)]))

    plt.plot([lo, hi], [lo, hi], color="black", lw=1)
    plt.xlabel("observed delta")
    plt.ylabel("regression-predicted delta")
    plt.tight_layout()
    plt.savefig(out_dir / f"regression_{safe_name}_delta_pred_vs_obs.png", dpi=180)
    plt.close()

    plt.figure(figsize=(6, 4))
    plt.hist(pred - y, bins=40)
    plt.xlabel("regression residual in delta")
    plt.ylabel("count")
    plt.tight_layout()
    plt.savefig(out_dir / f"regression_{safe_name}_residual_hist.png", dpi=180)
    plt.close()


def _adjusted_r2(r2: float, n: int, p: int) -> float:
    if not math.isfinite(float(r2)) or n <= p + 1:
        return float("nan")
    return float(1.0 - (1.0 - r2) * (n - 1) / (n - p - 1))


def run_delta_regressions(
    edge_rows: Sequence[dict],
    out_dir: Path,
) -> Tuple[List[dict], List[dict]]:
    reg_rows = [r for r in edge_rows if int(r.get("valid_obs", 0)) == 1]

    if not reg_rows:
        return [], []

    specs = [
        ("Model A", ["log_radius", "log_length"]),
        ("Model B", ["log_G_pois"]),
        ("Model C", ["log_radius", "log_length", "abs_Q_obs", "degree_src", "degree_dst"]),
        (
            "Model D",
            [
                "log_radius",
                "log_length",
                "abs_Q_obs",
                "degree_src",
                "degree_dst",
                "distance_to_A",
                "distance_to_V",
            ],
        ),
        (
            "Model E",
            [
                "log_radius",
                "log_length",
                "abs_Q_obs",
                "harmonic_ratio",
                "phase_dispersion_H1",
            ],
        ),
    ]

    y = np.asarray([float(r["delta"]) for r in reg_rows], dtype=float)

    summary_rows: List[dict] = []
    coef_rows: List[dict] = []
    pred_rows: List[dict] = []
    model_c_standardized: Optional[Tuple[List[str], np.ndarray]] = None

    for model_name, features in specs:
        X = np.asarray([[float(r[f]) for f in features] for r in reg_rows], dtype=float)

        mask = np.isfinite(y) & np.all(np.isfinite(X), axis=1)
        X_fit = X[mask]
        y_fit = y[mask]

        if X_fit.shape[0] <= len(features):
            continue

        coef, intercept, pred, r2 = _linear_fit(X_fit, y_fit)

        X_std, _, _ = _standardize(X_fit)
        y_std, _, _ = _standardize(y_fit[:, None])
        coef_std, intercept_std, _pred_std, r2_std = _linear_fit(X_std, y_std[:, 0])

        adj = _adjusted_r2(r2, int(X_fit.shape[0]), len(features))

        summary_rows.append({
            "model": model_name,
            "features": ";".join(features),
            "n_edges": int(X_fit.shape[0]),
            "intercept": intercept,
            "R2": r2,
            "adjusted_R2": adj,
            "standardized_intercept": intercept_std,
            "standardized_R2": r2_std,
            "standardized_adjusted_R2": _adjusted_r2(
                r2_std,
                int(X_fit.shape[0]),
                len(features),
            ),
        })

        for feature, raw_c, std_c in zip(features, coef, coef_std):
            coef_rows.append({
                "model": model_name,
                "feature": feature,
                "coefficient": float(raw_c),
                "standardized_coefficient": float(std_c),
                "intercept": intercept,
                "R2": r2,
                "adjusted_R2": adj,
                "n_edges": int(X_fit.shape[0]),
                "feature_names": ";".join(features),
            })

        used_rows = [r for r, keep in zip(reg_rows, mask) if bool(keep)]
        for r, y_true, y_pred in zip(used_rows, y_fit, pred):
            pred_rows.append({
                "model": model_name,
                "edge_id": int(r["edge_id"]),
                "delta": float(y_true),
                "predicted_delta": float(y_pred),
                "regression_residual": float(y_true - y_pred),
            })

        plot_regression_fit(reg_rows, model_name, y_fit, pred, out_dir)

        safe_name = model_name.lower().replace(" ", "_")
        order = np.argsort(np.abs(coef_std))

        plt.figure(figsize=(7, 4))
        plt.barh([features[i] for i in order], [float(coef_std[i]) for i in order])
        plt.xlabel("standardized coefficient")
        plt.tight_layout()
        plt.savefig(out_dir / f"regression_{safe_name}_standardized_coefficients.png", dpi=180)
        plt.close()

        if model_name == "Model C":
            model_c_standardized = (features, coef_std)

    if model_c_standardized is not None:
        features, coef_std = model_c_standardized
        order = np.argsort(np.abs(coef_std))

        plt.figure(figsize=(7, 4))
        plt.barh([features[i] for i in order], [float(coef_std[i]) for i in order])
        plt.xlabel("standardized coefficient")
        plt.tight_layout()
        plt.savefig(out_dir / "regression_model_c_standardized_coefficients.png", dpi=180)
        plt.close()

    write_csv(out_dir / "delta_regression_summary.csv", summary_rows)
    write_csv(out_dir / "delta_regression_coefficients.csv", coef_rows)
    write_csv(out_dir / "delta_regression_predictions.csv", pred_rows)

    return summary_rows, coef_rows


def interpretation_summary(
    label: str,
    validation_history: Sequence[dict],
    regression_summary: Sequence[dict],
    regression_coefficients: Sequence[dict],
) -> None:
    print(f"\n[{label}] Interpretation summary")

    if validation_history:
        first = validation_history[0]
        last = validation_history[-1]
        best = min(validation_history, key=lambda r: float(r["val_loss"]))

        trend = (
            "decreased"
            if float(last["val_loss"]) < float(first["val_loss"])
            else "did not decrease"
        )

        print(
            f"  Validation loss {trend}: "
            f"{float(first['val_loss']):.4g} -> {float(last['val_loss']):.4g}; "
            f"best epoch {int(best['epoch'])} val_loss={float(best['val_loss']):.4g}."
        )
    else:
        print("  No held-out validation split for this run.")

    finite_models = [
        r for r in regression_summary
        if math.isfinite(float(r.get("R2", float("nan"))))
    ]

    if finite_models:
        best = max(finite_models, key=lambda r: float(r["R2"]))
        print(
            f"  Best delta regression: {best['model']} "
            f"(R^2={float(best['R2']):.3f}, n={int(best['n_edges'])})."
        )
    else:
        print("  Delta regressions did not have enough finite edges to fit.")

    model_c = [
        r for r in regression_coefficients
        if r.get("model") == "Model C"
        and math.isfinite(float(r.get("standardized_coefficient", float("nan"))))
    ]

    if model_c:
        ordered = sorted(
            model_c,
            key=lambda r: abs(float(r["standardized_coefficient"])),
            reverse=True,
        )[:3]

        text = ", ".join(
            f"{r['feature']}={float(r['standardized_coefficient']):+.3f}"
            for r in ordered
        )

        print(f"  Largest standardized Model C predictors: {text}.")

        names = {r["feature"] for r in ordered}
        if "log_radius" in names or "log_G_pois" in names:
            print(
                "  Strong radius/log_G_pois dependence may indicate an "
                "effective conductance law differing from ideal Poiseuille scaling."
            )


def _edge_midpoints(rows: Sequence[dict]) -> Tuple[np.ndarray, np.ndarray]:
    xs = []
    ys = []

    for r in rows:
        xs.append(safe_float(r.get("x_mid")))
        ys.append(safe_float(r.get("y_mid")))

    return np.asarray(xs, dtype=float), np.asarray(ys, dtype=float)


def add_edge_midpoints(data: MosaicData, rows: Sequence[dict]) -> None:
    xy = data.node_xy.cpu().numpy()
    src, dst = data.edge_index.cpu().numpy()

    for r in rows:
        ei = int(r["edge_id"])
        s_i = int(src[ei])
        d_i = int(dst[ei])

        r["x_mid"] = float(0.5 * (xy[s_i, 0] + xy[d_i, 0]))
        r["y_mid"] = float(0.5 * (xy[s_i, 1] + xy[d_i, 1]))


def write_top_edge_tables(rows: Sequence[dict], out_dir: Path) -> None:
    def keep(row: dict) -> dict:
        keys = [
            "edge_id",
            "source",
            "target",
            "radius_m",
            "length_m",
            "G_pois",
            "G_hat",
            "C",
            "delta",
            "Q_obs_nL_s",
            "Q_hat_nL_s",
            "residual_nL_s",
            "pressure_drop_Pa",
            "degree_src",
            "degree_dst",
        ]
        return {k: row.get(k, "") for k in keys}

    valid = [r for r in rows if int(r.get("valid_obs", 0)) == 1]

    specs = [
        ("top50_largest_C.csv", lambda r: safe_float(r.get("C")), True),
        ("top50_smallest_C.csv", lambda r: safe_float(r.get("C")), False),
        ("top50_largest_abs_delta.csv", lambda r: abs(safe_float(r.get("delta"))), True),
        ("top50_largest_abs_residual.csv", lambda r: abs(safe_float(r.get("residual_nL_s"))), True),
        ("top50_largest_abs_Q_obs.csv", lambda r: abs(safe_float(r.get("Q_obs_nL_s"))), True),
    ]

    for filename, key_fn, reverse in specs:
        sorted_rows = sorted(
            [r for r in valid if math.isfinite(key_fn(r))],
            key=key_fn,
            reverse=reverse,
        )[:50]

        write_csv(out_dir / filename, [keep(r) for r in sorted_rows])


def write_top_pressure_nodes(node_rows: Sequence[dict], out_dir: Path) -> None:
    rows = sorted(
        [r for r in node_rows if math.isfinite(safe_float(r.get("pressure_Pa")))],
        key=lambda r: safe_float(r.get("pressure_Pa")),
        reverse=True,
    )[:50]

    fields = [
        "node_id",
        "pressure_Pa",
        "standardized_degree_feature",
        "source_sink_nL_s",
        "x",
        "y",
        "distance_to_A",
        "distance_to_V",
        "boundary_kind",
    ]

    write_csv(out_dir / "top50_nodes_by_pressure.csv", [
        {k: r.get(k, "") for k in fields}
        for r in rows
    ])
