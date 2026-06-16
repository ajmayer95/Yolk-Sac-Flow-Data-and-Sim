"""Physics-embedded GNN for whole-mosaic DC edge-flow prediction.

The model does not predict flow directly in the physics branch.  It predicts
an edge conductance correction

    G_hat[e] = G_pois[e] * exp(delta[e])

then solves a resistive network pressure problem and reconstructs

    Q_hat[e] = G_hat[e] * (p_src - p_dst).

All flow diagnostics are reported in nL/s.  Internal graph solves use SI units.
"""
from __future__ import annotations

import argparse
import copy
import csv
import importlib
import json
import math
import os
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

os.environ.setdefault("MPLCONFIGDIR", "/tmp/gnn_edge_mpl")

try:
    import numpy as np
except Exception as exc:  # pragma: no cover
    raise SystemExit(
        "numpy is required. Install the project dependencies first.\n"
        f"Original import error: {exc}"
    )

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except Exception as exc:  # pragma: no cover
    raise SystemExit(
        "matplotlib is required for diagnostics plots.\n"
        f"Original import error: {exc}"
    )

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = PROJECT_ROOT.parent / "emb1" / "config.json"
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

try:
    import torch
    import torch.nn as nn
except Exception as exc:  # pragma: no cover
    raise SystemExit(
        "PyTorch is required. On Apple Silicon, install the native wheel.\n"
        f"Original import error: {exc}"
    )

try:
    from tqdm.auto import tqdm
except Exception:  # pragma: no cover
    tqdm = None

try:
    import networkx as nx
except Exception:  # pragma: no cover
    nx = None

from pertile.analysis.local_pressure_inference import _edge_geometry  # noqa: E402
from synthetic_validation_neumann_bc import MU, PX_SIZE_M, nL_per_m3  # noqa: E402
from tile_mosaic_simulation import load_graph_from_args  # noqa: E402


@dataclass
class MosaicData:
    node_ids: List[object]
    edge_ids: List[Tuple[object, object]]
    edge_index: torch.Tensor
    x_node: torch.Tensor
    x_edge: torch.Tensor
    q_obs: torch.Tensor
    valid_mask: torch.Tensor
    radius_m: torch.Tensor
    length_m: torch.Tensor
    g_pois: torch.Tensor
    source_sink: torch.Tensor
    node_xy: torch.Tensor
    boundary_kind: List[str]
    feature_stats: dict
    ref_node_index: int

    def to(self, device: torch.device) -> "MosaicData":
        out = {}
        for key, value in self.__dict__.items():
            out[key] = value.to(device) if torch.is_tensor(value) else value
        return MosaicData(**out)


def _install_numpy_pickle_compat() -> None:
    aliases = {
        "numpy._core": "numpy.core",
        "numpy._core.numeric": "numpy.core.numeric",
        "numpy._core.multiarray": "numpy.core.multiarray",
        "numpy._core._multiarray_umath": "numpy.core._multiarray_umath",
        "numpy._core.umath": "numpy.core.umath",
        "numpy._core.fromnumeric": "numpy.core.fromnumeric",
    }
    for new_name, old_name in aliases.items():
        if new_name not in sys.modules:
            try:
                sys.modules[new_name] = importlib.import_module(old_name)
            except Exception:
                pass


def set_seed(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.backends.mps.is_available():
        try:
            torch.mps.manual_seed(int(seed))
        except Exception:
            pass


def resolve_device(name: str) -> torch.device:
    if name == "cpu":
        return torch.device("cpu")
    if name == "mps":
        if not torch.backends.mps.is_available():
            raise SystemExit("Requested --device mps, but MPS is unavailable.")
        return torch.device("mps")
    return torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")


def safe_float(value, default=float("nan")) -> float:
    try:
        x = float(value)
    except (TypeError, ValueError):
        return default
    return x if math.isfinite(x) else default


def edge_q_dc_nls(edge_data: dict, u, v) -> float:
    q = (
        edge_data.get("Q_DC")
        or edge_data.get("mean_Q_piv")
        or edge_data.get("mean_Q")
        or edge_data.get("mean_Q_nL_s")
    )
    q = safe_float(q)
    if not math.isfinite(q):
        return float("nan")
    ff = edge_data.get("flow_from")
    ft = edge_data.get("flow_to")
    if ff is None or ft is None:
        return q
    return q if (ff == u and ft == v) else -q


def _harmonic_features(edge_data: dict, sign: float) -> List[float]:
    feats: List[float] = []
    for h in (1, 2, 3):
        amp = edge_data.get(f"Q_H{h}_amp")
        phase = edge_data.get(f"Q_H{h}_phi")
        snr = edge_data.get(f"Q_H{h}_snr_db")
        if h == 1 and (amp is None or phase is None):
            amp = edge_data.get("amp_Q_h1_piv", edge_data.get("amp_Q"))
            phase = edge_data.get("phase_h1_piv", edge_data.get("phase"))
        amp_f = safe_float(amp, 0.0)
        phase_f = safe_float(phase, 0.0)
        snr_f = safe_float(snr, 0.0)
        feats.extend([
            math.log(max(abs(amp_f), 1e-30)),
            sign * amp_f * math.cos(phase_f),
            sign * amp_f * math.sin(phase_f),
            snr_f,
        ])
    return feats


def boundary_injections(graph, node_ids: Sequence[object],
                        node_index: Dict[object, int]) -> Tuple[np.ndarray, List[str], int]:
    s = np.zeros(len(node_ids), dtype=np.float32)
    kinds = [""] * len(node_ids)
    sink_ref: Optional[int] = None
    boundary_records = []
    boundary_nodes = {n for n, d in graph.nodes(data=True)
                      if d.get("boundary_type") in ("source", "sink")}
    for bn in boundary_nodes:
        bdata = graph.nodes[bn]
        nbrs = [n for n in graph.neighbors(bn) if n in node_index]
        if len(nbrs) != 1:
            print(f"WARNING: boundary node {bn} has {len(nbrs)} usable neighbors; skipping")
            continue
        nb = nbrs[0]
        ed = graph.edges[bn, nb]
        mean_q = (
            ed.get("Q_DC")
            or ed.get("mean_Q_piv")
            or ed.get("mean_Q")
            or ed.get("mean_Q_nL_s")
        )
        mean_q = safe_float(mean_q)
        if not math.isfinite(mean_q):
            print(f"WARNING: boundary edge {bn}<->{nb} has no Q_DC; skipping")
            continue
        ff = ed.get("flow_from")
        ft = ed.get("flow_to")
        if ff == bn:
            sign_inject = 1.0
        elif ft == bn:
            sign_inject = -1.0
        else:
            sign_inject = 1.0 if bdata.get("boundary_type") == "source" else -1.0
        boundary_records.append((nb, bdata.get("boundary_type"), mean_q * sign_inject / nL_per_m3))

    if not boundary_records:
        for u, v, ed in graph.edges(data=True):
            if u not in node_index or v not in node_index:
                continue
            q = edge_q_dc_nls(ed, u, v)
            if not math.isfinite(q):
                continue
            s[node_index[u]] += float(q / nL_per_m3)
            s[node_index[v]] -= float(q / nL_per_m3)
        if np.any(np.isfinite(s)) and float(np.sum(np.abs(s))) > 0.0:
            if abs(float(s.sum())) > 1e-18:
                s -= float(s.sum()) / max(len(s), 1)
            ref_idx = int(np.argmin(np.abs(s)))
            for i, value in enumerate(s):
                if abs(float(value)) > 0.0:
                    kinds[i] = "observed_divergence"
            print(
                "Boundary DC injections: no explicit source/sink metadata; "
                "using observed edge-flow divergence fallback, "
                f"sum={float(s.sum()) * nL_per_m3:+.3e} nL/s"
            )
            return s, kinds, ref_idx
        raise SystemExit("No boundary source/sink vessels or observed DC edge-flow divergence were found.")

    raw = np.asarray([r[2] for r in boundary_records], dtype=np.float64)
    balanced = raw - raw.mean()
    for (nb, kind, _), value in zip(boundary_records, balanced):
        idx = node_index[nb]
        s[idx] += float(value)
        kinds[idx] = str(kind)
        if kind == "sink" and sink_ref is None:
            sink_ref = idx

    if abs(float(s.sum())) > 1e-18:
        s -= float(s.sum()) / max(len(boundary_records), 1)
    if sink_ref is None:
        sink_candidates = np.where(s < 0)[0]
        sink_ref = int(sink_candidates[0]) if len(sink_candidates) else int(np.argmax(np.abs(s)))

    print(
        f"Boundary DC injections: {len(boundary_records)} vessels, "
        f"sum={float(s.sum()) * nL_per_m3:+.3e} nL/s"
    )
    return s, kinds, int(sink_ref)


def build_mosaic_data(graph, include_harmonic_features: bool) -> MosaicData:
    boundary_nodes = {n for n, d in graph.nodes(data=True)
                      if d.get("boundary_type") in ("source", "sink")}
    node_ids = [n for n in graph.nodes() if n not in boundary_nodes]
    node_index = {n: i for i, n in enumerate(node_ids)}

    edge_ids: List[Tuple[object, object]] = []
    src: List[int] = []
    dst: List[int] = []
    radii: List[float] = []
    lengths: List[float] = []
    g_pois: List[float] = []
    q_obs: List[float] = []
    valid_obs: List[bool] = []
    raw_edge_features: List[List[float]] = []

    deg = dict(graph.degree())
    for u, v, ed in graph.edges(data=True):
        if u not in node_index or v not in node_index:
            continue
        r_m, l_m = _edge_geometry(ed, PX_SIZE_M)
        r_m = max(safe_float(r_m), 1e-12)
        l_m = max(safe_float(l_m), 1e-12)
        g = math.pi * r_m ** 4 / (8.0 * MU * l_m)
        q_nls = edge_q_dc_nls(ed, u, v)
        sign = 1.0
        ff = ed.get("flow_from")
        ft = ed.get("flow_to")
        if ff is not None and ft is not None and not (ff == u and ft == v):
            sign = -1.0

        feats = [
            math.log(r_m),
            math.log(l_m),
            math.log(max(r_m ** 4 / l_m, 1e-60)),
            math.log(max(g, 1e-60)),
            float(deg.get(u, 0)),
            float(deg.get(v, 0)),
        ]
        if include_harmonic_features:
            feats.extend(_harmonic_features(ed, sign))

        edge_ids.append((u, v))
        src.append(node_index[u])
        dst.append(node_index[v])
        radii.append(r_m)
        lengths.append(l_m)
        g_pois.append(g)
        q_obs.append(q_nls / nL_per_m3 if math.isfinite(q_nls) else 0.0)
        valid_obs.append(math.isfinite(q_nls))
        raw_edge_features.append(feats)

    if not edge_ids:
        raise SystemExit("No usable interior mosaic edges were found.")

    s, boundary_kind, ref_idx = boundary_injections(graph, node_ids, node_index)
    node_rows = []
    node_xy = []
    for n in node_ids:
        nd = graph.nodes[n]
        i = node_index[n]
        x = safe_float(nd.get("x", nd.get("graph_x")), float("nan"))
        y = safe_float(nd.get("y", nd.get("graph_y")), float("nan"))
        kind = boundary_kind[i]
        node_rows.append([
            float(deg.get(n, 0)),
            float(s[i] * nL_per_m3),
            1.0 if kind == "source" else 0.0,
            1.0 if kind == "sink" else 0.0,
            x if math.isfinite(x) else 0.0,
            y if math.isfinite(y) else 0.0,
        ])
        node_xy.append([x, y])

    q_arr = np.asarray(q_obs, dtype=np.float32)
    valid = np.asarray(valid_obs, dtype=bool)
    q_arr = np.where(valid, q_arr, 0.0).astype(np.float32)
    x_node = np.asarray(node_rows, dtype=np.float32)
    x_edge = np.asarray(raw_edge_features, dtype=np.float32)

    node_mean = x_node.mean(axis=0)
    node_std = np.maximum(x_node.std(axis=0), 1e-12)
    edge_mean = x_edge.mean(axis=0)
    edge_std = np.maximum(x_edge.std(axis=0), 1e-12)
    x_node = (x_node - node_mean) / node_std
    x_edge = (x_edge - edge_mean) / edge_std

    stats = {
        "node_mean": node_mean.tolist(),
        "node_std": node_std.tolist(),
        "edge_mean": edge_mean.tolist(),
        "edge_std": edge_std.tolist(),
    }

    return MosaicData(
        node_ids=node_ids,
        edge_ids=edge_ids,
        edge_index=torch.tensor([src, dst], dtype=torch.long),
        x_node=torch.tensor(x_node, dtype=torch.float32),
        x_edge=torch.tensor(x_edge, dtype=torch.float32),
        q_obs=torch.tensor(q_arr, dtype=torch.float32),
        valid_mask=torch.tensor(valid.tolist(), dtype=torch.bool),
        radius_m=torch.tensor(radii, dtype=torch.float32),
        length_m=torch.tensor(lengths, dtype=torch.float32),
        g_pois=torch.tensor(g_pois, dtype=torch.float32),
        source_sink=torch.tensor(s, dtype=torch.float32),
        node_xy=torch.tensor(node_xy, dtype=torch.float32),
        boundary_kind=boundary_kind,
        feature_stats=stats,
        ref_node_index=ref_idx,
    )


class EdgeMPNNLayer(nn.Module):
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.msg = nn.Sequential(
            nn.Linear(3 * hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.node_upd = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.edge_upd = nn.Sequential(
            nn.Linear(3 * hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.node_norm = nn.LayerNorm(hidden_dim)
        self.edge_norm = nn.LayerNorm(hidden_dim)

    def forward(self, h: torch.Tensor, e: torch.Tensor,
                edge_index: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        src, dst = edge_index
        m_fwd = self.msg(torch.cat([h[src], h[dst], e], dim=-1))
        m_rev = self.msg(torch.cat([h[dst], h[src], e], dim=-1))
        agg = torch.zeros_like(h)
        agg.index_add_(0, dst, m_fwd)
        agg.index_add_(0, src, m_rev)
        h_new = self.node_norm(h + self.node_upd(torch.cat([h, agg], dim=-1)))
        e_new = self.edge_norm(e + self.edge_upd(torch.cat([h_new[src], h_new[dst], e], dim=-1)))
        return h_new, e_new


class EdgeCorrectionGNN(nn.Module):
    def __init__(self, node_dim: int, edge_dim: int,
                 hidden_dim: int = 64, n_layers: int = 2):
        super().__init__()
        self.node_enc = nn.Sequential(nn.Linear(node_dim, hidden_dim), nn.SiLU())
        self.edge_enc = nn.Sequential(nn.Linear(edge_dim, hidden_dim), nn.SiLU())
        self.layers = nn.ModuleList([EdgeMPNNLayer(hidden_dim) for _ in range(n_layers)])
        self.delta_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, data: MosaicData) -> torch.Tensor:
        h = self.node_enc(data.x_node)
        e = self.edge_enc(data.x_edge)
        for layer in self.layers:
            h, e = layer(h, e, data.edge_index)
        return self.delta_head(e).squeeze(-1).clamp(-8.0, 8.0)


class DirectFlowGNN(nn.Module):
    def __init__(self, node_dim: int, edge_dim: int,
                 hidden_dim: int = 64, n_layers: int = 2):
        super().__init__()
        self.node_enc = nn.Sequential(nn.Linear(node_dim, hidden_dim), nn.SiLU())
        self.edge_enc = nn.Sequential(nn.Linear(edge_dim, hidden_dim), nn.SiLU())
        self.layers = nn.ModuleList([EdgeMPNNLayer(hidden_dim) for _ in range(n_layers)])
        self.q_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, data: MosaicData) -> torch.Tensor:
        h = self.node_enc(data.x_node)
        e = self.edge_enc(data.x_edge)
        for layer in self.layers:
            h, e = layer(h, e, data.edge_index)
        return self.q_head(e).squeeze(-1) / nL_per_m3


def solve_pressures(data: MosaicData, g_hat: torch.Tensor,
                    jitter: float) -> torch.Tensor:
    n_nodes = data.x_node.shape[0]
    src, dst = data.edge_index
    L = torch.zeros((n_nodes, n_nodes), device=g_hat.device, dtype=g_hat.dtype)
    L.index_put_((src, src), g_hat, accumulate=True)
    L.index_put_((dst, dst), g_hat, accumulate=True)
    L.index_put_((src, dst), -g_hat, accumulate=True)
    L.index_put_((dst, src), -g_hat, accumulate=True)

    ref = int(data.ref_node_index)
    keep = torch.ones(n_nodes, dtype=torch.bool, device=g_hat.device)
    keep[ref] = False
    L_red = L[keep][:, keep]
    rhs = data.source_sink[keep].to(device=g_hat.device, dtype=g_hat.dtype)
    L_red = L_red + torch.eye(L_red.shape[0], device=g_hat.device, dtype=g_hat.dtype) * float(jitter)
    p_red = torch.linalg.solve(L_red, rhs)
    p = torch.zeros(n_nodes, device=g_hat.device, dtype=g_hat.dtype)
    p[keep] = p_red
    return p


def physics_forward(data: MosaicData, delta: torch.Tensor,
                    jitter: float) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    g_hat = data.g_pois.to(delta.device) * torch.exp(delta)
    p = solve_pressures(data, g_hat, jitter=jitter)
    src, dst = data.edge_index
    q_hat = g_hat * (p[src] - p[dst])
    return q_hat, p, g_hat


def poisson_baseline(data: MosaicData, jitter: float) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    delta = torch.zeros_like(data.g_pois)
    return physics_forward(data, delta, jitter)


def split_masks(data: MosaicData, val_fraction: float, seed: int) -> Tuple[torch.Tensor, torch.Tensor]:
    valid_idx = torch.where(data.valid_mask)[0].cpu().numpy()
    rng = np.random.default_rng(int(seed))
    rng.shuffle(valid_idx)
    n_val = max(1, int(round(float(val_fraction) * len(valid_idx)))) if len(valid_idx) else 0
    val_idx = set(int(i) for i in valid_idx[:n_val])
    train = data.valid_mask.clone()
    val = torch.zeros_like(data.valid_mask)
    for i in val_idx:
        train[i] = False
        val[i] = True
    return train, val


def _loss_mse_nls(q_hat: torch.Tensor, q_obs: torch.Tensor,
                  mask: torch.Tensor) -> torch.Tensor:
    if not bool(mask.any()):
        return q_hat.new_tensor(0.0)
    resid = (q_hat - q_obs) * nL_per_m3
    return (resid[mask] ** 2).mean()


def _rmse_nrmse_nls(q_hat: torch.Tensor, q_obs: torch.Tensor,
                    mask: torch.Tensor) -> Tuple[float, float]:
    if not bool(mask.any()):
        return float("nan"), float("nan")
    pred = q_hat.detach()
    obs = q_obs.detach()
    resid = (pred - obs) * nL_per_m3
    obs_nls = obs * nL_per_m3
    rmse = torch.sqrt((resid[mask] ** 2).mean())
    obs_rms = torch.sqrt((obs_nls[mask] ** 2).mean()).clamp_min(1e-30)
    return float(rmse.cpu()), float((rmse / obs_rms).cpu())


def train_physics_model(data: MosaicData, train_mask: torch.Tensor, args,
                        device: torch.device, label: str,
                        val_mask: Optional[torch.Tensor] = None) -> Tuple[EdgeCorrectionGNN, List[dict], List[dict]]:
    model = EdgeCorrectionGNN(
        data.x_node.shape[1], data.x_edge.shape[1],
        hidden_dim=args.hidden_dim, n_layers=args.layers).to(device)
    opt_cls = torch.optim.AdamW if args.optimizer == "adamw" else torch.optim.Adam
    opt = opt_cls(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    history: List[dict] = []
    validation_history: List[dict] = []
    d = data.to(device)
    train_mask = train_mask.to(device)
    val_mask_dev = val_mask.to(device) if val_mask is not None else None
    has_validation = val_mask_dev is not None and bool(val_mask_dev.any())
    iterator: Iterable[int] = range(1, args.epochs + 1)
    if args.use_tqdm and tqdm is not None:
        iterator = tqdm(iterator, desc=f"{label}/physics", dynamic_ncols=True)
    for epoch in iterator:
        opt.zero_grad()
        delta = model(d)
        q_hat, _, _ = physics_forward(d, delta, args.jitter)
        q_loss = _loss_mse_nls(q_hat, d.q_obs, train_mask)
        delta_loss = (delta ** 2).mean()
        loss = q_loss + float(args.lambda_delta) * delta_loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        opt.step()
        row = {
            "epoch": epoch,
            "model": "physics_gnn",
            "loss": float(loss.detach().cpu()),
            "q_loss": float(q_loss.detach().cpu()),
            "delta_loss": float(delta_loss.detach().cpu()),
        }
        history.append(row)
        if has_validation:
            with torch.no_grad():
                eval_delta = model(d)
                eval_q_hat, _, _ = physics_forward(d, eval_delta, args.jitter)
                train_loss = _loss_mse_nls(eval_q_hat, d.q_obs, train_mask)
                val_loss = _loss_mse_nls(eval_q_hat, d.q_obs, val_mask_dev)
                train_rmse, train_nrmse = _rmse_nrmse_nls(eval_q_hat, d.q_obs, train_mask)
                val_rmse, val_nrmse = _rmse_nrmse_nls(eval_q_hat, d.q_obs, val_mask_dev)
            validation_history.append({
                "epoch": epoch,
                "train_loss": float(train_loss.cpu()),
                "val_loss": float(val_loss.cpu()),
                "train_rmse": train_rmse,
                "val_rmse": val_rmse,
                "train_nrmse": train_nrmse,
                "val_nrmse": val_nrmse,
            })
        if hasattr(iterator, "set_postfix"):
            postfix = {"loss": f"{row['loss']:.3e}", "q": f"{row['q_loss']:.3e}"}
            if has_validation and validation_history:
                postfix["val"] = f"{validation_history[-1]['val_loss']:.3e}"
            iterator.set_postfix(**postfix)
    return model, history, validation_history


def train_direct_model(data: MosaicData, train_mask: torch.Tensor, args,
                       device: torch.device, label: str) -> Tuple[DirectFlowGNN, List[dict]]:
    model = DirectFlowGNN(
        data.x_node.shape[1], data.x_edge.shape[1],
        hidden_dim=args.hidden_dim, n_layers=args.layers).to(device)
    opt_cls = torch.optim.AdamW if args.optimizer == "adamw" else torch.optim.Adam
    opt = opt_cls(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    history: List[dict] = []
    d = data.to(device)
    train_mask = train_mask.to(device)
    iterator: Iterable[int] = range(1, args.epochs + 1)
    if args.use_tqdm and tqdm is not None:
        iterator = tqdm(iterator, desc=f"{label}/direct", dynamic_ncols=True)
    for epoch in iterator:
        opt.zero_grad()
        q_hat = model(d)
        loss = _loss_mse_nls(q_hat, d.q_obs, train_mask)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        opt.step()
        row = {"epoch": epoch, "model": "direct_gnn", "loss": float(loss.detach().cpu()), "q_loss": float(loss.detach().cpu()), "delta_loss": float("nan")}
        history.append(row)
        if hasattr(iterator, "set_postfix"):
            iterator.set_postfix(loss=f"{row['loss']:.3e}")
    return model, history


@torch.no_grad()
def evaluate_arrays(data: MosaicData, q_hat: torch.Tensor,
                    mask: torch.Tensor) -> dict:
    q = data.q_obs.detach().cpu().numpy() * nL_per_m3
    pred = q_hat.detach().cpu().numpy() * nL_per_m3
    m = mask.detach().cpu().numpy().astype(bool)
    if not np.any(m):
        return {"n": 0, "RMSE_nL_s": np.nan, "normalized_RMSE": np.nan, "MAE_nL_s": np.nan, "pearson_corr": np.nan, "R2": np.nan}
    resid = pred[m] - q[m]
    rmse = float(np.sqrt(np.mean(resid ** 2)))
    mae = float(np.mean(np.abs(resid)))
    obs_rms = float(np.sqrt(np.mean(q[m] ** 2)))
    if np.std(q[m]) <= 1e-30 or np.std(pred[m]) <= 1e-30:
        corr = np.nan
    else:
        corr = float(np.corrcoef(q[m], pred[m])[0, 1])
    ss_res = float(np.sum((q[m] - pred[m]) ** 2))
    ss_tot = float(np.sum((q[m] - np.mean(q[m])) ** 2))
    return {
        "n": int(np.sum(m)),
        "RMSE_nL_s": rmse,
        "normalized_RMSE": float(rmse / max(obs_rms, 1e-30)),
        "MAE_nL_s": mae,
        "pearson_corr": corr,
        "R2": float(1.0 - ss_res / ss_tot) if ss_tot > 1e-30 else np.nan,
    }


@torch.no_grad()
def mass_residual_rmse(data: MosaicData, q_hat: torch.Tensor) -> float:
    src, dst = data.edge_index.cpu()
    net = torch.zeros(data.x_node.shape[0], dtype=torch.float32)
    q_cpu = q_hat.detach().cpu() * nL_per_m3
    net.index_add_(0, src, q_cpu)
    net.index_add_(0, dst, -q_cpu)
    residual = net - data.source_sink.cpu() * nL_per_m3
    return float(torch.sqrt((residual ** 2).mean()).cpu())


@torch.no_grad()
def mass_residual_norm(data: MosaicData, q_hat: torch.Tensor) -> float:
    src, dst = data.edge_index.cpu()
    net = torch.zeros(data.x_node.shape[0], dtype=torch.float32)
    q_cpu = q_hat.detach().cpu() * nL_per_m3
    net.index_add_(0, src, q_cpu)
    net.index_add_(0, dst, -q_cpu)
    residual = net - data.source_sink.cpu() * nL_per_m3
    return float(torch.linalg.vector_norm(residual).cpu())


def write_csv(path: Path, rows: Sequence[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, obj: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, allow_nan=True)


@torch.no_grad()
def collect_edge_rows(data: MosaicData, q_hat: torch.Tensor, p: torch.Tensor,
                      g_hat: torch.Tensor, train_mask: torch.Tensor,
                      val_mask: torch.Tensor) -> List[dict]:
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
        row["distance_to_A"] = float(dist_a[i]) if math.isfinite(float(dist_a[i])) else float("nan")
        row["distance_to_V"] = float(dist_v[i]) if math.isfinite(float(dist_v[i])) else float("nan")


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
    deg = np.bincount(np.concatenate([src, dst]), minlength=data.x_node.shape[0]).astype(float)
    return deg, deg[src], deg[dst]


def _shortest_distances_from_sources(data: MosaicData, source_indices: Sequence[int]) -> np.ndarray:
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
    deg, deg_src, deg_dst = _edge_endpoint_degrees(data)
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
                out[int(ei)]["node_betweenness_src"] = float(node_bc.get(int(s_i), float("nan")))
                out[int(ei)]["node_betweenness_dst"] = float(node_bc.get(int(d_i), float("nan")))
                out[int(ei)]["edge_betweenness"] = float(edge_bc.get(key, edge_bc.get(rev, float("nan"))))
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
        q1 = sign * amp1 * complex(math.cos(phi1), math.sin(phi1)) if math.isfinite(amp1) and math.isfinite(phi1) else complex(float("nan"), float("nan"))
        q2 = sign * amp2 * complex(math.cos(phi2), math.sin(phi2)) if math.isfinite(amp2) and math.isfinite(phi2) else complex(float("nan"), float("nan"))
        phases_h1[ei] = math.atan2(q1.imag, q1.real) if math.isfinite(q1.real) and math.isfinite(q1.imag) else float("nan")
        out[ei] = {
            "abs_H1": float(abs(q1)) if math.isfinite(q1.real) else float("nan"),
            "abs_H2": float(abs(q2)) if math.isfinite(q2.real) else float("nan"),
            "harmonic_ratio": float(abs(q2) / (abs(q1) + 1e-30)) if math.isfinite(q1.real) and math.isfinite(q2.real) else float("nan"),
            "phase_H1": phases_h1[ei],
            "phase_H2": math.atan2(q2.imag, q2.real) if math.isfinite(q2.real) and math.isfinite(q2.imag) else float("nan"),
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
            for j in neigh if math.isfinite(float(phases_h1[j]))
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


def plot_loss(history: Sequence[dict], out_dir: Path) -> None:
    if not history:
        return
    plt.figure(figsize=(7, 4))
    for model_name in sorted({r["model"] for r in history}):
        rows = [r for r in history if r["model"] == model_name]
        plt.plot([r["epoch"] for r in rows], [r["loss"] for r in rows], label=model_name)
    plt.xlabel("epoch")
    plt.ylabel("training loss")
    plt.yscale("log")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "training_loss.png", dpi=180)
    plt.close()


def plot_validation_history(history: Sequence[dict], out_dir: Path) -> None:
    if not history:
        return
    epochs = [int(r["epoch"]) for r in history]
    plt.figure(figsize=(7, 4))
    plt.plot(epochs, [float(r["train_loss"]) for r in history], label="train")
    plt.plot(epochs, [float(r["val_loss"]) for r in history], label="validation")
    plt.xlabel("epoch")
    plt.ylabel("flow MSE (nL/s)^2")
    plt.yscale("log")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "validation_loss.png", dpi=180)
    plt.close()


def _linear_fit(X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, float, np.ndarray, float]:
    try:
        from sklearn.linear_model import LinearRegression
        model = LinearRegression()
        model.fit(X, y)
        pred = model.predict(X)
        coef = np.asarray(model.coef_, dtype=float)
        intercept = float(model.intercept_)
    except Exception:
        X_aug = np.column_stack([np.ones(X.shape[0]), X])
        beta, *_ = np.linalg.lstsq(X_aug, y, rcond=None)
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


def collect_regression_edges(data: MosaicData,
                             edge_rows: Sequence[dict]) -> List[dict]:
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


def plot_regression_fit(rows: Sequence[dict], model_name: str,
                        y: np.ndarray, pred: np.ndarray,
                        out_dir: Path) -> None:
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


def run_delta_regressions(edge_rows: Sequence[dict], out_dir: Path) -> Tuple[List[dict], List[dict]]:
    reg_rows = [r for r in edge_rows if int(r.get("valid_obs", 0)) == 1]
    if not reg_rows:
        return [], []

    specs = [
        ("Model A", ["log_radius", "log_length"]),
        ("Model B", ["log_G_pois"]),
        ("Model C", ["log_radius", "log_length", "abs_Q_obs",
                     "degree_src", "degree_dst"]),
        ("Model D", ["log_radius", "log_length", "abs_Q_obs",
                     "degree_src", "degree_dst", "distance_to_A", "distance_to_V"]),
        ("Model E", ["log_radius", "log_length", "abs_Q_obs",
                     "harmonic_ratio", "phase_dispersion_H1"]),
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
        coef_std, intercept_std, pred_std, r2_std = _linear_fit(X_std, y_std[:, 0])
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
            "standardized_adjusted_R2": _adjusted_r2(r2_std, int(X_fit.shape[0]), len(features)),
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


def interpretation_summary(label: str, validation_history: Sequence[dict],
                           regression_summary: Sequence[dict],
                           regression_coefficients: Sequence[dict]) -> None:
    print(f"\n[{label}] Interpretation summary")
    if validation_history:
        first = validation_history[0]
        last = validation_history[-1]
        best = min(validation_history, key=lambda r: float(r["val_loss"]))
        trend = "decreased" if float(last["val_loss"]) < float(first["val_loss"]) else "did not decrease"
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


def plot_prediction(edge_rows: Sequence[dict], out_dir: Path, prefix: str) -> None:
    rows = [r for r in edge_rows if int(r["valid_obs"]) == 1]
    if not rows:
        return
    q_obs = np.asarray([r["Q_obs_nL_s"] for r in rows], dtype=float)
    q_hat = np.asarray([r["Q_hat_nL_s"] for r in rows], dtype=float)
    resid = q_obs - q_hat
    radius = np.asarray([r["radius_m"] for r in rows], dtype=float)
    length = np.asarray([r["length_m"] for r in rows], dtype=float)
    g_pois = np.asarray([r["G_pois"] for r in rows], dtype=float)

    plt.figure(figsize=(5, 5))
    plt.scatter(q_obs, q_hat, s=12, alpha=0.6)
    lo = float(np.nanmin([q_obs.min(), q_hat.min()]))
    hi = float(np.nanmax([q_obs.max(), q_hat.max()]))
    plt.plot([lo, hi], [lo, hi], color="black", lw=1)
    plt.xlabel("observed Q_DC (nL/s)")
    plt.ylabel("predicted Q_DC (nL/s)")
    plt.tight_layout()
    plt.savefig(out_dir / f"{prefix}_q_pred_vs_obs.png", dpi=180)
    plt.close()

    plt.figure(figsize=(6, 4))
    plt.hist(resid, bins=50)
    plt.xlabel("residual Q_DC (nL/s)")
    plt.ylabel("count")
    plt.tight_layout()
    plt.savefig(out_dir / f"{prefix}_residual_hist.png", dpi=180)
    plt.close()

    c_vals = []
    for row in rows:
        try:
            c_vals.append(float(row["C"]))
        except (TypeError, ValueError):
            c_vals.append(float("nan"))
    c = np.asarray(c_vals, dtype=float)
    if np.isfinite(c).any():
        plt.figure(figsize=(6, 4))
        plt.hist(c[np.isfinite(c)], bins=50)
        plt.xlabel("C = G_hat / G_pois")
        plt.ylabel("count")
        plt.tight_layout()
        plt.savefig(out_dir / f"{prefix}_C_hist.png", dpi=180)
        plt.close()

        for x, name, xlabel, logx in (
            (radius, "radius", "radius (m)", True),
            (length, "length", "length (m)", True),
            (g_pois, "G_pois", "Poiseuille conductance", True),
        ):
            plt.figure(figsize=(5.5, 4))
            plt.scatter(x, c, s=12, alpha=0.55)
            if logx:
                plt.xscale("log")
            plt.yscale("log")
            plt.xlabel(xlabel)
            plt.ylabel("C = G_hat / G_pois")
            plt.tight_layout()
            plt.savefig(out_dir / f"{prefix}_C_vs_{name}.png", dpi=180)
            plt.close()


def _edge_midpoints(rows: Sequence[dict]) -> Tuple[np.ndarray, np.ndarray]:
    xs = []
    ys = []
    for r in rows:
        x = safe_float(r.get("x_mid"))
        y = safe_float(r.get("y_mid"))
        xs.append(x)
        ys.append(y)
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


def _plot_edge_map(rows: Sequence[dict], key: str, out_dir: Path,
                   filename: str, label: str) -> None:
    vals = np.asarray([safe_float(r.get(key)) for r in rows], dtype=float)
    x, y = _edge_midpoints(rows)
    mask = np.isfinite(vals) & np.isfinite(x) & np.isfinite(y)
    if not np.any(mask):
        return
    plt.figure(figsize=(6, 5))
    sc = plt.scatter(x[mask], y[mask], c=vals[mask], s=14)
    plt.gca().invert_yaxis()
    plt.gca().set_aspect("equal", adjustable="box")
    plt.colorbar(sc, label=label)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.tight_layout()
    plt.savefig(out_dir / filename, dpi=180)
    plt.close()


def _scatter(rows: Sequence[dict], x_key: str, y_key: str,
             out_dir: Path, filename: str, xlabel: str, ylabel: str,
             logx: bool = False, logy: bool = False) -> None:
    x = np.asarray([safe_float(r.get(x_key)) for r in rows], dtype=float)
    y = np.asarray([safe_float(r.get(y_key)) for r in rows], dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    if not np.any(mask):
        return
    if logx:
        mask &= x > 0
    if logy:
        mask &= y > 0
    if not np.any(mask):
        return
    plt.figure(figsize=(5.5, 4))
    plt.scatter(x[mask], y[mask], s=12, alpha=0.55)
    if logx:
        plt.xscale("log")
    if logy:
        plt.yscale("log")
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.tight_layout()
    plt.savefig(out_dir / filename, dpi=180)
    plt.close()


def _hist(rows: Sequence[dict], key: str, out_dir: Path,
          filename: str, xlabel: str, logx: bool = False) -> None:
    vals = np.asarray([safe_float(r.get(key)) for r in rows], dtype=float)
    vals = vals[np.isfinite(vals)]
    if logx:
        vals = vals[vals > 0]
    if vals.size == 0:
        return
    plt.figure(figsize=(6, 4))
    plt.hist(vals, bins=50)
    if logx:
        plt.xscale("log")
    plt.xlabel(xlabel)
    plt.ylabel("count")
    plt.tight_layout()
    plt.savefig(out_dir / filename, dpi=180)
    plt.close()


def _binned_error(rows: Sequence[dict], by_key: str, out_dir: Path,
                  filename: str, xlabel: str) -> None:
    vals = np.asarray([safe_float(r.get(by_key)) for r in rows], dtype=float)
    err = np.asarray([safe_float(r.get("abs_error_nL_s")) for r in rows], dtype=float)
    mask = np.isfinite(vals) & np.isfinite(err)
    vals = vals[mask]
    err = err[mask]
    if vals.size < 5:
        return
    qs = np.unique(np.nanquantile(vals, np.linspace(0.0, 1.0, 6)))
    if qs.size < 3:
        return
    plot_rows = []
    for lo, hi in zip(qs[:-1], qs[1:]):
        m = (vals >= lo) & (vals <= hi if hi == qs[-1] else vals < hi)
        if np.any(m):
            plot_rows.append({
                "bin_center": float(np.nanmean(vals[m])),
                "mean_abs_error": float(np.nanmean(err[m])),
                "n": int(np.sum(m)),
                "lo": float(lo),
                "hi": float(hi),
            })
    write_csv(out_dir / filename.replace(".png", ".csv"), plot_rows)
    if not plot_rows:
        return
    plt.figure(figsize=(6, 4))
    plt.plot([r["bin_center"] for r in plot_rows],
             [r["mean_abs_error"] for r in plot_rows], marker="o")
    plt.xlabel(xlabel)
    plt.ylabel("mean |Q_obs - Q_hat| (nL/s)")
    plt.tight_layout()
    plt.savefig(out_dir / filename, dpi=180)
    plt.close()


def plot_group_box(rows: Sequence[dict], value_key: str, group_key: str,
                   out_dir: Path, filename: str, ylabel: str) -> None:
    groups = sorted({str(r.get(group_key, "")) for r in rows if str(r.get(group_key, ""))})
    data = []
    labels = []
    for g in groups:
        vals = [safe_float(r.get(value_key)) for r in rows if str(r.get(group_key, "")) == g]
        vals = [v for v in vals if math.isfinite(v)]
        if vals:
            data.append(vals)
            labels.append(g)
    if not data:
        return
    plt.figure(figsize=(7, 4))
    plt.boxplot(data, labels=labels, showfliers=False)
    plt.xticks(rotation=25, ha="right")
    plt.ylabel(ylabel)
    plt.tight_layout()
    plt.savefig(out_dir / filename, dpi=180)
    plt.close()


def plot_diagnostic_suite(rows: Sequence[dict], node_rows: Sequence[dict],
                          conservation_rows: Sequence[dict], out_dir: Path) -> None:
    valid = [r for r in rows if int(r.get("valid_obs", 0)) == 1]
    if not valid:
        return
    _scatter(valid, "Q_obs_nL_s", "residual_nL_s", out_dir,
             "residual_vs_Q_obs.png", "Q_obs (nL/s)", "Q_obs - Q_hat (nL/s)")
    _scatter(valid, "abs_Q_obs", "abs_error_nL_s", out_dir,
             "abs_error_vs_abs_Q_obs.png", "|Q_obs| (nL/s)", "|error| (nL/s)")
    _binned_error(valid, "abs_Q_obs", out_dir, "binned_error_by_abs_Q_obs.png", "|Q_obs|")
    _binned_error(valid, "radius_m", out_dir, "binned_error_by_radius.png", "radius (m)")
    _binned_error(valid, "G_pois", out_dir, "binned_error_by_G_pois.png", "G_pois")
    _plot_edge_map(valid, "residual_nL_s", out_dir, "signed_error_map.png", "Q_obs - Q_hat (nL/s)")
    _plot_edge_map(valid, "abs_error_nL_s", out_dir, "absolute_error_map.png", "|error| (nL/s)")

    _hist(valid, "delta", out_dir, "delta_hist.png", "delta = log C")
    _hist(valid, "C", out_dir, "C_hist_linear.png", "C")
    _hist(valid, "C", out_dir, "C_hist_logx.png", "C", logx=True)
    _scatter(valid, "log_radius", "delta", out_dir, "delta_vs_log_radius.png", "log radius", "delta")
    _scatter(valid, "log_length", "delta", out_dir, "delta_vs_log_length.png", "log length", "delta")
    _scatter(valid, "log_G_pois", "delta", out_dir, "delta_vs_log_G_pois.png", "log G_pois", "delta")
    _scatter(valid, "radius_m", "C", out_dir, "C_vs_radius.png", "radius (m)", "C", logx=True, logy=True)
    _scatter(valid, "length_m", "C", out_dir, "C_vs_length.png", "length (m)", "C", logx=True, logy=True)
    _scatter(valid, "G_pois", "C", out_dir, "C_vs_G_pois.png", "G_pois", "C", logx=True, logy=True)
    _scatter(valid, "abs_Q_obs", "delta", out_dir, "delta_vs_abs_Q_obs.png", "|Q_obs| (nL/s)", "delta")
    _scatter(valid, "pressure_drop_Pa", "delta", out_dir, "delta_vs_pressure_drop.png", "pressure drop (Pa)", "delta")
    _scatter(valid, "residual_nL_s", "delta", out_dir, "delta_vs_residual.png", "Q_obs - Q_hat (nL/s)", "delta")
    _plot_edge_map(valid, "C", out_dir, "C_map.png", "C")
    _plot_edge_map(valid, "delta", out_dir, "delta_map.png", "delta")

    _hist(valid, "pressure_drop_Pa", out_dir, "pressure_drop_hist.png", "pressure drop (Pa)")
    _scatter(valid, "Q_hat_nL_s", "pressure_drop_Pa", out_dir,
             "pressure_drop_vs_flow.png", "Q_hat (nL/s)", "pressure drop (Pa)")
    _scatter(valid, "radius_m", "pressure_drop_Pa", out_dir,
             "pressure_drop_vs_radius.png", "radius (m)", "pressure drop (Pa)", logx=True)
    _plot_edge_map(valid, "pressure_drop_Pa", out_dir, "pressure_drop_map.png", "pressure drop (Pa)")

    _scatter(valid, "distance_to_A", "delta", out_dir, "delta_vs_distance_to_A.png", "distance to A/source", "delta")
    _scatter(valid, "distance_to_V", "delta", out_dir, "delta_vs_distance_to_V.png", "distance to V/sink", "delta")
    _scatter(valid, "distance_to_A", "residual_nL_s", out_dir, "residual_vs_distance_to_A.png", "distance to A/source", "residual")
    _scatter(valid, "distance_to_V", "residual_nL_s", out_dir, "residual_vs_distance_to_V.png", "distance to V/sink", "residual")
    plot_group_box(valid, "delta", "topology_class", out_dir, "delta_by_topology_class.png", "delta")
    plot_group_box(valid, "residual_nL_s", "topology_class", out_dir, "residual_by_topology_class.png", "residual (nL/s)")
    plot_group_box(valid, "C", "topology_class", out_dir, "C_by_topology_class.png", "C")
    plot_group_box([r for r in valid if r.get("split") == "val"], "abs_error_nL_s", "topology_class",
                   out_dir, "validation_error_by_topology_class.png", "|validation error| (nL/s)")

    for key, filename, xlabel in (
        ("abs_H1", "delta_vs_abs_H1.png", "|H1|"),
        ("abs_H2", "delta_vs_abs_H2.png", "|H2|"),
        ("harmonic_ratio", "delta_vs_harmonic_ratio.png", "|H2|/(|H1|+eps)"),
        ("phase_dispersion_H1", "delta_vs_phase_dispersion_H1.png", "H1 phase dispersion"),
    ):
        _scatter(valid, key, "delta", out_dir, filename, xlabel, "delta")
    _scatter(valid, "harmonic_ratio", "residual_nL_s", out_dir,
             "residual_vs_harmonic_ratio.png", "harmonic ratio", "residual (nL/s)")
    _scatter(valid, "harmonic_ratio", "C", out_dir,
             "C_vs_harmonic_ratio.png", "harmonic ratio", "C", logy=True)

    _hist(conservation_rows, "conservation_residual_nL_s", out_dir,
          "conservation_residual_hist.png", "B Q_hat - s (nL/s)")
    _plot_node_map(conservation_rows, "conservation_residual_nL_s", out_dir,
                   "conservation_residual_spatial.png", "B Q_hat - s (nL/s)")


def _plot_node_map(rows: Sequence[dict], key: str, out_dir: Path,
                   filename: str, label: str) -> None:
    vals = np.asarray([safe_float(r.get(key)) for r in rows], dtype=float)
    x = np.asarray([safe_float(r.get("x")) for r in rows], dtype=float)
    y = np.asarray([safe_float(r.get("y")) for r in rows], dtype=float)
    mask = np.isfinite(vals) & np.isfinite(x) & np.isfinite(y)
    if not np.any(mask):
        return
    plt.figure(figsize=(6, 5))
    sc = plt.scatter(x[mask], y[mask], c=vals[mask], s=14)
    plt.gca().invert_yaxis()
    plt.gca().set_aspect("equal", adjustable="box")
    plt.colorbar(sc, label=label)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.tight_layout()
    plt.savefig(out_dir / filename, dpi=180)
    plt.close()


def write_top_edge_tables(rows: Sequence[dict], out_dir: Path) -> None:
    def keep(row: dict) -> dict:
        keys = [
            "edge_id", "source", "target", "radius_m", "length_m", "G_pois",
            "G_hat", "C", "delta", "Q_obs_nL_s", "Q_hat_nL_s",
            "residual_nL_s", "pressure_drop_Pa", "degree_src", "degree_dst",
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
    fields = ["node_id", "pressure_Pa", "standardized_degree_feature",
              "source_sink_nL_s", "x", "y", "distance_to_A", "distance_to_V",
              "boundary_kind"]
    write_csv(out_dir / "top50_nodes_by_pressure.csv", [
        {k: r.get(k, "") for k in fields} for r in rows
    ])


def plot_pressure(node_rows: Sequence[dict], edge_rows: Sequence[dict], out_dir: Path,
                  prefix: str) -> None:
    p = np.asarray([r["pressure_Pa"] for r in node_rows], dtype=float)
    plt.figure(figsize=(6, 4))
    plt.hist(p[np.isfinite(p)], bins=50)
    plt.xlabel("pressure (Pa)")
    plt.ylabel("count")
    plt.tight_layout()
    plt.savefig(out_dir / f"{prefix}_pressure_hist.png", dpi=180)
    plt.close()

    spatial = [
        r for r in node_rows
        if math.isfinite(float(r["x"])) and math.isfinite(float(r["y"]))
    ]
    if spatial:
        plt.figure(figsize=(6, 5))
        sc = plt.scatter([r["x"] for r in spatial], [r["y"] for r in spatial],
                         c=[r["pressure_Pa"] for r in spatial], s=12)
        plt.gca().invert_yaxis()
        plt.gca().set_aspect("equal", adjustable="box")
        plt.colorbar(sc, label="pressure (Pa)")
        plt.xlabel("x")
        plt.ylabel("y")
        plt.tight_layout()
        plt.savefig(out_dir / f"{prefix}_pressure_spatial.png", dpi=180)
        plt.close()

    _scatter(node_rows, "distance_to_A", "pressure_Pa", out_dir,
             f"{prefix}_pressure_vs_distance_to_A.png", "distance to A/source", "pressure (Pa)")
    _scatter(node_rows, "distance_to_V", "pressure_Pa", out_dir,
             f"{prefix}_pressure_vs_distance_to_V.png", "distance to V/sink", "pressure (Pa)")


def metrics_rows(data: MosaicData, outputs: Dict[str, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
                 train_mask: torch.Tensor, val_mask: torch.Tensor) -> List[dict]:
    rows = []
    for name, (q_hat, _p, _g) in outputs.items():
        for split, mask in (("train", train_mask), ("val", val_mask), ("all", data.valid_mask)):
            if split == "val" and not bool(val_mask.any()):
                continue
            row = {"model": name, "split": split}
            row.update(evaluate_arrays(data, q_hat, mask))
            row["mass_residual_RMSE_nL_s"] = mass_residual_rmse(data, q_hat)
            rows.append(row)
    return rows


def run_experiment(base_data: MosaicData, args, device: torch.device,
                   out_dir: Path, label: str,
                   train_mask: torch.Tensor, val_mask: torch.Tensor,
                   graph=None, train_direct: bool = True) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    data = base_data
    history: List[dict] = []

    has_validation = bool(val_mask.any())
    physics_model, hist, validation_history = train_physics_model(
        data, train_mask, args, device, label,
        val_mask=val_mask if has_validation else None)
    physics_history = list(hist)
    history.extend(hist)
    direct_model = None
    if train_direct:
        direct_model, hist = train_direct_model(data, train_mask, args, device, label)
        history.extend(hist)

    d = data.to(device)
    with torch.no_grad():
        delta = physics_model(d)
        physics_out = physics_forward(d, delta, args.jitter)
        pois_out = poisson_baseline(d, args.jitter)
        if direct_model is not None:
            direct_q = direct_model(d)
            zero_p = torch.full((d.x_node.shape[0],), float("nan"), device=device)
            direct_out = (direct_q, zero_p, torch.full_like(d.g_pois, float("nan")))

    outputs = {
        "poiseuille": pois_out,
        "physics_gnn": physics_out,
    }
    if direct_model is not None:
        outputs["direct_gnn"] = direct_out
    metrics = metrics_rows(data, outputs, train_mask, val_mask)
    write_csv(out_dir / "metrics.csv", metrics)
    write_json(out_dir / "metrics.json", metrics)
    write_csv(out_dir / "training_history.csv", history)
    if validation_history:
        write_csv(out_dir / "validation_history.csv", validation_history)
    write_json(out_dir / "feature_stats.json", data.feature_stats)

    q_hat, p, g_hat = physics_out
    edge_rows = collect_edge_rows(data, q_hat, p, g_hat, train_mask, val_mask)
    add_edge_midpoints(data, edge_rows)
    node_rows = collect_node_rows(data, p)
    add_node_distance_diagnostics(data, node_rows)
    write_csv(out_dir / "edge_predictions_physics_gnn.csv", edge_rows)
    write_csv(out_dir / "node_pressures_physics_gnn.csv", node_rows)
    write_top_pressure_nodes(node_rows, out_dir)
    conservation_rows = collect_conservation_rows(data, q_hat)
    write_csv(out_dir / "node_conservation_residuals.csv", conservation_rows)
    max_cons = max((abs(float(r["conservation_residual_nL_s"])) for r in conservation_rows), default=float("nan"))
    print(f"[{label}] max |B Q_hat - s| = {max_cons:.4e} nL/s")
    for r in conservation_rows:
        if r.get("boundary_kind") in ("source", "sink"):
            print(
                f"[{label}] boundary {r['node_id']} {r['boundary_kind']}: "
                f"pred={float(r['predicted_net_flow_nL_s']):+.4g} nL/s "
                f"target={float(r['source_sink_value_nL_s']):+.4g} nL/s"
            )

    enriched_edge_rows = enrich_edge_rows(data, graph, edge_rows) if graph is not None else edge_rows
    write_csv(out_dir / "edge_diagnostics_physics_gnn.csv", enriched_edge_rows)
    write_csv(out_dir / "harmonic_diagnostics.csv", [
        {k: r.get(k, "") for k in [
            "edge_id", "source", "target", "abs_H1", "abs_H2",
            "harmonic_ratio", "phase_H1", "phase_H2", "phase_dispersion_H1",
            "delta", "C", "residual_nL_s",
        ]}
        for r in enriched_edge_rows
    ])
    write_top_edge_tables(enriched_edge_rows, out_dir)

    regression_edge_rows = collect_regression_edges(data, enriched_edge_rows)
    write_csv(out_dir / "delta_regression_edges.csv", regression_edge_rows)
    regression_summary, regression_coefficients = run_delta_regressions(
        regression_edge_rows, out_dir)

    q_pois, p_pois, g_pois = pois_out
    write_csv(
        out_dir / "edge_predictions_poiseuille.csv",
        collect_edge_rows(data, q_pois, p_pois, g_pois, train_mask, val_mask),
    )
    write_csv(out_dir / "node_pressures_poiseuille.csv", collect_node_rows(data, p_pois))

    if direct_model is not None:
        direct_edge_rows = collect_edge_rows(data, direct_q, p, g_hat, train_mask, val_mask)
        for row in direct_edge_rows:
            row["G_hat"] = ""
            row["delta"] = ""
            row["C"] = ""
            row["pressure_drop_Pa"] = ""
        write_csv(out_dir / "edge_predictions_direct_gnn.csv", direct_edge_rows)

    ckpt = {
        "physics_model_state_dict": physics_model.state_dict(),
        "args": vars(args),
        "node_ids": [str(n) for n in data.node_ids],
        "edge_ids": [(str(u), str(v)) for u, v in data.edge_ids],
        "feature_stats": data.feature_stats,
    }
    if direct_model is not None:
        ckpt["direct_model_state_dict"] = direct_model.state_dict()
    torch.save(ckpt, out_dir / "models.pt")

    plot_loss(history, out_dir)
    plot_validation_history(validation_history, out_dir)
    plot_prediction(edge_rows, out_dir, "physics_gnn")
    plot_pressure(node_rows, edge_rows, out_dir, "physics_gnn")
    plot_diagnostic_suite(enriched_edge_rows, node_rows, conservation_rows, out_dir)
    if direct_model is not None:
        plot_prediction(direct_edge_rows, out_dir, "direct_gnn")
    interpretation_summary(
        label, validation_history, regression_summary, regression_coefficients)

    train_metrics = evaluate_arrays(data, q_hat, train_mask)
    val_metrics = evaluate_arrays(data, q_hat, val_mask) if bool(val_mask.any()) else {
        "RMSE_nL_s": float("nan"), "normalized_RMSE": float("nan"),
        "MAE_nL_s": float("nan"), "pearson_corr": float("nan"),
    }
    delta_np = delta.detach().cpu().numpy()
    c_np = np.exp(delta_np)
    p_np = p.detach().cpu().numpy()
    summary = {
        "K": int(args.layers),
        "hidden_dim": int(args.hidden_dim),
        "lambda_delta": float(args.lambda_delta),
        "seed": int(args.seed),
        "train_loss_final": float(validation_history[-1]["train_loss"]) if validation_history else float(physics_history[-1]["q_loss"]),
        "val_loss_final": float(validation_history[-1]["val_loss"]) if validation_history else float("nan"),
        "train_RMSE": float(train_metrics["RMSE_nL_s"]),
        "val_RMSE": float(val_metrics["RMSE_nL_s"]),
        "train_NRMSE": float(train_metrics["normalized_RMSE"]),
        "val_NRMSE": float(val_metrics["normalized_RMSE"]),
        "train_MAE": float(train_metrics["MAE_nL_s"]),
        "val_MAE": float(val_metrics["MAE_nL_s"]),
        "train_Pearson": float(train_metrics["pearson_corr"]),
        "val_Pearson": float(val_metrics["pearson_corr"]),
        "mass_residual_norm": mass_residual_norm(data, q_hat),
        "mean_abs_delta": float(np.nanmean(np.abs(delta_np))),
        "median_abs_delta": float(np.nanmedian(np.abs(delta_np))),
        "mean_C": float(np.nanmean(c_np)),
        "median_C": float(np.nanmedian(c_np)),
        "p95_C": float(np.nanpercentile(c_np, 95)),
        "max_C": float(np.nanmax(c_np)),
        "pressure_min": float(np.nanmin(p_np)),
        "pressure_median": float(np.nanmedian(p_np)),
        "pressure_max": float(np.nanmax(p_np)),
        "pressure_range": float(np.nanmax(p_np) - np.nanmin(p_np)),
        "number_edges": int(len(data.edge_ids)),
        "number_nodes": int(len(data.node_ids)),
        "out_dir": str(out_dir),
    }
    write_csv(out_dir / "run_summary.csv", [summary])
    return summary


def _lambda_label(value: float) -> str:
    return f"{float(value):g}".replace(".", "p").replace("-", "m")


def plot_sweep_summary(rows: Sequence[dict], out_dir: Path) -> None:
    if not rows:
        return
    for key, filename, xlabel, logx in (
        ("K", "sweep_val_nrmse_vs_K.png", "message-passing layers K", False),
        ("hidden_dim", "sweep_val_nrmse_vs_hidden_dim.png", "hidden dimension", False),
        ("lambda_delta", "sweep_val_nrmse_vs_lambda_delta.png", "lambda_delta", True),
    ):
        xs = np.asarray([safe_float(r.get(key)) for r in rows], dtype=float)
        ys = np.asarray([safe_float(r.get("val_NRMSE")) for r in rows], dtype=float)
        mask = np.isfinite(xs) & np.isfinite(ys)
        if logx:
            mask &= xs > 0
        if not np.any(mask):
            continue
        plt.figure(figsize=(6, 4))
        plt.scatter(xs[mask], ys[mask], s=40, alpha=0.75)
        groups = sorted(set(float(x) for x in xs[mask]))
        med = [float(np.nanmedian(ys[mask & (xs == g)])) for g in groups]
        plt.plot(groups, med, color="black", lw=1.2, marker="o", label="median")
        if logx:
            plt.xscale("log")
        plt.xlabel(xlabel)
        plt.ylabel("final validation NRMSE")
        plt.legend()
        plt.tight_layout()
        plt.savefig(out_dir / filename, dpi=180)
        plt.close()


def run_sweep(data: MosaicData, graph, args, device: torch.device,
              out_root: Path) -> List[dict]:
    train_mask, val_mask = split_masks(data, args.val_fraction, args.seed)
    summaries: List[dict] = []
    total = (
        len(args.K_values) * len(args.hidden_dim_values)
        * len(args.lambda_delta_values) * len(args.seeds)
    )
    idx = 0
    for seed in args.seeds:
        for k in args.K_values:
            for hidden_dim in args.hidden_dim_values:
                for lambda_delta in args.lambda_delta_values:
                    idx += 1
                    run_args = copy.copy(args)
                    run_args.seed = int(seed)
                    run_args.layers = int(k)
                    run_args.hidden_dim = int(hidden_dim)
                    run_args.lambda_delta = float(lambda_delta)
                    set_seed(run_args.seed)
                    run_dir = (
                        out_root
                        / f"K{int(k)}_hidden{int(hidden_dim)}_lambda{_lambda_label(float(lambda_delta))}_seed{int(seed)}"
                    )
                    print(
                        f"\n=== Sweep {idx}/{total}: K={k}, hidden={hidden_dim}, "
                        f"lambda_delta={lambda_delta:g}, seed={seed} ==="
                    )
                    summary = run_experiment(
                        data, run_args, device, run_dir,
                        label=f"K{k}_h{hidden_dim}_l{lambda_delta:g}_s{seed}",
                        train_mask=train_mask,
                        val_mask=val_mask,
                        graph=graph,
                        train_direct=False,
                    )
                    summaries.append(summary)
                    write_csv(out_root / "sweep_summary.csv", summaries)
                    plot_sweep_summary(summaries, out_root)
    return summaries


def parse_args(argv: Optional[Sequence[str]] = None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default=str(DEFAULT_CONFIG))
    ap.add_argument("--graph", default=None)
    ap.add_argument("--out-dir", default=str(PROJECT_ROOT / "renders" / "gnn_edge_dc"))
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--hidden-dim", type=int, default=64)
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--lambda-delta", type=float, default=1e-3)
    ap.add_argument("--grad-clip", type=float, default=5.0)
    ap.add_argument("--jitter", type=float, default=1e-18)
    ap.add_argument("--val-fraction", type=float, default=0.15)
    ap.add_argument("--optimizer", choices=("adamw", "adam"), default="adamw")
    ap.add_argument("--device", choices=("auto", "cpu", "mps"), default="auto")
    ap.add_argument("--torch-threads", type=int, default=1)
    ap.add_argument("--include-harmonic-features", action="store_true",
                    help="include H1/H2/H3 amplitude/phase/SNR descriptors; never includes DC target")
    ap.add_argument("--sweep", action="store_true",
                    help="run K/hidden/lambda_delta/seed masked-validation sweep")
    ap.add_argument("--K-values", nargs="*", type=int, default=[0, 1, 2, 3, 4])
    ap.add_argument("--hidden-dim-values", nargs="*", type=int, default=[32, 64, 128])
    ap.add_argument("--lambda-delta-values", nargs="*", type=float, default=[1e-4, 1e-3, 1e-2])
    ap.add_argument("--seeds", nargs="*", type=int, default=[0])
    ap.add_argument("--no-tqdm", dest="use_tqdm", action="store_false")
    ap.set_defaults(use_tqdm=True)
    return ap.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    set_seed(args.seed)
    try:
        torch.set_num_threads(max(int(args.torch_threads), 1))
        torch.set_num_interop_threads(1)
    except Exception:
        pass
    _install_numpy_pickle_compat()
    out_root = Path(args.out_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    graph, graph_path = load_graph_from_args(args)
    data = build_mosaic_data(graph, include_harmonic_features=args.include_harmonic_features)
    device = resolve_device(args.device)
    print(f"Loaded {graph_path}")
    print(f"Built mosaic graph data: {len(data.node_ids)} nodes, {len(data.edge_ids)} edges")
    print(f"Valid DC edge observations: {int(data.valid_mask.sum())}")
    print(f"Training on {device}")

    write_json(out_root / "run_config.json", {
        "args": vars(args),
        "graph": str(graph_path),
        "n_nodes": len(data.node_ids),
        "n_edges": len(data.edge_ids),
        "n_valid_edges": int(data.valid_mask.sum()),
    })

    if args.sweep:
        summaries = run_sweep(data, graph, args, device, out_root)
        write_csv(out_root / "sweep_summary.csv", summaries)
        plot_sweep_summary(summaries, out_root)
        print(f"Done. Sweep outputs written to {out_root}")
        return 0

    all_train = data.valid_mask.clone()
    no_val = torch.zeros_like(data.valid_mask)
    run_experiment(data, args, device, out_root / "no_cross_validation",
                   "no_cv", all_train, no_val, graph=graph)

    train_mask, val_mask = split_masks(data, args.val_fraction, args.seed)
    run_experiment(data, args, device, out_root / "masked_edge_validation_15pct",
                   "masked_15pct", train_mask, val_mask, graph=graph)

    print(f"Done. Outputs written to {out_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
