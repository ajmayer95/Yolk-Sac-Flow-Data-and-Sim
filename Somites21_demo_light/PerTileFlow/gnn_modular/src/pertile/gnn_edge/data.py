"""Data construction for the modular GNN edge-flow workflow.

This module converts the NetworkX vascular mosaic graph into `MosaicData`,
the tensor container consumed by the GNN, physics solver, losses, and
evaluation code.

It owns:
- node/edge indexing
- geometric feature extraction
- DC flow observation extraction
- harmonic observation extraction
- boundary source/sink injections
- feature normalization
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

from pertile.analysis.local_pressure_inference import _edge_geometry

from .constants import MU, PX_SIZE_M, nL_per_m3
from .utils import safe_float


@dataclass
class MosaicData:
    node_ids: List[object]
    edge_ids: List[Tuple[object, object]]
    edge_index: torch.Tensor
    x_node: torch.Tensor
    x_edge: torch.Tensor
    q_obs: torch.Tensor
    q_harmonic_obs: torch.Tensor
    harmonic_valid_mask: torch.Tensor
    harmonic_loss_weight: torch.Tensor
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


def selected_harmonics(flow_components: str) -> Tuple[int, ...]:
    mode = str(flow_components).strip().lower().replace("_", "-")
    if mode in ("dc", "0"):
        return ()
    if mode in ("dc-h1", "dc+h1", "h1", "1"):
        return (1,)
    if mode in ("dc-h1-h2", "dc+h1+h2", "h2", "2"):
        return (1, 2)
    raise SystemExit(
        "Unsupported --flow-components value. Use dc, dc-h1, or dc-h1-h2."
    )


def _harmonic_observation(edge_data: dict, sign: float, h: int) -> Tuple[complex, bool, float]:
    amp = edge_data.get(f"Q_H{h}_amp")
    phase = edge_data.get(f"Q_H{h}_phi")
    snr = edge_data.get(f"Q_H{h}_snr_db")
    if h == 1 and (amp is None or phase is None):
        amp = edge_data.get("amp_Q_h1_piv", edge_data.get("amp_Q"))
        phase = edge_data.get("phase_h1_piv", edge_data.get("phase"))
    elif h == 2 and (amp is None or phase is None):
        amp = edge_data.get("amp_Q_h2_piv")
        phase = edge_data.get("phase_h2_piv")
    amp_f = safe_float(amp)
    phase_f = safe_float(phase)
    snr_f = safe_float(snr, 0.0)
    if not (math.isfinite(amp_f) and math.isfinite(phase_f)):
        return complex(0.0, 0.0), False, 0.0
    q = sign * amp_f * complex(math.cos(phase_f), math.sin(phase_f))
    snr_linear = 10.0 ** (snr_f / 20.0) if math.isfinite(snr_f) else 1.0
    sigma_nls = max(abs(q) / max(snr_linear, 1e-6), 1e-6)
    return q, True, 1.0 / sigma_nls


def _harmonic_features(edge_data: dict, sign: float,
                       harmonics: Sequence[int]) -> List[float]:
    feats: List[float] = []
    for h in harmonics:
        amp = edge_data.get(f"Q_H{h}_amp")
        phase = edge_data.get(f"Q_H{h}_phi")
        snr = edge_data.get(f"Q_H{h}_snr_db")
        if h == 1 and (amp is None or phase is None):
            amp = edge_data.get("amp_Q_h1_piv", edge_data.get("amp_Q"))
            phase = edge_data.get("phase_h1_piv", edge_data.get("phase"))
        elif h == 2 and (amp is None or phase is None):
            amp = edge_data.get("amp_Q_h2_piv")
            phase = edge_data.get("phase_h2_piv")
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
        raise SystemExit("No boundary source/sink vessels with DC flow metadata were found.")

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


def build_mosaic_data(graph, flow_components: str) -> MosaicData:
    harmonics = selected_harmonics(flow_components)
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
    q_harmonic_obs: List[List[List[float]]] = []
    harmonic_valid: List[List[bool]] = []
    harmonic_weight: List[List[float]] = []
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
        if harmonics:
            feats.extend(_harmonic_features(ed, sign, harmonics))

        h_obs_row: List[List[float]] = []
        h_valid_row: List[bool] = []
        h_weight_row: List[float] = []
        for h in harmonics:
            q_h_nls, valid_h, weight_h = _harmonic_observation(ed, sign, h)
            h_obs_row.append([
                float(q_h_nls.real / nL_per_m3),
                float(q_h_nls.imag / nL_per_m3),
            ])
            h_valid_row.append(bool(valid_h))
            h_weight_row.append(float(weight_h))

        edge_ids.append((u, v))
        src.append(node_index[u])
        dst.append(node_index[v])
        radii.append(r_m)
        lengths.append(l_m)
        g_pois.append(g)
        q_obs.append(q_nls / nL_per_m3 if math.isfinite(q_nls) else 0.0)
        q_harmonic_obs.append(h_obs_row)
        harmonic_valid.append(h_valid_row)
        harmonic_weight.append(h_weight_row)
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
    if harmonics:
        q_h_arr = np.asarray(q_harmonic_obs, dtype=np.float32)
        h_valid_arr = np.asarray(harmonic_valid, dtype=bool)
        h_weight_arr = np.asarray(harmonic_weight, dtype=np.float32)
    else:
        q_h_arr = np.zeros((len(edge_ids), 0, 2), dtype=np.float32)
        h_valid_arr = np.zeros((len(edge_ids), 0), dtype=bool)
        h_weight_arr = np.zeros((len(edge_ids), 0), dtype=np.float32)
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
        "flow_components": str(flow_components),
        "harmonics": [int(h) for h in harmonics],
    }

    return MosaicData(
        node_ids=node_ids,
        edge_ids=edge_ids,
        edge_index=torch.tensor([src, dst], dtype=torch.long),
        x_node=torch.tensor(x_node, dtype=torch.float32),
        x_edge=torch.tensor(x_edge, dtype=torch.float32),
        q_obs=torch.tensor(q_arr, dtype=torch.float32),
        q_harmonic_obs=torch.tensor(q_h_arr, dtype=torch.float32),
        harmonic_valid_mask=torch.tensor(h_valid_arr.tolist(), dtype=torch.bool),
        harmonic_loss_weight=torch.tensor(h_weight_arr, dtype=torch.float32),
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
