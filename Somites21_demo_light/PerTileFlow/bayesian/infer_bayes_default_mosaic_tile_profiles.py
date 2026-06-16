"""Section-9 Bayesian tile distensibility inference dashboard.

This script implements the v2 Bayesian treatment from Section 9 of
``Transport_Networks_With_Compliance`` for measured tile data.  It uses the
H1 complex flow channel, integrates over unknown tile-boundary forcing
analytically, and evaluates the marginal posterior over D on a log grid.

Default assumptions:
  * measured vessel geometry and PIV-derived tile edge flow observations
  * H1-only Bayesian likelihood, because D enters the pulsatile channel
  * log-normal prior on D centered at 1.5e-3 1/Pa with tau=ln(10)
  * zero-mean Gaussian H1 boundary forcing prior with sigma_b=7 Pa
  * H1 relative variances from cached H1 Z statistics by default
  * H1 boundary forcing is marginalized in closed form at each D

Outputs:
  * bayes_global_posterior_constant_D.csv
  * bayes_tile_posteriors.csv
  * bayes_tile_posterior_summary.csv
  * infer_bayes_default_mosaic_tile_profiles.html
"""
from __future__ import annotations

import argparse
import base64
import csv
import html
import io
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
    _observations,
    _transfer,
    _sigma_vectors,
)
from inspect_tile import build_tile_problem
from tile_mosaic_simulation import choose_tiles, load_graph_from_args
from synthetic_validation_neumann_bc import nL_per_m3


DEFAULT_TILE_PROFILE_HARMONICS = (1, 2)
DEFAULT_BAYES_HARMONICS = (1,)
DEFAULT_LOGD_PRIOR_MEAN = math.log(1.5e-3)
DEFAULT_LOGD_PRIOR_TAU = math.log(10.0)
DEFAULT_BOUNDARY_SIGMA_PA = 7.0
DEFAULT_SIGMA_H1_NL_S = 0.012
DEFAULT_SIGMA_H2_NL_S = 0.030
DEFAULT_PIXEL_SIZE_MM = 0.0017
PERIPHERY_TILES = {
    1, 2, 3, 4, 16, 17, 18, 31, 32, 43, 44, 53, 52, 51, 50, 49, 48,
    38, 37, 25, 24, 11, 10, 8,
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
    if "log_posterior" in profile[0]:
        best = max(profile, key=lambda r: float(r["log_posterior"]))
    else:
        best = min(profile, key=lambda r: float(r["chi2"]))
    return float(best["D"])


def _tile_location(tile_id: int) -> str:
    return "periphery" if int(tile_id) in PERIPHERY_TILES else "interior"


def _safe_float(value, default=float("nan")) -> float:
    try:
        x = float(value)
    except (TypeError, ValueError):
        return default
    return x if math.isfinite(x) else default


def _ac_harmonics(args) -> tuple[int, ...]:
    return (1, 2) if getattr(args, "use_second_harmonic", False) else (1,)


def _harmonic_label(args) -> str:
    return "+".join(f"H{h}" for h in _ac_harmonics(args))


def _bayes_objective(args) -> str:
    return ("bayes_h1h2_marginal_boundary"
            if getattr(args, "use_second_harmonic", False)
            else "bayes_h1_marginal_boundary")


def _bayes_out_name(base: str, args) -> str:
    path = Path(base)
    parts = []
    if getattr(args, "cluster", False):
        parts.append("cluster")
    if getattr(args, "use_second_harmonic", False):
        parts.append("h1h2")
    suffix = "_" + "_".join(parts) if parts else ""
    return f"{path.stem}{suffix}{path.suffix}"


def _tile_bbox_from_graph(graph, tile_id: int):
    nodes = set()
    for u, v, d in graph.edges(data=True):
        for m in d.get("measurements_piv") or []:
            try:
                tid = int(m.get("tile_id"))
            except (TypeError, ValueError):
                continue
            if tid == int(tile_id):
                nodes.update([u, v])
    if not nodes:
        return None
    xs = [float(graph.nodes[n].get("x", 0.0)) for n in nodes]
    ys = [float(graph.nodes[n].get("y", 0.0)) for n in nodes]
    return {
        "xmin": min(xs), "xmax": max(xs),
        "ymin": min(ys), "ymax": max(ys),
        "cx": 0.5 * (min(xs) + max(xs)),
        "cy": 0.5 * (min(ys) + max(ys)),
    }


def _bbox_touches(a: dict, b: dict, tol: float) -> bool:
    return (
        a["xmin"] <= b["xmax"] + tol
        and a["xmax"] + tol >= b["xmin"]
        and a["ymin"] <= b["ymax"] + tol
        and a["ymax"] + tol >= b["ymin"]
    )


def _cluster_assignments(graph, tiles: Sequence[int]) -> dict:
    """Subdivide interior/periphery tiles by source/sink proximity rules."""
    tile_ids = sorted(int(t) for t in tiles)
    bboxes = {
        tid: _tile_bbox_from_graph(graph, tid)
        for tid in tile_ids
    }
    valid_bboxes = [b for b in bboxes.values() if b is not None]
    if not valid_bboxes:
        return {}
    spans = [
        min(b["xmax"] - b["xmin"], b["ymax"] - b["ymin"])
        for b in valid_bboxes
        if (b["xmax"] > b["xmin"] and b["ymax"] > b["ymin"])
    ]
    tol = 0.05 * float(np.median(spans)) if spans else 0.0
    tile14 = bboxes.get(14)

    sources = [
        d for _, d in graph.nodes(data=True)
        if d.get("boundary_type") == "source"
        and d.get("x") is not None and d.get("y") is not None
    ]
    source_line = None
    if len(sources) >= 2:
        s1, s2 = sources[:2]
        x1, y1 = float(s1["x"]), float(s1["y"])
        x2, y2 = float(s2["x"]), float(s2["y"])
        source_line = (x1, y1, x2, y2)

    out = {}
    for tid in tile_ids:
        loc = _tile_location(tid)
        box = bboxes.get(tid)
        cluster = ""
        if loc == "interior":
            if tid == 14:
                cluster = "proximal"
            elif tile14 is not None and box is not None:
                cluster = "proximal" if _bbox_touches(box, tile14, tol) else "distal"
            else:
                cluster = "distal"
        else:
            if source_line is not None and box is not None:
                x1, y1, x2, y2 = source_line
                if abs(y2 - y1) > 1e-12:
                    x_on_line = x1 + (box["cy"] - y1) * (x2 - x1) / (y2 - y1)
                else:
                    x_on_line = 0.5 * (x1 + x2)
                cluster = "venous_end" if box["cx"] < x_on_line else "arterial_end"
            else:
                cluster = "periphery_unclassified"
        out[tid] = {
            "cluster": cluster,
            "location_cluster": f"{loc}_{cluster}" if cluster else loc,
        }
    return out


def _phase_wrap(value: float) -> float:
    return float((float(value) + math.pi) % (2.0 * math.pi) - math.pi)


def _resolve_config_path(config_path: str, key: str) -> Path | None:
    cfg_path = Path(config_path).resolve()
    try:
        cfg = json.loads(cfg_path.read_text())
    except Exception:
        return None
    value = cfg.get(key)
    if not value:
        return None
    p = Path(value)
    if not p.is_absolute():
        p = cfg_path.parent / p
    return p.resolve()


def _load_tile_positions(path: Path | None) -> tuple[dict, float, float]:
    if path is None:
        return {}, 0.0, 0.0
    try:
        raw = json.loads(path.read_text())
    except Exception:
        return {}, 0.0, 0.0
    tiles = {}
    for tid, entry in raw.get("tiles", {}).items():
        try:
            key = int(tid)
        except ValueError:
            continue
        tiles[key] = {
            "translate_x": float(entry.get("translate_x", 0.0)),
            "translate_y": float(entry.get("translate_y", 0.0)),
            "scale_x": float(entry.get("scale_x", 1.0)),
            "scale_y": float(entry.get("scale_y", 1.0)),
        }
    off_x = min((t["translate_x"] for t in tiles.values()), default=0.0)
    off_y = min((t["translate_y"] for t in tiles.values()), default=0.0)
    return tiles, off_x, off_y


def _load_mosaic_image(config_path: str):
    tiff_path = _resolve_config_path(config_path, "mosaic_tiff")
    if tiff_path is None:
        return None, None
    try:
        import tifffile
        img = tifffile.imread(tiff_path)
    except Exception as e:
        print(f"  Could not load tile visualization TIFF ({e}).")
        return None, tiff_path
    if img.ndim > 2:
        img = img[0]
    return np.asarray(img), tiff_path


def _png_data_url(gray: np.ndarray, max_size: int = 720) -> tuple[str, int, int, float]:
    import matplotlib.pyplot as plt

    arr = np.asarray(gray, dtype=float)
    if arr.size == 0:
        return "", 0, 0, 1.0
    lo, hi = np.nanpercentile(arr, [1, 99])
    if not math.isfinite(lo) or not math.isfinite(hi) or hi <= lo:
        lo, hi = float(np.nanmin(arr)), float(np.nanmax(arr))
    if hi <= lo:
        hi = lo + 1.0
    arr = np.clip((arr - lo) / (hi - lo), 0.0, 1.0)
    h, w = arr.shape
    scale = min(1.0, float(max_size) / max(h, w, 1))
    out_h = max(1, int(round(h * scale)))
    out_w = max(1, int(round(w * scale)))
    if scale < 1.0:
        yy = np.linspace(0, h - 1, out_h).astype(int)
        xx = np.linspace(0, w - 1, out_w).astype(int)
        arr = arr[np.ix_(yy, xx)]
    buf = io.BytesIO()
    plt.imsave(buf, arr, cmap="gray", format="png", vmin=0.0, vmax=1.0)
    encoded = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:image/png;base64,{encoded}", out_w, out_h, scale


def _tile_crop_bounds(graph, mosaic_shape: tuple[int, int], tile_id: int,
                      tile_positions: dict, off_x: float,
                      off_y: float) -> tuple[int, int, int, int]:
    h, w = mosaic_shape
    entry = tile_positions.get(int(tile_id))
    if entry:
        x0 = int(round(entry["translate_x"] - off_x))
        y0 = int(round(entry["translate_y"] - off_y))
        x1 = int(round(x0 + 640 * entry["scale_x"]))
        y1 = int(round(y0 + 704 * entry["scale_y"]))
        box = _tile_bbox_from_graph(graph, int(tile_id))
        if box is not None:
            pad = 35
            x0 = min(x0, int(math.floor(box["xmin"] - pad)))
            x1 = max(x1, int(math.ceil(box["xmax"] + pad)))
            y0 = min(y0, int(math.floor(h - box["ymax"] - pad)))
            y1 = max(y1, int(math.ceil(h - box["ymin"] + pad)))
    else:
        box = _tile_bbox_from_graph(graph, int(tile_id))
        if box is None:
            return 0, 0, w, h
        pad = 40
        x0 = int(math.floor(box["xmin"] - pad))
        x1 = int(math.ceil(box["xmax"] + pad))
        y0 = int(math.floor(h - box["ymax"] - pad))
        y1 = int(math.ceil(h - box["ymin"] + pad))
    return max(0, x0), max(0, y0), min(w, x1), min(h, y1)


def _measurement_for_tile(edge_data: dict, tile_id: int):
    for m in edge_data.get("measurements_piv") or []:
        try:
            if int(m.get("tile_id")) == int(tile_id):
                return m
        except (TypeError, ValueError):
            continue
    return None


def _direct_harmonic_from_qt(measurement: dict, harmonics: Sequence[int],
                             dt: float = 1.0 / 250.0) -> dict:
    """Project tile-specific Q(t) at exact f0 to avoid FFT leakage."""
    if measurement is None or measurement.get("Q_t") is None:
        return {}
    try:
        q = np.asarray(measurement.get("Q_t"), dtype=float)
        f0 = float(measurement.get("f0_hz"))
    except (TypeError, ValueError):
        return {}
    if q.size < 4 or not math.isfinite(f0) or f0 <= 0:
        return {}
    t = np.arange(q.size, dtype=float) * float(dt)
    out = {}
    fit_terms = []
    for k in sorted(set(int(h) for h in harmonics)):
        omega = 2.0 * math.pi * k * f0
        A = (2.0 / q.size) * float(np.sum(q * np.cos(omega * t)))
        B = (2.0 / q.size) * float(np.sum(q * np.sin(omega * t)))
        amp = float(math.hypot(A, B))
        phi = _phase_wrap(math.atan2(-B, A))
        out[int(k)] = {
            "amp_nl_s": amp,
            "phase_rad": phi,
            "A": A,
            "B": B,
            "f0_hz": f0,
        }
        fit_terms.append(amp * np.cos(omega * t - phi))

    if fit_terms:
        fit = float(np.mean(q)) + np.sum(fit_terms, axis=0)
        resid = q - fit
        try:
            spec = np.fft.rfft(resid)
            freqs = np.fft.rfftfreq(q.size, d=float(dt))
            spec[freqs > 3.5 * f0] = 0
            resid_bl = np.fft.irfft(spec, n=q.size)
            ms_bl = float(np.mean(resid_bl * resid_bl))
            if math.isfinite(ms_bl) and ms_bl > 0:
                for k, vals in out.items():
                    vals["snr_db"] = float(
                        10.0 * math.log10(0.5 * vals["amp_nl_s"] ** 2 / ms_bl))
        except Exception:
            pass
    return out


def _tile_diagnostics(graph, mosaic: np.ndarray | None, tile_id: int,
                      harmonics: Sequence[int], tile_positions: dict,
                      off_x: float, off_y: float,
                      pixel_size_mm: float) -> dict:
    """Tile image crop plus measured phase/amplitude edge overlays."""
    prob = build_tile_problem(graph, int(tile_id))
    fallback_obs = None
    image = {}
    crop = None
    x0 = y0 = 0
    scale = 1.0
    if mosaic is not None:
        x0, y0, x1, y1 = _tile_crop_bounds(
            graph, tuple(mosaic.shape[:2]), int(tile_id), tile_positions,
            off_x, off_y)
        crop = np.asarray(mosaic[y0:y1, x0:x1])
        data_url, display_w, display_h, scale = _png_data_url(crop)
        image = {
            "data_url": data_url,
            "width": display_w,
            "height": display_h,
            "crop_x": x0,
            "crop_y": y0,
            "scale": scale,
        }
    edges = []
    mosaic_h = int(mosaic.shape[0]) if mosaic is not None else int(
        max((_safe_float(d.get("y"), 0.0)
             for _, d in graph.nodes(data=True)), default=0.0) + 1)
    for i, (u, v) in enumerate(prob["edges_in"]):
        x1 = _safe_float(graph.nodes[u].get("x"))
        gy1 = _safe_float(graph.nodes[u].get("y"))
        x2 = _safe_float(graph.nodes[v].get("x"))
        gy2 = _safe_float(graph.nodes[v].get("y"))
        if not all(math.isfinite(z) for z in (x1, gy1, x2, gy2)):
            continue
        row1 = mosaic_h - gy1
        row2 = mosaic_h - gy2
        ed = graph.edges[u, v]
        length_px = _safe_float(ed.get("length"))
        if not math.isfinite(length_px) or length_px <= 0:
            length_px = math.hypot(x2 - x1, gy2 - gy1)
        length_mm = max(float(length_px) * float(pixel_size_mm), 1e-12)
        row = {
            "edge_id": f"{i}",
            "edge_label": f"{i}: {u}-{v}",
            "node_u": str(u),
            "node_v": str(v),
            "x1": float((x1 - x0) * scale),
            "y1": float((row1 - y0) * scale),
            "x2": float((x2 - x0) * scale),
            "y2": float((row2 - y0) * scale),
            "x": float((0.5 * (x1 + x2) - x0) * scale),
            "y": float((0.5 * (row1 + row2) - y0) * scale),
            "length_px": float(length_px),
            "length_mm": float(length_mm),
        }
        any_valid = False
        harmonic_values = _direct_harmonic_from_qt(
            _measurement_for_tile(graph.edges[u, v], int(tile_id)), harmonics)
        for h in harmonics:
            vals = harmonic_values.get(int(h))
            if vals is not None:
                amp = float(vals["amp_nl_s"])
                phase = float(vals["phase_rad"])
                snr = vals.get("snr_db")
            else:
                if fallback_obs is None:
                    fallback_obs = _bayes_observations(graph, prob, harmonics)
                valid = np.asarray(
                    fallback_obs["valid"].get(int(h), []), dtype=bool)
                if i >= valid.size or not valid[i]:
                    continue
                q = complex(np.asarray(
                    fallback_obs["q_h"][int(h)], dtype=complex)[i])
                amp = abs(q) * nL_per_m3
                phase = _phase_wrap(math.atan2(q.imag, q.real))
                snr = None
            row[f"h{int(h)}_amp_nl_s"] = float(amp)
            row[f"h{int(h)}_phase_rad"] = float(phase)
            if vals is not None and vals.get("f0_hz") is not None:
                row[f"h{int(h)}_f0_hz"] = float(vals["f0_hz"])
            if snr is not None and math.isfinite(float(snr)):
                row[f"h{int(h)}_snr_db"] = float(snr)
            any_valid = True
        if any_valid:
            edges.append(row)
    nodes = _node_phase_summary(edges, harmonics)
    _attach_edge_phase_gradients(edges, nodes, harmonics)
    _attach_edge_phase_jumps(edges, nodes, harmonics)
    return {
        "tile_id": int(tile_id),
        "image": image,
        "edges": edges,
        "nodes": nodes,
        "pixel_size_mm": float(pixel_size_mm),
        "phase_gradients": _phase_gradient_summary(edges, harmonics),
    }


def _harmonic_weight(edge: dict, h: int) -> float:
    amp = _safe_float(edge.get(f"h{int(h)}_amp_nl_s"), 0.0)
    snr = _safe_float(edge.get(f"h{int(h)}_snr_db"), float("nan"))
    snr_w = 1.0
    if math.isfinite(snr):
        snr_w = max(0.05, min(5.0, 10.0 ** (snr / 20.0)))
    return max(float(amp), 1e-12) * snr_w


def _node_phase_summary(edges: List[dict],
                        harmonics: Sequence[int]) -> List[dict]:
    nodes = {}
    for edge in edges:
        for node_key, x_key, y_key in (("node_u", "x1", "y1"),
                                       ("node_v", "x2", "y2")):
            nid = str(edge[node_key])
            rec = nodes.setdefault(nid, {
                "node_id": nid,
                "x": float(edge[x_key]),
                "y": float(edge[y_key]),
            })
            for h in harmonics:
                p = edge.get(f"h{int(h)}_phase_rad")
                if p is None:
                    continue
                w = _harmonic_weight(edge, int(h))
                rec.setdefault(f"h{int(h)}_phase_terms", []).append(
                    (float(p), float(w)))
    out = []
    for rec in nodes.values():
        clean = {
            "node_id": rec["node_id"],
            "x": rec["x"],
            "y": rec["y"],
        }
        for h in harmonics:
            terms = rec.get(f"h{int(h)}_phase_terms", [])
            if not terms:
                continue
            sx = sum(w * math.cos(p) for p, w in terms)
            sy = sum(w * math.sin(p) for p, w in terms)
            sw = sum(w for _, w in terms)
            clean[f"h{int(h)}_phase_rad"] = _phase_wrap(math.atan2(sy, sx))
            clean[f"h{int(h)}_phase_weight"] = float(sw)
            clean[f"h{int(h)}_n_edges"] = int(len(terms))
        out.append(clean)
    return out


def _attach_edge_phase_gradients(edges: List[dict], nodes: List[dict],
                                 harmonics: Sequence[int]) -> None:
    node_map = {str(n["node_id"]): n for n in nodes}
    for edge in edges:
        length_mm = max(_safe_float(edge.get("length_mm")), 1e-12)
        nu = node_map.get(str(edge.get("node_u")))
        nv = node_map.get(str(edge.get("node_v")))
        if not nu or not nv:
            continue
        for h in harmonics:
            pu = nu.get(f"h{int(h)}_phase_rad")
            pv = nv.get(f"h{int(h)}_phase_rad")
            if pu is None or pv is None:
                continue
            dphi = _circ_delta(float(pv), float(pu))
            grad = dphi / length_mm
            edge[f"h{int(h)}_node_start_phase_rad"] = float(pu)
            edge[f"h{int(h)}_node_end_phase_rad"] = float(pv)
            edge[f"h{int(h)}_edge_dphi_rad"] = float(dphi)
            edge[f"h{int(h)}_edge_gradient_rad_per_mm"] = float(grad)
            edge[f"h{int(h)}_edge_abs_gradient_rad_per_mm"] = float(abs(grad))


def _attach_edge_phase_jumps(edges: List[dict], nodes: List[dict],
                             harmonics: Sequence[int]) -> None:
    incident = {}
    for edge in edges:
        for node_key in ("node_u", "node_v"):
            incident.setdefault(str(edge.get(node_key)), []).append(edge)
    for node in nodes:
        incident_edges = incident.get(str(node["node_id"]), [])
        for h in harmonics:
            vals = [
                float(e[f"h{int(h)}_phase_rad"])
                for e in incident_edges
                if e.get(f"h{int(h)}_phase_rad") is not None
            ]
            if len(vals) < 2:
                continue
            sx = sum(math.cos(v) for v in vals)
            sy = sum(math.sin(v) for v in vals)
            r = math.hypot(sx, sy) / max(len(vals), 1)
            node[f"h{int(h)}_edge_phase_dispersion"] = float(1.0 - r)
            node[f"h{int(h)}_incident_phase_n"] = int(len(vals))

    for edge in edges:
        for h in harmonics:
            p = edge.get(f"h{int(h)}_phase_rad")
            if p is None:
                continue
            jumps = []
            jumps_per_mm = []
            for node_key in ("node_u", "node_v"):
                for other in incident.get(str(edge.get(node_key)), []):
                    if other is edge:
                        continue
                    q = other.get(f"h{int(h)}_phase_rad")
                    if q is None:
                        continue
                    d = abs(_circ_delta(float(q), float(p)))
                    jumps.append(d)
                    dist = math.hypot(float(other["x"]) - float(edge["x"]),
                                      float(other["y"]) - float(edge["y"]))
                    # Use displayed crop geometry converted by this edge's
                    # length ratio; this is a diagnostic normalization only.
                    px_to_mm = (
                        float(edge.get("length_mm", 0.0))
                        / max(float(edge.get("length_px", 0.0)), 1e-12)
                    )
                    dist_mm = max(dist * px_to_mm, 1e-12)
                    jumps_per_mm.append(d / dist_mm)
            if jumps:
                arr = np.asarray(jumps, dtype=float)
                edge[f"h{int(h)}_neighbor_phase_jump_median_rad"] = float(
                    np.median(arr))
                edge[f"h{int(h)}_neighbor_phase_jump_max_rad"] = float(
                    np.max(arr))
                edge[f"h{int(h)}_neighbor_phase_jump_n"] = int(arr.size)
            if jumps_per_mm:
                arr = np.asarray(jumps_per_mm, dtype=float)
                edge[f"h{int(h)}_neighbor_phase_jump_median_rad_per_mm"] = float(
                    np.median(arr))
                edge[f"h{int(h)}_neighbor_phase_jump_max_rad_per_mm"] = float(
                    np.max(arr))


def _circ_delta(a: float, b: float) -> float:
    return _phase_wrap(float(a) - float(b))


def _phase_gradient_summary(edges: List[dict],
                            harmonics: Sequence[int]) -> dict:
    out = {}
    for h in harmonics:
        key = f"h{int(h)}_phase_rad"
        vals = []
        for i, e1 in enumerate(edges):
            p1 = e1.get(key)
            if p1 is None:
                continue
            nodes1 = {e1.get("node_u"), e1.get("node_v")}
            for e2 in edges[i + 1:]:
                p2 = e2.get(key)
                if p2 is None:
                    continue
                if not nodes1.intersection({e2.get("node_u"), e2.get("node_v")}):
                    continue
                dist = math.hypot(float(e1["x"]) - float(e2["x"]),
                                  float(e1["y"]) - float(e2["y"]))
                if dist <= 1e-9:
                    continue
                vals.append(abs(_circ_delta(float(p1), float(p2))) / dist)
        arr = np.asarray(vals, dtype=float)
        if arr.size:
            out[f"h{int(h)}"] = {
                "n_pairs": int(arr.size),
                "median_rad_per_px": float(np.median(arr)),
                "p90_rad_per_px": float(np.percentile(arr, 90)),
                "max_rad_per_px": float(np.max(arr)),
            }
        else:
            out[f"h{int(h)}"] = {
                "n_pairs": 0,
                "median_rad_per_px": float("nan"),
                "p90_rad_per_px": float("nan"),
                "max_rad_per_px": float("nan"),
            }
    return out


def _edge_harmonic_q(graph, edge, harmonic: int) -> float:
    u, v = edge
    ed = graph.edges[u, v]
    z_key = f"_h_Z_H{int(harmonic)}"
    if ed.get(z_key) is not None:
        z = _safe_float(ed.get(z_key))
        if math.isfinite(z):
            return max(z * z, 0.0)
    for key in ("snr_f0_piv", "psd_snr_f0", "snr_db"):
        val = _safe_float(ed.get(key))
        if math.isfinite(val):
            return max(10.0 ** (val / 10.0), 0.0)
    return 1.0


def _relative_variances_harmonic(graph, prob: dict, valid: np.ndarray,
                                 harmonic: int, args) -> np.ndarray:
    """Section 9-style AC relative variances, normalized to geometric mean 1."""
    vals = np.ones(len(prob["edges_in"]), dtype=float)
    weight_source = (args.h2_weight_source if int(harmonic) == 2
                     else args.h1_weight_source)
    for i, edge in enumerate(prob["edges_in"]):
        ed = graph.edges[edge[0], edge[1]]
        if weight_source == "uniform":
            q = 1.0
        elif weight_source == "snr_db":
            q = float("nan")
            for key in ("snr_f0_piv", "psd_snr_f0", "snr_db"):
                db = _safe_float(ed.get(key))
                if math.isfinite(db):
                    q = max(10.0 ** (db / 10.0), 0.0)
                    break
            if not math.isfinite(q):
                q = _edge_harmonic_q(graph, edge, harmonic)
        else:
            q = _edge_harmonic_q(graph, edge, harmonic)
        q = max(float(q), float(args.snr_q_floor))
        vals[i] = float(args.relative_variance_floor) + 1.0 / q

    valid_vals = vals[np.asarray(valid, dtype=bool)]
    valid_vals = valid_vals[np.isfinite(valid_vals) & (valid_vals > 0)]
    if valid_vals.size:
        gmean = float(np.exp(np.mean(np.log(valid_vals))))
        if math.isfinite(gmean) and gmean > 0:
            vals = vals / gmean
    return np.maximum(vals, 1e-12)


def _bayes_observations(graph, prob: dict, harmonics: Sequence[int]) -> dict:
    """Measured observations with an H2 fallback for newer cached fields."""
    obs = _observations(graph, prob, harmonics)
    if 2 not in harmonics:
        return obs

    valid_h2 = np.asarray(obs["valid"].get(2, []), dtype=bool)
    if valid_h2.size and valid_h2.any():
        return obs

    q_h2 = np.zeros(len(prob["edges_in"]), dtype=complex)
    valid = np.zeros(len(prob["edges_in"]), dtype=bool)
    for i, (u, v) in enumerate(prob["edges_in"]):
        ed = graph.edges[u, v]
        amp = ed.get("Q_H2_amp") or ed.get("amp_Q_h2_piv")
        phase = ed.get("Q_H2_phi") or ed.get("phase_h2_piv")
        if amp is None or phase is None:
            amp = ed.get("_h_amp_H2")
            phase = ed.get("_h_phase_H2")
        if (amp is None or phase is None or not np.isfinite(amp)
                or not np.isfinite(phase)):
            continue
        ff = ed.get("flow_from")
        ft = ed.get("flow_to")
        sign = 1.0 if (ff == u and ft == v) else -1.0
        q_h2[i] = float(amp) * np.exp(1j * float(phase)) * sign / nL_per_m3
        valid[i] = True
    obs["q_h"][2] = q_h2
    obs["valid"][2] = valid
    return obs


def _real_harmonic_system(graph, prob: dict, D: float, harmonic: int,
                          obs: dict, transfer: dict, args):
    valid = np.asarray(obs["valid"][int(harmonic)], dtype=bool)
    if not valid.any():
        raise ValueError(f"tile has no valid H{int(harmonic)} observations")
    q = np.asarray(obs["q_h"][int(harmonic)], dtype=complex)[valid]
    T_complex = transfer[int(harmonic)][valid]
    y = np.concatenate([q.real, q.imag])
    A = np.block([
        [T_complex.real, -T_complex.imag],
        [T_complex.imag, T_complex.real],
    ])
    v_edge = _relative_variances_harmonic(
        graph, prob, obs["valid"][int(harmonic)], int(harmonic), args)[valid]
    # Circular complex Gaussian split: real and imaginary variances each v/2.
    rdiag = np.concatenate([0.5 * v_edge, 0.5 * v_edge])
    sigma_b = float(args.boundary_sigma_pa)
    # H1 forcing prior split matches the real/imag measurement split.
    pdiag = np.full(A.shape[1], 0.5 * sigma_b * sigma_b, dtype=float)
    return y, A, rdiag, pdiag, int(valid.sum())


def _cholesky_loglike(y: np.ndarray, C: np.ndarray) -> tuple[float, np.ndarray]:
    jitter = 0.0
    eye = np.eye(C.shape[0])
    for _ in range(8):
        try:
            L = np.linalg.cholesky(C + jitter * eye)
            z = np.linalg.solve(L, y)
            alpha = np.linalg.solve(L.T, z)
            logdet = 2.0 * float(np.sum(np.log(np.diag(L))))
            ll = -0.5 * logdet - 0.5 * float(y @ alpha)
            return ll, alpha
        except np.linalg.LinAlgError:
            scale = float(np.nanmean(np.diag(C))) if C.size else 1.0
            jitter = max(1e-30, (10.0 if jitter else 1e-12) * max(scale, 1e-30))
    raise np.linalg.LinAlgError("failed Cholesky factorization")


def _bayes_harmonic_at_D(graph, prob: dict, D: float, harmonic: int,
                         obs: dict, transfer: dict, args) -> dict:
    y, A, rdiag, pdiag, n_h = _real_harmonic_system(
        graph, prob, D, int(harmonic), obs, transfer, args)
    sigma_nl_s = (float(args.sigma_h2_nl_s) if int(harmonic) == 2
                  else float(args.sigma_h1_nl_s))
    sigma = sigma_nl_s / nL_per_m3
    sigma = max(sigma, 1e-30)
    C_prior = (A * pdiag[None, :]) @ A.T
    alpha = None
    for _ in range(max(int(args.sigma_rescale_iters), 0) + 1):
        C = sigma * sigma * np.diag(rdiag) + C_prior
        loglike, alpha = _cholesky_loglike(y, C)
        # Conditional posterior mean b = P A^T C^-1 y.
        b_hat = pdiag * (A.T @ alpha)
        resid = y - A @ b_hat
        chi_rel = float(np.sum((resid * resid) / (sigma * sigma * rdiag)))
        if not args.rescale_sigma_h1:
            break
        target = max(y.size, 1)
        scale = math.sqrt(max(chi_rel, 1e-30) / target)
        sigma_new = max(sigma * scale, 1e-30)
        if abs(math.log(sigma_new / sigma)) < 1e-4:
            sigma = sigma_new
            break
        sigma = sigma_new
    C = sigma * sigma * np.diag(rdiag) + C_prior
    loglike, alpha = _cholesky_loglike(y, C)
    b_hat = pdiag * (A.T @ alpha)
    resid = y - A @ b_hat
    chi_rel = float(np.sum((resid * resid) / (sigma * sigma * rdiag)))
    h = int(harmonic)
    return {
        "log_likelihood": float(loglike),
        f"sigma_h{h}": float(sigma),
        f"sigma_h{h}_nL_s": float(sigma * nL_per_m3),
        f"chi2_h{h}_rescaled": chi_rel,
        f"chi2_h{h}_red": chi_rel / max(y.size, 1),
        f"n_h{h}": n_h,
        "n_obs": int(y.size),
        "n_boundary": int(len(prob["boundary_nodes"])),
    }


def _bayes_ac_at_D(graph, prob: dict, D: float, args) -> dict:
    harmonics = _ac_harmonics(args)
    obs = _bayes_observations(graph, prob, harmonics)
    transfer = _transfer(prob, float(D), harmonics)
    out = {
        "log_likelihood": 0.0,
        "n_obs": 0,
        "n_boundary": int(len(prob["boundary_nodes"])),
    }
    for h in harmonics:
        try:
            item = _bayes_harmonic_at_D(graph, prob, float(D), int(h),
                                        obs, transfer, args)
        except ValueError:
            if int(h) == 1:
                raise
            continue
        out["log_likelihood"] += float(item["log_likelihood"])
        out["n_obs"] += int(item["n_obs"])
        for key, value in item.items():
            if key not in ("log_likelihood", "n_obs", "n_boundary"):
                out[key] = value
    out.setdefault("n_h2", 0)
    out.setdefault("sigma_h2_nL_s", float("nan"))
    out.setdefault("chi2_h2_red", float("nan"))
    return out


def _logD_prior(D: float, args) -> float:
    x = math.log(float(D))
    mu = math.log(float(args.logD_prior_median))
    tau = float(args.logD_prior_tau)
    return -0.5 * ((x - mu) / tau) ** 2


def _normalize_log_profile(rows: List[dict]) -> None:
    finite = np.array([math.isfinite(float(r["log_posterior"]))
                       for r in rows], dtype=bool)
    if not finite.any():
        return
    vals = np.array([float(r["log_posterior"]) for r in rows], dtype=float)
    m = float(np.nanmax(vals[finite]))
    weights = np.where(finite, np.exp(vals - m), 0.0)
    total = float(np.sum(weights))
    if total <= 0:
        return
    for r, w in zip(rows, weights):
        r["posterior_prob"] = float(w / total)
        r["neg2_delta_logpost"] = float(-2.0 * (r["log_posterior"] - m))
        r["delta_chi2"] = r["neg2_delta_logpost"]


def _posterior_summary(tile_id: int, profile: List[dict], prob: dict) -> dict:
    finite = [r for r in profile if math.isfinite(float(r["log_posterior"]))]
    if not finite:
        raise ValueError("no finite posterior grid points")
    best = max(finite, key=lambda r: float(r["log_posterior"]))
    probs = np.array([float(r.get("posterior_prob", 0.0)) for r in profile])
    Ds = np.array([float(r["D"]) for r in profile])

    def q(prob):
        return _log_grid_weighted_quantile(Ds, probs, float(prob))

    lr = np.array([float(r.get("neg2_delta_logpost", float("inf")))
                   for r in profile])
    ok = np.isfinite(lr) & (lr <= 6.0)
    if ok.any():
        lo = float(np.min(Ds[ok]))
        hi = float(np.max(Ds[ok]))
        width = float(np.log10(hi / lo)) if lo > 0 else float("nan")
    else:
        lo = hi = width = float("nan")
    boundary = (np.isclose(float(best["D"]), float(Ds.min()))
                or np.isclose(float(best["D"]), float(Ds.max())))
    curvature = _posterior_curvature_logD(profile)
    ci_width = (
        float(math.log10(q(0.975) / q(0.025)))
        if q(0.025) > 0 and q(0.975) > 0 else float("nan")
    )
    return {
        "tile_id": int(tile_id),
        "ablation": _bayes_objective_from_profile(best),
        "harmonics": best.get("harmonics", "H1"),
        "D_hat": float(best["D"]),
        "D_p025": q(0.025),
        "D_p500": q(0.5),
        "D_p975": q(0.975),
        "credible_width_decades_95": ci_width,
        "D_lo_lr3": lo,
        "D_hi_lr3": hi,
        "width_decades_lr3": width,
        "mode_at_grid_boundary": bool(boundary),
        "max_neg2_delta_logpost": float(np.nanmax(lr[np.isfinite(lr)])),
        "posterior_curvature_logD": curvature,
        "log_likelihood_at_mode": float(best["log_likelihood"]),
        "log_prior_at_mode": float(best["log_prior"]),
        "log_posterior_at_mode": float(best["log_posterior"]),
        "sigma_h1_nL_s_at_mode": float(best["sigma_h1_nL_s"]),
        "chi2_h1_red_at_mode": float(best["chi2_h1_red"]),
        "sigma_h2_nL_s_at_mode": float(best.get("sigma_h2_nL_s", float("nan"))),
        "chi2_h2_red_at_mode": float(best.get("chi2_h2_red", float("nan"))),
        "n_obs": int(best["n_obs"]),
        "n_h1": int(best["n_h1"]),
        "n_h2": int(best.get("n_h2", 0)),
        "n_boundary": int(best["n_boundary"]),
        "n_params_marginalized": int(
            2 * len(prob["boundary_nodes"]) *
            len(str(best.get("harmonics", "H1")).split("+"))),
    }


def _log_grid_weighted_quantile(ds: np.ndarray, probs: np.ndarray,
                                q: float) -> float:
    """Weighted quantile for a log-spaced D grid."""
    ds = np.asarray(ds, dtype=float)
    probs = np.asarray(probs, dtype=float)
    ok = np.isfinite(ds) & (ds > 0) & np.isfinite(probs) & (probs >= 0)
    ds = ds[ok]
    probs = probs[ok]
    if not ds.size or float(np.sum(probs)) <= 0:
        return float("nan")
    order = np.argsort(ds)
    ds = ds[order]
    probs = probs[order] / float(np.sum(probs))
    logd = np.log(ds)
    if ds.size == 1:
        return float(ds[0])
    edges = np.empty(ds.size + 1, dtype=float)
    edges[1:-1] = 0.5 * (logd[:-1] + logd[1:])
    edges[0] = logd[0] - 0.5 * (logd[1] - logd[0])
    edges[-1] = logd[-1] + 0.5 * (logd[-1] - logd[-2])
    target = min(max(float(q), 0.0), 1.0)
    cdf_prev = 0.0
    for i, mass in enumerate(probs):
        cdf_next = cdf_prev + float(mass)
        if target <= cdf_next or i == len(probs) - 1:
            frac = 0.0 if mass <= 0 else (target - cdf_prev) / float(mass)
            frac = min(max(frac, 0.0), 1.0)
            return float(math.exp(edges[i] + frac * (edges[i + 1] - edges[i])))
        cdf_prev = cdf_next
    return float(ds[-1])


def _posterior_curvature_logD(profile: List[dict]) -> float:
    finite = [r for r in profile if math.isfinite(float(r["log_posterior"]))]
    if len(finite) < 3:
        return float("nan")
    vals = np.array([float(r["log_posterior"]) for r in finite], dtype=float)
    xs = np.array([math.log(float(r["D"])) for r in finite], dtype=float)
    idx = int(np.argmax(vals))
    if idx == 0 or idx == len(finite) - 1:
        return float("nan")
    sl = slice(idx - 1, idx + 2)
    try:
        a, _, _ = np.polyfit(xs[sl], vals[sl], 2)
    except Exception:
        return float("nan")
    # log posterior ~= const - 0.5 * I * (logD-logDhat)^2
    curv = -2.0 * float(a)
    return curv if math.isfinite(curv) and curv >= 0 else float("nan")


def _bayes_objective_from_profile(row: dict) -> str:
    return ("bayes_h1h2_marginal_boundary"
            if row.get("harmonics") == "H1+H2"
            else "bayes_h1_marginal_boundary")


def _profile_tile_bayes(graph, tile_id: int, D_grid: np.ndarray, args):
    prob = build_tile_problem(graph, int(tile_id))
    rows = []
    for D in D_grid:
        item = _bayes_ac_at_D(graph, prob, float(D), args)
        item["D"] = float(D)
        item["harmonics"] = _harmonic_label(args)
        item["log_prior"] = _logD_prior(float(D), args)
        item["log_posterior"] = item["log_likelihood"] + item["log_prior"]
        rows.append(item)
    _normalize_log_profile(rows)
    metrics = _posterior_summary(int(tile_id), rows, prob)
    metrics["observation_source"] = "measured_graph"
    metrics["objective"] = _bayes_objective(args)
    metrics["weight_mode"] = (f"h1:{args.h1_weight_source};"
                              f"h2:{args.h2_weight_source}"
                              if args.use_second_harmonic
                              else args.h1_weight_source)
    return rows, metrics


def _combine_bayes_profiles(tile_profiles: Dict[int, List[dict]],
                            args) -> List[dict]:
    by_d = {}
    for profile in tile_profiles.values():
        for row in profile:
            d = float(row["D"])
            by_d.setdefault(d, 0.0)
            # Shared-D posterior: multiply tile likelihoods, apply one D prior.
            by_d[d] += float(row["log_likelihood"])
    rows = []
    for d, ll in sorted(by_d.items()):
        lp = _logD_prior(d, args)
        rows.append({
            "D": d,
            "log_likelihood": ll,
            "log_prior": lp,
            "log_posterior": ll + lp,
        })
    _normalize_log_profile(rows)
    return rows


def _write_csv(path: Path, rows: List[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({k for r in rows for k in r.keys()})
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _safe_percentile(vals: Sequence[float], q: float) -> float:
    arr = np.asarray([v for v in vals if math.isfinite(float(v))], dtype=float)
    return float(np.percentile(arr, q)) if arr.size else float("nan")


def _tile_phase_metric_row(tile_id: int, diag: dict) -> dict:
    row = {"tile_id": int(tile_id)}
    for h in (1, 2):
        hk = f"h{h}"
        edges = diag.get("edges") or []
        nodes = diag.get("nodes") or []

        jumps = [
            _safe_float(e.get(f"{hk}_neighbor_phase_jump_median_rad"))
            for e in edges
        ]
        jumps = [v for v in jumps if math.isfinite(v)]
        jumps_pm = [
            _safe_float(e.get(f"{hk}_neighbor_phase_jump_median_rad_per_mm"))
            for e in edges
        ]
        jumps_pm = [v for v in jumps_pm if math.isfinite(v)]
        grads = [
            _safe_float(e.get(f"{hk}_edge_abs_gradient_rad_per_mm"))
            for e in edges
        ]
        grads = [v for v in grads if math.isfinite(v)]
        disp = [
            _safe_float(n.get(f"{hk}_edge_phase_dispersion"))
            for n in nodes
        ]
        disp = [v for v in disp if math.isfinite(v)]
        snr = [
            _safe_float(e.get(f"{hk}_snr_db"))
            for e in edges
        ]
        snr = [v for v in snr if math.isfinite(v)]

        row[f"{hk}_edge_jump_median_rad"] = _safe_percentile(jumps, 50)
        row[f"{hk}_edge_jump_p90_rad"] = _safe_percentile(jumps, 90)
        row[f"{hk}_edge_jump_max_rad"] = max(jumps) if jumps else float("nan")
        row[f"{hk}_edge_jump_median_rad_per_mm"] = _safe_percentile(jumps_pm, 50)
        row[f"{hk}_edge_jump_p90_rad_per_mm"] = _safe_percentile(jumps_pm, 90)
        row[f"{hk}_node_dispersion_median"] = _safe_percentile(disp, 50)
        row[f"{hk}_node_dispersion_p90"] = _safe_percentile(disp, 90)
        row[f"{hk}_within_edge_gradient_median_rad_per_mm"] = _safe_percentile(grads, 50)
        row[f"{hk}_within_edge_gradient_p90_rad_per_mm"] = _safe_percentile(grads, 90)
        row[f"{hk}_snr_median_db"] = _safe_percentile(snr, 50)
        row[f"{hk}_n_edges_phase"] = int(sum(
            e.get(f"{hk}_phase_rad") is not None for e in edges))

        robust = _robust_phase_plane_metrics(edges, h)
        for key, value in robust.items():
            row[f"{hk}_{key}"] = value
    return row


def _weighted_plane_fit(points: np.ndarray, z: np.ndarray,
                        w: np.ndarray):
    X = np.column_stack([points[:, 0], points[:, 1], np.ones(points.shape[0])])
    sw = np.sqrt(np.maximum(w, 1e-12))
    try:
        beta, *_ = np.linalg.lstsq(X * sw[:, None], z * sw, rcond=None)
    except Exception:
        return None
    return beta


def _robust_phase_plane_metrics(edges: List[dict], harmonic: int) -> dict:
    hk = f"h{int(harmonic)}"
    items = [
        e for e in edges
        if e.get(f"{hk}_phase_rad") is not None
        and math.isfinite(_safe_float(e.get("x")))
        and math.isfinite(_safe_float(e.get("y")))
    ]
    if len(items) < 3:
        return {
            "robust_gradient_rad_per_px": float("nan"),
            "robust_gradient_rad_per_mm": float("nan"),
            "robust_residual_median_rad": float("nan"),
            "robust_residual_p90_rad": float("nan"),
            "robust_n_edges": int(len(items)),
        }
    phases = np.array([float(e[f"{hk}_phase_rad"]) for e in items])
    sx = float(np.sum(np.cos(phases)))
    sy = float(np.sum(np.sin(phases)))
    mode = math.atan2(sy, sx)
    z = np.array([_circ_delta(float(p), mode) for p in phases], dtype=float)
    pts = np.array([[float(e["x"]), float(e["y"])] for e in items], dtype=float)
    amps = np.array([
        max(_safe_float(e.get(f"{hk}_amp_nl_s"), 0.0), 0.0)
        for e in items
    ], dtype=float)
    max_amp = max(float(np.nanmax(amps)), 1e-12)
    snr = np.array([
        _safe_float(e.get(f"{hk}_snr_db"), float("nan"))
        for e in items
    ], dtype=float)
    snr_w = np.where(np.isfinite(snr), np.clip(10.0 ** (snr / 20.0), 0.05, 5.0), 0.4)
    w0 = np.maximum(0.05, np.sqrt(amps / max_amp)) * snr_w
    w = w0.copy()
    beta = None
    for _ in range(8):
        beta = _weighted_plane_fit(pts, z, w)
        if beta is None:
            break
        pred = beta[0] * pts[:, 0] + beta[1] * pts[:, 1] + beta[2]
        resid = z - pred
        mad = float(np.median(np.abs(resid)))
        c = max(1.345 * 1.4826 * mad, 1e-9)
        w = w0 * np.minimum(1.0, c / np.maximum(np.abs(resid), 1e-9))
    if beta is None:
        return {
            "robust_gradient_rad_per_px": float("nan"),
            "robust_gradient_rad_per_mm": float("nan"),
            "robust_residual_median_rad": float("nan"),
            "robust_residual_p90_rad": float("nan"),
            "robust_n_edges": int(len(items)),
        }
    pred = beta[0] * pts[:, 0] + beta[1] * pts[:, 1] + beta[2]
    resid_abs = np.abs(z - pred)
    # The display coordinates are already scaled; use edge length ratios to
    # estimate a representative mm/display-pixel calibration for this tile.
    ratios = []
    for e in items:
        display_len = math.hypot(float(e["x2"]) - float(e["x1"]),
                                 float(e["y2"]) - float(e["y1"]))
        length_mm = _safe_float(e.get("length_mm"))
        if math.isfinite(length_mm) and display_len > 0:
            ratios.append(length_mm / display_len)
    mm_per_px = float(np.median(ratios)) if ratios else DEFAULT_PIXEL_SIZE_MM
    grad_px = float(math.hypot(float(beta[0]), float(beta[1])))
    return {
        "robust_gradient_rad_per_px": grad_px,
        "robust_gradient_rad_per_mm": grad_px / max(mm_per_px, 1e-12),
        "robust_residual_median_rad": float(np.median(resid_abs)),
        "robust_residual_p90_rad": float(np.percentile(resid_abs, 90)),
        "robust_n_edges": int(len(items)),
    }


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
<title>Bayesian Tile Distensibility Inference</title>
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
.diagnosticCanvas { height: 320px; }
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
button {
  height: 34px;
  border: 1px solid var(--line);
  border-radius: 6px;
  background: #edf2f6;
  color: var(--ink);
  padding: 0 12px;
  cursor: pointer;
}
button:hover { background: #e1e9f0; }
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
  <h1>Bayesian Tile Distensibility Inference</h1>
  <div class="meta" id="meta"></div>
</header>
<main>
  <div class="gridTop">
    <section>
      <h2>Shared Distensibility Posterior Across All Tiles</h2>
      <canvas id="globalCanvas"></canvas>
      <div class="note">The blue curve is the shared-D posterior from multiplying tile H1 marginal likelihoods and applying one log-normal D prior. Best D values are posterior modes for all tiles, interior-only tiles, and periphery-only tiles.</div>
    </section>
    <section>
      <h2>All Tile Posteriors Overlayed</h2>
      <div class="inlineControls">
        <label>Show
          <select id="allFilter">
            <option value="all">All tiles</option>
            <option value="interior">Interior</option>
            <option value="periphery">Periphery</option>
            <option value="proximal">Interior proximal</option>
            <option value="distal">Interior distal</option>
            <option value="venous_end">Periphery venous end</option>
            <option value="arterial_end">Periphery arterial end</option>
          </select>
        </label>
      </div>
      <canvas id="allCanvas"></canvas>
      <div class="note">Each line is one tile's H1 marginal posterior, shown as -2 delta log posterior relative to that tile's mode. Reference lines show shared-D posterior modes for all, interior, and periphery tiles.</div>
    </section>
  </div>
  <div class="grid2">
    <section>
      <h2>Selected Tile Posterior Overlay</h2>
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
  <section id="diagnosticsSection" style="display:none">
    <h2>Tile Phase and Flow Diagnostics</h2>
    <div class="controls">
      <label>Tile<select id="diagTile"></select></label>
      <label>Harmonic<select id="diagHarmonic"></select></label>
      <label>View<select id="diagView">
        <option value="phase_time">Phase over time</option>
        <option value="phase_map">Phase map</option>
        <option value="amp_map">Amplitude map</option>
        <option value="edge_phase_gradient">Phase gradient</option>
        <option value="edge_phase_jump">Edge-to-edge phase change</option>
        <option value="phase_contour">Phase contour</option>
        <option value="phase_gradient">Robust phase gradient</option>
      </select></label>
      <label>Time<input id="diagTime" type="range" min="0" max="1" step="0.01" value="0"></label>
      <label>Animate<button id="diagPlay" type="button">Play</button></label>
      <label>Phase filter<select id="phaseFilter">
        <option value="snr_amp">SNR + amplitude</option>
        <option value="amp">Amplitude only</option>
        <option value="snr">SNR only</option>
        <option value="none">None</option>
      </select></label>
    </div>
    <canvas id="diagnosticCanvas" class="diagnosticCanvas"></canvas>
    <div class="note" id="phaseGradientSummary"></div>
    <div class="note" id="edgeHoverInfo"></div>
    <div class="note">The image is the stitched mosaic tile crop used by the readonly viewer. Phase map is static vessel timing; phase-over-time animates the fitted harmonic signal through one normalized cardiac cycle.</div>
    <div class="grid2" style="margin-top:12px">
      <section>
        <h2>Harmonic Distributions In Selected Tile</h2>
        <canvas id="harmonicDistributionCanvas" class="diagnosticCanvas"></canvas>
        <div class="note">Amplitude histograms show H1/H2 magnitude across measured tile edges. Phase histograms show fitted timing across the same edges.</div>
      </section>
      <section>
        <h2>Selected Edge Harmonics</h2>
        <div class="controls">
          <label>Edge 1<select id="diagEdgeA"></select></label>
          <label>Edge 2<select id="diagEdgeB"></select></label>
        </div>
        <canvas id="edgeCompareCanvas" class="diagnosticCanvas"></canvas>
        <div class="note">Bars compare H1/H2 amplitude, phase, and direct-projection SNR for the selected vessel segments.</div>
      </section>
    </div>
    <section style="margin-top:12px">
      <h2>Unwrapped Phase vs Distance</h2>
      <div class="controls">
        <label>Inlet node<select id="pathInlet"></select></label>
        <label>Outlet node<select id="pathOutlet"></select></label>
      </div>
      <canvas id="phasePathCanvas" class="diagnosticCanvas"></canvas>
      <div class="note" id="phasePathSummary"></div>
    </section>
  </section>
  <section id="phaseIdentSection" style="display:none">
    <h2>Edge-to-Edge Phase Change vs Distensibility Inference</h2>
    <div class="controls">
      <label>Tiles<select id="phaseScatterGroup">
        <option value="all">All</option>
        <option value="interior">Interior</option>
        <option value="periphery">Periphery</option>
      </select></label>
      <label>Phase metric<select id="phaseScatterX"></select></label>
      <label>Inference metric<select id="phaseScatterY"></select></label>
    </div>
    <canvas id="phaseIdentCanvas"></canvas>
    <div class="note">Each labeled point is one tile. Edge-to-edge phase change is summarized across neighboring vessel segments within the tile; interval error bars are drawn when D_hat is plotted.</div>
  </section>
</main>
<script id="payload" type="application/json">__DATA__</script>
<script>
const data = JSON.parse(document.getElementById("payload").textContent);
const colors = ["#2868b7", "#c74e45", "#2c8a68", "#c27a22"];
const allColors = ["#8aa7c8", "#c8a18a", "#8bc0a9", "#b6a2ce", "#d8ba74", "#9cb5ba"];
const tileIds = Object.keys(data.tileProfiles).map(Number).sort((a,b)=>a-b).map(String);
let diagTimer = null;
let diagRenderState = null;

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
  if (data.tileDiagnostics && Object.keys(data.tileDiagnostics).length) {
    byId("diagnosticsSection").style.display = "";
    const diagTile = byId("diagTile");
    diagTile.innerHTML = "";
    for (const tid of tileIds) {
      if (!data.tileDiagnostics[tid]) continue;
      const opt = document.createElement("option");
      opt.value = tid;
      opt.textContent = tid.padStart(3, "0");
      diagTile.appendChild(opt);
    }
    const diagHarmonic = byId("diagHarmonic");
    diagHarmonic.innerHTML = "";
    for (const h of [1, 2]) {
      const opt = document.createElement("option");
      opt.value = `h${h}`;
      opt.textContent = `H${h}`;
      diagHarmonic.appendChild(opt);
    }
    const diagView = byId("diagView");
    diagView.innerHTML = "";
    const viewOptions = [
      ["phase_time", "Phase over time"],
      ["phase_map", "Phase map"],
      ["amp_map", "Amplitude map"],
      ["edge_phase_gradient", "Phase gradient"],
      ["edge_phase_jump", "Edge-to-edge phase change"],
      ["phase_contour", "Phase contour"],
      ["phase_gradient", "Robust phase gradient"],
    ];
    for (const [value, label] of viewOptions) {
      const opt = document.createElement("option");
      opt.value = value;
      opt.textContent = label;
      diagView.appendChild(opt);
    }
    for (const id of ["diagTile", "diagHarmonic", "diagView", "diagTime", "phaseFilter"]) {
      byId(id).addEventListener("change", render);
      byId(id).addEventListener("input", render);
    }
    byId("diagTile").addEventListener("change", setupDiagnosticEdges);
    byId("diagTile").addEventListener("change", setupDiagnosticNodes);
    byId("diagEdgeA").addEventListener("change", render);
    byId("diagEdgeB").addEventListener("change", render);
    byId("pathInlet").addEventListener("change", render);
    byId("pathOutlet").addEventListener("change", render);
    byId("diagnosticCanvas").addEventListener("mousemove", updateEdgeHover);
    byId("diagPlay").addEventListener("click", toggleDiagnosticPlay);
    setupDiagnosticEdges();
    setupDiagnosticNodes();
  }
  setupPhaseIdentifiability();
}
function setupPhaseIdentifiability() {
  if (!data.phaseIdentifiability || !data.phaseIdentifiability.length) return;
  byId("phaseIdentSection").style.display = "";
  const phaseMetrics = [
    ["h1_edge_jump_p90_rad_per_mm", "H1 edge-to-edge phase change p90 (rad/mm)"],
    ["h1_edge_jump_median_rad_per_mm", "H1 edge-to-edge phase change median (rad/mm)"],
    ["h1_edge_jump_p90_rad", "H1 edge-to-edge phase change p90 (rad)"],
    ["h1_edge_jump_median_rad", "H1 edge-to-edge phase change median (rad)"],
    ["h2_edge_jump_p90_rad_per_mm", "H2 edge-to-edge phase change p90 (rad/mm)"],
    ["h2_edge_jump_median_rad_per_mm", "H2 edge-to-edge phase change median (rad/mm)"],
    ["h2_edge_jump_p90_rad", "H2 edge-to-edge phase change p90 (rad)"],
    ["h2_edge_jump_median_rad", "H2 edge-to-edge phase change median (rad)"],
  ];
  const inferenceMetrics = [
    ["posterior_curvature_logD", "posterior curvature / Fisher-like information"],
    ["credible_width_decades_95", "95% credible interval width (decades)"],
    ["width_decades_lr3", "likelihood-ratio width, delta=3 (decades)"],
    ["D_hat", "D_hat"],
  ];
  for (const [id, fallback, metrics] of [
      ["phaseScatterX", "h1_edge_jump_p90_rad_per_mm", phaseMetrics],
      ["phaseScatterY", "posterior_curvature_logD", inferenceMetrics]]) {
    const sel = byId(id);
    sel.innerHTML = "";
    for (const [key, label] of metrics) {
      const opt = document.createElement("option");
      opt.value = key;
      opt.textContent = label;
      sel.appendChild(opt);
    }
    sel.value = fallback;
    sel.addEventListener("change", render);
  }
  byId("phaseScatterGroup").addEventListener("change", render);
}
function setupDiagnosticEdges() {
  if (!data.tileDiagnostics) return;
  const tid = byId("diagTile").value;
  const diag = data.tileDiagnostics[tid];
  if (!diag) return;
  const edges = (diag.edges || []).filter(e =>
    e.h1_amp_nl_s !== undefined || e.h2_amp_nl_s !== undefined);
  for (const id of ["diagEdgeA", "diagEdgeB"]) {
    const sel = byId(id);
    const old = sel.value;
    sel.innerHTML = "";
    edges.forEach((edge, i) => {
      const opt = document.createElement("option");
      opt.value = edge.edge_id;
      opt.textContent = edge.edge_label || `edge ${edge.edge_id}`;
      sel.appendChild(opt);
      if (!old && ((id === "diagEdgeA" && i === 0) ||
                   (id === "diagEdgeB" && i === Math.min(1, edges.length - 1)))) {
        sel.value = opt.value;
      }
    });
    if (old && edges.some(e => String(e.edge_id) === old)) sel.value = old;
  }
}
function setupDiagnosticNodes() {
  if (!data.tileDiagnostics) return;
  const tid = byId("diagTile").value;
  const diag = data.tileDiagnostics[tid];
  if (!diag) return;
  const nodes = (diag.nodes || []).filter(n =>
    n.h1_phase_rad !== undefined || n.h2_phase_rad !== undefined);
  for (const id of ["pathInlet", "pathOutlet"]) {
    const sel = byId(id);
    const old = sel.value;
    sel.innerHTML = "";
    nodes.forEach((node, i) => {
      const opt = document.createElement("option");
      opt.value = String(node.node_id);
      opt.textContent = String(node.node_id);
      sel.appendChild(opt);
      if (!old && ((id === "pathInlet" && i === 0) ||
                   (id === "pathOutlet" && i === Math.max(0, nodes.length - 1)))) {
        sel.value = opt.value;
      }
    });
    if (old && nodes.some(n => String(n.node_id) === old)) sel.value = old;
  }
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
  axes(ctx,w,h,"D (1/Pa, log scale)","-2 delta log posterior");
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
    if (filter === "interior" || filter === "periphery") {
      return data.tileLocation[tid] === filter;
    }
    return data.tileCluster && data.tileCluster[tid] === filter;
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
function phaseColor(rad) {
  const hue = ((Number(rad) + Math.PI) / (2 * Math.PI)) * 360;
  return `hsl(${hue}, 75%, 45%)`;
}
function valueColor(v, vmin, vmax) {
  const t = Math.max(0, Math.min(1, (Number(v) - vmin) / ((vmax - vmin) || 1)));
  const hue = 215 - 170 * t;
  return `hsl(${hue}, 70%, 42%)`;
}
function circDelta(a, b) {
  return Math.atan2(Math.sin(Number(a) - Number(b)), Math.cos(Number(a) - Number(b)));
}
function circularMean(phases) {
  let sx = 0, sy = 0;
  for (const p of phases) {
    sx += Math.cos(Number(p));
    sy += Math.sin(Number(p));
  }
  return Math.atan2(sy, sx);
}
function circularMode(phases, bins=36) {
  if (!phases.length) return 0;
  const counts = Array(bins).fill(0);
  for (const p of phases) {
    const idx = Math.max(0, Math.min(bins - 1,
      Math.floor((Number(p) + Math.PI) / (2 * Math.PI) * bins)));
    counts[idx] += 1;
  }
  const maxIdx = counts.indexOf(Math.max(...counts));
  const center = -Math.PI + (maxIdx + 0.5) / bins * 2 * Math.PI;
  const local = phases.filter(p => Math.abs(circDelta(p, center)) <= Math.PI / 6);
  return local.length ? circularMean(local) : center;
}
function drawColorbar(ctx, x, y, h, mode, label, minVal, maxVal) {
  const w = 14;
  for (let i = 0; i < h; i++) {
    const t = 1 - i / Math.max(1, h - 1);
    let color;
    if (mode === "phase") color = phaseColor(-Math.PI + t * 2 * Math.PI);
    else color = valueColor(minVal + t * (maxVal - minVal), minVal, maxVal);
    ctx.fillStyle = color;
    ctx.fillRect(x, y + i, w, 1);
  }
  ctx.strokeStyle = "#18222d";
  ctx.strokeRect(x, y, w, h);
  ctx.fillStyle = "#18222d";
  ctx.font = "11px sans-serif";
  ctx.fillText(label, x - 4, y - 6);
  ctx.fillStyle = "#647486";
  ctx.fillText(mode === "phase" ? "+pi" : num(maxVal), x + w + 4, y + 4);
  ctx.fillText(mode === "phase" ? "0" : "", x + w + 4, y + h / 2 + 4);
  ctx.fillText(mode === "phase" ? "-pi" : num(minVal), x + w + 4, y + h);
}
function interpPoint(p1, p2, v1, v2, level) {
  const t = (level - v1) / ((v2 - v1) || 1e-12);
  return [p1[0] + t * (p2[0] - p1[0]), p1[1] + t * (p2[1] - p1[1])];
}
function drawPhaseContours(ctx, rows, phaseKey, sx, sy, ox, oy, iw, ih, fit) {
  const phases = rows.map(r => Number(r[phaseKey])).filter(Number.isFinite);
  const mode = circularMode(phases);
  const levels = [
    {v: -Math.PI / 3, label: "mode - pi/3", color: "#00e5ff"},
    {v: -Math.PI / 6, label: "mode - pi/6", color: "#74ff00"},
    {v: 0, label: "mode", color: "#ffffff"},
    {v: Math.PI / 6, label: "mode + pi/6", color: "#ffeb00"},
    {v: Math.PI / 3, label: "mode + pi/3", color: "#ff4dff"},
  ];
  const step = Math.max(8, Math.floor(Math.min(iw * fit, ih * fit) / 65));
  const nx = Math.max(3, Math.floor(iw * fit / step));
  const ny = Math.max(3, Math.floor(ih * fit / step));
  const vals = [];
  for (let j = 0; j <= ny; j++) {
    vals[j] = [];
    for (let i = 0; i <= nx; i++) {
      const px = ox + i / nx * iw * fit;
      const py = oy + j / ny * ih * fit;
      let cx = 0, cy = 0, den = 0;
      for (const r of rows) {
        const dx = px - sx(r.x);
        const dy = py - sy(r.y);
        const wt = 1 / (dx * dx + dy * dy + 120);
        const ph = Number(r[phaseKey]);
        cx += wt * Math.cos(circDelta(ph, mode));
        cy += wt * Math.sin(circDelta(ph, mode));
        den += wt;
      }
      vals[j][i] = Math.atan2(cy / den, cx / den);
    }
  }
  for (const level of levels) {
    let labeled = false;
    for (let j = 0; j < ny; j++) {
      for (let i = 0; i < nx; i++) {
        const x0 = ox + i / nx * iw * fit;
        const x1 = ox + (i + 1) / nx * iw * fit;
        const y0 = oy + j / ny * ih * fit;
        const y1 = oy + (j + 1) / ny * ih * fit;
        const corners = [
          [[x0, y0], vals[j][i]],
          [[x1, y0], vals[j][i + 1]],
          [[x1, y1], vals[j + 1][i + 1]],
          [[x0, y1], vals[j + 1][i]],
        ];
        const pts = [];
        for (let e = 0; e < 4; e++) {
          const a = corners[e], b = corners[(e + 1) % 4];
          if ((a[1] <= level.v && b[1] > level.v) ||
              (a[1] > level.v && b[1] <= level.v)) {
            pts.push(interpPoint(a[0], b[0], a[1], b[1], level.v));
          }
        }
        if (pts.length >= 2) {
          ctx.lineCap = "round";
          ctx.strokeStyle = "rgba(0,0,0,0.95)";
          ctx.lineWidth = 5.5;
          ctx.beginPath();
          ctx.moveTo(pts[0][0], pts[0][1]);
          ctx.lineTo(pts[1][0], pts[1][1]);
          ctx.stroke();
          ctx.strokeStyle = "rgba(255,255,255,0.9)";
          ctx.lineWidth = 3.5;
          ctx.beginPath();
          ctx.moveTo(pts[0][0], pts[0][1]);
          ctx.lineTo(pts[1][0], pts[1][1]);
          ctx.stroke();
          ctx.strokeStyle = level.color;
          ctx.lineWidth = 2.0;
          ctx.beginPath();
          ctx.moveTo(pts[0][0], pts[0][1]);
          ctx.lineTo(pts[1][0], pts[1][1]);
          ctx.stroke();
          if (!labeled) {
            const tx = pts[0][0] + 4;
            const ty = pts[0][1] - 4;
            ctx.font = "11px sans-serif";
            ctx.lineWidth = 3;
            ctx.strokeStyle = "rgba(0,0,0,0.9)";
            ctx.strokeText(level.label, tx, ty);
            ctx.fillStyle = level.color;
            ctx.fillText(level.label, tx, ty);
            labeled = true;
          }
        }
      }
    }
  }
  return mode;
}
function quantile(values, q) {
  const vals = values.filter(Number.isFinite).sort((a,b)=>a-b);
  if (!vals.length) return NaN;
  const pos = (vals.length - 1) * q;
  const lo = Math.floor(pos), hi = Math.ceil(pos);
  if (lo === hi) return vals[lo];
  return vals[lo] * (hi - pos) + vals[hi] * (pos - lo);
}
function phaseFilterRows(rows, harmonic, mode) {
  const ampKey = `${harmonic}_amp_nl_s`;
  const snrKey = `${harmonic}_snr_db`;
  const amps = rows.map(r => Number(r[ampKey])).filter(Number.isFinite);
  const ampCut = quantile(amps, 0.5);
  return rows.filter(r => {
    const amp = Number(r[ampKey]);
    const snr = Number(r[snrKey]);
    const ampOk = mode === "none" || mode === "snr" ||
                  (!Number.isFinite(ampCut) || amp >= ampCut);
    const snrOk = mode === "none" || mode === "amp" ||
                  (Number.isFinite(snr) && snr >= 0);
    return ampOk && snrOk;
  });
}
function solve3(A, b) {
  const M = A.map((row, i) => row.concat([b[i]]));
  for (let col = 0; col < 3; col++) {
    let piv = col;
    for (let r = col + 1; r < 3; r++) {
      if (Math.abs(M[r][col]) > Math.abs(M[piv][col])) piv = r;
    }
    if (Math.abs(M[piv][col]) < 1e-12) return null;
    [M[col], M[piv]] = [M[piv], M[col]];
    const div = M[col][col];
    for (let c = col; c < 4; c++) M[col][c] /= div;
    for (let r = 0; r < 3; r++) {
      if (r === col) continue;
      const f = M[r][col];
      for (let c = col; c < 4; c++) M[r][c] -= f * M[col][c];
    }
  }
  return [M[0][3], M[1][3], M[2][3]];
}
function fitWeightedPlane(points, weights) {
  const A = [[0,0,0],[0,0,0],[0,0,0]];
  const b = [0,0,0];
  points.forEach((p, i) => {
    const w = weights[i];
    const x = p.x, y = p.y, z = p.z;
    const v = [x, y, 1];
    for (let r = 0; r < 3; r++) {
      b[r] += w * v[r] * z;
      for (let c = 0; c < 3; c++) A[r][c] += w * v[r] * v[c];
    }
  });
  return solve3(A, b);
}
function robustPhasePlane(rows, harmonic) {
  const phaseKey = `${harmonic}_phase_rad`;
  const ampKey = `${harmonic}_amp_nl_s`;
  const snrKey = `${harmonic}_snr_db`;
  const phases = rows.map(r => Number(r[phaseKey])).filter(Number.isFinite);
  if (phases.length < 3) return null;
  const modePhase = circularMode(phases);
  const maxAmp = Math.max(...rows.map(r => Number(r[ampKey])).filter(Number.isFinite), 1e-12);
  const points = rows.map(r => {
    const amp = Number(r[ampKey]);
    const snr = Number(r[snrKey]);
    const snrWeight = Number.isFinite(snr) ? Math.max(0.05, Math.min(5, Math.pow(10, snr / 20))) : 0.4;
    return {
      x: Number(r.x),
      y: Number(r.y),
      z: circDelta(Number(r[phaseKey]), modePhase),
      w0: Math.max(0.05, Math.sqrt(Math.max(amp, 0) / maxAmp)) * snrWeight,
    };
  });
  let weights = points.map(p => p.w0);
  let beta = null;
  for (let iter = 0; iter < 8; iter++) {
    beta = fitWeightedPlane(points, weights);
    if (!beta) return null;
    const residuals = points.map(p => p.z - (beta[0] * p.x + beta[1] * p.y + beta[2]));
    const abs = residuals.map(Math.abs);
    const mad = quantile(abs, 0.5) || 1e-6;
    const c = 1.345 * 1.4826 * mad;
    weights = points.map((p, i) => p.w0 * Math.min(1, c / Math.max(Math.abs(residuals[i]), 1e-9)));
  }
  const residuals = points.map(p => p.z - (beta[0] * p.x + beta[1] * p.y + beta[2]));
  const abs = residuals.map(Math.abs);
  return {
    beta,
    modePhase,
    points,
    grad: Math.hypot(beta[0], beta[1]),
    angle: Math.atan2(beta[1], beta[0]),
    medianResidual: quantile(abs, 0.5),
    p90Residual: quantile(abs, 0.9),
  };
}
function drawRobustPhaseGradient(ctx, rows, harmonic, sx, sy, ox, oy, iw, ih, fit) {
  const filtered = phaseFilterRows(rows, harmonic, byId("phaseFilter").value || "snr_amp");
  const fitObj = robustPhasePlane(filtered, harmonic);
  if (!fitObj) return null;
  const levels = [
    {v: -Math.PI / 3, label: "mode - pi/3", color: "#00e5ff"},
    {v: -Math.PI / 6, label: "mode - pi/6", color: "#74ff00"},
    {v: 0, label: "mode", color: "#ffffff"},
    {v: Math.PI / 6, label: "mode + pi/6", color: "#ffeb00"},
    {v: Math.PI / 3, label: "mode + pi/3", color: "#ff4dff"},
  ];
  const [a, b, c] = fitObj.beta;
  for (const level of levels) {
    const pts = [];
    const bounds = [
      [[ox, oy], [ox + iw * fit, oy]],
      [[ox + iw * fit, oy], [ox + iw * fit, oy + ih * fit]],
      [[ox + iw * fit, oy + ih * fit], [ox, oy + ih * fit]],
      [[ox, oy + ih * fit], [ox, oy]],
    ];
    for (const [p1, p2] of bounds) {
      const v1 = a * ((p1[0] - ox) / fit) + b * ((p1[1] - oy) / fit) + c;
      const v2 = a * ((p2[0] - ox) / fit) + b * ((p2[1] - oy) / fit) + c;
      if ((v1 <= level.v && v2 >= level.v) || (v1 >= level.v && v2 <= level.v)) {
        pts.push(interpPoint(p1, p2, v1, v2, level.v));
      }
    }
    if (pts.length >= 2) {
      ctx.lineCap = "round";
      ctx.strokeStyle = "rgba(0,0,0,0.95)";
      ctx.lineWidth = 6;
      ctx.beginPath(); ctx.moveTo(pts[0][0], pts[0][1]); ctx.lineTo(pts[1][0], pts[1][1]); ctx.stroke();
      ctx.strokeStyle = "rgba(255,255,255,0.95)";
      ctx.lineWidth = 4;
      ctx.beginPath(); ctx.moveTo(pts[0][0], pts[0][1]); ctx.lineTo(pts[1][0], pts[1][1]); ctx.stroke();
      ctx.strokeStyle = level.color;
      ctx.lineWidth = 2.3;
      ctx.beginPath(); ctx.moveTo(pts[0][0], pts[0][1]); ctx.lineTo(pts[1][0], pts[1][1]); ctx.stroke();
      ctx.font = "11px sans-serif";
      ctx.strokeStyle = "rgba(0,0,0,0.9)";
      ctx.lineWidth = 3;
      ctx.strokeText(level.label, pts[0][0] + 4, pts[0][1] - 4);
      ctx.fillStyle = level.color;
      ctx.fillText(level.label, pts[0][0] + 4, pts[0][1] - 4);
    }
  }
  const cx = ox + iw * fit * 0.5;
  const cy = oy + ih * fit * 0.5;
  const len = Math.min(iw * fit, ih * fit) * 0.22;
  const dx = Math.cos(fitObj.angle) * len;
  const dy = Math.sin(fitObj.angle) * len;
  ctx.strokeStyle = "rgba(0,0,0,0.95)";
  ctx.lineWidth = 8;
  ctx.beginPath(); ctx.moveTo(cx - dx * 0.5, cy - dy * 0.5); ctx.lineTo(cx + dx * 0.5, cy + dy * 0.5); ctx.stroke();
  ctx.strokeStyle = "#ffffff";
  ctx.lineWidth = 4;
  ctx.beginPath(); ctx.moveTo(cx - dx * 0.5, cy - dy * 0.5); ctx.lineTo(cx + dx * 0.5, cy + dy * 0.5); ctx.stroke();
  ctx.fillStyle = "#ffffff";
  ctx.strokeStyle = "rgba(0,0,0,0.95)";
  ctx.lineWidth = 2;
  ctx.beginPath();
  ctx.moveTo(cx + dx * 0.5, cy + dy * 0.5);
  ctx.lineTo(cx + dx * 0.5 - 10 * Math.cos(fitObj.angle - 0.6), cy + dy * 0.5 - 10 * Math.sin(fitObj.angle - 0.6));
  ctx.lineTo(cx + dx * 0.5 - 10 * Math.cos(fitObj.angle + 0.6), cy + dy * 0.5 - 10 * Math.sin(fitObj.angle + 0.6));
  ctx.closePath(); ctx.fill(); ctx.stroke();
  return {...fitObj, nUsed: filtered.length, nTotal: rows.length};
}
function drawSelectedEdgeHighlights(ctx, diag, sx, sy) {
  const ids = [byId("diagEdgeA")?.value, byId("diagEdgeB")?.value]
    .filter(v => v !== undefined && v !== null && v !== "");
  const colors = ["#ffffff", "#ff2d2d"];
  ids.forEach((id, i) => {
    const edge = (diag.edges || []).find(e => String(e.edge_id) === String(id));
    if (!edge) return;
    ctx.lineCap = "round";
    ctx.strokeStyle = "rgba(0,0,0,0.95)";
    ctx.lineWidth = 11;
    ctx.beginPath();
    ctx.moveTo(sx(edge.x1), sy(edge.y1));
    ctx.lineTo(sx(edge.x2), sy(edge.y2));
    ctx.stroke();
    ctx.strokeStyle = colors[i] || "#ffffff";
    ctx.lineWidth = 7;
    ctx.beginPath();
    ctx.moveTo(sx(edge.x1), sy(edge.y1));
    ctx.lineTo(sx(edge.x2), sy(edge.y2));
    ctx.stroke();
    ctx.fillStyle = "rgba(0,0,0,0.95)";
    ctx.beginPath();
    ctx.arc(sx(edge.x), sy(edge.y), 10, 0, Math.PI * 2);
    ctx.fill();
    ctx.fillStyle = colors[i] || "#ffffff";
    ctx.font = "bold 13px sans-serif";
    ctx.textAlign = "center";
    ctx.textBaseline = "middle";
    ctx.fillText(String(i + 1), sx(edge.x), sy(edge.y));
    ctx.textAlign = "left";
    ctx.textBaseline = "alphabetic";
  });
}
function nearestPointDistance(px, py, x1, y1, x2, y2) {
  const dx = x2 - x1, dy = y2 - y1;
  const den = dx * dx + dy * dy || 1;
  const t = Math.max(0, Math.min(1, ((px - x1) * dx + (py - y1) * dy) / den));
  const x = x1 + t * dx, y = y1 + t * dy;
  return Math.hypot(px - x, py - y);
}
function updateEdgeHover(evt) {
  if (!diagRenderState) return;
  const rect = evt.currentTarget.getBoundingClientRect();
  const px = evt.clientX - rect.left;
  const py = evt.clientY - rect.top;
  const {diag, harmonic, sx, sy} = diagRenderState;
  let best = null, bestD = Infinity;
  for (const e of diag.edges || []) {
    const d = nearestPointDistance(px, py, sx(e.x1), sy(e.y1), sx(e.x2), sy(e.y2));
    if (d < bestD) { best = e; bestD = d; }
  }
  if (!best || bestD > 12) {
    byId("edgeHoverInfo").textContent = "";
    return;
  }
  const p0 = best[`${harmonic}_node_start_phase_rad`];
  const p1 = best[`${harmonic}_node_end_phase_rad`];
  const dphi = best[`${harmonic}_edge_dphi_rad`];
  const grad = best[`${harmonic}_edge_gradient_rad_per_mm`];
  const jump = best[`${harmonic}_neighbor_phase_jump_median_rad`];
  const jumpMax = best[`${harmonic}_neighbor_phase_jump_max_rad`];
  byId("edgeHoverInfo").textContent =
    `${best.edge_label}: start phase=${num(p0)} rad, end phase=${num(p1)} rad, ` +
    `wrapped dphi=${num(dphi)} rad, gradient=${num(grad)} rad/mm, ` +
    `neighbor jump median=${num(jump)} rad, max=${num(jumpMax)} rad`;
}
function buildAdjacency(diag, harmonic) {
  const adj = {};
  for (const e of diag.edges || []) {
    const u = String(e.node_u), v = String(e.node_v);
    if (e[`${harmonic}_node_start_phase_rad`] === undefined ||
        e[`${harmonic}_node_end_phase_rad`] === undefined) continue;
    const len = Number(e.length_mm);
    if (!Number.isFinite(len) || len <= 0) continue;
    (adj[u] ||= []).push({node: v, edge: e, length: len});
    (adj[v] ||= []).push({node: u, edge: e, length: len});
  }
  return adj;
}
function shortestPath(diag, harmonic, start, goal) {
  const adj = buildAdjacency(diag, harmonic);
  const dist = {[start]: 0};
  const prev = {};
  const queue = new Set(Object.keys(adj));
  if (!queue.has(start) || !queue.has(goal)) return null;
  while (queue.size) {
    let u = null, best = Infinity;
    for (const n of queue) {
      const d = dist[n] ?? Infinity;
      if (d < best) { best = d; u = n; }
    }
    if (u === null || !Number.isFinite(best)) break;
    queue.delete(u);
    if (u === goal) break;
    for (const item of adj[u] || []) {
      if (!queue.has(item.node)) continue;
      const alt = best + item.length;
      if (alt < (dist[item.node] ?? Infinity)) {
        dist[item.node] = alt;
        prev[item.node] = {node: u, edge: item.edge};
      }
    }
  }
  if (!(goal in dist)) return null;
  const nodes = [goal];
  const edges = [];
  let cur = goal;
  while (cur !== start) {
    const p = prev[cur];
    if (!p) return null;
    edges.unshift(p.edge);
    cur = p.node;
    nodes.unshift(cur);
  }
  return {nodes, edges, distance: dist[goal]};
}
function nodeMap(diag) {
  const out = {};
  for (const n of diag.nodes || []) out[String(n.node_id)] = n;
  return out;
}
function unwrap(values) {
  if (!values.length) return [];
  const out = [Number(values[0])];
  for (let i = 1; i < values.length; i++) {
    out.push(out[i - 1] + circDelta(Number(values[i]), Number(values[i - 1])));
  }
  return out;
}
function linearFit(xs, ys) {
  const n = xs.length;
  if (n < 2) return null;
  const mx = xs.reduce((a,b)=>a+b,0) / n;
  const my = ys.reduce((a,b)=>a+b,0) / n;
  let sxx = 0, sxy = 0;
  for (let i = 0; i < n; i++) {
    sxx += (xs[i] - mx) ** 2;
    sxy += (xs[i] - mx) * (ys[i] - my);
  }
  if (sxx <= 1e-12) return null;
  const slope = sxy / sxx;
  return {slope, intercept: my - slope * mx};
}
function drawPhaseGradientEdges(ctx, rows, harmonic, sx, sy) {
  const key = `${harmonic}_edge_abs_gradient_rad_per_mm`;
  const vals = rows.map(e => Number(e[key])).filter(Number.isFinite);
  const vmax = Math.max(...vals, 1e-12);
  for (const e of rows) {
    const g = Number(e[key]);
    if (!Number.isFinite(g)) continue;
    ctx.strokeStyle = valueColor(g, 0, vmax);
    ctx.globalAlpha = 0.92;
    ctx.lineWidth = Math.max(2.5, 6 * Math.sqrt(Math.max(g, 0) / vmax));
    ctx.beginPath();
    ctx.moveTo(sx(e.x1), sy(e.y1));
    ctx.lineTo(sx(e.x2), sy(e.y2));
    ctx.stroke();
  }
  ctx.globalAlpha = 1;
  return vmax;
}
function drawEdgePhaseJumps(ctx, diag, rows, harmonic, sx, sy) {
  const key = `${harmonic}_neighbor_phase_jump_median_rad`;
  const vals = rows.map(e => Number(e[key])).filter(Number.isFinite);
  const vmax = Math.max(...vals, Math.PI / 6);
  for (const e of rows) {
    const jump = Number(e[key]);
    if (!Number.isFinite(jump)) continue;
    ctx.strokeStyle = valueColor(jump, 0, vmax);
    ctx.globalAlpha = 0.94;
    ctx.lineWidth = Math.max(2.5, 7 * Math.sqrt(Math.max(jump, 0) / vmax));
    ctx.beginPath();
    ctx.moveTo(sx(e.x1), sy(e.y1));
    ctx.lineTo(sx(e.x2), sy(e.y2));
    ctx.stroke();
  }
  const nodeKey = `${harmonic}_edge_phase_dispersion`;
  const nodeVals = (diag.nodes || []).map(n => Number(n[nodeKey])).filter(Number.isFinite);
  const nodeMax = Math.max(...nodeVals, 1e-9);
  for (const n of diag.nodes || []) {
    const disp = Number(n[nodeKey]);
    if (!Number.isFinite(disp) || disp <= 0) continue;
    const r = 3 + 9 * Math.sqrt(disp / nodeMax);
    ctx.fillStyle = valueColor(disp, 0, nodeMax);
    ctx.strokeStyle = "rgba(0,0,0,0.85)";
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.arc(sx(n.x), sy(n.y), r, 0, Math.PI * 2);
    ctx.fill();
    ctx.stroke();
  }
  ctx.globalAlpha = 1;
  return vmax;
}
function drawPathOnTile(ctx, diag, harmonic, sx, sy) {
  const start = byId("pathInlet").value;
  const goal = byId("pathOutlet").value;
  const path = shortestPath(diag, harmonic, start, goal);
  if (!path) return null;
  ctx.lineCap = "round";
  for (const e of path.edges) {
    ctx.strokeStyle = "rgba(0,0,0,0.95)";
    ctx.lineWidth = 9;
    ctx.beginPath(); ctx.moveTo(sx(e.x1), sy(e.y1)); ctx.lineTo(sx(e.x2), sy(e.y2)); ctx.stroke();
    ctx.strokeStyle = "#ffffff";
    ctx.lineWidth = 5;
    ctx.beginPath(); ctx.moveTo(sx(e.x1), sy(e.y1)); ctx.lineTo(sx(e.x2), sy(e.y2)); ctx.stroke();
  }
  return path;
}
function renderPhasePathPlot(diag, harmonic) {
  const canvas = byId("phasePathCanvas");
  const {ctx,w,h} = canvasCtx(canvas);
  clear(ctx,w,h);
  const start = byId("pathInlet").value;
  const goal = byId("pathOutlet").value;
  const path = shortestPath(diag, harmonic, start, goal);
  const nodes = nodeMap(diag);
  if (!path || path.nodes.length < 2) {
    ctx.fillStyle = "#18222d";
    ctx.fillText("No valid path for selected nodes/harmonic", 18, 28);
    byId("phasePathSummary").textContent = "";
    return;
  }
  const s = [0];
  for (const e of path.edges) s.push(s[s.length - 1] + Number(e.length_mm || 0));
  const rawPhase = path.nodes.map(n => Number(nodes[n]?.[`${harmonic}_phase_rad`]));
  if (rawPhase.some(v => !Number.isFinite(v))) {
    ctx.fillText("Path has nodes without phase estimates", 18, 28);
    return;
  }
  const ph = unwrap(rawPhase);
  const fit = linearFit(s, ph);
  const xmin = 0, xmax = Math.max(...s, 1e-9);
  const fitVals = fit ? [fit.intercept, fit.intercept + fit.slope * xmax] : [];
  const ymin = Math.min(...ph, ...fitVals);
  const ymax = Math.max(...ph, ...fitVals);
  const padY = Math.max(0.1, 0.08 * (ymax - ymin || 1));
  const sxp = x => 58 + (x - xmin) / ((xmax - xmin) || 1) * (w - 82);
  const syp = y => h - 44 - (y - ymin + padY) / ((ymax - ymin + 2 * padY) || 1) * (h - 68);
  axes(ctx,w,h,"cumulative distance (mm)","unwrapped phase (rad)");
  const yTicks = linearTickValues(Math.max(1, ymax - ymin)).map(v => v + ymin);
  drawTicks(ctx,w,h,linearTickValues(xmax),yTicks,sxp,syp);
  ctx.strokeStyle = "#2868b7";
  ctx.lineWidth = 2;
  ctx.beginPath();
  ph.forEach((p, i) => i ? ctx.lineTo(sxp(s[i]), syp(p)) : ctx.moveTo(sxp(s[i]), syp(p)));
  ctx.stroke();
  ctx.fillStyle = "#2868b7";
  ph.forEach((p, i) => { ctx.beginPath(); ctx.arc(sxp(s[i]), syp(p), 3, 0, Math.PI*2); ctx.fill(); });
  if (fit) {
    ctx.strokeStyle = "#c74e45";
    ctx.setLineDash([5,4]);
    ctx.beginPath();
    ctx.moveTo(sxp(0), syp(fit.intercept));
    ctx.lineTo(sxp(xmax), syp(fit.intercept + fit.slope * xmax));
    ctx.stroke();
    ctx.setLineDash([]);
  }
  const f0s = path.edges.flatMap(e => [
    Number(e[`${harmonic}_f0_hz`])
  ]).filter(Number.isFinite);
  const f0 = f0s.length ? f0s.reduce((a,b)=>a+b,0) / f0s.length : NaN;
  const omega = 2 * Math.PI * (Number.isFinite(f0) ? f0 : 0);
  const slope = fit ? fit.slope : NaN;
  const k = Number.isFinite(slope) ? -slope : NaN;
  const lag = ph[ph.length - 1] - ph[0];
  const c = Number.isFinite(k) && Math.abs(k) > 1e-12 ? omega / Math.abs(k) : NaN;
  byId("phasePathSummary").textContent =
    `path length=${num(xmax)} mm, total phase lag=${num(lag)} rad, ` +
    `k=${num(k)} rad/mm, f0=${num(f0)} Hz, estimated wave speed=${num(c)} mm/s`;
}
function pearson(rows, xKey, yKey) {
  const pts = rows.map(r => [Number(r[xKey]), Number(r[yKey])])
    .filter(p => Number.isFinite(p[0]) && Number.isFinite(p[1]));
  if (pts.length < 3) return NaN;
  const mx = pts.reduce((a,p)=>a+p[0],0) / pts.length;
  const my = pts.reduce((a,p)=>a+p[1],0) / pts.length;
  let sxx=0, syy=0, sxy=0;
  for (const [x,y] of pts) {
    sxx += (x-mx)**2; syy += (y-my)**2; sxy += (x-mx)*(y-my);
  }
  return sxy / Math.sqrt((sxx || 1) * (syy || 1));
}
function normalCdf(z) {
  const sign = z < 0 ? -1 : 1;
  const x = Math.abs(z) / Math.sqrt(2);
  const t = 1 / (1 + 0.3275911 * x);
  const a1 = 0.254829592, a2 = -0.284496736, a3 = 1.421413741;
  const a4 = -1.453152027, a5 = 1.061405429;
  const erf = sign * (1 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * Math.exp(-x * x));
  return 0.5 * (1 + erf);
}
function regressionStats(rows) {
  const pts = rows.map(r => [Number(r.x), Number(r.y)])
    .filter(p => Number.isFinite(p[0]) && Number.isFinite(p[1]));
  if (pts.length < 3) return null;
  const mx = pts.reduce((a,p)=>a+p[0],0) / pts.length;
  const my = pts.reduce((a,p)=>a+p[1],0) / pts.length;
  let sxx=0, syy=0, sxy=0;
  for (const [x,y] of pts) {
    sxx += (x-mx)**2; syy += (y-my)**2; sxy += (x-mx)*(y-my);
  }
  if (sxx <= 0 || syy <= 0) return null;
  const slope = sxy / sxx;
  const intercept = my - slope * mx;
  const r = sxy / Math.sqrt(sxx * syy);
  const r2 = r * r;
  const df = pts.length - 2;
  const t = Math.abs(r) * Math.sqrt(df / Math.max(1e-12, 1 - r2));
  const p = 2 * (1 - normalCdf(t));
  return {n: pts.length, slope, intercept, r, r2, p: Math.max(0, Math.min(1, p))};
}
function selectedLabel(id) {
  const sel = byId(id);
  return sel && sel.selectedOptions.length ? sel.selectedOptions[0].textContent : id;
}
function metricLabel(id) {
  const label = selectedLabel(id);
  const sel = byId(id);
  if (sel && sel.value === "D_hat") return "log10(D_hat)";
  return label;
}
function metricValue(row, key) {
  const value = Number(row[key]);
  if (!Number.isFinite(value)) return NaN;
  if (key === "D_hat") return value > 0 ? Math.log10(value) : NaN;
  return value;
}
function tickRange(minVal, maxVal) {
  const span = Math.max(maxVal - minVal, Math.abs(maxVal), Math.abs(minVal), 1e-12);
  const rawStep = span / 4;
  const pow = Math.pow(10, Math.floor(Math.log10(rawStep || 1)));
  const candidates = [1, 2, 5, 10].map(v => v * pow);
  let step = candidates[candidates.length - 1];
  for (const c of candidates) {
    if (rawStep <= c) { step = c; break; }
  }
  const start = Math.ceil(minVal / step) * step;
  const ticks = [];
  for (let v = start; v <= maxVal + step * 0.5; v += step) ticks.push(v);
  if (!ticks.length) ticks.push(minVal, maxVal);
  return ticks;
}
function intervalForMetric(row, key) {
  if (key !== "D_hat") return null;
  const lo = Number(row.D_p025);
  const hi = Number(row.D_p975);
  if (!Number.isFinite(lo) || !Number.isFinite(hi) || lo <= 0 || hi <= 0) return null;
  return [Math.log10(lo), Math.log10(hi)];
}
function renderPhaseIdentifiability() {
  if (!data.phaseIdentifiability || !data.phaseIdentifiability.length ||
      !byId("phaseIdentCanvas")) return;
  const xKey = byId("phaseScatterX").value;
  const yKey = byId("phaseScatterY").value;
  const group = byId("phaseScatterGroup").value;
  const rows = data.phaseIdentifiability
    .map(r => ({...r, x: metricValue(r, xKey), y: metricValue(r, yKey)}))
    .filter(r => Number.isFinite(r.x) && Number.isFinite(r.y))
    .filter(r => group === "all" || r.location === group);
  const {ctx,w,h} = canvasCtx(byId("phaseIdentCanvas"));
  clear(ctx,w,h);
  axes(ctx,w,h,metricLabel("phaseScatterX"),metricLabel("phaseScatterY"));
  if (!rows.length) {
    ctx.fillStyle = "#18222d";
    ctx.fillText("No finite rows for selected metrics", 72, 32);
    return;
  }
  let xVals = rows.map(r=>r.x);
  let yVals = rows.map(r=>r.y);
  for (const r of rows) {
    const xi = intervalForMetric(r, xKey);
    const yi = intervalForMetric(r, yKey);
    if (xi) xVals = xVals.concat(xi);
    if (yi) yVals = yVals.concat(yi);
  }
  const xmin = Math.min(...xVals);
  const xmax = Math.max(...xVals);
  const ymin = Math.min(...yVals);
  const ymax = Math.max(...yVals);
  const xpad = Math.max(1e-12, 0.08 * ((xmax - xmin) || Math.abs(xmax) || 1));
  const ypad = Math.max(1e-12, 0.08 * ((ymax - ymin) || Math.abs(ymax) || 1));
  const sxp = x => 58 + (x - xmin + xpad) / ((xmax - xmin + 2*xpad) || 1) * (w - 82);
  const syp = y => h - 44 - (y - ymin + ypad) / ((ymax - ymin + 2*ypad) || 1) * (h - 68);
  drawTicks(ctx,w,h,tickRange(xmin, xmax),tickRange(ymin, ymax),sxp,syp);
  const fit = regressionStats(rows);
  if (fit) {
    const x1 = xmin;
    const x2 = xmax;
    const y1 = fit.intercept + fit.slope * x1;
    const y2 = fit.intercept + fit.slope * x2;
    ctx.strokeStyle = "rgba(24,34,45,0.75)";
    ctx.lineWidth = 1.6;
    ctx.setLineDash([6, 4]);
    ctx.beginPath();
    ctx.moveTo(sxp(x1), syp(y1));
    ctx.lineTo(sxp(x2), syp(y2));
    ctx.stroke();
    ctx.setLineDash([]);
  }
  ctx.lineWidth = 1.4;
  for (const r of rows) {
    const x = sxp(r.x);
    const y = syp(r.y);
    const xi = intervalForMetric(r, xKey);
    const yi = intervalForMetric(r, yKey);
    ctx.strokeStyle = "rgba(24,34,45,0.45)";
    if (xi) {
      const xlo = sxp(xi[0]);
      const xhi = sxp(xi[1]);
      ctx.beginPath();
      ctx.moveTo(xlo, y);
      ctx.lineTo(xhi, y);
      ctx.moveTo(xlo, y - 4);
      ctx.lineTo(xlo, y + 4);
      ctx.moveTo(xhi, y - 4);
      ctx.lineTo(xhi, y + 4);
      ctx.stroke();
    }
    if (yi) {
      const ylo = syp(yi[0]);
      const yhi = syp(yi[1]);
      ctx.beginPath();
      ctx.moveTo(x, ylo);
      ctx.lineTo(x, yhi);
      ctx.moveTo(x - 4, ylo);
      ctx.lineTo(x + 4, ylo);
      ctx.moveTo(x - 4, yhi);
      ctx.lineTo(x + 4, yhi);
      ctx.stroke();
    }
    ctx.fillStyle = r.location === "periphery" ? "#c27a22" : "#2868b7";
    ctx.strokeStyle = "rgba(0,0,0,0.65)";
    ctx.beginPath();
    ctx.arc(x, y, 5, 0, Math.PI * 2);
    ctx.fill();
    ctx.stroke();
    ctx.fillStyle = "#18222d";
    ctx.font = "11px sans-serif";
    ctx.textAlign = "left";
    ctx.textBaseline = "middle";
    ctx.fillText(String(r.tile_id), x + 7, y - 7);
  }
  ctx.fillStyle = "#18222d";
  ctx.font = "12px sans-serif";
  ctx.textAlign = "left";
  ctx.textBaseline = "alphabetic";
  const fitText = fit
    ? `R^2=${num(fit.r2)}; p~${num(fit.p)}`
    : "fit unavailable";
  ctx.fillText(`tiles=${group}; n=${rows.length}; ${fitText}`, 70, 32);
  ctx.fillStyle = "#2868b7";
  ctx.fillText("interior", w - 148, 28);
  ctx.fillStyle = "#c27a22";
  ctx.fillText("periphery", w - 92, 28);
}
function toggleDiagnosticPlay() {
  const btn = byId("diagPlay");
  if (diagTimer) {
    clearInterval(diagTimer);
    diagTimer = null;
    btn.textContent = "Play";
    return;
  }
  byId("diagView").value = "phase_time";
  btn.textContent = "Pause";
  diagTimer = setInterval(() => {
    const slider = byId("diagTime");
    let t = Number(slider.value || 0) + 0.015;
    if (t > 1) t -= 1;
    slider.value = t.toFixed(2);
    renderDiagnostics();
  }, 60);
}
function renderDiagnostics() {
  if (!data.tileDiagnostics || !Object.keys(data.tileDiagnostics).length) return;
  const tid = byId("diagTile").value;
  const harmonic = byId("diagHarmonic").value || "h1";
  const view = byId("diagView").value;
  const diag = data.tileDiagnostics[tid];
  if (!diag) return;
  const rows = (diag.edges || []).filter(e => e[`${harmonic}_phase_rad`] !== undefined);
  const {ctx,w,h} = canvasCtx(byId("diagnosticCanvas"));
  clear(ctx,w,h);
  ctx.fillStyle = "#18222d";
  ctx.font = "13px sans-serif";
  if (!rows.length) {
    ctx.fillText(`No ${harmonic.toUpperCase()} measurements for tile ${tid}`, 18, 28);
    renderHarmonicDistributions(diag);
    renderEdgeComparison(diag);
    return;
  }
  const img = diag.image || {};
  const iw = Number(img.width || 1);
  const ih = Number(img.height || 1);
  const fit = Math.min((w - 20) / iw, (h - 44) / ih);
  const ox = 10 + Math.max(0, (w - 20 - iw * fit) / 2);
  const oy = 12;
  const sx = x => ox + Number(x) * fit;
  const sy = y => oy + Number(y) * fit;
  const drawImageThen = (after) => {
    if (img.data_url) {
      const im = new Image();
      im.onload = () => {
        clear(ctx,w,h);
        ctx.drawImage(im, ox, oy, iw * fit, ih * fit);
        after();
      };
      im.src = img.data_url;
    } else {
      after();
    }
  };
  const ampKey = `${harmonic}_amp_nl_s`;
  const phaseKey = `${harmonic}_phase_rad`;
  const ampsRaw = rows.map(r => Number(r[ampKey])).filter(Number.isFinite);
  const maxAmp = Math.max(...ampsRaw, 1e-12);
  const grad = (diag.phase_gradients || {})[harmonic] || {};
  byId("phaseGradientSummary").textContent =
    `${harmonic.toUpperCase()} phase gradient: median=${num(grad.median_rad_per_px)} rad/pixel, ` +
    `p90=${num(grad.p90_rad_per_px)} rad/pixel, max=${num(grad.max_rad_per_px)} rad/pixel, ` +
    `neighbor pairs=${grad.n_pairs || 0}`;
  drawImageThen(() => {
    const t = view === "phase_time" ? Number(byId("diagTime").value) : 0;
    let robustFit = null;
    let gradientMax = null;
    let jumpMax = null;
    if (view === "phase_contour") {
      drawPhaseContours(ctx, rows, phaseKey, sx, sy, ox, oy, iw, ih, fit);
    } else if (view === "phase_gradient") {
      robustFit = drawRobustPhaseGradient(ctx, rows, harmonic, sx, sy, ox, oy, iw, ih, fit);
      if (robustFit) {
        byId("phaseGradientSummary").textContent =
          `${harmonic.toUpperCase()} robust gradient: |grad|=${num(robustFit.grad)} rad/pixel, ` +
          `angle=${num(robustFit.angle)} rad, median residual=${num(robustFit.medianResidual)} rad, ` +
          `p90 residual=${num(robustFit.p90Residual)} rad, used=${robustFit.nUsed}/${robustFit.nTotal} edges`;
      }
    } else if (view === "edge_phase_gradient") {
      gradientMax = drawPhaseGradientEdges(ctx, rows, harmonic, sx, sy);
      byId("phaseGradientSummary").textContent =
        `${harmonic.toUpperCase()} edge phase-gradient view: edge color = |wrapped node-phase difference| / length; ` +
        `max=${num(gradientMax)} rad/mm`;
    } else if (view === "edge_phase_jump") {
      jumpMax = drawEdgePhaseJumps(ctx, diag, rows, harmonic, sx, sy);
      const vals = rows.map(e => Number(e[`${harmonic}_neighbor_phase_jump_median_rad`]))
        .filter(Number.isFinite);
      byId("phaseGradientSummary").textContent =
        `${harmonic.toUpperCase()} edge-to-edge phase-change view: edge color = median wrapped phase jump to neighboring edges; ` +
        `median=${num(quantile(vals, 0.5))} rad, p90=${num(quantile(vals, 0.9))} rad, max=${num(jumpMax)} rad`;
    }
    if (view !== "edge_phase_gradient" && view !== "edge_phase_jump") for (const r of rows) {
      const phase = Number(r[phaseKey]);
      const amp = Number(r[ampKey]);
      const val = amp * Math.cos(2 * Math.PI * t + phase);
      ctx.strokeStyle = view === "phase_gradient"
        ? "rgba(220,225,230,0.42)"
        : view === "phase_time"
        ? valueColor(val / maxAmp, -1, 1)
        : view === "amp_map"
          ? valueColor(amp, 0, maxAmp)
          : phaseColor(phase);
      ctx.globalAlpha = 0.9;
      ctx.lineWidth = Math.max(2.5, 7 * Math.sqrt(Math.max(amp, 0) / maxAmp));
      ctx.beginPath();
      ctx.moveTo(sx(r.x1), sy(r.y1));
      ctx.lineTo(sx(r.x2), sy(r.y2));
      ctx.stroke();
    }
    drawPathOnTile(ctx, diag, harmonic, sx, sy);
    drawSelectedEdgeHighlights(ctx, diag, sx, sy);
    ctx.globalAlpha = 1;
    ctx.fillStyle = "#647486";
    ctx.font = "12px sans-serif";
    const label = view === "phase_time"
      ? `time = ${t.toFixed(2)} cycle; color = instantaneous harmonic flow`
      : view === "amp_map"
        ? "color = fitted amplitude"
      : view === "edge_phase_gradient"
        ? "edge color = |phase gradient| in rad/mm"
      : view === "edge_phase_jump"
        ? "edge color = median phase jump to adjacent edges; node dots = junction phase dispersion"
        : view === "phase_gradient"
          ? "robust fitted phase plane; arrow = dominant phase-gradient direction"
        : view === "phase_contour"
          ? "contours = mode, mode +/- pi/6, mode +/- pi/3; lines = fitted phase"
        : "color = fitted phase";
    ctx.fillText(label, 16, h - 14);
    const cbMode = view === "amp_map" ? "amp" :
                   view === "edge_phase_gradient" ? "gradient" :
                   view === "edge_phase_jump" ? "jump" :
                   view === "phase_time" ? "value" : "phase";
    const cbLabel = view === "amp_map" ? "amp" :
                    view === "edge_phase_gradient" ? "Phase Gradient (rad/mm)" :
                    view === "edge_phase_jump" ? "Edge-to-edge phase jump (rad)" :
                    view === "phase_time" ? "Q(t)" : "phase";
    drawColorbar(ctx, Math.max(20, w - 70), 34, Math.min(190, h - 92),
                 cbMode === "phase" ? "phase" : "value",
                 cbLabel,
                 view === "phase_time" ? -1 : 0,
                 view === "phase_time" ? 1 :
                   view === "edge_phase_gradient" ? (gradientMax || 1) :
                   view === "edge_phase_jump" ? (jumpMax || Math.PI) : maxAmp);
    diagRenderState = {diag, harmonic, sx, sy};
  });
  renderHarmonicDistributions(diag);
  renderEdgeComparison(diag);
  renderPhasePathPlot(diag, harmonic);
}
function histogram(values, bins, minVal, maxVal) {
  const out = Array(bins).fill(0);
  const span = (maxVal - minVal) || 1;
  for (const v of values) {
    const idx = Math.max(0, Math.min(bins - 1, Math.floor((v - minVal) / span * bins)));
    out[idx] += 1;
  }
  return out;
}
function drawSmallBarChart(ctx, box, bins, color, title, xMin, xMax) {
  const [x0, y0, w, h] = box;
  const ymax = Math.max(1, ...bins);
  ctx.strokeStyle = "#d8e0e8";
  ctx.strokeRect(x0, y0, w, h);
  const bw = w / bins.length;
  bins.forEach((v, i) => {
    const bh = v / ymax * (h - 22);
    ctx.fillStyle = color;
    ctx.fillRect(x0 + i * bw + 1, y0 + h - bh - 18, Math.max(1, bw - 2), bh);
  });
  ctx.fillStyle = "#18222d";
  ctx.font = "12px sans-serif";
  ctx.fillText(title, x0 + 6, y0 + 14);
  ctx.fillStyle = "#647486";
  ctx.font = "10px sans-serif";
  ctx.fillText(num(xMin), x0 + 4, y0 + h - 4);
  ctx.textAlign = "right";
  ctx.fillText(num(xMax), x0 + w - 4, y0 + h - 4);
  ctx.textAlign = "left";
}
function renderHarmonicDistributions(diag) {
  const {ctx,w,h} = canvasCtx(byId("harmonicDistributionCanvas"));
  clear(ctx,w,h);
  const edges = diag.edges || [];
  const colorsH = {h1: "#2868b7", h2: "#c74e45"};
  const boxes = [
    [18, 22, w / 2 - 28, h / 2 - 34],
    [w / 2 + 10, 22, w / 2 - 28, h / 2 - 34],
    [18, h / 2 + 16, w / 2 - 28, h / 2 - 42],
    [w / 2 + 10, h / 2 + 16, w / 2 - 28, h / 2 - 42],
  ];
  const h1Amp = edges.map(e => Number(e.h1_amp_nl_s)).filter(Number.isFinite);
  const h2Amp = edges.map(e => Number(e.h2_amp_nl_s)).filter(Number.isFinite);
  const h1Ph = edges.map(e => Number(e.h1_phase_rad)).filter(Number.isFinite);
  const h2Ph = edges.map(e => Number(e.h2_phase_rad)).filter(Number.isFinite);
  const ampMax = Math.max(...h1Amp, ...h2Amp, 1e-12);
  drawSmallBarChart(ctx, boxes[0], histogram(h1Amp, 18, 0, ampMax),
                    colorsH.h1, `H1 amplitude n=${h1Amp.length}`, 0, ampMax);
  drawSmallBarChart(ctx, boxes[1], histogram(h2Amp, 18, 0, ampMax),
                    colorsH.h2, `H2 amplitude n=${h2Amp.length}`, 0, ampMax);
  drawSmallBarChart(ctx, boxes[2], histogram(h1Ph, 18, -Math.PI, Math.PI),
                    colorsH.h1, `H1 phase n=${h1Ph.length}`, -Math.PI, Math.PI);
  drawSmallBarChart(ctx, boxes[3], histogram(h2Ph, 18, -Math.PI, Math.PI),
                    colorsH.h2, `H2 phase n=${h2Ph.length}`, -Math.PI, Math.PI);
}
function renderEdgeComparison(diag) {
  const {ctx,w,h} = canvasCtx(byId("edgeCompareCanvas"));
  clear(ctx,w,h);
  const edges = diag.edges || [];
  const a = edges.find(e => String(e.edge_id) === byId("diagEdgeA").value);
  const b = edges.find(e => String(e.edge_id) === byId("diagEdgeB").value);
  const selected = [a, b].filter(Boolean);
  if (!selected.length) {
    ctx.fillStyle = "#18222d";
    ctx.fillText("No edge selected", 18, 28);
    return;
  }
  const ampMax = Math.max(...selected.flatMap(e => [
    Number(e.h1_amp_nl_s), Number(e.h2_amp_nl_s)
  ]).filter(Number.isFinite), 1e-12);
  const left = 56, top = 24, plotW = w - 86, plotH = h - 72;
  ctx.strokeStyle = "#d8e0e8";
  ctx.strokeRect(left, top, plotW, plotH);
  const groups = [
    ["H1 amp", "h1_amp_nl_s", 0, ampMax, "#2868b7"],
    ["H2 amp", "h2_amp_nl_s", 0, ampMax, "#c74e45"],
    ["H1 phase", "h1_phase_rad", -Math.PI, Math.PI, "#2c8a68"],
    ["H2 phase", "h2_phase_rad", -Math.PI, Math.PI, "#c27a22"],
    ["H1 SNR", "h1_snr_db", -10, 30, "#5f7f3d"],
    ["H2 SNR", "h2_snr_db", -10, 30, "#8a5aa8"],
  ];
  const groupW = plotW / groups.length;
  groups.forEach((g, gi) => {
    const [label, key, minVal, maxVal, color] = g;
    selected.forEach((edge, ei) => {
      const raw = Number(edge[key]);
      if (!Number.isFinite(raw)) return;
      const norm = Math.max(0, Math.min(1, (raw - minVal) / ((maxVal - minVal) || 1)));
      const bh = norm * (plotH - 28);
      const bw = groupW / 4;
      const x = left + gi * groupW + groupW * 0.28 + ei * (bw + 5);
      const y = top + plotH - bh - 20;
      ctx.fillStyle = ei === 0 ? color : "#6f63bd";
      ctx.fillRect(x, y, bw, bh);
      ctx.fillStyle = "#18222d";
      ctx.font = "10px sans-serif";
      ctx.fillText(num(raw), x - 2, Math.max(top + 10, y - 4));
    });
    ctx.fillStyle = "#647486";
    ctx.font = "11px sans-serif";
    ctx.textAlign = "center";
    ctx.fillText(label, left + gi * groupW + groupW / 2, top + plotH - 4);
    ctx.textAlign = "left";
  });
  ctx.fillStyle = "#18222d";
  ctx.font = "12px sans-serif";
  if (a) ctx.fillText(`Edge 1: ${a.edge_label}`, 12, h - 24);
  if (b) ctx.fillText(`Edge 2: ${b.edge_label}`, 12, h - 8);
}
function renderTable(selectedIds) {
  const rows = data.summary
    .filter(r => !selectedIds.length || selectedIds.includes(String(r.tile_id)))
    .sort((a,b) => Number(a.tile_id) - Number(b.tile_id));
  const cols = [
    ["tile_id", "tile"],
    ["location", "location"],
    ["cluster", "cluster"],
    ["D_hat", "D_hat"],
    ["D_p025", "D 2.5%"],
    ["D_p975", "D 97.5%"],
    ["width_decades_lr3", "width lr3"],
    ["chi2_h1_red_at_mode", "H1 red chi2"],
    ["sigma_h1_nL_s_at_mode", "sigma H1"],
    ["sigma_h2_nL_s_at_mode", "sigma H2"],
    ["mode_at_grid_boundary", "edge mode"],
    ["n_h1", "n h1"],
    ["n_h2", "n h2"],
    ["n_boundary", "n bnd"]
  ];
  const cell = (r, key) => {
    const value = r[key];
    if (key === "tile_id" || key === "location" || key === "cluster") return value ?? "";
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
  renderDiagnostics();
  renderPhaseIdentifiability();
}
function setup() {
  byId("meta").textContent = `objective=Bayes H1 marginal boundary; source=measured graph; all mode D=${num(data.bestD.all)}; interior mode D=${num(data.bestD.interior)}; periphery mode D=${num(data.bestD.periphery)}; tiles=${tileIds.length}`;
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
        description="Run Section-9 Bayesian tile distensibility inference.")
    ap.add_argument("--config", default="../emb1/config.json")
    ap.add_argument("--graph", default=None)
    ap.add_argument("--tiles", nargs="*", type=int, default=None)
    ap.add_argument("--all-tiles", action="store_true",
                    help="Run all measured tiles. This is also the default "
                         "when --tiles is omitted.")
    ap.add_argument("--out-dir", default=None,
                    help="Default: renders/meeting/"
                         "infer_bayes_default_mosaic_tile_profiles")
    ap.add_argument("--D-min", type=float, default=1e-5)
    ap.add_argument("--D-max", type=float, default=1e-1)
    ap.add_argument("--D-count", type=int, default=41)
    ap.add_argument("--use-second-harmonic", action="store_true",
                    help="Include H2 as an additional independent AC block "
                         "in the Bayesian marginal likelihood. Output "
                         "filenames receive an _h1h2 suffix.")
    ap.add_argument("--cluster", action="store_true",
                    help="Add source/sink-proximity tile clusters. Interior "
                         "tiles are split into proximal/distal relative to "
                         "tile 14 and overlapping tile boundaries. Periphery "
                         "tiles are split into venous_end/arterial_end by "
                         "whether their center lies left of the line between "
                         "the two source A nodes. Output filenames receive "
                         "a _cluster suffix.")
    ap.add_argument("--tile-visualization",
                    choices=["none", "phase"],
                    default="none",
                    help="Embed per-tile measured harmonic diagnostics in "
                         "the HTML. phase adds a time slider and phase "
                         "and amplitude overlays on the stitched tile crop.")
    ap.add_argument("--pixel-size-mm", type=float,
                    default=DEFAULT_PIXEL_SIZE_MM,
                    help="Pixel size in mm/pixel for phase-gradient and "
                         "path-distance diagnostics. Default is 0.0017 "
                         "mm/pixel from the PerTileFlow calibration.")
    ap.add_argument("--logD-prior-median", type=float,
                    default=math.exp(DEFAULT_LOGD_PRIOR_MEAN),
                    help="Median of the log-normal D prior in 1/Pa.")
    ap.add_argument("--logD-prior-tau", type=float,
                    default=DEFAULT_LOGD_PRIOR_TAU,
                    help="Standard deviation of log(D) prior.")
    ap.add_argument("--boundary-sigma-pa", type=float,
                    default=DEFAULT_BOUNDARY_SIGMA_PA,
                    help="H1 boundary forcing prior sigma in Pa.")
    ap.add_argument("--sigma-h1-nl-s", type=float,
                    default=DEFAULT_SIGMA_H1_NL_S,
                    help="Initial H1 absolute noise scale in nL/s.")
    ap.add_argument("--sigma-h2-nl-s", type=float,
                    default=DEFAULT_SIGMA_H2_NL_S,
                    help="Initial H2 absolute noise scale in nL/s.")
    ap.add_argument("--rescale-sigma-h1", action="store_true",
                    default=True,
                    help="Iteratively rescale H1 sigma to unit channel "
                         "reduced chi-square at each D.")
    ap.add_argument("--no-rescale-sigma-h1", dest="rescale_sigma_h1",
                    action="store_false",
                    help="Hold --sigma-h1-nl-s fixed instead of rescaling.")
    ap.add_argument("--sigma-rescale-iters", type=int, default=8)
    ap.add_argument("--h1-weight-source",
                    choices=["h1_z", "snr_db", "uniform"], default="h1_z",
                    help="Relative variance source. h1_z uses cached "
                         "_h_Z_H1 as q=Z^2; snr_db uses available dB SNR "
                         "fields; uniform disables SNR weighting.")
    ap.add_argument("--h2-weight-source",
                    choices=["h2_z", "snr_db", "uniform"], default="h2_z",
                    help="Relative variance source for H2 when "
                         "--use-second-harmonic is set. h2_z uses cached "
                         "_h_Z_H2 as q=Z^2.")
    ap.add_argument("--snr-q-floor", type=float, default=0.1,
                    help="Lower bound on linear SNR q in Eq. 85.")
    ap.add_argument("--relative-variance-floor", type=float, default=0.05,
                    help="Additive relative-variance floor v_min in Eq. 86.")
    args = ap.parse_args()

    out_dir = (Path(args.out_dir).resolve() if args.out_dir else
               PROJECT_ROOT / "renders" / "meeting"
               / "infer_bayes_default_mosaic_tile_profiles")
    out_dir.mkdir(parents=True, exist_ok=True)

    graph, graph_path = load_graph_from_args(args)
    tiles = choose_tiles(graph, args.tiles, all_tiles=(args.all_tiles
                         or not args.tiles))
    cluster_info = _cluster_assignments(graph, tiles) if args.cluster else {}
    D_grid = np.logspace(np.log10(args.D_min), np.log10(args.D_max),
                         int(args.D_count))

    print(f"Graph: {graph_path}")
    print(f"Tiles: {tiles}")
    print(f"Output: {out_dir}")
    print("Bayesian inference config:")
    print("  observation_source=measured_graph")
    print(f"  tile_harmonics={_ac_harmonics(args)}, "
          "boundary forcing marginalized")
    print(f"  D_grid=[{args.D_min:.3e}, {args.D_max:.3e}], "
          f"count={args.D_count}")
    print(f"  logD_prior_median={args.logD_prior_median:.3e}, "
          f"tau={args.logD_prior_tau:.3g}")
    print(f"  boundary_sigma={args.boundary_sigma_pa:.3g} Pa, "
          f"h1_weight_source={args.h1_weight_source}")
    print(f"  sigma_h1_initial={args.sigma_h1_nl_s:.3g} nL/s, "
          f"rescale={args.rescale_sigma_h1}")
    if args.use_second_harmonic:
        print(f"  sigma_h2_initial={args.sigma_h2_nl_s:.3g} nL/s, "
              f"h2_weight_source={args.h2_weight_source}")
    if args.cluster:
        counts = {}
        for info in cluster_info.values():
            counts[info["cluster"]] = counts.get(info["cluster"], 0) + 1
        print(f"  cluster=True; cluster_counts={counts}")
    if args.tile_visualization != "none":
        print(f"  tile_visualization={args.tile_visualization}")
        print(f"  pixel_size={args.pixel_size_mm:.4g} mm/pixel")

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
            profile, metrics = _profile_tile_bayes(
                graph, int(tile_id), D_grid, args)
        except Exception as e:
            import traceback
            traceback.print_exc()
            summary_rows.append({
                "tile_id": int(tile_id), "error": f"{type(e).__name__}: {e}"})
            continue
        tile_profiles[int(tile_id)] = profile
        metrics["location"] = _tile_location(int(tile_id))
        metrics["cluster"] = cluster_info.get(
            int(tile_id), {}).get("cluster", "")
        metrics["location_cluster"] = cluster_info.get(
            int(tile_id), {}).get("location_cluster", metrics["location"])
        summary_rows.append(metrics)
        for p in profile:
            profile_rows.append({
                "tile_id": int(tile_id),
                "observation_source": "measured_graph",
                "location": metrics["location"],
                "cluster": metrics["cluster"],
                "location_cluster": metrics["location_cluster"],
                "objective": metrics["objective"],
                "weight_mode": metrics["weight_mode"],
                "D": float(p["D"]),
                "log_likelihood": float(p["log_likelihood"]),
                "log_prior": float(p["log_prior"]),
                "log_posterior": float(p["log_posterior"]),
                "posterior_prob": float(p.get("posterior_prob", 0.0)),
                "neg2_delta_logpost": float(p["neg2_delta_logpost"]),
                "delta_chi2": float(p["delta_chi2"]),
                "sigma_h1_nL_s": float(p["sigma_h1_nL_s"]),
                "chi2_h1_red": float(p["chi2_h1_red"]),
                "sigma_h2_nL_s": float(p.get("sigma_h2_nL_s", float("nan"))),
                "chi2_h2_red": float(p.get("chi2_h2_red", float("nan"))),
                "n_h1": int(p.get("n_h1", 0)),
                "n_h2": int(p.get("n_h2", 0)),
                "harmonics": p.get("harmonics", _harmonic_label(args)),
            })
        print(f"  D_hat={metrics['D_hat']:.3e}  "
              f"H1_chi2_red={metrics['chi2_h1_red_at_mode']:.3g}  "
              f"H2_chi2_red={metrics['chi2_h2_red_at_mode']:.3g}  "
              f"LR3_width={metrics['width_decades_lr3']:.3g} decades  "
              f"edge_mode={metrics['mode_at_grid_boundary']}")

    global_rows = _combine_bayes_profiles(tile_profiles, args)
    interior_profiles = {
        tid: prof for tid, prof in tile_profiles.items()
        if _tile_location(tid) == "interior"
    }
    periphery_profiles = {
        tid: prof for tid, prof in tile_profiles.items()
        if _tile_location(tid) == "periphery"
    }
    interior_rows = _combine_bayes_profiles(interior_profiles, args)
    periphery_rows = _combine_bayes_profiles(periphery_profiles, args)
    cluster_profiles = {}
    cluster_rows = {}
    if args.cluster:
        for cluster in ("proximal", "distal", "venous_end", "arterial_end"):
            subset = {
                tid: prof for tid, prof in tile_profiles.items()
                if cluster_info.get(tid, {}).get("cluster") == cluster
            }
            cluster_profiles[cluster] = subset
            cluster_rows[cluster] = _combine_bayes_profiles(subset, args)
    best_D = {
        "all": _best_D(global_rows),
        "interior": _best_D(interior_rows),
        "periphery": _best_D(periphery_rows),
    }
    if args.cluster:
        for cluster, rows in cluster_rows.items():
            best_D[cluster] = _best_D(rows)
    for rows, group in ((global_rows, "all"), (interior_rows, "interior"),
                        (periphery_rows, "periphery")):
        for row in rows:
            row["observation_source"] = "measured_graph"
            row["objective"] = _bayes_objective(args)
            row["weight_mode"] = (f"h1:{args.h1_weight_source};"
                                  f"h2:{args.h2_weight_source}"
                                  if args.use_second_harmonic
                                  else args.h1_weight_source)
            row["tile_group"] = group
    if args.cluster:
        for cluster, rows in cluster_rows.items():
            for row in rows:
                row["observation_source"] = "measured_graph"
                row["objective"] = _bayes_objective(args)
                row["weight_mode"] = (f"h1:{args.h1_weight_source};"
                                      f"h2:{args.h2_weight_source}"
                                      if args.use_second_harmonic
                                      else args.h1_weight_source)
                row["tile_group"] = cluster
    csv_outputs = [
        (out_dir / _bayes_out_name("bayes_global_posterior_constant_D.csv", args),
         global_rows),
        (out_dir / _bayes_out_name("bayes_interior_posterior_constant_D.csv", args),
         interior_rows),
        (out_dir / _bayes_out_name("bayes_periphery_posterior_constant_D.csv", args),
         periphery_rows),
        (out_dir / _bayes_out_name("bayes_tile_posteriors.csv", args),
         profile_rows),
        (out_dir / _bayes_out_name("bayes_tile_posterior_summary.csv", args),
         summary_rows),
    ]
    if args.cluster:
        for cluster, rows in cluster_rows.items():
            csv_outputs.append((
                out_dir / _bayes_out_name(
                    f"bayes_{cluster}_posterior_constant_D.csv", args),
                rows))
    for path, rows in csv_outputs:
        _write_csv(path, rows)
        print(f"Wrote {path}")

    tile_payload = {}
    for tile_id, profile in tile_profiles.items():
        tile_payload[str(tile_id)] = [
            {"D": float(p["D"]),
             "log_posterior": float(p["log_posterior"]),
             "posterior_prob": float(p.get("posterior_prob", 0.0)),
             "delta_chi2": float(p["delta_chi2"])}
            for p in profile
        ]
    tile_diagnostics = {}
    if args.tile_visualization != "none":
        mosaic, tiff_path = _load_mosaic_image(args.config)
        tile_positions, off_x, off_y = _load_tile_positions(
            _resolve_config_path(args.config, "tile_positions"))
        if tiff_path is not None:
            print(f"  tile visualization TIFF={tiff_path}")
        for tile_id in tile_profiles:
            tile_diagnostics[str(tile_id)] = _tile_diagnostics(
                graph, mosaic, int(tile_id), (1, 2),
                tile_positions, off_x, off_y, float(args.pixel_size_mm))
    phase_identifiability_rows = []
    if tile_diagnostics:
        summary_by_tile = {
            int(r["tile_id"]): r for r in summary_rows if "tile_id" in r
        }
        for tile_id, diag in tile_diagnostics.items():
            tid = int(tile_id)
            row = dict(summary_by_tile.get(tid, {}))
            row.update(_tile_phase_metric_row(tid, diag))
            phase_identifiability_rows.append(row)
        phase_path = out_dir / _bayes_out_name(
            "bayes_tile_phase_identifiability_metrics.csv", args)
        _write_csv(phase_path, phase_identifiability_rows)
        print(f"Wrote {phase_path}")
    payload = {
        "defaults": {
            "observation_source": "measured_graph",
            "tile_harmonics": [int(h) for h in _ac_harmonics(args)],
            "D_min": float(args.D_min),
            "D_max": float(args.D_max),
            "D_count": int(args.D_count),
            "objective": _bayes_objective(args),
            "logD_prior_median": float(args.logD_prior_median),
            "logD_prior_tau": float(args.logD_prior_tau),
            "boundary_sigma_pa": float(args.boundary_sigma_pa),
            "sigma_h1_nl_s": float(args.sigma_h1_nl_s),
            "sigma_h2_nl_s": float(args.sigma_h2_nl_s),
            "rescale_sigma_h1": bool(args.rescale_sigma_h1),
            "h1_weight_source": args.h1_weight_source,
            "h2_weight_source": args.h2_weight_source,
            "use_second_harmonic": bool(args.use_second_harmonic),
            "cluster": bool(args.cluster),
            "tile_visualization": args.tile_visualization,
            "pixel_size_mm": float(args.pixel_size_mm),
            "cluster_definitions": {
                "proximal": "tile 14 plus interior tiles whose graph-derived tile bounding boxes touch tile 14",
                "distal": "remaining interior tiles",
                "venous_end": "periphery tiles whose center is left of the line between the two source A nodes",
                "arterial_end": "remaining periphery tiles",
            } if args.cluster else {},
            "snr_q_floor": float(args.snr_q_floor),
            "relative_variance_floor": float(args.relative_variance_floor),
        },
        "globalProfile": global_rows,
        "groupProfiles": {
            "interior": interior_rows,
            "periphery": periphery_rows,
        },
        "clusterProfiles": cluster_rows,
        "bestD": best_D,
        "tileProfiles": tile_payload,
        "tileLocation": {str(t): _tile_location(t) for t in tile_profiles},
        "tileCluster": {
            str(t): cluster_info.get(int(t), {}).get("cluster", "")
            for t in tile_profiles
        },
        "tileDiagnostics": tile_diagnostics,
        "phaseIdentifiability": phase_identifiability_rows,
        "summary": summary_rows,
    }
    html_path = out_dir / _bayes_out_name(
        "infer_bayes_default_mosaic_tile_profiles.html", args)
    _write_dashboard(html_path, payload)
    print(f"Wrote {html_path}")
    print(f"Done in {(time.time() - t0) / 60.0:.1f} min")


if __name__ == "__main__":
    main()
