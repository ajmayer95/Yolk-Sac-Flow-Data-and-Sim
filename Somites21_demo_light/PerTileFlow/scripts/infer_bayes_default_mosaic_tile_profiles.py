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
    suffix = "_h1h2" if getattr(args, "use_second_harmonic", False) else ""
    return f"{path.stem}{suffix}{path.suffix}"


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
        amp = ed.get("amp_Q_h2_piv")
        phase = ed.get("phase_h2_piv")
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
    order = np.argsort(Ds)
    cdf = np.cumsum(probs[order])
    cdf = cdf / cdf[-1] if cdf.size and cdf[-1] > 0 else cdf

    def q(prob):
        if not cdf.size:
            return float("nan")
        return float(np.interp(prob, cdf, Ds[order]))

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
    return {
        "tile_id": int(tile_id),
        "ablation": _bayes_objective_from_profile(best),
        "harmonics": best.get("harmonics", "H1"),
        "D_hat": float(best["D"]),
        "D_p025": q(0.025),
        "D_p500": q(0.5),
        "D_p975": q(0.975),
        "D_lo_lr3": lo,
        "D_hi_lr3": hi,
        "width_decades_lr3": width,
        "mode_at_grid_boundary": bool(boundary),
        "max_neg2_delta_logpost": float(np.nanmax(lr[np.isfinite(lr)])),
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
        summary_rows.append(metrics)
        for p in profile:
            profile_rows.append({
                "tile_id": int(tile_id),
                "observation_source": "measured_graph",
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
    best_D = {
        "all": _best_D(global_rows),
        "interior": _best_D(interior_rows),
        "periphery": _best_D(periphery_rows),
    }
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
    csv_paths = [
        out_dir / _bayes_out_name("bayes_global_posterior_constant_D.csv", args),
        out_dir / _bayes_out_name("bayes_interior_posterior_constant_D.csv", args),
        out_dir / _bayes_out_name("bayes_periphery_posterior_constant_D.csv", args),
        out_dir / _bayes_out_name("bayes_tile_posteriors.csv", args),
        out_dir / _bayes_out_name("bayes_tile_posterior_summary.csv", args),
    ]
    for path, rows in zip(csv_paths, [
            global_rows, interior_rows, periphery_rows, profile_rows,
            summary_rows]):
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
            "snr_q_floor": float(args.snr_q_floor),
            "relative_variance_floor": float(args.relative_variance_floor),
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
    html_path = out_dir / _bayes_out_name(
        "infer_bayes_default_mosaic_tile_profiles.html", args)
    _write_dashboard(html_path, payload)
    print(f"Wrote {html_path}")
    print(f"Done in {(time.time() - t0) / 60.0:.1f} min")


if __name__ == "__main__":
    main()
