"""Production-like joint inversion pipeline.

Single entry point used by inspect_tile.py (Script A),
rerun_with_multiplicative_noise.py (Script C), and the meeting
notebook.  Encapsulates:

  • joint_lm with DC + H1 + H2 (configurable via `harmonics`)
  • per-channel noise floor σ²_e = a_c² (configurable initial a_c)
  • optional outer FGLS refit of a_c from squared residuals
  • per-tile carve extraction via build_tile_problem

This is the canonical inversion the scripts use after the 2026-05-28
consolidation.  See renders/meeting/STATUS_SUMMARY.md for context.
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from synthetic_validation_neumann_bc import (
    joint_lm, MU, F0_HZ, PX_SIZE_M, nL_per_m3,
)
from pertile.analysis.local_pressure_inference import (
    _build_admittance_system, _compute_transfer_matrices,
)
from inspect_tile import build_tile_problem
from h2_sensitivity_check import fetch_h2_phasor

# Initial noise floors (nL/s) — manuscript Section 7.2 medians used as
# a sensible starting point; FGLS refit converges per-tile from these.
DEFAULT_NOISE_INIT_NL = dict(dc=0.061, h1=0.012, h2=0.030)
DEFAULT_N_OUTER = 2          # one inner LM + one FGLS refit + one inner LM
DEFAULT_HARMONICS = (1, 2)


def production_fit(graph, tile_id, *, harmonics=DEFAULT_HARMONICS,
                    noise_init_nL=None, n_outer=DEFAULT_N_OUTER,
                    D_init=1.3e-3, verbose=False,
                    use_heteroscedastic=True,
                    b_floor_nL=None):
    """Run the production-like inversion on one tile.

    Noise model — two paths, controlled by `use_heteroscedastic`:

      • False (legacy):  σ_e is a per-channel CONSTANT, refit each
        FGLS pass as RMS residual.  Equivalent to assuming purely
        additive noise.
      • True (default):  σ²_e = a_n² + (b_n · |Q_e|)² per channel.
        Both a_n (additive floor, low-|Q| regime) AND b_n (multiplicative
        coefficient, high-|Q| regime) are FGLS-estimated from the
        squared inversion residuals via `fit_noise_model(form=
        'variance_linear')`.  b_n is then floored at the user-supplied
        KCL measurement-noise value so the inversion can't claim
        smaller-than-measured noise.  Default floor:
        `{'dc': 0.29, 'h1': 0, 'h2': 0}` per the manuscript Section 7
        KCL fit.

    Returns dict (extra keys when heteroscedastic):
      D_hat, sigma_D, rel_sigma_D, chi2, iters, converged,
      P_DC, P_H[h] for h in harmonics,
      a_DC_fit, a_H1_fit, a_H2_fit (additive σ floor in nL/s),
      b_DC_fit, b_H1_fit, b_H2_fit (multiplicative coef, dimensionless;
        present only when use_heteroscedastic),
      prob (the tile_problem dict),
      T (transfer matrices at converged D̂),
      r_dc, r_h, sigma_dc_final, sigma_h_final (per-edge arrays),
      noise_floored (dict {channel: bool} — True iff KCL floor was
        binding on that channel, present only when heteroscedastic),
    Or {error: '...'} if the fit failed.
    """
    if b_floor_nL is None:
        b_floor_nL = {'dc': 0.29, 'h1': 0.0, 'h2': 0.0}
    if noise_init_nL is None:
        noise_init_nL = dict(DEFAULT_NOISE_INIT_NL)
    try:
        prob = build_tile_problem(graph, tile_id)
    except Exception as e:
        return dict(error=f"build_tile_problem failed: {e}")

    n_edges = len(prob["edges_in"])
    Q_DC = np.where(prob["valid_dc"], prob["Q_DC_obs"], 0)
    Q_H1 = np.where(prob["valid_h1"], prob["Q_H1_obs"], 0)
    Q_H = {1: Q_H1}
    valid_extras = {}
    if 2 in harmonics:
        Q_H2_full, valid_h2 = fetch_h2_phasor(
            graph, prob["edges_in"], prob["valid_h1"], prob["Q_DC_obs"])
        Q_H2 = np.where(valid_h2, Q_H2_full, 0)
        Q_H[2] = Q_H2
        valid_extras["h2"] = valid_h2

    # Initialise per-channel σ vectors from noise_init
    sig_dc = np.full(n_edges, max(noise_init_nL["dc"], 1e-9) / nL_per_m3)
    sig_h1 = np.full(n_edges, max(noise_init_nL["h1"], 1e-9) / nL_per_m3)
    sig_h_dict = {1: sig_h1}
    if 2 in harmonics:
        sig_h2 = np.full(n_edges, max(noise_init_nL["h2"], 1e-9) / nL_per_m3)
        sig_h_dict[2] = sig_h2

    lm = None
    a_fit = dict(noise_init_nL)
    # b_n (multiplicative coefficient, dimensionless) — initialised at 0
    # for the first pass; refit FGLS-style alongside a_n when
    # `use_heteroscedastic=True`.
    b_fit = {'dc': 0.0, 'h1': 0.0, 'h2': 0.0}
    floored = {}
    outer_history = []
    # Helper: import the noise-model fitter lazily so legacy callers
    # (with `use_heteroscedastic=False`) don't take the import cost.
    if use_heteroscedastic:
        from pertile.analysis.inference import fit_noise_model
    for outer in range(n_outer):
        try:
            lm = joint_lm(
                prob["sub"], prob["edges_in"],
                prob["boundary_nodes"], prob["interior_nodes"],
                Q_DC, Q_H, sig_dc, sig_h_dict,
                ac_harmonics=tuple(harmonics), pin_dc=True,
                pin_idx=prob["pin_idx"], D_init=D_init, verbose=verbose)
        except Exception as e:
            return dict(error=f"joint_lm failed at outer {outer}: {e}")
        # Snapshot the outer-iteration state (post-inner-LM, pre-refit)
        outer_history.append(dict(
            outer=outer,
            a_DC=a_fit.get("dc"),
            a_H1=a_fit.get("h1"),
            a_H2=a_fit.get("h2"),
            D_hat=float(lm["D_hat"]),
            sigma_D=float(lm["sigma_D"]),
            chi2=float(lm["chi2"]),
            n_inner_iter=int(lm["iters"]),
            converged_inner=bool(lm["converged"]),
        ))

        # FGLS refit: build T at converged D̂, compute residuals,
        # refit noise model per channel.  Two paths (see docstring).
        if outer < n_outer - 1:
            ab = _build_admittance_system(
                prob["sub"], prob["edges_in"],
                prob["boundary_nodes"], prob["interior_nodes"],
                float(lm["D_hat"]), MU, F0_HZ,
                tuple(harmonics), PX_SIZE_M)
            T = _compute_transfer_matrices(
                ab, prob["edges_in"], prob["boundary_nodes"],
                prob["interior_nodes"], verbose=False)
            # ── DC channel ──
            r_dc = (Q_DC[prob["valid_dc"]]
                    - (T[0] @ lm["P_DC"]).real[prob["valid_dc"]])
            Q_dc_valid = Q_DC[prob["valid_dc"]]
            if use_heteroscedastic:
                # Fit σ²_DC = a² + b²·|Q|² in nL/s units, per memory.
                fit = fit_noise_model(
                    np.abs(r_dc) * nL_per_m3,
                    np.abs(Q_dc_valid) * nL_per_m3,
                    form='variance_linear')
                a_dc_nL = float(np.sqrt(max(fit['a'], 1e-12)))
                b_dc_raw = float(np.sqrt(max(fit['b'], 0.0)))
                b_dc_floor = float(b_floor_nL.get('dc', 0.0))
                b_dc = max(b_dc_raw, b_dc_floor)
                floored['dc'] = b_dc_raw < b_dc_floor
                a_fit['dc'] = a_dc_nL
                b_fit['dc'] = b_dc
                # Per-edge σ_e (SI) — needs |Q_e| in SI then scaled
                sig_dc = np.sqrt((a_dc_nL / nL_per_m3) ** 2
                                  + (b_dc * np.abs(Q_DC)) ** 2)
            else:
                a_fit["dc"] = float(np.sqrt(
                    max(np.mean(r_dc**2), 1e-30))) * nL_per_m3
                sig_dc = np.full(n_edges, a_fit["dc"] / nL_per_m3)
            # ── H1 channel ──
            r_h1 = (Q_H[1][prob["valid_h1"]]
                    - (T[1] @ lm["P_H"][1])[prob["valid_h1"]])
            Q_h1_valid = Q_H[1][prob["valid_h1"]]
            if use_heteroscedastic:
                fit = fit_noise_model(
                    np.abs(r_h1) * nL_per_m3,
                    np.abs(Q_h1_valid) * nL_per_m3,
                    form='variance_linear')
                a_h1_nL = float(np.sqrt(max(fit['a'], 1e-12)))
                b_h1_raw = float(np.sqrt(max(fit['b'], 0.0)))
                b_h1_floor = float(b_floor_nL.get('h1', 0.0))
                b_h1 = max(b_h1_raw, b_h1_floor)
                floored['h1'] = b_h1_raw < b_h1_floor
                a_fit['h1'] = a_h1_nL
                b_fit['h1'] = b_h1
                sig_h_dict[1] = np.sqrt((a_h1_nL / nL_per_m3) ** 2
                                          + (b_h1 * np.abs(Q_H[1])) ** 2)
            else:
                a_fit["h1"] = float(np.sqrt(max(
                    np.mean(np.abs(r_h1)**2) / 2, 1e-30))) * nL_per_m3
                sig_h_dict[1] = np.full(n_edges, a_fit["h1"] / nL_per_m3)
            # ── H2 channel (optional) ──
            if 2 in harmonics:
                valid_h2 = valid_extras["h2"]
                r_h2 = (Q_H[2][valid_h2]
                        - (T[2] @ lm["P_H"][2])[valid_h2])
                Q_h2_valid = Q_H[2][valid_h2]
                if use_heteroscedastic:
                    fit = fit_noise_model(
                        np.abs(r_h2) * nL_per_m3,
                        np.abs(Q_h2_valid) * nL_per_m3,
                        form='variance_linear')
                    a_h2_nL = float(np.sqrt(max(fit['a'], 1e-12)))
                    b_h2_raw = float(np.sqrt(max(fit['b'], 0.0)))
                    b_h2_floor = float(b_floor_nL.get('h2', 0.0))
                    b_h2 = max(b_h2_raw, b_h2_floor)
                    floored['h2'] = b_h2_raw < b_h2_floor
                    a_fit['h2'] = a_h2_nL
                    b_fit['h2'] = b_h2
                    sig_h_dict[2] = np.sqrt((a_h2_nL / nL_per_m3) ** 2
                                              + (b_h2 * np.abs(Q_H[2])) ** 2)
                else:
                    a_fit["h2"] = float(np.sqrt(max(
                        np.mean(np.abs(r_h2)**2) / 2, 1e-30))) * nL_per_m3
                    sig_h_dict[2] = np.full(n_edges, a_fit["h2"]
                                              / nL_per_m3)

    # Final assembly: at the converged θ, build T once more for the
    # caller (so the inspector can compute profile likelihood etc).
    ab = _build_admittance_system(
        prob["sub"], prob["edges_in"],
        prob["boundary_nodes"], prob["interior_nodes"],
        float(lm["D_hat"]), MU, F0_HZ, tuple(harmonics), PX_SIZE_M)
    T = _compute_transfer_matrices(
        ab, prob["edges_in"], prob["boundary_nodes"],
        prob["interior_nodes"], verbose=False)
    r_dc = (Q_DC[prob["valid_dc"]]
             - (T[0] @ lm["P_DC"]).real[prob["valid_dc"]])
    r_h = {1: (Q_H[1][prob["valid_h1"]]
                - (T[1] @ lm["P_H"][1])[prob["valid_h1"]])}
    if 2 in harmonics:
        valid_h2 = valid_extras["h2"]
        r_h[2] = (Q_H[2][valid_h2]
                   - (T[2] @ lm["P_H"][2])[valid_h2])

    return dict(
        D_hat=float(lm["D_hat"]),
        sigma_D=float(lm["sigma_D"]),
        rel_sigma_D=float(lm["sigma_D"]) / float(lm["D_hat"])
            if lm["D_hat"] > 0 else float("nan"),
        chi2=float(lm["chi2"]),
        iters=int(lm["iters"]),
        converged=bool(lm["converged"]),
        P_DC=lm["P_DC"], P_H=lm["P_H"],
        a_DC_fit=a_fit.get("dc"),
        a_H1_fit=a_fit.get("h1"),
        a_H2_fit=a_fit.get("h2", None) if 2 in harmonics else None,
        b_DC_fit=b_fit.get("dc") if use_heteroscedastic else None,
        b_H1_fit=b_fit.get("h1") if use_heteroscedastic else None,
        b_H2_fit=(b_fit.get("h2") if (use_heteroscedastic
                                       and 2 in harmonics)
                  else None),
        noise_floored=dict(floored) if use_heteroscedastic else None,
        use_heteroscedastic=bool(use_heteroscedastic),
        b_floor_nL=dict(b_floor_nL) if use_heteroscedastic else None,
        prob=prob, T=T,
        r_dc=r_dc, r_h=r_h,
        sigma_dc_final=sig_dc,
        sigma_h_final=sig_h_dict,
        harmonics=tuple(harmonics),
        n_outer=n_outer,
        outer_history=outer_history,
        # Inner LM trajectory from the FINAL outer iteration — per-iter
        # (D, chi2, accept) dicts.  Useful for overlaying the optimizer
        # path on the profile-likelihood plot to spot local-minimum
        # traps or convergence to the wrong basin.
        lm_history=list(lm.get("history") or []),
        error="",
    )
