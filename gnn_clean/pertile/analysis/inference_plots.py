"""Diagnostic figures for `pertile.analysis.inference`.

Kept in a separate module so the inference engine has no matplotlib
dependency.  Callers do:

    from pertile.analysis import inference as inf
    from pertile.analysis import inference_plots as ip
    result = inf.run_inference(graph, spec, forward_solver=...)
    fig = ip.plot_inference_result(result, title='Tile 26')
    fig.savefig(...)

Adding a new diagnostic for a new parameter:
    1. Add a panel-builder function `_panel_<name>(ax, result, ...)`.
    2. Append it to `_PANELS_FOR` keyed on which params are active.
"""

from __future__ import annotations

from typing import Optional

import numpy as np


def plot_inference_result(
    result,
    *,
    title: Optional[str] = None,
    figsize: tuple = (13, 9),
):
    """Build the standard 2×2 diagnostic figure for an `InferenceResult`.

    Panels:
      (0,0) |Q_pred| vs |Q_meas|  scatter colored by harmonic + identity
      (0,1) |residual| vs |Q_meas| with one-sided σ-bands per harmonic
      (1,0) summary text — α, D, τ, χ²/dof, ρ, noise-model coeffs
      (1,1) basis sensitivity — |b^(0)| vs |b^(p)| for the first non-α LINEAR
            param (typically D); shows DC-rows clustered at b^(p)≈0 and
            H1-rows fanned out — the geometric reason D is identifiable.

    Returns the matplotlib `Figure`.
    """
    import matplotlib.pyplot as plt

    if result.design_matrix is None or result.Qm_rotated is None:
        raise ValueError(
            'InferenceResult missing design_matrix / Qm_rotated — '
            'cannot plot.  Make sure the solver populates these (the '
            'closed-form routes do; stub routes do not).')

    dm = result.design_matrix
    Qm_rot = result.Qm_rotated
    resid = result.residuals if result.residuals is not None else (
        Qm_rot - sum(
            result.params.get(c, 0.0) * dm.basis[c]
            for c in result.convergence.get('coef_order', [])
        )
    )
    h_idx = dm.harmonic_idx
    coef_order = result.convergence.get('coef_order', list(dm.basis.keys()))
    f0 = result.convergence.get('f0_tile', 0.0)
    omega0 = 2.0 * np.pi * f0 if f0 > 0 else 1.0

    fig, axes = plt.subplots(2, 2, figsize=figsize, constrained_layout=True)

    # Predicted Q in (u,v) convention
    Q_pred = sum(result.params.get(c, 0.0) * dm.basis[c]
                  for c in coef_order)
    Q_pred_mag = np.abs(Q_pred)
    Qm_mag = np.abs(Qm_rot)
    resid_mag = np.abs(resid)

    n_dc = int(np.sum(h_idx == 0))
    n_h1 = int(np.sum(h_idx == 1))

    color_dc = '#5A4FCF'    # purple
    color_h1 = '#FF7F0E'    # orange
    m_dc = h_idx == 0
    m_h1 = h_idx == 1

    # ── (0,0) |Q_pred| vs |Q_meas| ───────────────────────────────────
    ax = axes[0, 0]
    if m_dc.any():
        ax.scatter(Q_pred_mag[m_dc], Qm_mag[m_dc], color=color_dc,
                    s=42, edgecolor='black', lw=0.5, alpha=0.85,
                    zorder=4, label=f'DC (N={n_dc})')
    if m_h1.any():
        ax.scatter(Q_pred_mag[m_h1], Qm_mag[m_h1], color=color_h1,
                    s=42, edgecolor='black', lw=0.5, alpha=0.85,
                    zorder=4, label=f'H1 (N={n_h1})')
    if Q_pred_mag.size and Qm_mag.size:
        lim = max(Q_pred_mag.max(), Qm_mag.max()) * 1.05
    else:
        lim = 1.0
    ax.plot([0, lim], [0, lim], color='gray', lw=1.2, ls='--',
             alpha=0.6, label='identity')
    ax.set_xlim(0, lim)
    ax.set_ylim(0, lim)
    ax.set_aspect('equal')
    ax.set_xlabel(r'fit:  $|\hat Q|$  [nL/s]', fontsize=12)
    ax.set_ylabel(r'$|Q_{\mathrm{meas}}|$  [nL/s]', fontsize=12)
    ax.set_title(f'χ²/dof = {result.chi2_red:.2f}', fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)
    ax.tick_params(labelsize=11)

    # ── (0,1) |residual| vs |Q_meas|  with σ bands per harmonic ─────
    ax = axes[0, 1]
    nm = next(iter(result.noise_model.values()), {})
    nm_dc = nm.get(0)
    nm_h1 = nm.get(1)
    if m_dc.any():
        ax.scatter(Qm_mag[m_dc], resid_mag[m_dc], color=color_dc,
                    s=42, edgecolor='black', lw=0.5, alpha=0.85,
                    zorder=4, label='DC')
    if m_h1.any():
        ax.scatter(Qm_mag[m_h1], resid_mag[m_h1], color=color_h1,
                    s=42, edgecolor='black', lw=0.5, alpha=0.85,
                    zorder=4, label='H1')
    qmax = float(Qm_mag.max() * 1.05) if Qm_mag.size else 1.0
    q_grid = np.linspace(0, qmax, 100)
    if nm_dc:
        from .inference import evaluate_noise_model
        s_dc = evaluate_noise_model(nm_dc, q_grid)
        ax.fill_between(q_grid, 0.0, s_dc, color=color_dc, alpha=0.12,
                         label=f'DC σ ({nm_dc.get("form","linear")})')
    if nm_h1:
        from .inference import evaluate_noise_model
        s_h1 = evaluate_noise_model(nm_h1, q_grid)
        ax.fill_between(q_grid, 0.0, s_h1, color=color_h1, alpha=0.12,
                         label=f'H1 σ ({nm_h1.get("form","linear")})')
    ax.set_xlim(0, qmax)
    ax.set_ylim(bottom=0)
    ax.set_xlabel(r'$|Q_{\mathrm{meas}}|$  [nL/s]', fontsize=12)
    ax.set_ylabel(r'$|Q_{\mathrm{meas}} - \hat Q|$  [nL/s]', fontsize=12)
    ax.set_title('|Residual| vs |Q_meas|', fontsize=12)
    ax.legend(fontsize=9, loc='best')
    ax.grid(alpha=0.3)
    ax.tick_params(labelsize=11)

    # ── (1,0) summary text ─────────────────────────────────────────
    ax = axes[1, 0]
    ax.axis('off')
    lines: list[str] = []
    if title:
        lines.append(title)
        lines.append('')
    # Natural params
    for name in ('alpha', 'D', 'tau', 'mu'):
        if name in result.params:
            v = result.params[name]
            s = result.sigma.get(name, float('nan'))
            if name == 'tau':
                tau_deg = float(np.degrees(omega0 * v))
                lines.append(f'  τ̂  = {v * 1e3:+.2f} ms  '
                              f'({tau_deg:+.1f}° at f₀)')
            elif name == 'D':
                lines.append(f'  D̂  = {v:.3e} 1/Pa')
                lines.append(f'         ± {s:.3e}')
            elif name == 'mu':
                lines.append(f'  µ̂  = {v:.3e} Pa·s')
                lines.append(f'         ± {s:.3e}')
            else:
                lines.append(f'  {name:5s}= {v:6.4f}  ± {s:.4f}')
    p0 = result.convergence.get('p0', {})
    for k, v in p0.items():
        if k == 'alpha':
            continue
        lines.append(f'  {k}₀ (anchor) = {v:.3e}')
    lines.append('')
    lines.append(f'  χ²/dof       = {result.chi2_red:.3f}')
    lines.append(f'  N obs (real) = {result.n_obs_real}  '
                  f'(dof = {result.dof}, params = {result.n_params_fit})')
    nph = result.convergence.get('n_per_harmonic', {})
    if nph:
        lines.append(f'  N rows       = {result.n_obs_complex}  '
                      f'(' + ', '.join(f'{("DC" if k==0 else f"H{k}")}'
                                          f'={int(v)}'
                                          for k, v in nph.items()) + ')')
    if result.cov is not None and len(coef_order) >= 2:
        c = np.asarray(result.cov)
        sig = np.sqrt(np.maximum(np.diag(c), 0.0))
        if sig[0] > 0 and sig[1] > 0:
            rho = float(c[0, 1] / (sig[0] * sig[1]))
            lines.append(f'  ρ({coef_order[0]}, {coef_order[1]}) '
                          f'= {rho:+.3f}')
    if 'tau_warm_start_ssrs' in result.convergence:
        lines.append('')
        lines.append('  τ warm-start SSRs: '
                      + ', '.join(f'{x:.3g}' for x in
                                  result.convergence['tau_warm_start_ssrs']))
    if nm_dc or nm_h1:
        lines.append('')
        lines.append('  Per-harmonic noise σ(|Q|):')
        for n_label, nmm in (('DC', nm_dc), ('H1', nm_h1)):
            if not nmm:
                continue
            form = nmm.get('form', 'linear')
            if form == 'linear':
                lines.append(f'    {n_label}: '
                              f'{nmm["a"]:.3g} + {nmm["b"]:.3g}·|Q|')
            elif form == 'quadrature':
                lines.append(f'    {n_label}: √('
                              f'{nmm["a"]:.3g}² + ({nmm["b"]:.3g}·|Q|)²)')
            elif form == 'powerlaw':
                lines.append(f'    {n_label}: '
                              f'{nmm["c"]:.3g}·|Q|^{nmm["p"]:.3f}')
    ax.text(0.0, 1.0, '\n'.join(lines), transform=ax.transAxes,
             fontsize=11, family='monospace', verticalalignment='top')

    # ── (1,1) basis sensitivity ────────────────────────────────────
    # Plots |b^(0)| vs |b^(p)| where b^(p) is the first non-α basis
    # column (β_D in the typical (α, D) fit).  Generalizes if a
    # different reparam is the most informative — show the param with
    # the largest range of |b^(p)|.
    ax = axes[1, 1]
    non_alpha_cols = [c for c in coef_order if c != 'alpha']
    if non_alpha_cols and 'alpha' in dm.basis:
        # pick the column with largest dynamic range
        best = max(non_alpha_cols,
                    key=lambda c: float(np.ptp(np.abs(dm.basis[c]))))
        b0_mag = np.abs(dm.basis['alpha'])
        bp_mag = np.abs(dm.basis[best])
        if m_dc.any():
            ax.scatter(b0_mag[m_dc], bp_mag[m_dc], color=color_dc,
                        s=42, edgecolor='black', lw=0.5, alpha=0.85,
                        zorder=4, label=f'DC (N={n_dc})')
        if m_h1.any():
            ax.scatter(b0_mag[m_h1], bp_mag[m_h1], color=color_h1,
                        s=42, edgecolor='black', lw=0.5, alpha=0.85,
                        zorder=4, label=f'H1 (N={n_h1})')
        ax.axhline(0, color='gray', lw=0.6, alpha=0.5)
        ax.set_ylim(bottom=0)
        ax.set_xlabel(r'$|b^{(α)}_e|$  =  $|\hat Q_{\mathrm{sim}}|$  [nL/s]',
                       fontsize=12)
        natural = best.replace('beta_', '')
        ax.set_ylabel(rf'$|b^{{({natural})}}_e|$  =  '
                       rf'$|\partial \hat Q_{{\mathrm{{sim}}}}/\partial {natural}|$',
                       fontsize=12)
        ax.set_title(f'Basis sensitivity (first non-α: {natural})',
                      fontsize=12)
        ax.legend(fontsize=10)
        ax.grid(alpha=0.3)
        ax.tick_params(labelsize=11)
    else:
        ax.text(0.5, 0.5, '(no non-α LINEAR param)',
                 ha='center', va='center', transform=ax.transAxes,
                 fontsize=11, color='gray')
        ax.set_axis_off()

    suptitle = title or 'Inference result'
    method = result.convergence.get('method', '?')
    fig.suptitle(f'{suptitle}  —  {method}',
                  fontweight='bold', fontsize=13)

    return fig


# ──────────────────────────────────────────────────────────────────
# Two-stage feasible GLS diagnostic figure
# ──────────────────────────────────────────────────────────────────


def plot_two_stage_gls(
    result,
    *,
    title: Optional[str] = None,
    figsize: tuple = (8, 14),
    layout: str = 'vertical',
    pooled_residuals: Optional[dict] = None,
):
    """Three-panel figure illustrating the two-stage feasible GLS used
    in `_solve_closed_form_*`:

      Panel 1 (top)    : Stage 1 — unweighted fit, predicted vs measured
      Panel 2 (middle) : Noise model fit on |r|² vs |Q|² with overlay
      Panel 3 (bottom) : Stage 2 — weighted fit, predicted vs measured,
                          point sizes ∝ weights = 1/σ²

    Parameters
    ----------
    result : InferenceResult
        Must have `stage1_*` populated (closed-form routes do this).
    title : str, optional
    figsize : (w, h)
    layout : 'vertical' | 'horizontal'
    pooled_residuals : dict, optional
        For Panel 2: pooled residuals across multiple tiles.  Schema:
            {'Qm_mag_dc':   np.ndarray,
             'resid_mag_dc': np.ndarray,
             'Qm_mag_h1':   np.ndarray,
             'resid_mag_h1': np.ndarray}
        If None, falls back to single-tile residuals on `result`.

    Convention: x-axis shows real part of fit and measurement (cleanest
    for the regression intuition).  H1 measurements have already been
    τ-rotated and sign-corrected, so their real component carries
    the bulk of the signal after alignment.
    """
    import matplotlib.pyplot as plt

    if (result.design_matrix is None
            or result.stage1_theta is None
            or result.stage1_residuals is None
            or result.stage1_Qm_rotated is None):
        raise ValueError(
            'InferenceResult missing stage-1 captures; cannot draw '
            'two-stage GLS figure.  Re-run with the closed-form engine.')

    dm = result.design_matrix
    coef_order = result.convergence.get('coef_order', list(dm.basis.keys()))
    h_idx = dm.harmonic_idx
    m_dc = h_idx == 0
    m_h1 = h_idx == 1
    n_dc = int(m_dc.sum())
    n_h1 = int(m_h1.sum())

    color_dc = '#5A4FCF'   # blue/purple
    color_h1 = '#FF7F0E'   # orange

    # ── Stage 1 — predicted/measured (real parts) ──
    stage1_Qm = result.stage1_Qm_rotated
    pred_stage1 = sum(result.stage1_theta[c] * dm.basis[c]
                       for c in coef_order)
    x1_dc = np.real(pred_stage1[m_dc])
    y1_dc = np.real(stage1_Qm[m_dc])
    x1_h1 = np.real(pred_stage1[m_h1])
    y1_h1 = np.real(stage1_Qm[m_h1])

    # ── Stage 2 — predicted/measured (final τ-aligned, real parts) ──
    Qm_final = (result.Qm_rotated if result.Qm_rotated is not None
                else stage1_Qm)
    pred_stage2 = sum(result.params.get(c, 0.0) * dm.basis[c]
                       for c in coef_order)
    x2_dc = np.real(pred_stage2[m_dc])
    y2_dc = np.real(Qm_final[m_dc])
    x2_h1 = np.real(pred_stage2[m_h1])
    y2_h1 = np.real(Qm_final[m_h1])

    # Common axis bounds across panels 1 and 3
    all_x = np.concatenate([x1_dc, x1_h1, x2_dc, x2_h1])
    all_y = np.concatenate([y1_dc, y1_h1, y2_dc, y2_h1])
    if all_x.size and all_y.size:
        lo = float(min(all_x.min(), all_y.min()))
        hi = float(max(all_x.max(), all_y.max()))
        pad = 0.05 * (hi - lo if hi > lo else 1.0)
        xy_lo, xy_hi = lo - pad, hi + pad
    else:
        xy_lo, xy_hi = -1, 1

    # Stage-2 point weights ∝ 1/σ² (visualize the weighting)
    w = (1.0 / np.maximum(result.sigma_e, 1e-12) ** 2
          if result.sigma_e is not None else np.ones(len(stage1_Qm)))
    # Normalize to a sensible point-size range (10 pt² to 200 pt²)
    if w.size and w.max() > w.min():
        w_size = 10 + 190 * (w - w.min()) / (w.max() - w.min())
    else:
        w_size = np.full_like(w, 42.0)

    # ── Pooled residuals for Panel 2 ──
    if pooled_residuals is not None:
        q_dc_p = np.asarray(pooled_residuals.get('Qm_mag_dc', []))
        r_dc_p = np.asarray(pooled_residuals.get('resid_mag_dc', []))
        q_h1_p = np.asarray(pooled_residuals.get('Qm_mag_h1', []))
        r_h1_p = np.asarray(pooled_residuals.get('resid_mag_h1', []))
        pooled_label = '  (pooled)'
    else:
        # Fallback: use single-tile stage-1 residuals
        Qm_mag_all = np.abs(stage1_Qm)
        r_mag_all = np.abs(result.stage1_residuals)
        q_dc_p = Qm_mag_all[m_dc]
        r_dc_p = r_mag_all[m_dc]
        q_h1_p = Qm_mag_all[m_h1]
        r_h1_p = r_mag_all[m_h1]
        pooled_label = '  (single tile)'

    nm = next(iter(result.noise_model.values()), {})
    nm_dc = nm.get(0)
    nm_h1 = nm.get(1)

    # ── Build figure ──────────────────────────────────────────────
    if layout == 'horizontal':
        fig, axes = plt.subplots(1, 3, figsize=(figsize[1], figsize[0]),
                                  constrained_layout=True)
    else:
        fig, axes = plt.subplots(3, 1, figsize=figsize,
                                  constrained_layout=True)

    # ── Panel 1: Stage 1 unweighted fit ───────────────────────────
    ax1 = axes[0]
    if n_dc:
        ax1.scatter(x1_dc, y1_dc, color=color_dc, s=42,
                     edgecolor='black', lw=0.5, alpha=0.85, marker='o',
                     zorder=4, label=f'DC (N={n_dc})')
    if n_h1:
        ax1.scatter(x1_h1, y1_h1, color=color_h1, s=46,
                     edgecolor='black', lw=0.5, alpha=0.85, marker='^',
                     zorder=4, label=f'H1 (N={n_h1})')
    ax1.plot([xy_lo, xy_hi], [xy_lo, xy_hi], color='gray',
              lw=1.2, ls='--', alpha=0.6, label='y = x')
    ax1.set_xlim(xy_lo, xy_hi); ax1.set_ylim(xy_lo, xy_hi)
    ax1.set_aspect('equal')
    ax1.set_xlabel(r'fit:  $\alpha_0\,b^{(0)} + \beta_0\,b^{(D)}$  '
                    '[real, nL/s]', fontsize=12)
    ax1.set_ylabel(r'$\mathrm{Re}\,Q_e^{(\mathrm{exp})}$  [nL/s]',
                    fontsize=12)
    ax1.set_title('Stage 1 — Unweighted fit', fontsize=13,
                   fontweight='bold')
    s1 = result.stage1_theta
    ann1 = (r'$\alpha_0$ = ' + f'{s1.get("alpha", float("nan")):.3f}'
            + (r',  $\beta_0$ = '
               + f'{s1.get("beta_D", float("nan")):.3g}'
               if 'beta_D' in s1 else ''))
    ax1.text(0.03, 0.97, ann1, transform=ax1.transAxes,
              ha='left', va='top', fontsize=11,
              family='monospace',
              bbox=dict(boxstyle='round', fc='white', ec='gray',
                        alpha=0.85))
    ax1.legend(loc='lower right', fontsize=10)
    ax1.grid(alpha=0.3)
    ax1.tick_params(labelsize=11)

    # ── Panel 2: Noise model fit on |r|² vs |Q|² ──────────────────
    ax2 = axes[1]
    if q_dc_p.size:
        ax2.scatter(q_dc_p ** 2, r_dc_p ** 2, color=color_dc, s=20,
                     edgecolor='none', alpha=0.5, marker='o',
                     label=f'DC ({pooled_label.strip()}, N={q_dc_p.size})')
    if q_h1_p.size:
        ax2.scatter(q_h1_p ** 2, r_h1_p ** 2, color=color_h1, s=22,
                     edgecolor='none', alpha=0.5, marker='^',
                     label=f'H1 ({pooled_label.strip()}, N={q_h1_p.size})')
    # Overlay fitted noise models — σ² = c + d·|Q|²
    q2_lim = max(
        float(q_dc_p.max() ** 2) if q_dc_p.size else 0.0,
        float(q_h1_p.max() ** 2) if q_h1_p.size else 0.0,
        1e-6,
    )
    q2_grid = np.linspace(0, q2_lim, 200)
    if nm_dc is not None:
        from .inference import evaluate_noise_model
        sigma_dc = evaluate_noise_model(nm_dc, np.sqrt(q2_grid))
        ax2.plot(q2_grid, sigma_dc ** 2, color=color_dc, lw=2.0,
                  ls='-', alpha=0.95, zorder=5)
    if nm_h1 is not None:
        from .inference import evaluate_noise_model
        sigma_h1 = evaluate_noise_model(nm_h1, np.sqrt(q2_grid))
        ax2.plot(q2_grid, sigma_h1 ** 2, color=color_h1, lw=2.0,
                  ls='-', alpha=0.95, zorder=5)
    ax2.set_xlabel(r'$|Q_e^{(\mathrm{exp})}|^2$  [nL²/s²]', fontsize=12)
    ax2.set_ylabel(r'$|r_e|^2$  [nL²/s²]', fontsize=12)
    ax2.set_title(f'Noise model fit{pooled_label}', fontsize=13,
                   fontweight='bold')
    # Annotation showing fitted parameters
    ann_lines = []
    if nm_dc is not None:
        if nm_dc.get('form') == 'variance_linear':
            ann_lines.append(
                r'$\sigma_0^2$ = ' + f'{nm_dc["a"]:.3g}'
                + r' + ' + f'{nm_dc["b"]:.3g}' + r'$|Q|^2$'
            )
        else:
            ann_lines.append(f'DC: {nm_dc.get("form")}')
    if nm_h1 is not None:
        if nm_h1.get('form') == 'variance_linear':
            ann_lines.append(
                r'$\sigma_1^2$ = ' + f'{nm_h1["a"]:.3g}'
                + r' + ' + f'{nm_h1["b"]:.3g}' + r'$|Q|^2$'
            )
        else:
            ann_lines.append(f'H1: {nm_h1.get("form")}')
    if ann_lines:
        ax2.text(0.03, 0.97, '\n'.join(ann_lines),
                  transform=ax2.transAxes, ha='left', va='top',
                  fontsize=11, family='monospace',
                  bbox=dict(boxstyle='round', fc='white', ec='gray',
                            alpha=0.85))
    ax2.legend(loc='lower right', fontsize=10)
    ax2.grid(alpha=0.3)
    ax2.tick_params(labelsize=11)
    # Decide log scale by dynamic range
    if q2_lim > 1e3 * (q2_grid[1] if len(q2_grid) > 1 else 1):
        ax2.set_xscale('log')
        ax2.set_yscale('log')

    # ── Panel 3: Stage 2 weighted fit ──────────────────────────────
    ax3 = axes[2]
    if n_dc:
        ax3.scatter(x2_dc, y2_dc, color=color_dc, s=w_size[m_dc],
                     edgecolor='black', lw=0.5, alpha=0.80, marker='o',
                     zorder=4, label=f'DC (N={n_dc})')
    if n_h1:
        ax3.scatter(x2_h1, y2_h1, color=color_h1, s=w_size[m_h1],
                     edgecolor='black', lw=0.5, alpha=0.80, marker='^',
                     zorder=4, label=f'H1 (N={n_h1})')
    ax3.plot([xy_lo, xy_hi], [xy_lo, xy_hi], color='gray',
              lw=1.2, ls='--', alpha=0.6, label='y = x')
    ax3.set_xlim(xy_lo, xy_hi); ax3.set_ylim(xy_lo, xy_hi)
    ax3.set_aspect('equal')
    ax3.set_xlabel(r'fit:  $\hat\alpha\,b^{(0)} + \hat\beta\,b^{(D)}$  '
                    '[real, nL/s]', fontsize=12)
    ax3.set_ylabel(r'$\mathrm{Re}\,Q_e^{(\mathrm{exp})}$  [nL/s]',
                    fontsize=12)
    ax3.set_title('Stage 2 — Weighted fit  '
                   r'(point size $\propto 1/\sigma^2$)',
                   fontsize=13, fontweight='bold')
    alpha_hat = result.params.get('alpha', float('nan'))
    D_hat = result.params.get('D', float('nan'))
    beta_D_hat = result.params.get('beta_D',
                                     result.stage1_theta.get('beta_D',
                                                              float('nan')))
    ann3 = (r'$\hat\alpha$ = ' + f'{alpha_hat:.3f}'
            + r',  $\hat\beta$ = ' + f'{beta_D_hat:.3g}'
            + '\n' + r'$\hat D$ = ' + f'{D_hat:.3e} 1/Pa'
            + '\n' + r'$\chi^2/\mathrm{dof}$ = ' + f'{result.chi2_red:.3f}')
    ax3.text(0.03, 0.97, ann3, transform=ax3.transAxes,
              ha='left', va='top', fontsize=11, family='monospace',
              bbox=dict(boxstyle='round', fc='white', ec='gray',
                        alpha=0.85))
    ax3.legend(loc='lower right', fontsize=10)
    ax3.grid(alpha=0.3)
    ax3.tick_params(labelsize=11)

    suptitle = title or 'Two-stage feasible GLS'
    fig.suptitle(suptitle, fontweight='bold', fontsize=14)

    return fig


# ──────────────────────────────────────────────────────────────────
# Helpers for pooled-tile residual collection (Panel 2)
# ──────────────────────────────────────────────────────────────────


def collect_pooled_residuals(
    graph,
    forward_solver,
    *,
    tiles: Optional[list] = None,
    eps_D: float = 0.10,
    sim_state_get=None,
    sim_state_set=None,
    verbose: bool = False,
) -> dict:
    """Collect Stage-1 |r|² and |Q|² across multiple tiles for the
    noise-fit panel.

    Runs Pass 1 (unweighted) of `_solve_closed_form_with_tau` for each
    tile and stacks the magnitudes.

    Returns a dict consumable by `plot_two_stage_gls(pooled_residuals=...)`.
    """
    from .inference import (InferenceSpec, _build_design_matrix,
                              _phase_rotate, _solve_kxk_wls,
                              _solve_tau_closed_form,
                              _coef_order_from_spec)

    # Discover tiles from graph if not given.
    if tiles is None:
        seen = set()
        for _, _, d in graph.edges(data=True):
            for m in d.get('measurements_piv', []) or []:
                tid = m.get('tile_id')
                if tid is not None:
                    seen.add(int(tid))
        tiles = sorted(seen)

    Qm_mag_dc, resid_mag_dc = [], []
    Qm_mag_h1, resid_mag_h1 = [], []

    for tid in tiles:
        spec = InferenceSpec(
            fit_alpha=True, fit_D=True, fit_tau=True,
            eps_D=eps_D, scope='single_tile', focus_tile=int(tid),
            harmonics=(1,), verbose=False, save_to_graph=False,
            save_figure=False,
        )
        try:
            dm = _build_design_matrix(graph, spec, forward_solver,
                                        sim_state_get=sim_state_get,
                                        sim_state_set=sim_state_set)
        except Exception as _e:
            if verbose:
                print(f'  [pool] tile {tid} skipped: {_e}')
            continue
        if dm.Qm.size == 0:
            continue
        coef_order = _coef_order_from_spec(spec)
        omega0 = 2.0 * np.pi * dm.f0_tile if dm.f0_tile > 0 else 1.0
        # Single warm-start at τ=0 is enough for pooled noise statistics
        Qm_rot = _phase_rotate(dm.Qm, dm.harmonic_idx, omega0, 0.0)
        try:
            theta, _, _ = _solve_kxk_wls(Qm_rot, dm.basis,
                                           np.ones(len(Qm_rot)),
                                           coef_order)
        except np.linalg.LinAlgError:
            continue
        tau0 = _solve_tau_closed_form(dm.Qm, dm.basis, theta, coef_order,
                                        np.ones(len(Qm_rot)),
                                        dm.harmonic_idx, omega0)
        Qm_rot = _phase_rotate(dm.Qm, dm.harmonic_idx, omega0, tau0)
        pred = sum(theta[i] * dm.basis[c]
                    for i, c in enumerate(coef_order))
        resid = Qm_rot - pred
        h_idx = dm.harmonic_idx
        Qm_mag_dc.append(np.abs(Qm_rot[h_idx == 0]))
        resid_mag_dc.append(np.abs(resid[h_idx == 0]))
        Qm_mag_h1.append(np.abs(Qm_rot[h_idx == 1]))
        resid_mag_h1.append(np.abs(resid[h_idx == 1]))
        if verbose:
            print(f'  [pool] tile {tid}: '
                  f'DC={int((h_idx==0).sum())}, '
                  f'H1={int((h_idx==1).sum())}')

    return {
        'Qm_mag_dc': (np.concatenate(Qm_mag_dc) if Qm_mag_dc
                       else np.array([])),
        'resid_mag_dc': (np.concatenate(resid_mag_dc) if resid_mag_dc
                          else np.array([])),
        'Qm_mag_h1': (np.concatenate(Qm_mag_h1) if Qm_mag_h1
                       else np.array([])),
        'resid_mag_h1': (np.concatenate(resid_mag_h1) if resid_mag_h1
                          else np.array([])),
    }


# ──────────────────────────────────────────────────────────────────
# Multi-tile comparison figure
# ──────────────────────────────────────────────────────────────────


def plot_multi_tile_summary(
    multi_result: dict,
    *,
    title: Optional[str] = None,
    figsize: tuple = (14, 10),
):
    """Comparison figure for `run_inference_multi_tile` output.

    2×2 layout:
      (0,0) α̂ ± σ_α per tile
      (0,1) D̂ ± σ_D per tile
      (1,0) τ̂ per tile  (in degrees at f₀)
      (1,1) χ²/dof per tile  (with dashed horizontal at 1.0)

    Tiles ordered by tile id.  Failed tiles shown as gaps.
    """
    import matplotlib.pyplot as plt

    tiles = sorted(multi_result['results'].keys())
    if not tiles:
        raise ValueError('No successful tile fits to plot.')

    alpha_vals, alpha_errs = [], []
    D_vals, D_errs = [], []
    tau_vals_deg = []
    chi2_vals = []
    n_obs = []

    for tid in tiles:
        r = multi_result['results'][tid]
        alpha_vals.append(float(r.params.get('alpha', np.nan)))
        alpha_errs.append(float(r.sigma.get('alpha', np.nan)))
        D_vals.append(float(r.params.get('D', np.nan)))
        D_errs.append(float(r.sigma.get('D', np.nan)))
        tau_s = float(r.params.get('tau', 0.0))
        f0 = float(r.convergence.get('f0_tile', 0.0))
        tau_deg = (np.degrees(2 * np.pi * f0 * tau_s)
                    if f0 > 0 else float('nan'))
        # Wrap into [-180, 180]
        tau_deg = ((tau_deg + 180.0) % 360.0) - 180.0
        tau_vals_deg.append(tau_deg)
        chi2_vals.append(float(r.chi2_red))
        n_obs.append(int(r.n_obs_complex))

    alpha_vals = np.array(alpha_vals)
    alpha_errs = np.array(alpha_errs)
    D_vals = np.array(D_vals)
    D_errs = np.array(D_errs)
    tau_vals_deg = np.array(tau_vals_deg)
    chi2_vals = np.array(chi2_vals)

    fig, axes = plt.subplots(2, 2, figsize=figsize,
                              constrained_layout=True)

    # (0,0) α̂ per tile
    ax = axes[0, 0]
    ax.errorbar(tiles, alpha_vals, yerr=alpha_errs, fmt='o',
                 color='#5A4FCF', ecolor='#5A4FCF',
                 markersize=6, capsize=3, lw=1.0, alpha=0.85)
    med = float(np.nanmedian(alpha_vals))
    ax.axhline(med, color='gray', lw=1.0, ls='--', alpha=0.6,
                label=f'median = {med:.3f}')
    ax.axhline(1.0, color='black', lw=0.8, ls=':', alpha=0.5,
                label='α = 1 (sim matches meas)')
    ax.set_xlabel('Tile id', fontsize=12)
    ax.set_ylabel(r'$\hat\alpha$', fontsize=13)
    ax.set_title(r'$\hat\alpha$ per tile  (sim ↔ measurement scale)',
                  fontsize=12)
    ax.legend(fontsize=10, loc='best')
    ax.grid(alpha=0.3)
    ax.tick_params(labelsize=11)

    # (0,1) D̂ per tile
    ax = axes[0, 1]
    ax.errorbar(tiles, D_vals, yerr=D_errs, fmt='s',
                 color='#FF7F0E', ecolor='#FF7F0E',
                 markersize=6, capsize=3, lw=1.0, alpha=0.85)
    medD = float(np.nanmedian(D_vals))
    ax.axhline(medD, color='gray', lw=1.0, ls='--', alpha=0.6,
                label=f'median = {medD:.2e}')
    # D₀ anchor (constant across tiles since same forward sim)
    p0 = next(iter(multi_result['results'].values())).convergence.get(
        'p0', {})
    D0 = p0.get('D')
    if D0 is not None:
        ax.axhline(float(D0), color='black', lw=0.8, ls=':',
                    alpha=0.5, label=f'D₀ = {float(D0):.2e}')
    ax.set_xlabel('Tile id', fontsize=12)
    ax.set_ylabel(r'$\hat D$  [1/Pa]', fontsize=13)
    ax.set_title(r'$\hat D$ per tile  (distensibility)', fontsize=12)
    ax.legend(fontsize=10, loc='best')
    ax.grid(alpha=0.3)
    ax.tick_params(labelsize=11)

    # (1,0) τ̂ per tile (degrees)
    ax = axes[1, 0]
    ax.scatter(tiles, tau_vals_deg, s=42, color='#1F9E45',
                edgecolor='black', lw=0.5, alpha=0.85)
    ax.axhline(0, color='black', lw=0.8, ls=':', alpha=0.5)
    ax.set_xlabel('Tile id', fontsize=12)
    ax.set_ylabel(r'$\hat\tau$  [degrees at f₀]', fontsize=13)
    ax.set_ylim(-185, 185)
    ax.set_title(r'$\hat\tau$ per tile  '
                  '(should be uncorrelated across tiles)', fontsize=12)
    ax.grid(alpha=0.3)
    ax.tick_params(labelsize=11)

    # (1,1) χ²/dof per tile
    ax = axes[1, 1]
    colors = ['#3CB371' if c < 1.5
               else ('#FF8C00' if c < 3.0 else '#D62728')
               for c in chi2_vals]
    ax.bar(tiles, chi2_vals, color=colors, edgecolor='black', lw=0.4,
            alpha=0.85)
    ax.axhline(1.0, color='black', lw=0.8, ls='--', alpha=0.6,
                label='χ²/dof = 1')
    ax.set_xlabel('Tile id', fontsize=12)
    ax.set_ylabel(r'$\chi^2/\mathrm{dof}$', fontsize=13)
    ax.set_title(r'$\chi^2/\mathrm{dof}$ per tile  '
                  '(green<1.5, orange<3, red≥3)', fontsize=12)
    ax.legend(fontsize=10, loc='best')
    ax.grid(alpha=0.3, axis='y')
    ax.tick_params(labelsize=11)

    if multi_result.get('failures'):
        n_fail = len(multi_result['failures'])
        suptitle = (title or 'Multi-tile (α, D, τ) MLE summary') \
            + f'  —  {len(tiles)} succeeded, {n_fail} failed'
    else:
        suptitle = (title or 'Multi-tile (α, D, τ) MLE summary') \
            + f'  —  {len(tiles)} tiles'
    fig.suptitle(suptitle, fontweight='bold', fontsize=14)

    return fig
