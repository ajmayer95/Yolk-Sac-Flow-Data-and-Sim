"""Re-run harmonic fits on stored Q(t) signals to refresh all fit-derived
edge fields without re-running PIV extraction from videos.

The expensive part of batch (Farneback OF + profile fitting → Q_t) is
skipped; this only re-runs `fit_harmonics` on `measurements_piv[i]['Q_t']`
plus the downstream RPSI/eta/PI/quality_tier/aggregation steps.

Useful when you've added a new harmonic-fit metric (e.g. `snr_ac_fit_db`,
`snr_harm_fit_db` in the 2026-06-04 reanalysis) and want to backfill it on
a graph whose `Q_t_piv` is still trustworthy.

Used by:
  - "Refit harmonics from Q(t)" button in mosaic_app
  - `pertile.cli.refit_harmonics_from_qt` CLI
"""
from __future__ import annotations
from typing import Tuple

import numpy as np

from .config import FRAME_DT_S, N_HARMONICS
from .harmonic import fit_harmonics
from .transmission_line import compute_rpsi_from_qt, compute_eta_from_qt


def refit_harmonics_for_graph(
        G,
        *,
        verbose: bool = True,
        progress_every: int = 200,
        re_estimate_f0: bool = False,
        fmin_hz: float = 0.5,
        fmax_hz: float = 4.0,
) -> Tuple[int, int]:
    """Refit harmonics on every edge's stored Q_t and rewrite all
    fit-derived fields in place.

    - Per-measurement (`edge['measurements_piv'][i]`):
        mean_Q, amp_Q, PI, RPSI, eta, harmonics,
        snr_pulse, snr_harm_fit_db, snr_ac_fit_db,
        phase, amp_Q_h2, phase_h2, amp_Q_h3, phase_h3,
        pulse_frac, H1_frac, H2_frac, H3_frac,
        quality_tier, fit_success.
    - Top-level edge (best-by-snr_pulse aggregation): everything
      above suffixed with `_piv`.

    Doesn't touch `Q_t` itself.  Returns
    `(n_edges_updated, n_measurements_refit)`.

    When `re_estimate_f0=True`, first re-runs `consensus_f0_stacked`
    on each tile's stored Q(t) with `fmin_hz` / `fmax_hz` to get a
    fresh per-tile heart rate, updates `G.graph['tile_f0_piv']`, and
    uses the new f0 for the harmonic refit (overrides the per-
    measurement `f0_hz` cached from a prior batch run).
    """
    tile_f0_override = None
    if re_estimate_f0:
        tile_f0_override = _re_estimate_tile_f0(
            G, fmin_hz=fmin_hz, fmax_hz=fmax_hz, verbose=verbose)
        # Persist the new consensus into graph metadata
        G.graph['tile_f0_piv'] = dict(tile_f0_override)

    n_edges = 0
    n_meas = 0
    for k_e, (u, v, d) in enumerate(G.edges(data=True)):
        mlist = d.get('measurements_piv')
        if not mlist:
            continue
        any_updated = False
        for m in mlist:
            if _refit_one_measurement(m, tile_f0_override=tile_f0_override):
                any_updated = True
                n_meas += 1
        if any_updated:
            _aggregate_piv_top_level(d, mlist)
            n_edges += 1
        if verbose and progress_every and (k_e + 1) % progress_every == 0:
            print(f'  …processed {k_e + 1} edges  '
                   f'({n_edges} updated, {n_meas} measurements refit)')

    if verbose:
        print(f'Refit done: {n_edges} edges, {n_meas} measurements updated.')
    return n_edges, n_meas


def _re_estimate_tile_f0(
        G,
        *,
        fmin_hz: float,
        fmax_hz: float,
        verbose: bool = True,
) -> dict:
    """Re-estimate per-tile heart rate from each tile's stored Q(t)
    signals using `estimate_f0_in_band` + `consensus_f0_stacked`.

    Returns a `{tile_id: f0_hz}` dict.  Prints a diff against the
    current `G.graph['tile_f0_piv']` so the caller can see which
    tiles changed and by how much.
    """
    from .harmonic import estimate_f0_in_band, consensus_f0_stacked

    # Group all Q(t) by tile_id
    tile_qts: dict = {}
    for u, v, d in G.edges(data=True):
        for m in d.get('measurements_piv') or []:
            tid = m.get('tile_id')
            qt = m.get('Q_t')
            if tid is None or qt is None:
                continue
            try:
                arr = np.asarray(qt, dtype=float)
                finite = arr[np.isfinite(arr)]
                if len(finite) < 20 or np.nanstd(finite) < 1e-12:
                    continue
                tile_qts.setdefault(int(tid), []).append(finite)
            except Exception:
                continue

    if verbose:
        print(f'Re-estimating f0 for {len(tile_qts)} tiles  '
               f'(fmin={fmin_hz} Hz, fmax={fmax_hz} Hz)')

    old_f0s = dict(G.graph.get('tile_f0_piv', {}))
    new_f0s: dict = {}
    for tid in sorted(tile_qts):
        periodograms = []
        for qt in tile_qts[tid]:
            try:
                _, freqs, power = estimate_f0_in_band(
                    qt, FRAME_DT_S,
                    fmin_hz=fmin_hz, fmax_hz=fmax_hz,
                    return_periodogram=True)
                if len(freqs) and len(power) and np.any(power > 0):
                    periodograms.append((freqs, power))
            except Exception:
                continue
        if periodograms:
            try:
                f0 = float(consensus_f0_stacked(
                    periodograms, fmin_hz=fmin_hz, fmax_hz=fmax_hz))
                if np.isfinite(f0):
                    new_f0s[tid] = f0
            except Exception:
                continue

    if verbose:
        n_changed = 0
        n_big = 0
        for tid in sorted(new_f0s):
            new_v = new_f0s[tid]
            old_v = old_f0s.get(tid)
            if old_v is None:
                print(f'  tile {tid:>3}: (new) → {new_v:.3f} Hz')
                continue
            rel = abs(new_v - old_v) / max(abs(old_v), 1e-9)
            if rel > 0.05:
                n_changed += 1
                if rel > 0.20: n_big += 1
                print(f'  tile {tid:>3}: {old_v:.3f} → {new_v:.3f} Hz  '
                       f'(Δ={(new_v-old_v):+.3f}, {rel:.1%})')
        n_unchanged = len(new_f0s) - n_changed
        print(f'  Summary: {n_changed}/{len(new_f0s)} tiles changed by >5% '
               f'({n_big} by >20%), {n_unchanged} essentially unchanged.')

    return new_f0s


# -- private helpers ---------------------------------------------------------

def _refit_one_measurement(m: dict, *, tile_f0_override: dict = None) -> bool:
    """Rerun fit_harmonics on m['Q_t'] and update m in place.
    Returns True if fields were updated, False if skipped.

    If `tile_f0_override` is given, use `tile_f0_override[m['tile_id']]`
    as the f0 instead of m['f0_hz'] (and update m['f0_hz'] in place).
    """
    Q_t = m.get('Q_t')
    if Q_t is None:
        return False
    # f0 — use override if provided, else cached value on the measurement
    f0 = None
    if tile_f0_override is not None:
        tid = m.get('tile_id')
        if tid is not None:
            f0 = tile_f0_override.get(int(tid))
    if f0 is None:
        f0 = m.get('f0_hz')
    if f0 is None:
        return False
    try:
        f0 = float(f0)
    except (TypeError, ValueError):
        return False
    if not np.isfinite(f0) or f0 <= 0:
        return False
    m['f0_hz'] = f0     # record the f0 we actually used
    try:
        hr = fit_harmonics(Q_t, FRAME_DT_S, f0,
                            K=N_HARMONICS, loss='huber', include_dc=True)
    except Exception:
        return False

    # ── Primary fit fields ─────────────────────────────────────────────
    mean_Q = float(hr.get('a0', np.nan))
    if not np.isfinite(mean_Q):
        mean_Q = float(np.nanmean(Q_t))
    m['mean_Q'] = mean_Q
    m['amp_Q']  = float(hr.get('amp1', np.nan))
    m['snr_pulse']       = float(hr.get('hr_snr_db',      np.nan))
    m['snr_harm_fit_db'] = float(hr.get('snr_harm_fit_db', np.nan))
    m['snr_ac_fit_db']   = float(hr.get('snr_ac_fit_db',   np.nan))
    harmonics = hr.get('harmonics', []) or []
    m['harmonics'] = harmonics

    # ── Per-harmonic amp/phase break-outs (raw, not relative) ─────────
    # H1 phase is the existing 'phase' field for back-compat.
    if harmonics:
        m['phase'] = float(np.degrees(harmonics[0]['phi']) % 360)
    if len(harmonics) >= 2:
        m['amp_Q_h2'] = float(harmonics[1]['amp'])
        m['phase_h2'] = float(np.degrees(harmonics[1]['phi']) % 360)
    if len(harmonics) >= 3:
        m['amp_Q_h3'] = float(harmonics[2]['amp'])
        m['phase_h3'] = float(np.degrees(harmonics[2]['phi']) % 360)

    # ── Power-fraction decomposition (∼ synthetic_viewer convention) ─
    # P(t) = a0 + Σ Aₙ cos(nωt) + Bₙ sin(nωt)
    # ⟨Q²⟩ = a0² + ½·Σ|Qₙ|²  where |Qₙ|² = Aₙ² + Bₙ²
    P_DC = mean_Q * mean_Q
    P_h  = [0.5 * float(h.get('amp', 0.0)) ** 2 for h in harmonics]
    P_osc = float(sum(P_h))
    P_tot = P_DC + P_osc
    if P_tot > 1e-30:
        m['pulse_frac'] = P_osc / P_tot
        if P_osc > 1e-30:
            # Per-harmonic fraction of OSCILLATORY content (waveform shape)
            m['H1_frac'] = (P_h[0] / P_osc) if len(P_h) >= 1 else float('nan')
            m['H2_frac'] = (P_h[1] / P_osc) if len(P_h) >= 2 else float('nan')
            m['H3_frac'] = (P_h[2] / P_osc) if len(P_h) >= 3 else float('nan')
        else:
            m['H1_frac'] = m['H2_frac'] = m['H3_frac'] = float('nan')
    else:
        m['pulse_frac'] = float('nan')
        m['H1_frac'] = m['H2_frac'] = m['H3_frac'] = float('nan')

    # ── PI ─────────────────────────────────────────────────────────────
    amp_Q = m['amp_Q']
    m['PI'] = (float(2.0 * amp_Q / mean_Q)
                if (np.isfinite(amp_Q) and mean_Q > 1e-12) else float('nan'))

    # ── RPSI / eta from Q_t (cycle-averaged metrics) ──────────────────
    try:
        m['RPSI'] = float(compute_rpsi_from_qt(Q_t, f0, FRAME_DT_S))
    except Exception:
        m['RPSI'] = float('nan')
    try:
        m['eta'] = float(compute_eta_from_qt(Q_t, f0, FRAME_DT_S))
    except Exception:
        m['eta'] = float('nan')

    # ── Quality tier (mirrors batch_analyze_mosaic_piv logic) ─────────
    snr_pulse = m.get('snr_pulse', float('nan'))
    fit_success = bool(np.isfinite(mean_Q) and mean_Q > 0)
    m['fit_success'] = fit_success
    if fit_success and np.isfinite(snr_pulse) and snr_pulse >= 3.0:
        m['quality_tier'] = 'A'
    elif fit_success:
        m['quality_tier'] = 'B'
    else:
        m['quality_tier'] = 'X'

    return True


def _aggregate_piv_top_level(d: dict, mlist: list) -> None:
    """Pick the best measurement (by snr_pulse, ignoring tier='X') and
    write its values into the top-level edge `_piv` attributes."""
    valid = [m for m in mlist if m.get('quality_tier') in ('A', 'B')]
    if not valid:
        valid = mlist   # fall back to whatever we have
    if not valid:
        return
    best = max(valid, key=lambda m: m.get('snr_pulse', -np.inf))

    # Existing top-level fields the batch writes (kept verbatim)
    d['mean_Q_piv']          = best.get('mean_Q', float('nan'))
    d['amp_Q_piv']           = best.get('amp_Q',  float('nan'))
    d['PI_piv']              = best.get('PI',     float('nan'))
    d['RPSI_piv']            = best.get('RPSI',   float('nan'))
    d['eta_piv']             = best.get('eta',    float('nan'))
    d['snr_pulse_piv']       = best.get('snr_pulse',       float('nan'))
    d['snr_f0_piv']          = best.get('snr_f0',          float('nan'))
    d['snr_harm_fit_db_piv'] = best.get('snr_harm_fit_db', float('nan'))
    d['snr_ac_fit_db_piv']   = best.get('snr_ac_fit_db',   float('nan'))
    d['quality_tier_piv']    = best.get('quality_tier', 'X')
    d['v_max_piv']           = best.get('v_max', float('nan'))
    d['f0_hz_piv']           = best.get('f0_hz', float('nan'))
    d['phase_piv']           = best.get('phase', float('nan'))
    d['flow_from_piv']       = best.get('flow_from')
    d['flow_to_piv']         = best.get('flow_to')
    d['Q_t_piv']             = best.get('Q_t')
    # New top-level fields (per-harmonic + power fractions)
    d['amp_Q_h2_piv']        = best.get('amp_Q_h2', float('nan'))
    d['phase_h2_piv']        = best.get('phase_h2', float('nan'))
    d['amp_Q_h3_piv']        = best.get('amp_Q_h3', float('nan'))
    d['phase_h3_piv']        = best.get('phase_h3', float('nan'))
    d['pulse_frac_piv']      = best.get('pulse_frac', float('nan'))
    d['H1_frac_piv']         = best.get('H1_frac',    float('nan'))
    d['H2_frac_piv']         = best.get('H2_frac',    float('nan'))
    d['H3_frac_piv']         = best.get('H3_frac',    float('nan'))
