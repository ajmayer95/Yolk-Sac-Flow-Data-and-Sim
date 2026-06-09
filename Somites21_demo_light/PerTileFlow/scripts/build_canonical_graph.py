"""Build a canonical mosaic graph with a clean, unambiguous schema.

Input:  current mosaic_graph_analyzed.gpickle (cloud-drive, with legacy
        fields + redundant kymograph/PIV/linear/murray/sim/gz variants)
Output: mosaic_graph_canonical.gpickle (clean schema, only honest fields)

Schema produced
───────────────

Top-level edge attrs:
  radius, length, flow_from, flow_to
  Q_t_piv                              # raw signal (canonical tile)
  f0_hz                                 # consensus f₀ (canonical tile)
  tile_canonical                        # tile_id whose measurement was promoted
  Q_DC,          Q_DC_snr_db
  Q_H1_amp,      Q_H1_phi,      Q_H1_snr_db
  Q_H2_amp,      Q_H2_phi,      Q_H2_snr_db
  Q_H3_amp,      Q_H3_phi,      Q_H3_snr_db
  snr_ac_fit_db, snr_harm_fit_db

Per-tile `measurements_piv` list — each entry uses the same flat keys:
  tile_id, f0_hz, Q_t,
  Q_DC,          Q_DC_snr_db,
  Q_H1_amp, Q_H1_phi, Q_H1_snr_db,
  Q_H2_amp, Q_H2_phi, Q_H2_snr_db,
  Q_H3_amp, Q_H3_phi, Q_H3_snr_db,
  snr_ac_fit_db, snr_harm_fit_db,
  harmonics                             # raw fit list with k/A/B/sigmas/P_sig

Node attrs kept: x, y, boundary_type (when set), tile_id (if any).
Graph attrs kept: tile_f0_piv, mosaic_height, mosaic_width,
                  exclusion_blue_sources, exclusion_blue_sinks.

Canonical tile promotion rule:
  m_best = max(measurements_piv, key=lambda m: m['Q_H1_snr_db'])
  (Highest H1 SNR — cardiac fundamental is what carries D₀ identification.)

Usage
─────
    python scripts/build_canonical_graph.py SRC_GRAPH OUT_GRAPH [--dry-run] [--conservative]

Flags:
  --dry-run        Print plan + first-edge before/after diff, don't write.
  --conservative   Only drop the four proven-misleading fields (amp_Q,
                   phase, amp_Q_h1_piv, phase_h1_piv) and their h2/h3
                   variants.  Default is aggressive (drop the full
                   legacy *_kymo/_linear/_murray/_gz/_sim cluster too).
"""
from __future__ import annotations
import argparse
import pickle
import sys
from pathlib import Path
from collections import defaultdict

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from pertile.analysis.harmonic import fit_harmonics

FRAME_DT_S = 1.0 / 250.0


# ── Fields to drop ──────────────────────────────────────────────────────

# Proven misleading by direct refit comparison (134% amp / 142° phase off).
PROVEN_MISLEADING = {
    "amp_Q", "phase",
    "amp_Q_h1_piv", "phase_h1_piv",
    "amp_Q_h2_piv", "phase_h2_piv",        # same convention by association
    "amp_Q_h3_piv", "phase_h3_piv",
}

# Legacy duplicate variants of mean_Q / amp_Q / phase / Q_t from older pipelines.
LEGACY_AGGRESSIVE = {
    # mean_Q variants
    "mean_Q", "mean_Q_kymo", "mean_Q_linear", "mean_Q_murray",
    "mean_Q_gz", "mean_Q_sim", "mean_Q_piv", "mean_Q_piv_murray",
    "mean_Q_nL_s", "mean_Q_source",
    # amp_Q variants
    "amp_Q_gz", "amp_Q_linear", "amp_Q_murray", "amp_Q_sim",
    "amp_Q_piv", "amp_Q_piv_murray",
    # phase variants
    "phase_global", "phase_linear", "phase_raw_linear", "phase_sim",
    "phase_h2_rel", "phase_piv",
    # Q_t variants
    "Q_t", "Q_t_murray", "Q_t_piv_murray", "Q_min_sign",
    "Q_rms_ac_linear", "Q_rms_ac_sim", "Q_stored_mag_sim",
    # derived statistics that are recomputable from harmonics
    "H1_frac_piv", "H2_frac_piv", "H3_frac_piv",
    "PI_piv", "pulse_frac_piv", "RPSI_piv", "eta_piv",
    "WSS_amp", "v_max_piv",
    # legacy SNR/quality
    "snr_pulse", "snr_pulse_piv", "snr_f0", "snr_f0_piv",
    "snr_harm_fit_db_piv", "snr_ac_fit_db_piv",
    "quality_tier_piv",
    # redundant flow direction
    "flow_from_piv", "flow_to_piv",
    # f0
    "f0_hz_piv",
}


# ── Helpers ────────────────────────────────────────────────────────────

def refit_harmonics_from_Q_t(Q_t, f0, K=3):
    """Re-run fit_harmonics from the raw Q_t.  Returns the result dict or
    None if Q_t is too short / bad."""
    Q_t = np.asarray(Q_t, dtype=float)
    Q_t = Q_t[np.isfinite(Q_t)]
    if len(Q_t) < 32:
        return None
    try:
        return fit_harmonics(Q_t, FRAME_DT_S, float(f0), K=K, loss='huber',
                              include_dc=True)
    except Exception:
        return None


def build_per_tile_entry(m_old, f0_consensus):
    """Convert one measurements_piv entry into the canonical per-tile dict.

    Refits harmonics from Q_t if `harmonics` is missing/empty (e.g. demo bundle).
    Returns None if Q_t isn't usable.
    """
    Q_t = m_old.get("Q_t")
    if Q_t is None:
        Q_t = m_old.get("Q_t_piv")
    f0_m = m_old.get("f0_hz")
    if f0_m is None:
        f0_m = f0_consensus
    if Q_t is None:
        return None
    needs_refit = (not m_old.get("harmonics"))
    if needs_refit:
        ref = refit_harmonics_from_Q_t(Q_t, f0_m)
        if ref is None:
            return None
        harmonics_full = ref["harmonics"]
        # Mean from refit (= a0 from the fit, post mean-subtraction).
        Q_DC = float(ref.get("a0", np.nan))
        # If the fit didn't include DC, fall back to numerical mean.
        if not np.isfinite(Q_DC):
            Q_DC = float(np.nanmean(np.asarray(Q_t, float)))
        snr_h_k = {f"snr_h{k}_db": float(ref.get(f"snr_h{k}_db", np.nan))
                   for k in (1, 2, 3)}
        snr_ac = float(ref.get("snr_ac_fit_db", np.nan))
        snr_harm = float(ref.get("snr_harm_fit_db", np.nan))
        snr_dc = float(ref.get("snr_dc_fit_db", np.nan))
    else:
        harmonics_full = list(m_old["harmonics"])
        Q_DC = float(m_old.get("mean_Q", np.nan))
        # The stored harmonics list may not have snr_db per harmonic
        # (older graphs).  Compute from P_sig and ms_bl if missing —
        # but ms_bl isn't on the stored data.  Easier: rerun fit_harmonics
        # to get the consistent SNR set.  Cheap if Q_t is available.
        ref = refit_harmonics_from_Q_t(Q_t, f0_m)
        if ref is not None:
            harmonics_full = ref["harmonics"]
            Q_DC = float(ref.get("a0", Q_DC))
            snr_h_k = {f"snr_h{k}_db": float(ref.get(f"snr_h{k}_db", np.nan))
                       for k in (1, 2, 3)}
            snr_ac = float(ref.get("snr_ac_fit_db", np.nan))
            snr_harm = float(ref.get("snr_harm_fit_db", np.nan))
            snr_dc = float(ref.get("snr_dc_fit_db", np.nan))
        else:
            # No Q_t — fall back to whatever's in the stored harmonics
            snr_h_k = {f"snr_h{k}_db": np.nan for k in (1, 2, 3)}
            snr_ac = float(m_old.get("snr_ac_fit_db", np.nan))
            snr_harm = float(m_old.get("snr_harm_fit_db", np.nan))
            snr_dc = np.nan

    def harm(i):
        if i < len(harmonics_full):
            h = harmonics_full[i]
            return float(h.get("amp", np.nan)), float(h.get("phi", np.nan))
        return float("nan"), float("nan")

    Q_H1_amp, Q_H1_phi = harm(0)
    Q_H2_amp, Q_H2_phi = harm(1)
    Q_H3_amp, Q_H3_phi = harm(2)
    return dict(
        tile_id=int(m_old["tile_id"]),
        f0_hz=float(f0_m),
        Q_t=np.asarray(Q_t, dtype=float),
        Q_DC=Q_DC,
        Q_DC_snr_db=snr_dc,
        Q_H1_amp=Q_H1_amp, Q_H1_phi=Q_H1_phi,
        Q_H1_snr_db=snr_h_k["snr_h1_db"],
        Q_H2_amp=Q_H2_amp, Q_H2_phi=Q_H2_phi,
        Q_H2_snr_db=snr_h_k["snr_h2_db"],
        Q_H3_amp=Q_H3_amp, Q_H3_phi=Q_H3_phi,
        Q_H3_snr_db=snr_h_k["snr_h3_db"],
        snr_ac_fit_db=snr_ac,
        snr_harm_fit_db=snr_harm,
        harmonics=harmonics_full,
    )


def build_top_level_from_canonical(m_best, ed_old):
    """Top-level edge attrs from the canonical (highest-Q_H1_snr_db) tile."""
    out = dict(
        radius=float(ed_old.get("radius", np.nan)),
        length=float(ed_old.get("length", np.nan)),
        flow_from=ed_old.get("flow_from"),
        flow_to=ed_old.get("flow_to"),
        Q_t_piv=m_best["Q_t"],
        f0_hz=m_best["f0_hz"],
        tile_canonical=m_best["tile_id"],
    )
    # Copy the canonical Q values with the same key names
    for k in ("Q_DC", "Q_DC_snr_db",
               "Q_H1_amp", "Q_H1_phi", "Q_H1_snr_db",
               "Q_H2_amp", "Q_H2_phi", "Q_H2_snr_db",
               "Q_H3_amp", "Q_H3_phi", "Q_H3_snr_db",
               "snr_ac_fit_db", "snr_harm_fit_db"):
        out[k] = m_best[k]
    return out


# ── Main ────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("src", type=Path, help="input gpickle path")
    ap.add_argument("out", type=Path, help="output gpickle path")
    ap.add_argument("--dry-run", action="store_true",
                     help="print plan + first-edge diff, don't write")
    ap.add_argument("--conservative", action="store_true",
                     help="only drop proven-misleading fields, keep legacy")
    args = ap.parse_args()

    drop_set = PROVEN_MISLEADING.copy()
    if not args.conservative:
        drop_set |= LEGACY_AGGRESSIVE

    print(f"Loading {args.src}")
    G = pickle.load(open(args.src, "rb"))
    print(f"  {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    tile_f0 = G.graph.get("tile_f0_piv") or {}
    print(f"  tile_f0_piv has {len(tile_f0)} entries")
    print(f"  drop policy: {'conservative' if args.conservative else 'aggressive'} "
          f"({len(drop_set)} fields)")

    # Count of fields per edge before vs after, for the audit trail
    before_counts = defaultdict(int)
    after_counts = defaultdict(int)
    n_edges_processed = 0
    n_edges_skipped = 0
    n_refit = 0

    # Build canonical edges
    for u, v, d in list(G.edges(data=True)):
        # Track field provenance
        for k in d.keys(): before_counts[k] += 1

        old_meas = d.get("measurements_piv") or []
        if not old_meas:
            # No tile measurements — drop everything except geometry?  For
            # safety, just leave the edge alone (skip canonical fill).
            n_edges_skipped += 1
            continue

        # Rebuild per-tile measurements with canonical schema
        new_meas = []
        for m in old_meas:
            tid = m.get("tile_id")
            if tid is None: continue
            f0_consensus = tile_f0.get(tid, m.get("f0_hz"))
            if f0_consensus is None: continue
            entry = build_per_tile_entry(m, f0_consensus)
            if entry is None: continue
            new_meas.append(entry)
            n_refit += 1
        if not new_meas:
            n_edges_skipped += 1
            continue

        # Pick canonical tile = highest Q_H1_snr_db (NaN → -inf)
        def _key(m):
            v = m.get("Q_H1_snr_db", float("nan"))
            return v if np.isfinite(v) else -np.inf
        m_best = max(new_meas, key=_key)

        # Build new edge dict
        new_ed = build_top_level_from_canonical(m_best, d)
        new_ed["measurements_piv"] = new_meas
        # Keep raw radius/length/flow_from/flow_to (already in new_ed)
        # Plus boundary type, etc. that downstream code cares about — if any

        # Drop misleading + legacy fields from the original.  Anything we
        # didn't explicitly produce above + isn't in drop_set gets carried
        # over verbatim (defensive: don't lose data we don't understand yet).
        for k, val in d.items():
            if k in drop_set: continue
            if k in new_ed: continue  # canonical already set
            if k == "measurements_piv": continue
            new_ed[k] = val

        for k in new_ed.keys(): after_counts[k] += 1

        # Replace the edge dict's contents in place (preserve identity for nx)
        d.clear(); d.update(new_ed)
        n_edges_processed += 1

    print(f"\n  edges processed: {n_edges_processed}, "
          f"skipped (no measurements): {n_edges_skipped}, "
          f"per-tile entries refit: {n_refit}")

    # Audit
    print(f"\n  Fields per edge (canonical schema):")
    dropped = []
    for k in sorted(before_counts.keys()):
        b = before_counts[k]; a = after_counts.get(k, 0)
        if a == 0:
            dropped.append(k)
        elif b != a:
            print(f"    {k:<28s}: {b} → {a}")
    if dropped:
        print(f"\n  Dropped {len(dropped)} field(s):")
        for k in sorted(dropped):
            print(f"    - {k}")
    new_only = sorted(k for k in after_counts if k not in before_counts)
    if new_only:
        print(f"\n  Added {len(new_only)} field(s):")
        for k in new_only:
            print(f"    + {k}")

    if args.dry_run:
        print(f"\n[dry-run] not writing {args.out}")
        return

    print(f"\n  Writing {args.out}")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "wb") as f:
        pickle.dump(G, f)
    print(f"  Wrote {args.out.stat().st_size / 1024 / 1024:.0f} MB")


if __name__ == "__main__":
    main()
