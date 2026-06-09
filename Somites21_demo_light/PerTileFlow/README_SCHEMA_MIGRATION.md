# Canonical graph schema migration (v0.3.0)

The mosaic graph this codebase consumes has been migrated to a new
canonical schema.  This document explains what changed, why, and what
you need to do.

## TL;DR

1. Download `Somites{21,27}_canonical_graph_v0.3.0.zip` from the
   [v0.3.0 release](https://github.com/ajmayer95/Yolk-Sac-Flow-Data-and-Sim/releases/tag/v0.3.0).
2. Unzip and drop `mosaic_graph_canonical.gpickle` into your
   `emb1/analyzed/` folder.
3. Run scripts as usual.  `inspect_tile.py` now defaults to the
   canonical filename; reader logic is backwards-compatible (falls back
   to legacy `amp_Q_h1_piv` and kymo `amp_Q` if canonical fields are
   absent).

## What changed

The legacy graph carried up to **three different "H1 phasor" fields
with conflicting values**.  A direct refit verification on a sample of
measured edges showed:

| Legacy field | median \|Δamp\|/refit | median \|Δφ\| |
|---|---|---|
| `m['harmonics'][0]` (per-tile measurement list) | **0.7%** | **1.0°** |
| `amp_Q_h1_piv` / `phase_h1_piv` (top-level edge attr) | 133.6% | 141.9° |
| `amp_Q` / `phase` (kymograph-era edge attr) | varies | varies |

Only `m['harmonics'][0]` matches a clean refit of `Q_t` at the
unified per-tile fundamental.  The other fields encoded different (and
in some cases unidentified) computations — code reading the wrong
field got the wrong answer.

The canonical schema exposes only the honest field, under a single
unambiguous name (`Q_H1_amp` / `Q_H1_phi`).  Full schema in
[`SCHEMA.md`](https://github.com/ajmayer95/Yolk-Sac-Flow-Data-and-Sim/blob/main/SCHEMA.md).

## Empirical impact on this codebase

The same `inspect_tile.py` joint-LM inference on stage-21 tile 4:

| H1 source | D̂ | σ/D̂ | χ²_red |
|---|---|---|---|
| `amp_Q` (kymo, prior default) | 1.52×10⁻³ /Pa | 38% | (poor) |
| `amp_Q_h1_piv` (intermediate) | 1.08×10⁻³ /Pa | **117%** | (worse) |
| `Q_H1_amp` (canonical, this release) | **2.26×10⁻³ /Pa** | **30%** | **0.52** |

The canonical-schema value lands closer to the production network-wide
median (3.98×10⁻³ /Pa) and converges with substantially tighter σ.
The remaining ~3× gap from the Bayesian b-marginalized inference
(D̂ = 6.81×10⁻³ /Pa on the same tile) reflects methodology architecture
differences — full vs measured-only carve, joint LM vs Bayesian, etc.
Not a data artifact.

## What was patched in this PR

- `scripts/inspect_tile.py` — `build_tile_problem()` reader now reads:
  - DC: `Q_DC` → `mean_Q_piv` → `mean_Q` → `mean_Q_nL_s`
  - H1: `Q_H1_amp`/`Q_H1_phi` → `amp_Q_h1_piv`/`phase_h1_piv` → `amp_Q`/`phase`
- `scripts/inspect_tile.py` — default `GRAPH_PATH` now points at
  `mosaic_graph_canonical.gpickle` (relative).  Override with the
  positional argument or environment as needed.
- `pertile/analysis/harmonic.py` — `fit_harmonics` now also returns
  per-harmonic SNRs (`snr_h1_db`, `snr_h2_db`, `snr_h3_db`) and DC-only
  SNR (`snr_dc_fit_db`).  Read-only addition; backwards-compatible.
- `pertile/viewer/mosaic_readonly_app.py` — supports canonical fields
  in the View tab (new per-harmonic SNR Property, aggregate SNR_AC /
  SNR_total Properties, per-harmonic PI, dissipation percentile filter,
  edge value labels).  Legacy graphs still work via fallback chain.
- `scripts/build_canonical_graph.py` — new migration script.  Run on a
  legacy graph to produce a canonical one.

## What was NOT patched

Other scripts in `scripts/` may also read legacy field names:

- `synthetic_validation_neumann_bc.py`
- `infer_bayes_default_mosaic_tile_profiles.py`
- `h2_sensitivity_check.py`
- `tile_mosaic_simulation.py`

These weren't patched in this PR because they use slightly different
reader patterns and may need per-script attention.  If you actively
use any of them, the same canonical-first fallback chain applies —
read `Q_DC`, `Q_H1_amp`, `Q_H1_phi` first, fall back to legacy.

## Backwards compatibility

The reader patches use a canonical→legacy fallback chain, so the same
script works on:

- Old legacy graphs (kymo `amp_Q` route)
- The pre-canonical demo bundle (legacy `amp_Q_h1_piv` route)
- The new canonical graph (`Q_H1_amp` route)

No need to update both code AND graph atomically.

## Questions

If anything's unclear, check:

- [SCHEMA.md](https://github.com/ajmayer95/Yolk-Sac-Flow-Data-and-Sim/blob/main/SCHEMA.md) — canonical schema reference
- [CHANGELOG.md](https://github.com/ajmayer95/Yolk-Sac-Flow-Data-and-Sim/blob/main/CHANGELOG.md) — v0.3.0 release notes
