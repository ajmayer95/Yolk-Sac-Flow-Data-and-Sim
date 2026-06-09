# Changelog

## v0.3.0 — Canonical graph schema migration

This release replaces the legacy mosaic-graph schema with a clean,
unambiguous canonical schema.  See `SCHEMA.md` for the full field list.

**Headline change:** the H1 phasor (and the rest of the per-harmonic
flow data) is now stored under one well-defined name (`Q_H1_amp` /
`Q_H1_phi`), derived from a re-fit of `Q_t` at the unified per-tile
fundamental frequency.  Verified by direct refit to within 0.7%
amplitude / 1° phase.  The legacy graph carried up to three
differently-named "H1 phasor" fields with conflicting values — only
one was honest.

### Why

A direct refit comparison on a sample of measured edges showed:

| Legacy field | median \|Δamp\|/refit | median \|Δφ\| |
|---|---|---|
| `m['harmonics'][0]` (used by production v3 inference) | **0.7%** | **1.0°** |
| `amp_Q_h1_piv` / `phase_h1_piv` (top-level edge attr) | 133.6% | 141.9° |
| `amp_Q` / `phase` (kymograph-era edge attr) | varies | varies |

Code reading the wrong field got the wrong answer.  The canonical
schema exposes only the honest field, under a single unambiguous name.

### What's in this release

- **`scripts/build_canonical_graph.py`** — migration script that
  produces a `mosaic_graph_canonical.gpickle` from a legacy graph.
  Aggressive cleanup (drops ~50 legacy fields) by default; pass
  `--conservative` to keep all but the four proven-misleading ones.
- **Updated `pertile.analysis.harmonic`** — `fit_harmonics` now also
  returns per-harmonic SNRs (`snr_h1_db`, `snr_h2_db`, `snr_h3_db`)
  and DC-only SNR (`snr_dc_fit_db`).
- **Updated `pertile.viewer.mosaic_readonly_app`** —
  - New per-harmonic `SNR (dB)` Property (replaces the ambiguous
    `Resolution (Z = amp/SE)` for typical use)
  - New aggregate `SNR_AC (dB)` and `SNR_total (dB)` Properties
    (read canonical `snr_ac_fit_db` / `snr_harm_fit_db`)
  - `Pulsatility index` is now per-harmonic (PI@H1, PI@H2, PI@H3)
    with the Gosling-convention `2·|Q_Hk| / |Q_DC|`
  - The bottom-percentile filter on the View tab now gates by
    canonical `Q_H1_snr_db` (dB) by default; falls back to the
    legacy `Var(fit)/Var(resid)` ratio on older graphs
  - New percentile filter on viscous dissipation `Φ` with a tooltip
    showing the network-wide distribution
  - New `Show edge value labels (single tile)` toggle in the Display
    group
  - Removed deprecated `Resolution (Z = amp/SE)` and
    `Var(fit)/Var(resid) ratio (legacy)` Properties — redundant with
    the new dB SNRs
- **`SCHEMA.md`** — new top-level reference for the canonical schema.

### Release assets

The canonical graphs ship as small standalone ZIPs (one per stage):

| Asset | Stage | Size |
|---|---|---|
| `Somites15_canonical_graph_v0.3.0.zip` | HH-15 (pre-perfusion) | ~48 MB |
| `Somites21_canonical_graph_v0.3.0.zip` | HH-21 | ~88 MB |
| `Somites27_canonical_graph_v0.3.0.zip` | HH-27 | ~94 MB |

Drop the canonical graph into your existing v0.2.0 bundle at
`emb1/analyzed/mosaic_graph_canonical.gpickle`.  The full v0.2.0
bundle ZIPs (with raw videos + stitched TIFFs) remain valid — only
the graph file changes.

### Migration for existing users

1. Download the canonical graph for your stage from the v0.3.0 release.
2. Place it next to your existing analyzed graph in `emb1/analyzed/`.
3. Launch the viewer:
   - With a `config.json` that points at `mosaic_graph_canonical.gpickle`:
     `python -m pertile.viewer.mosaic_readonly_app --config emb1/config.json`
   - Or pass the canonical graph as the first positional argument.
4. Update any custom analysis code to read canonical fields per
   `SCHEMA.md` — typically a 3-line change (read `Q_H1_amp` instead of
   `amp_Q` or `amp_Q_h1_piv`).

### Backwards compatibility

- The viewer falls back to legacy `*_piv` fields automatically when
  the canonical fields are absent.  Old graphs render correctly.
- The canonical builder leaves the original gpickle untouched —
  it writes a new file alongside.

### Methodology validation

The schema migration was tested end-to-end by re-running both the
production v3 Bayesian inference and the intern's joint-LM frequentist
inference on the canonical stage-21 graph for tile 4.  v3's D̂ is
unchanged at 6.81×10⁻³ /Pa (already used the honest field via
`m['harmonics'][0]`).  The intern's pipeline moved from 1.52×10⁻³ →
2.26×10⁻³, into the production-sweep IQR neighborhood and closer to
the network-wide median (3.98×10⁻³).  Remaining ~3× spread between
the two pipelines reflects methodology architecture (carve / fit
framework / harmonics included / BC treatment), not data.

## v0.2.0

- Adds Somites15 stage (pre-perfusion).
- SNR fixes.

## v0.1.0

- Initial release: Somites21 and Somites27 bundles.
