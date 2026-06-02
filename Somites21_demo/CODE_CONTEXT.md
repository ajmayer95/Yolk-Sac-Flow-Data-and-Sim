# Code context for AI agents and humans

This bundle ships data for the **PerTileFlow** read-only viewer.  If you
or an AI assistant (Codex, Claude Code, Cursor agent, etc.) are going to
work with the underlying source, this file gives the orientation that
saves you the first half hour of grepping.

For launch instructions and bundle contents, see `LAUNCH.txt`.  This
file is about the **code**, not the bundle.

## Runtime dependencies

```
pip install napari[pyqt5] numpy scipy matplotlib tifffile \
            networkx scikit-image opencv-python qtpy
pip install -e <path-to-PerTileFlow>       # the package itself
```

Optional (only needed for the Bayesian-MCMC inference path):
`pip install numpyro arviz`.  Everything else (pickle, json, pathlib,
argparse, typing) is stdlib.

---

## What this codebase does

Quantifies blood flow in embryonic yolk-sac vasculature from **250 fps
fluorescence videos** of fluorescent microspheres.  Each video covers
one tile of the embryo (roughly 700×640 px at 1.7 µm/px).  Tiles are
stitched into a mosaic, vessels are segmented, the network is extracted
as a graph, and per-edge flow rates Q(t) are recovered via optical-flow
or kymograph analysis.  Downstream the network is treated as a 1D
transmission-line vascular circuit driven by measured boundary
conditions at arterial inflows (A) and venous outflows (V).

There are two viewers:

- **`pertile.viewer.mosaic_app`** — the editing viewer (writes back to
  the gpickle).  Not used by this bundle.
- **`pertile.viewer.mosaic_readonly_app`** — what this bundle launches.
  Self-contained, never mutates the gpickle on disk.

---

## Repo layout

```
PerTileFlow/
├── pertile/                       # main package
│   ├── analysis/                  # heavy lifting (per-edge + network)
│   │   ├── harmonic.py            # fit_harmonics() — DC + H1 + H2 + H3, Huber loss
│   │   ├── of_local.py            # cheap per-edge Farneback OF → Q(t)
│   │   ├── of_profile.py          # the "unified" OF used by the viewer's
│   │   │                          # Run-OF button: refined centerline,
│   │   │                          # flow-derived tangent, radial profile,
│   │   │                          # Poiseuille fit, mean velocity field
│   │   ├── flow.py                # get_chain_coords() and related geometry
│   │   ├── transmission_line.py   # 1D vascular solver, RPSI/eta, kappa_L
│   │   ├── kirchhoff.py           # network DC-conservation solve
│   │   ├── bayesian_mcmc.py       # NumPyro/PyMC global inference
│   │   ├── flow_consistency.py    # signed flow direction across the network
│   │   └── config.py              # PX_SIZE_UM, FRAME_DT_S, FMIN_HZ, …
│   ├── cli/
│   │   └── batch_analyze_videos.py  # offline pipeline entry point (~10K lines)
│   ├── filter/                    # bandpass + spatial cleanup
│   ├── io/
│   │   ├── tiff.py                # load_tiff_stack(), cut_before_fade()
│   │   └── graph.py               # gpickle I/O
│   ├── segmentation/              # vessel skeleton from probability maps
│   ├── stitch/                    # tile → mosaic affine stitching
│   ├── vectorize/                 # mask → graph
│   └── viewer/
│       ├── app.py                 # per-tile viewer
│       ├── mosaic_app.py          # editing mosaic viewer
│       ├── mosaic_readonly_app.py # ← read-only viewer (what we ship)
│       └── mosaic/                # mixin subpackage (shared editing logic)
├── scripts/                       # one-off analysis (inspect_tile.py,
│                                  # production_fit.py, …)
├── notebooks/                     # tutorials, calibration sweeps
├── configs/default.json           # calibration, bandpass, quality thresholds
├── memory/                        # methodology notes (read these — they
│                                  # encode why the code looks the way it does)
├── docs/                          # deeper docs
├── WORKFLOW.md                    # full 7-stage pipeline narrative
├── CODEBASE_SUMMARY.md            # module overview
└── CLAUDE.md                      # AI-agent session memory + conventions
```

---

## The read-only viewer at a glance

Single ~5500-line file: `pertile/viewer/mosaic_readonly_app.py`.

**Class:** `MosaicReadonlyApp`.  Long but logically ordered:

1. Module-level constants + helpers (lines ~1–500): `PROPERTY_DEFS`,
   `HARMONIC_CACHE_VERSION`, `_harmonic_class`, `_harmonic_snrs`,
   `_harmonic_fit_full`, `_combo_valid`, `_best_measurement`, …
2. `__init__` and state defaults (~500–650).
3. **Field resolver** (~650–900): `_resolve_field_value(u, v)` and
   `_resolve_tile_filtered_value(...)`.  This is the central lookup
   the colormap calls per edge per refresh.
4. Graph loading + per-tile cache plumbing (~900–1200).
5. UI build (`_setup_panel`, tab builders) (~1200–2700).
6. `_refresh_edges` / `_refresh_nodes` / `_refresh_cbar` (~1700–2000).
7. Simulation path (`_run_simulation`, BC building) (~3000–3500).
8. Inference adapter region (~2300–2700 and its render path further down).
9. Optical-flow click handler + 4-panel diagnostic (~4400–4900).
10. Event handlers, helper callbacks, `def main()` argparse (~5500+).

If you're an AI agent looking for a specific thing, grep first; the
file is well-commented and section headers use `# ──` lines.

---

## Data conventions baked into the code

| Quantity | Sign / convention |
| --- | --- |
| Q at sources | stored as `+Q_inflow` in `bc_harmonics` |
| Q at sinks | stored as `+Q_outflow` in `bc_harmonics` |
| Solver Q | "Q into network" — solver applies `−1` to sink BCs internally |
| Flux direction | `flow_from` → `flow_to` per edge, signed |
| Distensibility `D` | **areal** `ΔA/A = D·ΔP` (since 2026-05-18; older sweeps may be in radius convention, factor of 2) |
| Compliance per length `c` | `c = π R² D` (areal) |
| Pixel size | 1.7 µm/px (10x objective, NA 0.3) |
| Frame interval | 4 ms (250 fps) |
| Bandpass | 0.75 – 3.5 Hz (cardiac fundamental) |
| Viscosity | 3.5 mPa·s |

These are mostly defined in `pertile/analysis/config.py`.  When in
doubt about a unit, read that file.

---

## Harmonic content per edge

The viewer caches per-edge harmonic-fit results during the precompute
pass (`_precompute_harmonic_classes`), keyed under the schema version
`HARMONIC_CACHE_VERSION = 2`:

| Key | Meaning |
| --- | --- |
| `_h_amp_DC`, `_h_amp_H{k}` | raw fit amplitudes |
| `_h_phase_H{k}` | phase in radians |
| `_h_Z_DC`, `_h_Z_H{k}` | detection significance Z = Â / SE.  SE uses σ from **MAD** (robust). |
| `_h_sigma` | σ_MAD of residuals |
| `_h_total_snr` | `Var(fit) / Var(resid)` |
| `_h_r2` | R² of the harmonic fit |
| `_h_cache_ver` | sentinel for cache invalidation |
| `harmonic_class` | int 0–3 — highest contiguous AC harmonic with Z ≥ 3 |

Z is the raw `Â / SE` so it's Rayleigh-distributed under the null:
`P(Z > z) = exp(−z²/2)`.  Threshold tiers used in the UI:
≥ 4 excellent, ≥ 3 good, ≥ 2 marginal, < 2 unresolved.

> **Known limit:** `harmonic_class` silently assumes DC always passes,
> which is wrong for bidirectional vessels (`Z_DC ≈ 0` but
> `Z_H1` large).  See the comments in `_harmonic_class` for the
> deferred refactor.

---

## The 4-selector colormap model

Colour is built from `Source × Quantity × Property × Harmonic`.

- **Source**: `measured` or `sim`
- **Quantity**: `Q` or `P` (P is sim-only)
- **Property**: see `PROPERTY_DEFS` constant.  Each entry has a
  validity-matrix of which (Source, Quantity) tuples it accepts.
- **Harmonic**: `DC`, `H1`, `H2`, `H3` — only meaningful for
  harmonic-keyed properties.

`_combo_valid(...)` enforces the matrix; greyed-out combos in the UI
correspond to `_combo_valid → False`.  To add a new field:

1. Append a tuple to `PROPERTY_DEFS`.
2. Add a branch in `_resolve_field_value` returning the per-edge value.
3. If the field should support tile-filtering, add a branch in
   `_resolve_tile_filtered_value` too.

---

## Tile filter and per-tile resolver

When the user picks a single tile from the View-tab "Tile filter"
combo, `_resolve_field_value` short-circuits into
`_resolve_tile_filtered_value`, which reads from
`self._tile_harmonic_cache` (populated by
`_populate_tile_harmonic_cache(tile_id)`).  This cache stores the
per-tile harmonic-fit dict per edge — *not* the cached `_h_*` fields,
which are best-of-edge.

Edges with no measurement on the filtered tile render as the
"no data" grey (`NO_DATA_COLOR`).

---

## Quick recipes

**Add a CLI flag**: edit `def main()` near the bottom of the file; the
argparse block is straightforward.

**Run inference programmatically**: see `scripts/production_fit.py`.

**Re-build the gpickle from raw videos**: `pertile.cli.batch_analyze_videos`
— heavy, references `Mosaic/Graphs/mosaic_graph.gpickle` as input
and writes `mosaic_graph_analyzed.gpickle` with `measurements_piv`
populated per edge.

**Find where a per-edge attribute is written**: grep for the literal
attr name in `pertile/cli/batch_analyze_videos.py` and
`pertile/analysis/`.  Viewer code reads, doesn't write.

---

## Where to go for depth

| File | Contents |
| --- | --- |
| `WORKFLOW.md` | 7-stage pipeline narrative |
| `CODEBASE_SUMMARY.md` | Module overview + recent development focus |
| `CLAUDE.md` | Long-running AI-agent context, includes calibration notes |
| `memory/methods.md` | Current batch analysis + analyze-click methods |
| `memory/transmission_line_physics.md` | Corrected solver physics |
| `memory/local_pressure_inference_state.md` | Per-tile inference state |
| `docs/tile_stitching.md` | Stitching internals |
| `docs/mosaic_flow_simulation.md` | Simulation docs |
| `notebooks/*.py` | Tutorial scripts |

The `memory/` directory in particular is where the *reasoning behind
the code* lives — the WHY behind sign conventions, threshold choices,
convention switches.  Read those before making non-trivial changes.

---

## A note for AI agents

This codebase has gone through several methodology refactors recorded
in `CLAUDE.md` and `memory/`.  Before adding a new metric or changing
a threshold, **search those files for prior decisions** — there's
often a deliberately deferred refactor (e.g. the `harmonic_class`
DC-coupling, the continuous-Z field design) that affects how new code
should be structured.  Don't reinvent decisions that were made and
parked for a reason.

The read-only viewer's cache schema (`HARMONIC_CACHE_VERSION`) is the
canonical place to bump if you change what's stored per edge.  The
precompute step will recompute on next launch and persist back to the
gpickle atomically.
