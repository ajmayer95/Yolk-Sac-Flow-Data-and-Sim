# Agent migration prompt — v0.2.0 → v0.3.0 (canonical graph schema)

This file is a self-contained prompt you can paste into a coding agent
(Codex, Claude Code, Cursor, etc.) to apply the v0.3.0 schema
migration to a PerTileFlow install.  Drop everything below into the
agent's chat as-is.

---

## Prompt (paste below into your coding agent)

**Task:** Migrate this `PerTileFlow` install from v0.2.0 to v0.3.0.
v0.3.0 introduces a canonical mosaic-graph schema with refit-verified
flow fields; the legacy schema carried multiple differently-named "H1
phasor" fields with conflicting values.  Detailed reference:
https://github.com/ajmayer95/Yolk-Sac-Flow-Data-and-Sim/blob/main/SCHEMA.md

The migration has three parts.  Do all three; do not skip any.

### Part 1 — Drop in the updated code

Download `pertile_code_v0.3.0.zip` from:
https://github.com/ajmayer95/Yolk-Sac-Flow-Data-and-Sim/releases/tag/v0.3.0

Unzip it directly over the `PerTileFlow/` folder.  Three files get
replaced and one new file added:

- `pertile/analysis/harmonic.py` — `fit_harmonics()` now returns
  `snr_h1_db`, `snr_h2_db`, `snr_h3_db`, `snr_dc_fit_db` in addition
  to the existing fields.  Backwards-compatible (additive only).
- `pertile/viewer/mosaic_readonly_app.py` — viewer with canonical
  field readers, new Properties (SNR (dB) per-harmonic, SNR_AC,
  SNR_total, per-harmonic PI), dissipation filter, edge labels.
- `scripts/build_canonical_graph.py` — NEW migration script.

Concretely:

```bash
cd <path-to-your-bundle>/PerTileFlow
curl -L -o /tmp/pertile_code_v0.3.0.zip \
  https://github.com/ajmayer95/Yolk-Sac-Flow-Data-and-Sim/releases/download/v0.3.0/pertile_code_v0.3.0.zip
unzip -o /tmp/pertile_code_v0.3.0.zip
```

If `pertile` was installed with `pip install ./PerTileFlow` (not -e),
reinstall to pick up the source changes:

```bash
pip install --force-reinstall --no-deps ./PerTileFlow
```

### Part 2 — Drop in the canonical graph

Download the canonical graph for whichever stage this bundle is for:

| Stage | Asset |
|---|---|
| Somites15 | `Somites15_canonical_graph_v0.3.0.zip` |
| Somites21 | `Somites21_canonical_graph_v0.3.0.zip` |
| Somites27 | `Somites27_canonical_graph_v0.3.0.zip` |

From the same release URL.  Unzip and place
`mosaic_graph_canonical.gpickle` next to the existing graph at
`emb1/analyzed/`.  Then update `emb1/config.json` so `mosaic_graph`
points at the canonical file:

```json
"mosaic_graph": "analyzed/mosaic_graph_canonical.gpickle"
```

If the bundle uses absolute paths instead of `config.json`, update
those at the launch site instead.

### Part 3 — Patch any custom reader scripts in `scripts/`

If `scripts/` contains custom analysis or inference scripts (e.g.
`inspect_tile.py`, `synthetic_validation_neumann_bc.py`,
`production_fit.py`, `h2_sensitivity_check.py`,
`tile_mosaic_simulation.py`, `infer_bayes_*.py`), they may read
legacy field names.  Update them to use a **canonical → legacy
fallback chain** so they work on both old and new graphs.

**Specific replacement pattern.**  Anywhere in `scripts/` you see
code reading the H1 phasor or DC mean flow from an edge:

```python
# BEFORE (v0.2.0 reader)
mq = ed.get("mean_Q") or ed.get("mean_Q_nL_s")
amp = ed.get("amp_Q"); phase = ed.get("phase")
```

Replace with:

```python
# AFTER (v0.3.0 canonical → legacy fallback chain)
# DC: canonical Q_DC → legacy mean_Q_piv → kymo mean_Q
mq = (ed.get("Q_DC") or ed.get("mean_Q_piv")
      or ed.get("mean_Q") or ed.get("mean_Q_nL_s"))
# H1 phasor: canonical Q_H1_{amp,phi} → legacy amp_Q_h1_piv → kymo amp_Q
amp = ed.get("Q_H1_amp")
phase = ed.get("Q_H1_phi")
if amp is None or phase is None:
    amp = ed.get("amp_Q_h1_piv")
    phase = ed.get("phase_h1_piv")
if amp is None or phase is None:
    amp = ed.get("amp_Q"); phase = ed.get("phase")
```

For higher harmonics (H2, H3):

```python
# H2 amplitude / phase
amp_h2 = ed.get("Q_H2_amp") or ed.get("amp_Q_h2_piv")
phi_h2 = ed.get("Q_H2_phi") or ed.get("phase_h2_piv")
# Same pattern for H3 with Q_H3_amp / Q_H3_phi.
```

For per-tile measurements in `measurements_piv`, the same flat keys
work directly:

```python
# Per-tile measurement entry: use the same flat keys as top-level
m = next(m for m in ed['measurements_piv'] if m['tile_id'] == tile_id)
Q_DC_tile  = m['Q_DC']
Q_H1_tile  = m['Q_H1_amp'] * np.exp(1j * m['Q_H1_phi'])
# Detailed fit (sigmas, A/B coefficients) — only in canonical:
sigma_amp1 = m['harmonics'][0]['sigma_amp']
```

If a script also has a `GRAPH_PATH` constant pointing at
`mosaic_graph_analyzed.gpickle`, update it to
`mosaic_graph_canonical.gpickle` (or accept it as a CLI argument).

### Part 4 — Verify

After applying all three parts, run a smoke test.  From inside
the bundle:

```bash
python scripts/inspect_tile.py 4 --no-show
```

(or whichever tile / script makes sense for the bundle.)

Expected behaviour on the **stage-21** canonical graph for tile 4:
the script should produce `D̂ ≈ 2.26×10⁻³ /Pa` with `σ/D̂ ≈ 30%`
and `χ²_red ≈ 0.52`.  If it produces `0` valid H1 readings or a
`D̂ → grid floor`, the canonical field reads didn't take — re-check
Part 1 and Part 3.

You can also launch the viewer:

```bash
python -m pertile.viewer.mosaic_readonly_app \
    emb1/analyzed/mosaic_graph_canonical.gpickle \
    --tiff emb1/analyzed/stitched_linear.tif \
    --tile-positions emb1/analyzed/tile_positions_manual.json \
    --video-dir emb1/videos \
    --video-pattern "10x 250fps loc{vid}_C001H001S0001.tif"
```

In the View tab, the Property dropdown should include:

- `SNR (dB)` (per-harmonic, reads canonical `Q_Hk_snr_db`)
- `SNR_AC (dB)` (aggregate H1+H2+H3)
- `SNR_total (dB)` (aggregate DC+AC)
- `Pulsatility index` (now per-harmonic via the Harmonic dropdown)

In the Display group at the bottom, there should be a
`Show edge value labels (single tile)` checkbox.

### Common pitfalls

- **Viewer renders all grey** after migration: the reader is
  finding none of canonical / legacy / kymo fields.  Check that
  the canonical graph file actually has `Q_H1_amp` populated
  on a sample edge — open in Python and inspect.
- **`KeyError: 'PI'`** in some downstream code: legacy `m['PI']`
  is dropped on canonical.  Use the fallback
  `2 * m['Q_H1_amp'] / abs(m['Q_DC'])` (Gosling convention).
- **`pertile.analysis.harmonic` import error**: the drop-in only
  changed `harmonic.py`; if other modules in `pertile/analysis/`
  import from there, they should still work (signature unchanged,
  fields added only).
- **`amp_Q_h1_piv` still being read by some script**: that's
  the misleading legacy field (133% amp / 142° phase off from a
  direct refit).  ALWAYS prefer `Q_H1_amp` first in the
  fallback chain.

### Report back

When done, summarise:

1. Which files in `scripts/` were patched (paths + line numbers
   of the reads you changed).
2. Whether the smoke test produced the expected output (or what
   it produced if not).
3. Any script you couldn't auto-patch because the reader pattern
   didn't match the example above — paste the unchanged code and
   I'll review.

---

End of prompt.
