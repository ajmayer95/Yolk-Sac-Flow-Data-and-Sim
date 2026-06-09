# Mosaic graph schema (canonical, v0.3.0)

This document describes the canonical edge-attribute schema used by the
analyzed mosaic graphs shipped in the v0.3.0+ demo bundles.  Prior
bundles (v0.1.0, v0.2.0) used a legacy schema with many redundant
fields; the canonical schema replaces them with one unambiguous set.

The migration script is `scripts/build_canonical_graph.py` (run on a
legacy `mosaic_graph_analyzed.gpickle` to produce a canonical
`mosaic_graph_canonical.gpickle`).

## Why canonicalize?

Direct refit verification (see `notebooks/honest_h1_field_check.ipynb`)
showed that the legacy graph carried three different "H1 phasor" fields
with conflicting values:

| Legacy field | median \|Δamp\|/refit | median \|Δφ\| |
|---|---|---|
| `m['harmonics'][0]` (per-tile measurement list) | **0.7%** | **1.0°** |
| `amp_Q_h1_piv` / `phase_h1_piv` (top-level edge attr) | 133.6% | 141.9° |
| `amp_Q` / `phase` (kymograph-era edge attr) | varies | varies |

Only `m['harmonics'][0]` matches a clean from-scratch fit of `Q_t` at
the unified per-tile f₀ to within 1% / 1°.  The other fields encode
different (and in some cases unidentified) computations.  Code that
reads the wrong field gets the wrong answer.  The canonical schema
exposes only the honest field, under a single unambiguous name.

## Top-level edge attributes (canonical)

For each edge with at least one PIV measurement, the following
top-level attributes are populated by `build_canonical_graph.py`:

| Field | Units | Source |
|---|---|---|
| `radius`, `length` | px | unchanged |
| `flow_from`, `flow_to` | node ids | canonical edge direction |
| `Q_t_piv` | nL/s, shape (T,) | raw flow time series, best-tile measurement |
| `f0_hz` | Hz | per-tile consensus f₀ for the canonical tile |
| `tile_canonical` | int | which tile's measurement was promoted to top level |
| `Q_DC` | nL/s | mean flow (best-tile measurement) |
| `Q_DC_snr_db` | dB | DC SNR — `10·log10(a₀² / MS_BL)` |
| `Q_H1_amp`, `Q_H1_phi` | nL/s, rad | H1 phasor (unified-f₀ refit, honest field) |
| `Q_H1_snr_db` | dB | H1 SNR — `10·log10(P_H1 / MS_BL)` |
| `Q_H2_amp`, `Q_H2_phi`, `Q_H2_snr_db` | … | H2 phasor + SNR |
| `Q_H3_amp`, `Q_H3_phi`, `Q_H3_snr_db` | … | H3 phasor + SNR |
| `snr_ac_fit_db` | dB | aggregate AC SNR — `10·log10(Σ P_Hk / MS_BL)` |
| `snr_harm_fit_db` | dB | aggregate DC+AC SNR — `10·log10((a₀² + Σ P_Hk) / MS_BL)` |

`MS_BL` is the mean-squared residual band-limited to `f ≤ 3.5·f₀`.

## Per-tile measurements (`measurements_piv` list)

Each entry corresponds to a tile that measured this edge.  Uses the
**same flat key names** as the top-level attrs so reader code stays
consistent regardless of which level it's reading from:

```python
{
  tile_id,            # int
  f0_hz,              # Hz — this tile's consensus fundamental
  Q_t,                # nL/s, raw time series for this tile
  Q_DC, Q_DC_snr_db,
  Q_H1_amp, Q_H1_phi, Q_H1_snr_db,
  Q_H2_amp, Q_H2_phi, Q_H2_snr_db,
  Q_H3_amp, Q_H3_phi, Q_H3_snr_db,
  snr_ac_fit_db, snr_harm_fit_db,
  harmonics,          # list of K=3 detailed fit dicts (k, A, B, amp, phi,
                      # sigma_A, sigma_B, sigma_amp, sigma_phi, P_sig, snr_db)
}
```

The `harmonics` list keeps the detailed fit (with cosine/sine
coefficients and per-coefficient sigmas) for downstream code that
needs them.  For most use cases the flat keys are sufficient.

## Canonical-tile promotion rule

```python
m_best = max(measurements_piv, key=lambda m: m['Q_H1_snr_db'])
```

Top-level attrs are copied from `m_best`.  H1 SNR is the criterion
because cardiac-fundamental phase carries the dominant identification
signal for distensibility-aware inference.

## Fields dropped from the canonical schema

`build_canonical_graph.py` removes ~50 legacy fields in aggressive
mode.  Pass `--conservative` to keep all but the proven-misleading
four.  The proven-misleading ones (133% amp / 142° phase off from
direct refit, verified empirically):

- `amp_Q`, `phase` (kymograph-era)
- `amp_Q_h1_piv`, `phase_h1_piv` (mislabeled top-level promotion)
- `amp_Q_h2_piv`, `phase_h2_piv`, `amp_Q_h3_piv`, `phase_h3_piv` (same convention)

Other dropped fields are legacy variants (`*_kymo`, `*_linear`,
`*_murray`, `*_sim`, `*_gz`) and statistics trivially derivable from
the canonical fields (`PI_piv`, `pulse_frac_piv`, `RPSI_piv`,
`H{1,2,3}_frac_piv`, etc.).

## Reading from the canonical schema (code snippets)

```python
# Top-level edge attr (canonical tile)
import pickle
G = pickle.load(open('mosaic_graph_canonical.gpickle', 'rb'))
ed = G.edges[u, v]
Q_H1 = ed['Q_H1_amp'] * np.exp(1j * ed['Q_H1_phi'])  # H1 phasor

# Per-tile measurement (same flat keys)
m = next(m for m in ed['measurements_piv'] if m['tile_id'] == 22)
Q_H1_tile22 = m['Q_H1_amp'] * np.exp(1j * m['Q_H1_phi'])

# Detailed fit if needed (sigmas, coefficients)
h1 = m['harmonics'][0]
sigma_amp = h1['sigma_amp']; sigma_phi = h1['sigma_phi']
```

## Verifying a graph is canonical

The presence of `Q_H1_amp` on every measured edge is sufficient:

```python
import pickle
G = pickle.load(open('<graph>.gpickle', 'rb'))
n_canonical = sum(1 for _, _, d in G.edges(data=True) if 'Q_H1_amp' in d)
print(f"{n_canonical}/{G.number_of_edges()} edges have canonical Q_H1_amp")
```

If `n_canonical` ≥ measured-edge count, the graph is canonical.  If
it's zero, the graph is legacy — run `build_canonical_graph.py` first.
