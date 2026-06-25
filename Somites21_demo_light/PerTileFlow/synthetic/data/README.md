# Synthetic distensibility datasets

This directory contains compact whole-mosaic datasets for testing recovery of
vascular distensibility. The generator is
[`../scripts/make_synthetic.py`](../scripts/make_synthetic.py), and its
experiment settings are in [`../configs/`](../configs/).

## Forward model

The anatomical topology and vessel geometry come from
`mosaic_graph_analyzed.gpickle`. The generator reuses the established
`pertile.analysis.transmission_line.solve_transmission_line` implementation
from `Somites21_demo_light/PerTileFlow`, the same simulation path used by
`scripts/default_mosaic_tile_profiles.py`.

For each vessel `e`, the areal distensibility is

```text
D_e = D_0 (R_e / R_0)^α
```

The 6,441 solver-valid vessels have a median radius of **25.8541 µm**. The
experiments use the nearby round, fixed reference **R₀ = 25 µm** so the
normalization is stable and easy to report.

The whole mosaic is solved jointly for DC, H1, and H2. Boundary forcing follows
the default viewer simulation: measured boundary harmonic shapes are averaged
within the source and sink groups, with a total DC target flux of 1 nL/s split
equally within each group. The graph median cardiac frequency is used
(approximately 2.77334 Hz).

## Noise

The truth solve is reused across noise levels. For H1 and H2, observations use
relative circular complex Gaussian noise:

```text
v_obs(e,h) = v_true(e,h) + ε

sqrt(E[|ε|²]) = η |v_true(e,h)|
```

where `η` is the configured noise level. Independent real and imaginary
components each have standard deviation

```text
η |v_true(e,h)| / √2
```

DC remains noise-free in the base configuration. A fixed seed gives paired
noise realizations across parameter conditions and noise levels.

## Files and names

Generated datasets live directly in `data/synthetic/` as compressed NumPy
archives. Names encode all varying parameters:

```text
pl_d1e-03_a1_n10_s42.npz
│  │       │  │   └── random seed
│  │       │  └────── noise percent
│  │       └───────── alpha
│  └───────────────── D0
└──────────────────── power-law wall model
```

`manifest.csv` is the index of all generated files.

Each `.npz` is self-describing through `schema_version` and `metadata_json` and
contains:

* graph topology: node IDs, edge endpoint IDs and indices;
* geometry: radius, length, area, node coordinates, and tile membership;
* ground truth: per-edge distensibility, node pressure harmonics, edge flow
  harmonics, and edge velocity harmonics;
* observations: noisy edge velocity harmonics, their known noise scale, and a
  validity mask;
* boundary forcing: boundary node IDs, source/sink labels, and DC/H1/H2 flows;
* fixed edge split codes: `0=train`, `1=validation`, `2=test`.

Arrays use SI units. Harmonic arrays have shape `(item, 3)` with columns
`[DC, H1, H2]`; harmonic values are stored as complex numbers, including DC
for a uniform interface. Signs follow
`edge_source_node_id -> edge_target_node_id`.

The large source graph is referenced by path and SHA-256 digest in metadata,
not copied into every archive.

## Matching real data to this schema

Analysis code should consume this observation schema rather than branch on a
NetworkX graph versus a synthetic file. A future real-data preprocessing step
can write the same topology, geometry, tile membership, boundary, split,
harmonic, and observation arrays:

* populate `velocity_observed_m_s` from PIV DC/H1/H2 estimates;
* populate `observation_valid` from measurement availability and quality
  filtering;
* populate `velocity_noise_sigma_m_s` from PIV uncertainty or SNR;
* use `data_kind: real` in `metadata_json`;
* omit unavailable truth arrays or fill them with `NaN`;
* preserve the same edge orientation and SI units.

With that adapter, deterministic solvers, Bayesian solvers, GNNs, and metrics
can load synthetic and real mosaics through one interface. Only truth-dependent
metrics should be disabled for real data.

## Reproduction

From the `synthetic/` project root, using the existing project environment:

```bash
/mnt/home/sswee/miniforge3/envs/yolk-sac/bin/python scripts/make_synthetic.py
```

Pass `--overwrite` to regenerate existing archives. The generator first checks
`data/processed/mosaic_graph.gpickle`; until that graph is staged, it falls
back to the known source at
`/mnt/home/sswee/ceph/Somites21_demo/emb1/analyzed/mosaic_graph_analyzed.gpickle`.
