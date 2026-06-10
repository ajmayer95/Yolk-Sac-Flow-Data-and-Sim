pertile_code_v0.3.0 — drop-in code update for v0.2.0 demo bundles

This archive contains only the source files that changed between
v0.2.0 and v0.3.0 of the PerTileFlow demo bundles.  Use it if you
already have a v0.2.0 bundle unpacked and want to upgrade the viewer
without re-downloading the full ~1.5 GB bundle ZIP.

WHAT TO DO
==========

1. Locate your existing v0.2.0 bundle folder.  Example:

       /path/to/Somites21_demo/PerTileFlow/

2. Copy the files in this archive INTO the bundle, preserving paths:

       cp -r pertile/* /path/to/Somites21_demo/PerTileFlow/pertile/
       cp scripts/build_canonical_graph.py /path/to/Somites21_demo/PerTileFlow/scripts/

   (Replace `Somites21_demo` with `Somites15_demo` or `Somites27_demo`
   as appropriate.)

3. Download the canonical graph for your stage from the v0.3.0 release:

       Somites{15,21,27}_canonical_graph_v0.3.0.zip

   Unzip and drop the `.gpickle` into your bundle's `emb1/analyzed/`:

       /path/to/Somites21_demo/emb1/analyzed/mosaic_graph_canonical.gpickle

4. Edit emb1/config.json to point at the canonical graph:

       "mosaic_graph": "analyzed/mosaic_graph_canonical.gpickle"

   (Or pass the path directly to the viewer with the first positional
   argument.)

5. Re-launch the viewer.  No re-install required if you used
   `pip install -e ./PerTileFlow` originally; if you used
   `pip install ./PerTileFlow`, reinstall to pick up the source
   changes:

       pip install --force-reinstall --no-deps ./PerTileFlow

WHAT'S NEW IN v0.3.0
====================

- Canonical mosaic-graph schema (see SCHEMA.md in the repo root).
- Per-harmonic SNR computation in pertile.analysis.harmonic.
- New viewer Properties: SNR (dB), SNR_AC (dB), SNR_total (dB),
  per-harmonic Pulsatility index.
- New viewer features: dissipation percentile filter, edge value
  labels (single-tile mode).
- Canonical-field readers in the viewer with legacy *_piv fallback,
  so both old and new graphs render correctly.

The viewer falls back to legacy fields when canonical attrs are
absent.  You can use the v0.3.0 viewer with an old v0.2.0 graph if
you don't want to migrate everything at once — you just won't see
the new per-harmonic Properties populated.

DOCS
====

- SCHEMA.md     — canonical schema reference
- CHANGELOG.md  — v0.3.0 release notes

Both live at: https://github.com/ajmayer95/Yolk-Sac-Flow-Data-and-Sim
