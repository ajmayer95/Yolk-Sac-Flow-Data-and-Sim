# Yolk-sac flow demo datasets

Small, self-contained demonstration bundles of embryonic chick yolk-sac
vasculature with flow measurements.  Each bundle is fully portable:
**data + viewer code + one launch command**.  No second clone, no
compiler, no platform-specific build steps.

This repo holds the lightweight documentation only (<100 KB).  The
actual bundles ship as **release assets** on the
[Releases page](../../releases) — pick the stage you want.

**Latest: v0.3.0** introduces a canonical graph schema with
unambiguous, refit-verified flow fields.  See [`CHANGELOG.md`](CHANGELOG.md)
and [`SCHEMA.md`](SCHEMA.md) for the migration details.  Existing v0.2.0
bundles remain compatible — drop in the new canonical graph from the
v0.3.0 release at `emb1/analyzed/`.

If you also want the new viewer features (per-harmonic SNR Properties,
dissipation filter, edge value labels), download `pertile_code_v0.3.0.zip`
from the release and unzip it over your existing `PerTileFlow/`
folder — 94 KB drop-in, no re-download of the full bundle needed.

## Bundles

| Stage | Folder | v0.2.0 full bundle | v0.3.0 canonical-graph drop-in |
| :---: | :--- | :--- | :--- |
| HH-15 | [`Somites15_demo/`](Somites15_demo/) | `Somites15_demo.zip` (~870 MB) | `Somites15_canonical_graph_v0.3.0.zip` (~48 MB) |
| HH-21 | [`Somites21_demo/`](Somites21_demo/) | `Somites21_demo.zip` (~1.4 GB) | `Somites21_canonical_graph_v0.3.0.zip` (~88 MB) |
| HH-27 | [`Somites27_demo/`](Somites27_demo/) | `Somites27_demo.zip` (~1.7 GB) | `Somites27_canonical_graph_v0.3.0.zip` (~94 MB) |

Each subfolder in this repo contains the bundle's `README.md`,
`LAUNCH.txt`, `CODE_CONTEXT.md`, and `emb1/config.json` — the
lightweight files only.  The actual data + viewer source lives in the
zip on Releases.

## Requirements

- **Python 3.9 or newer.**  Works on macOS, Linux, and Windows.
- Either conda/mamba or Python's built-in `venv`.
- ~3 GB of free disk space per bundle (1.5 GB zip + 1.5 GB unzipped).

No C compiler, no Cython, no system libraries to install separately.

## Setup — option A (conda, recommended if you have it)

Works identically on macOS, Linux, and Windows.

```bash
# 1. Download a bundle zip from the Releases tab, unzip it.
unzip Somites21_demo.zip
cd Somites21_demo

# 2. Create + activate a fresh conda environment.
conda create -n yolk-sac python=3.11 -y
conda activate yolk-sac

# 3. Install the bundled viewer (pulls all runtime deps).
pip install ./PerTileFlow

# 4. Launch the viewer.
python -m pertile.viewer.mosaic_readonly_app --config emb1/config.json
```

`mamba` works as a drop-in replacement for `conda` if you have it
installed — same commands, faster solver.

## Setup — option B (pip + venv, no conda needed)

```bash
# 1. Unzip the bundle.
unzip Somites21_demo.zip
cd Somites21_demo

# 2. Create a virtual environment.  Same command on every OS:
python -m venv .venv

# 3. Activate it (OS-specific):
source .venv/bin/activate                # macOS / Linux (bash, zsh, fish)
.\.venv\Scripts\Activate.ps1             # Windows PowerShell
.\.venv\Scripts\activate.bat             # Windows cmd.exe

# 4. Install the bundled viewer (pulls all runtime deps).
python -m pip install --upgrade pip
python -m pip install ./PerTileFlow

# 5. Launch the viewer.
python -m pertile.viewer.mosaic_readonly_app --config emb1/config.json
```

That single `pip install ./PerTileFlow` brings in napari, PyQt5, numpy,
scipy, matplotlib, tifffile, networkx, scikit-image, opencv-python, and
qtpy automatically — they're declared in the bundle's `setup.py`.

Substitute `Somites27_demo` everywhere for the stage-27 bundle.  Each
bundle's `LAUNCH.txt` has CLI flag reference and troubleshooting notes.

## What's in each bundle zip

```
SomitesNN_demo/
├── README.md            stage-specific overview
├── LAUNCH.txt           launch instructions + CLI flags
├── CODE_CONTEXT.md      codebase orientation for humans + AI agents
├── PerTileFlow/         the viewer code (~680 KB, pip-installable)
└── emb1/
    ├── config.json      relative-path config consumed by the viewer
    ├── analyzed/        graph, stitched mosaic TIFF, tile-position JSON
    └── videos/          per-tile multi-page TIFFs (downsampled subset)
```

The `PerTileFlow/`, `analyzed/`, and `videos/` directories live only
inside the zips on Releases, not in this repo.  The bundled
`PerTileFlow` is a pure-Python slim build containing only the
read-only viewer + its runtime dependencies — none of the
mask-segmentation, vectorization, batch-analysis, or editing-viewer
code from the full PerTileFlow research codebase ships here.

## Troubleshooting

- **`zsh: no matches found: napari[pyqt5]`** — you wouldn't see this
  with the current install instructions, but if you do anywhere else,
  it's macOS zsh treating `[…]` as a glob.  Wrap in quotes
  (`"napari[pyqt5]"`) or prefix with `noglob`.
- **`ModuleNotFoundError: No module named 'PyQt5'` at viewer launch**
  — the PyQt5 wheel didn't install cleanly.  Try
  `pip install --force-reinstall pyqt5`.
- **Black napari window with no mosaic backdrop** — TIFF didn't load.
  Check that `emb1/analyzed/stitched_linear.tif` exists and is the
  ~45 MB it should be (zero-byte file means a partial unzip).
- **Linux: `platform plugin 'xcb' could not be loaded`** at startup —
  install system Qt deps: `sudo apt install libxcb-xinerama0 libxcb-cursor0`
  (Ubuntu/Debian) or the equivalent for your distro.
- **Slow first launch (~30 s)** — that's the harmonic-class precompute.
  Watch the terminal for the progress line; it caches the result back
  to the gpickle so subsequent launches load instantly.

## Limitations of the demo bundles

These are *demonstration* datasets — partial tile coverage (top 50% by
per-tile signal quality) and 2× spatially downsampled videos capped
at 600 frames each.  The bundled viewer build hides the optical-flow
re-analysis button because absolute-Q scaling is fragile under spatial
downsampling.  All other functionality — network coloring, harmonic
SNR panels, BC simulation, per-tile inference — uses the analyzed
graph directly and is unaffected by the video downsampling.

## Citation

If this dataset contributes to a publication, please cite the
PerTileFlow methodology paper (in preparation).

## License

Data: CC-BY-4.0 unless noted otherwise.
PerTileFlow viewer code (inside each bundle): see its own license file
when shipped.
