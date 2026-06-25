# Source modules and analysis workflow

This directory contains the reusable scientific and data-processing code for
the synthetic distensibility project. Command-line scripts in `../scripts/`
should be small wrappers around functions defined here.

## Analysis workflow

The intended execution order is below. Items marked **implemented** can be run
now; the remaining stages describe the interfaces to preserve as the project
grows.

### 1. Generate synthetic data — implemented

Run:

```bash
python scripts/make_synthetic.py
```

Configuration:

- `configs/synthetic_base.yaml`
- `configs/experiment_grid.yaml`

Implementation:

- `src/distensibility/simulation.py`

Outputs:

- `data/synthetic/pl_*.npz`
- `data/synthetic/manifest.csv`

This stage loads the whole vascular mosaic, evaluates
\(D_e=D_0(R_e/R_0)^\alpha\), runs DC/H1/H2 transmission-line simulations,
adds velocity noise, and stores graph-aligned truth and observations. The
reference radius is \(R_0=25\ \mu\mathrm{m}\).

### 2. Load and validate datasets — implemented

`src/distensibility/io.py` loads the `.npz` schema documented in
`data/README.md`. It exposes a common `VascularDataset` representation for:

- synthetic observations, including ground truth;
- processed real observations, where unavailable truth fields are absent or
  `NaN`; and
- common topology, geometry, harmonic, boundary, mask, and split arrays.

All implemented classical solvers use this loader rather than independently
interpreting archive field names.

### 3. Run classical inverse methods — implemented

Use `configs/solver_base.yaml` to run:

- tile-wise deterministic linear inference;
- whole-mosaic deterministic linear inference;
- tile-wise Bayesian inference; and
- whole-mosaic Bayesian inference.

These methods recover \(D_0\) and, when configured, \(\alpha\), using H1 or
H1+H2 velocity observations. The deterministic methods profile complex
boundary pressures by weighted least squares. The Bayesian methods marginalize
complex boundary pressures analytically under a Gaussian prior.

Run:

```bash
python scripts/run_solver.py DATASET --method METHOD [options]
```

Run artifacts are organized by method and input dataset under `outputs/runs/`,
`outputs/metrics/`, and `outputs/figures/`.

Tile-specific methods run every valid tile by default, matching the established
PerTileFlow tile-profile workflows. Use `--tiles 22 26 38` only when an
explicit subset is wanted for debugging or focused analysis.

The optional CUDA tile engine lives in `models/gpu_tile.py`. It batches
parameter points and solves tile nodal systems with complex
`torch.linalg.solve`, while reusing the same output orchestration and
best-point reconstruction. Entry points are:

- `scripts/run_linear_solver_gpu.py`
- `scripts/run_bayesian_solver_gpu.py`
- `scripts/run_all_tile_solvers_gpu.sbatch`
- `scripts/run_gnn_conditioned_tile_solvers_gpu.sbatch`

The CUDA tile engine supports unconditioned inference and DC neural-pressure
conditioning in `scaled`, `absolute`, or `off` mode. The conditioned GPU array
selects the best physics-informed GNN and K=0 edge-local MLP pressure profiles
using validation DC relative RMSE. Whole-mosaic methods remain on the sparse
SciPy CPU engine, and fixed neural H1/H2 pressure fields are not yet supported
by the GPU path.

Classical dashboards include full and \(y\leq10\) parameter-profile views.
For linear inference the 95% one-parameter profile threshold is
\(\Delta\chi^2=3.8415\), while the joint two-parameter 95% threshold is
\(\Delta\chi^2=5.9915\). The dashboard draws the latter as a contour on the
\(D_0\)-and-\(\alpha\) surface and shows both values on the profiles. Bayesian
95% intervals are computed from posterior quantiles and marked separately from
these likelihood-ratio-style references.

### 4. Train learned models — implemented

Use `configs/gnn_base.yaml` to train:

- the physics-informed GNN;
- the vanilla GCN baseline;
- the edge-local MLP baseline; and
- any simple feature-only baselines.

Models must use the same edge splits and observation arrays as the classical
solvers. Physics layers should call shared physics functions rather than carry
a second implementation of the vascular equations.

The implemented framework uses the fixed split codes stored in every
synthetic archive. The primary GNN and K=0 MLP predict edge conductance
corrections and use a differentiable graph pressure solve. The vanilla GCN is
a data-driven nodal-pressure baseline: it predicts gauge-fixed DC pressure,
then fixed Poiseuille conductance converts pressure drops into reconstructed
edge velocities. It receives no true-pressure supervision. Harmonic-informed
modes add direct H1 and H2 prediction heads as ablations while retaining the
learned DC pressure field.

Configuration:

- `configs/gnn_experiments.yaml`

Entry points:

- `scripts/train_gnn.py`
- `scripts/run_gnn_grid.py`
- `scripts/plot_best_gnn_runs.py`
- `scripts/make_gnn_108_dashboard.py`
- `scripts/make_classical_solver_comparison.py`

The plotting entry point selects the best physics-informed GNN,
pressure-decoding vanilla GCN, and K=0 edge-local MLP independently for each
of the 36 datasets using validation DC relative RMSE. Its reusable
implementation lives in `src/gnn_plotting.py` and produces model-specific
training, pressure, velocity, residual, and correction diagnostics, standalone
dashboards, and a combined comparison index. Historical direct-velocity
vanilla GCN outputs remain in their original run directories; only finite
pressure-decoder runs are eligible for the vanilla comparison.

`make_gnn_108_dashboard.py` then flattens the three selected-run manifests into
108 comparable rows and builds the interactive cross-model dashboard and CSV.

`make_classical_solver_comparison.py` performs the corresponding aggregation
for linear/Bayesian tile/mosaic inference, with and without K=0 or
physics-informed GNN pressure priors. It combines all 36 datasets into
filterable profile, parameter-recovery, coverage, reconstruction-error, and
summary-table views. It reads completed artifacts only and can therefore be
rerun as ongoing solver arrays add results.

### 5. Perform downstream parameter inference — implemented

Pass a neural `pressure_field.npz` to the common inverse runner with
`--pressure-field`. All four deterministic/Bayesian tile/mosaic methods use
the same conditioning implementation.

Current neural artifacts store DC pressure only. The recommended `scaled`
mode uses the DC field as a spatial prior for complex H1/H2 boundary pressures
while fitting a separate amplitude and phase for each harmonic. This is the
staged GNN-pressure-prior formulation used by the earlier PerTileFlow
experiments. `absolute` directly regularizes toward the saved values, and
`off` recovers the original free-pressure solver. If a future artifact stores
H1 or H2 pressure columns, those harmonics can be treated as fully known.

Example:

```bash
python scripts/run_solver.py \
  data/synthetic/pl_d1e-03_a1_n10_s42.npz \
  --method bayesian_mosaic \
  --alpha-mode solved \
  --harmonics h1_h2 \
  --pressure-field \
    outputs/runs/gnn/pl_d1e-03_a1_n10_s42/physics_informed_gnn__K2__dc_h1_h2__seed42 \
  --pressure-mode scaled
```

The full CPU pressure-conditioned campaign is launched by
`scripts/run_all_gnn_conditioned_solvers.sbatch`. The tile-only CUDA campaign,
restricted to the best physics-informed GNN and K=0 pressure profiles, is:

```bash
sbatch scripts/run_gnn_conditioned_tile_solvers_gpu.sbatch
```

### 6. Evaluate every method with common metrics — partially implemented

Compute the same metrics for all applicable methods:

- log-scale and relative \(D_0\) error;
- absolute \(\alpha\) error;
- confidence or credible interval coverage;
- profile or posterior width;
- parameter-grid boundary-hit rate;
- observed and held-out velocity reconstruction error; and
- pressure-field error when truth is available.

Real-data runs use reconstruction and diagnostic metrics but skip metrics that
require known ground truth.

The classical solver runner already writes these recovery, interval,
boundary-hit, and held-out reconstruction metrics to
`outputs/metrics/<method>/<dataset>/summary_metrics.json`. A standalone
`evaluate.py` entry point is still planned for applying the same definitions
to learned-model outputs.

### 7. Aggregate runs — planned

Combine run-level metrics across \(D_0\), \(\alpha\), noise, seed, harmonic
selection, spatial mode, and method. Save machine-readable summaries under
`outputs/metrics/` and publication-facing tables under `outputs/tables/`.

### 8. Plot and report results — planned

Create final figures under `outputs/figures/`, including:

- predicted-versus-true parameter recovery;
- posterior and profile surfaces;
- velocity and pressure reconstruction;
- uncertainty coverage;
- boundary-hit rates; and
- noise-sensitivity comparisons.

## Current source files

### `distensibility/__init__.py`

Defines the package's small public interface. It currently re-exports
`generate_experiment_grid()` so callers can write:

```python
from distensibility import generate_experiment_grid
```

### `distensibility/simulation.py`

Implements the complete current synthetic-generation stage. Its responsibilities
include:

- loading the YAML configuration;
- loading and hashing the mosaic graph;
- importing and reusing the existing transmission-line solver;
- extracting valid vessel geometry;
- computing the fixed-radius power-law distensibility;
- constructing default whole-mosaic boundary forcing;
- solving DC, H1, and H2 pressure and flow fields;
- converting flow to mean cross-sectional velocity;
- applying relative complex Gaussian noise;
- constructing reproducible edge splits;
- writing self-describing compressed `.npz` archives; and
- writing the dataset manifest.

The primary public function is:

```python
generate_experiment_grid(project_root, graph_path=None,
                         simulation_root=None, overwrite=False)
```

`scripts/make_synthetic.py` calls this function. Future tests, notebooks, or
batch orchestration can call it directly.

### `distensibility/io.py`

Defines the common `VascularDataset` representation, loads the versioned
compressed archives, exposes tile membership, and writes JSON-safe result
summaries.

### `distensibility/physics.py`

Defines tile and whole-mosaic spatial problems and constructs the distributed
harmonic pressure-to-velocity transfer operator. It uses the same compliant
transmission-line formulation as synthetic generation, including viscous
resistance, fluid inertance, and radius-dependent wall compliance.

### `distensibility/experiment.py`

Orchestrates one complete solver run, creates the method/dataset output
folders, saves effective configuration and numerical artifacts, aggregates
tile results, and invokes figure generation.

### `distensibility/plotting.py`

Creates pressure maps, velocity-error maps, and a self-contained HTML
dashboard for exploring the \(D_0\)-and-\(\alpha\) surface. The dashboard
contains full and \(y\leq10\) profile plots, true-parameter markers, and the
appropriate saved confidence or credible interval.

### `models/linear_tile.py`

Tile-specific deterministic solver. It profiles the complex pressure phasors
on each tile carve boundary and estimates \(D_0\) and optionally \(\alpha\).
The orchestration layer applies it independently to every valid tile unless a
tile subset is explicitly requested.

### `models/linear_mosaic.py`

Whole-mosaic deterministic solver. It uses the connected global pressure field
and the four external mosaic boundary nodes rather than artificial tile-cut
boundaries.

### `models/bayesian_tile.py`

Tile-specific Bayesian solver. It analytically marginalizes each tile's
complex boundary pressure phasors under the configured Gaussian prior.

### `models/bayesian_mosaic.py`

Whole-mosaic Bayesian solver. It analytically marginalizes the external mosaic
boundary pressure phasors while coupling all vessels through the global
harmonic operator.

### `models/_shared.py`

Contains numerical machinery shared by the four model modules: parameter-grid
construction, weighted complex least squares, Gaussian marginal likelihoods,
interval calculation, prediction reconstruction, and common metrics. This
file is not a standalone model entry point.

### `models/gnn.py`

Implements the full physics-informed GNN: node and edge encoders, directed
message passing, symmetric edge decoding, DC conductance corrections, and
optional H1/H2 heads.

### `models/baselines.py`

Implements the vanilla GCN direct-prediction baseline and the K=0 edge-local
MLP. Both expose the same output dictionary expected by the generic trainer.

### `gnn_losses.py`

Defines toggleable velocity, correction, pressure, harmonic reconstruction,
harmonic correction, and harmonic pressure penalties.

### `gnn_training.py`

Builds graph tensors from the shared `.npz` dataset schema, performs the
differentiable pressure solve, runs model-agnostic training and early
stopping, and computes split-specific reconstruction metrics.

### `experiment.py`

Provides GNN grid expansion, model construction, random seeding, device
selection, run-directory naming, resolved-config handling, and consistent
artifact saving. This is separate from `distensibility/experiment.py`, which
orchestrates the classical solvers.

## Design rules

- Put reusable logic in `src/`; keep CLI parsing in `scripts/`.
- Maintain one physics implementation shared by simulation, inference, and
  physics-informed models.
- Maintain one dataset interface shared by synthetic and real data.
- Maintain one metric implementation shared by every method.
- Keep paths relative to the `synthetic/` project root in saved configs.
- Save the effective config, random seed, predictions, and metrics for every
  run.

Update this README as each planned module becomes implemented.
