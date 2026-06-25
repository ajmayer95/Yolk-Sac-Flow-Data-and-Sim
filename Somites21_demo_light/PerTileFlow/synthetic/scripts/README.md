# Script entry points

This directory contains command-line entry points for the synthetic
distensibility project. Scripts should stay thin: they parse command-line
arguments, resolve project paths, and call reusable functions from `src/`.

Run commands from the `synthetic/` project root unless noted otherwise.

## Current scripts

### `make_synthetic.py`

Generates the configured whole-mosaic synthetic dataset grid.

It:

1. reads `configs/synthetic_base.yaml` and
   `configs/experiment_grid.yaml`;
2. loads the vascular mosaic graph;
3. calls the established PerTileFlow transmission-line simulator;
4. applies the power-law wall model
   \(D_e=D_0(R_e/R_0)^\alpha\);
5. adds the configured velocity noise;
6. assigns reproducible train/validation/test edge splits; and
7. writes compressed datasets and `manifest.csv` under
   `data/synthetic/`.

The script delegates the actual work to
`src/distensibility/simulation.py`, specifically
`generate_experiment_grid()`. Other scripts should reuse that source module
rather than import functions from `scripts/make_synthetic.py`.

Basic use:

```bash
/mnt/home/sswee/miniforge3/envs/yolk-sac/bin/python scripts/make_synthetic.py
```

Useful options:

```bash
# Replace parameter-matched datasets that already exist.
python scripts/make_synthetic.py --overwrite

# Use a different graph.
python scripts/make_synthetic.py --graph /path/to/graph.gpickle

# Use a different PerTileFlow installation for the transmission-line solver.
python scripts/make_synthetic.py \
  --simulation-root /path/to/PerTileFlow
```

Without `--overwrite`, complete existing parameter groups are retained. The
script first checks the configured relative graph path and then falls back to
the known Somites21 mosaic source graph.

### `run_solver.py`

Runs one of the four implemented classical inverse methods:

- `linear_tile`
- `linear_mosaic`
- `bayesian_tile`
- `bayesian_mosaic`

The method loads a synthetic or real-compatible `.npz` dataset, evaluates the
configured \(D_0\)-and-\(\alpha\) grid, and fits H1 or H1+H2. Deterministic
methods profile out complex boundary pressures by weighted least squares.
Bayesian methods analytically marginalize the boundary pressures under the
configured Gaussian pressure prior.

Example with solved alpha and H1+H2:

```bash
python scripts/run_solver.py \
  data/synthetic/pl_d1e-03_a1_n10_s42.npz \
  --method linear_mosaic \
  --alpha-mode solved \
  --harmonics h1_h2
```

Example with prescribed alpha:

```bash
python scripts/run_solver.py \
  data/synthetic/pl_d1e-03_a1_n10_s42.npz \
  --method bayesian_tile \
  --alpha-mode prescribed \
  --alpha 1 \
  --harmonics h1 \
  --tiles 22
```

`--tiles` limits tile-specific methods to selected tile IDs. `--num-D0` and
`--num-alpha` provide convenient grid-size overrides for smoke tests or
exploratory runs. When `--tiles` is omitted, tile-specific methods run all
valid tiles.

Outputs use the layout:

```text
outputs/
├── runs/<method>/<dataset>/<configuration>/
│   ├── solver_config.yaml
│   ├── run_manifest.json
│   ├── predictions.npz
│   ├── parameter_surfaces.npz
│   └── spatial_summary.csv
├── metrics/<method>/<dataset>/<configuration>/
│   └── summary_metrics.json
└── figures/<method>/<dataset>/<configuration>/
    ├── pressure_map.png
    ├── flow_error_map.png
    └── distensibility_dashboard.html
```

The four solver entry modules are:

- `src/models/linear_tile.py`
- `src/models/linear_mosaic.py`
- `src/models/bayesian_tile.py`
- `src/models/bayesian_mosaic.py`

They share data, physics, orchestration, plotting, and low-level numerical
utilities without combining the model entry points into one file.

### GPU tile solver entry points

`run_linear_solver_gpu.py` and `run_bayesian_solver_gpu.py` evaluate
tile-specific parameter grids using batched complex network solves on CUDA.
They preserve the existing metrics, prediction, figure, and dashboard formats,
but write under `linear_tile_gpu` and `bayesian_tile_gpu`.

```bash
python scripts/run_linear_solver_gpu.py \
  data/synthetic/pl_d1e-03_a1_n10_s42.npz \
  --alpha-mode solved --harmonics h1_h2

python scripts/run_bayesian_solver_gpu.py \
  data/synthetic/pl_d1e-03_a1_n10_s42.npz \
  --alpha-mode prescribed --alpha 1 --harmonics h1
```

Restrict a validation run and tune the CUDA parameter batch:

```bash
python scripts/run_linear_solver_gpu.py \
  data/synthetic/pl_d1e-03_a1_n10_s42.npz \
  --tiles 22 --num-D0 5 --num-alpha 5 \
  --alpha-mode solved --harmonics h1_h2 \
  --chunk-size 64
```

The GPU implementation currently targets tile solvers only. The whole mosaic
contains 4,388 nodes and requires a sparse CUDA solver; this environment has
CUDA PyTorch but no CuPy/JAX sparse backend. A dense whole-mosaic solve would
be slower and less memory-efficient than the existing SciPy sparse CPU path.
The GPU tile solvers accept a saved neural DC pressure field through
`--pressure-field`. The recommended `--pressure-mode scaled` uses its spatial
shape as a prior while retaining a free complex amplitude and phase for each
velocity harmonic. Fixed H1/H2 neural pressure fields are not yet supported by
the GPU path.

Each tile dashboard contains both a full profile plot and a second copy limited
to \(y=10\). Linear profiles show the one-parameter 95% likelihood-ratio cutoff
\(\Delta\chi^2=3.8415\). This is the chi-square cutoff with one degree of
freedom; a joint two-parameter 95% contour would instead use approximately
5.99. The dashboard draws both references on the profile plots and draws the
5.9915 joint contour on the two-dimensional \(D_0\)-and-\(\alpha\) surface.
Bayesian dashboards display these likelihood-ratio-style references on the
transformed posterior and separately mark the actual 95% posterior credible
interval saved by the solver.

Validate both GPU methods on one tile:

```bash
sbatch scripts/run_classical_gpu_smoke.sbatch
```

Run the complete tile-only GPU campaign:

```bash
sbatch scripts/run_all_tile_solvers_gpu.sbatch
```

This is a 36-task array, one dataset per task. Each task runs linear and
Bayesian tile solvers for prescribed alpha 0/1/2 and solved alpha, using H1 and
H1+H2. Completed summaries are skipped. To lower GPU memory usage:

```bash
sbatch --export=ALL,CHUNK_SIZE=32 \
  scripts/run_all_tile_solvers_gpu.sbatch
```

Run the GPU tile solvers using the best validation-selected physics-informed
GNN pressure profile and the K=0 edge-local MLP pressure profile:

```bash
sbatch scripts/run_gnn_conditioned_tile_solvers_gpu.sbatch
```

This is also a 36-task array, one dataset per task. For each dataset it selects
one pressure run per model using validation DC relative RMSE, then runs linear
and Bayesian tile inference for prescribed alpha 0/1/2 and solved alpha with
H1 and H1+H2. The resulting pressure maps, flow-error maps, parameter
surfaces, and dashboards are stored under `linear_tile_gpu` and
`bayesian_tile_gpu`; pressure-source suffixes keep these runs separate from
unconditioned results.

Preview the commands for the first dataset without solving:

```bash
SLURM_ARRAY_TASK_ID=0 DRY_RUN=1 \
  bash scripts/run_gnn_conditioned_tile_solvers_gpu.sbatch
```

Run one conditioned fit directly:

```bash
python scripts/run_linear_solver_gpu.py \
  data/synthetic/pl_d1e-03_a1_n10_s42.npz \
  --alpha-mode solved --harmonics h1_h2 \
  --pressure-field \
    outputs/runs/gnn/pl_d1e-03_a1_n10_s42/physics_informed_gnn__K4__dc_only__seed42 \
  --pressure-mode scaled
```

### `run_all_classical_solvers.sbatch`

Slurm job array for running every classical solver configuration over all
datasets listed in `data/synthetic/manifest.csv`.

The array covers:

- 36 synthetic datasets;
- four methods;
- prescribed alpha values `0`, `1`, and `2`, including matched and mismatched
  fits for every dataset;
- solved alpha;
- H1 only; and
- H1+H2.

This produces 1,152 solver runs, grouped into only 36 Slurm array jobs—one job
per dataset, with all 32 configurations run sequentially inside that job. This
avoids both array-size and submitted-job QOS limits.

The checked-in default runs at most eight dataset jobs simultaneously and
requests two CPUs, 16 GB RAM, and 24 hours per job. Thus, at most 16 CPUs are
active at once. The worst-case campaign reservation is 1,728 CPU-hours if all
36 jobs consume the full time limit; actual use should be lower.

Dry-run the array decoding and command construction locally:

```bash
SLURM_ARRAY_TASK_ID=0 DRY_RUN=1 \
  bash scripts/run_all_classical_solvers.sbatch
```

Submit:

```bash
sbatch scripts/run_all_classical_solvers.sbatch
```

Completed summary files are skipped by default. To replace existing outputs:

```bash
sbatch --export=ALL,SKIP_EXISTING=0 \
  scripts/run_all_classical_solvers.sbatch
```

Logs are written to `scripts/logs/`.

### `run_all_gnn_conditioned_solvers.sbatch`

Runs the same four classical tile/mosaic methods after conditioning their
boundary-pressure inference on completed neural pressure fields.

The current GNN implementation saves a real DC pressure field for all
`dc_only`, `dc_h1`, and `dc_h1_h2` experiments. H1/H2 in those run names refer
to velocity heads; they are not saved harmonic pressure fields. Consequently,
the downstream default is `scaled`: each fitted complex H1/H2 boundary
pressure is regularized toward the DC spatial shape, but retains a free complex
amplitude and phase. If a future pressure artifact includes harmonic indices 1
or 2, those pressure fields are used directly.

One Slurm array task owns one dataset. Within each pressure-producing model
family and harmonic mode, `select_best_gnn_pressure_runs.py` chooses the run
with the lowest saved `best_validation_loss`. Thus the physics-informed model
retains only its best K for each of `dc_only`, `dc_h1`, and `dc_h1_h2`; the
edge-local MLP has its single K=0 candidate. Test errors are never used for
model selection.

For each selected pressure run the task executes all 32 classical
configurations. The default array therefore remains exactly 36 jobs—one per
dataset—with at most two active simultaneously. It uses the `gen` partition's
maximum seven-day limit. If a dataset task reaches that limit, resubmitting
the array safely continues because completed summaries are skipped.

Preview one dataset:

```bash
SLURM_ARRAY_TASK_ID=0 DRY_RUN=1 \
  bash scripts/run_all_gnn_conditioned_solvers.sbatch
```

Submit:

```bash
sbatch scripts/run_all_gnn_conditioned_solvers.sbatch
```

Useful overrides include `PRESSURE_MODE=absolute`, `PRESSURE_WEIGHT=0.1`,
`PRESSURE_SIGMA_PA=100`, and a space-separated `PRESSURE_MODELS` selection.
Completed summaries are skipped by default.

### `train_gnn.py`

Trains one resolved GNN experiment on one synthetic dataset. It loads the
fixed train/validation/test edge splits, builds the selected model, performs
early-stopped training, evaluates every split, and writes:

```text
config.yaml
metrics.json
predicted_velocities.npz
pressure_field.npz
corrections.npz
checkpoint.pt
training_history.json
```

Example:

```bash
export LD_LIBRARY_PATH="/mnt/home/sswee/miniforge3/envs/yolk-sac/lib:${LD_LIBRARY_PATH:-}"

python scripts/train_gnn.py \
  data/synthetic/pl_d1e-03_a1_n10_s42.npz \
  --config outputs/runs/gnn/pl_d1e-03_a1_n10_s42/physics_informed_gnn__K2__dc_only__seed42/config.yaml
```

Normally, resolved run configs are created by `run_gnn_grid.py`.

The current `vanilla_gcn` is a pressure-decoding baseline rather than a direct
edge-velocity decoder. It predicts one gauge-fixed DC pressure per node and
reconstructs DC edge velocity using fixed Poiseuille conductance:

```text
GCN node embeddings → nodal pressure
nodal pressure drop × known conductance → edge flow → edge velocity
```

It is trained through velocity reconstruction only; synthetic true pressure is
not used as a supervised target. Harmonic-enabled configurations retain direct
H1/H2 velocity heads, while `pressure_field.npz` stores the inferred DC field
for downstream deterministic and Bayesian solvers. New runs use the
`__pressure_decoder` suffix so the earlier direct-velocity GCN results remain
available for comparison.

### `run_gnn_grid.py`

Reads `configs/gnn_experiments.yaml`, expands valid model/depth/harmonic
combinations, creates one output directory per run, saves its resolved config,
and invokes `train_gnn.py`.

The default grid contains 21 runs per dataset:

- physics-informed GNN at K=1, 2, and 4;
- vanilla GCN at K=1, 2, and 4;
- edge-local MLP at K=0;
- each under `dc_only`, `dc_h1`, and `dc_h1_h2`.

Preview commands without training:

```bash
python scripts/run_gnn_grid.py \
  --datasets data/synthetic/pl_d1e-03_a1_n10_s42.npz \
  --dry-run
```

Run one focused experiment:

```bash
python scripts/run_gnn_grid.py \
  --datasets data/synthetic/pl_d1e-03_a1_n10_s42.npz \
  --model physics_informed_gnn \
  --K 2 \
  --harmonic-mode dc_only \
  --device auto
```

Run the full configured grid over every manifest dataset:

```bash
python scripts/run_gnn_grid.py --skip-existing
```

### `run_synthetic_gnn_array.sbatch`

Runs the configured GNN grid as a 36-task GPU array, one dataset per task. It
now skips completed runs by default. Because pressure-decoding vanilla GCN runs
have distinct `__pressure_decoder` directories, submitting this array retains
the completed physics-GNN and K=0 runs while training the new GCN runs.

Optional environment filters are `MODEL`, `K_VALUES`, `HARMONIC_MODE`,
`EPOCHS`, and `SKIP_EXISTING`.

```bash
# Full configured grid, skipping already completed run directories.
sbatch scripts/run_synthetic_gnn_array.sbatch

# Only pressure-decoding vanilla GCN runs.
sbatch --export=ALL,MODEL=vanilla_gcn \
  scripts/run_synthetic_gnn_array.sbatch

# Example focused rerun.
sbatch --export=ALL,MODEL=vanilla_gcn,K_VALUES="2 4",HARMONIC_MODE=dc_only \
  scripts/run_synthetic_gnn_array.sbatch
```

### `run_vanilla_pressure_gcn_array.sbatch`

Temporary focused array for the new pressure-decoding vanilla GCN. Each of the
36 array tasks trains K=1–4 under `dc_only`, `dc_h1`, and `dc_h1_h2`, for 12
runs per dataset. Completed pressure-decoder runs are skipped, so ordinary
resubmission is safe.

Preview task zero:

```bash
SLURM_ARRAY_TASK_ID=0 DRY_RUN=true \
  bash scripts/run_vanilla_pressure_gcn_array.sbatch
```

Submit:

```bash
sbatch scripts/run_vanilla_pressure_gcn_array.sbatch
```

Force replacement of completed pressure-decoder runs:

```bash
sbatch --export=ALL,SKIP_EXISTING=false \
  scripts/run_vanilla_pressure_gcn_array.sbatch
```

### `plot_best_gnn_runs.py`

Selects the best configuration independently for the physics-informed GNN,
pressure-decoding vanilla GCN, and K=0 edge-local MLP. It creates one
diagnostic dashboard per model and dataset plus a combined comparison index.
Selection minimizes validation DC relative RMSE, a metric shared by
`dc_only`, `dc_h1`, and `dc_h1_h2`; test metrics are not used.

The default command processes all 36 manifest datasets:

```bash
export LD_LIBRARY_PATH="/mnt/home/sswee/miniforge3/envs/yolk-sac/lib:${LD_LIBRARY_PATH:-}"
python scripts/plot_best_gnn_runs.py
```

The default all-model output is:

```text
outputs/figures/gnn_comparison/
├── index.html
├── comparison_manifest.json
├── validation_comparison.png
├── physics_informed_gnn/
├── vanilla_gcn/
└── edge_local_mlp/
```

Each model folder contains its own `index.html`, manifest, and 36 dataset
folders with training, pressure, velocity, error, and model-specific
diagnostics. Vanilla GCN correction panels are explicitly marked not
applicable because it predicts pressure rather than conductance corrections.

Generate all three models for a single dataset:

```bash
python scripts/plot_best_gnn_runs.py \
  --datasets data/synthetic/pl_d1e-03_a1_n10_s42.npz
```

Generate selected model families:

```bash
python scripts/plot_best_gnn_runs.py \
  --models physics_informed_gnn vanilla_gcn \
  --output-root outputs/figures/gnn_vs_gcn
```

Generate one model family using the backward-compatible `--model` shortcut:

```bash
python scripts/plot_best_gnn_runs.py \
  --model vanilla_gcn \
  --output-root outputs/figures/vanilla_gcn_best

python scripts/plot_best_gnn_runs.py \
  --model edge_local_mlp \
  --output-root outputs/figures/edge_local_mlp_best
```

Use `--require-complete` in automated workflows to fail if any requested model
is unfinished. Without it, incomplete model/dataset combinations are shown as
`pending` in the comparison index.

Open the overall index from the project root:

```bash
python -m http.server 8000
```

Then browse to
`http://localhost:8000/outputs/figures/gnn_comparison/index.html`. On a remote
cluster, forward port 8000 through SSH or open an individual `dashboard.html`
directly in a browser.

### `make_gnn_108_dashboard.py`

Builds one interactive dashboard over the complete set of 108 selected neural
results: 36 datasets multiplied by the physics-informed GNN, vanilla pressure
GCN, and K=0 edge-local MLP.

```bash
python scripts/make_gnn_108_dashboard.py
```

It reads the manifests produced by `plot_best_gnn_runs.py` and writes:

```text
outputs/figures/gnn_comparison/
├── all_108_dashboard.html
└── all_108_results.csv
```

The dashboard filters by model, true `D0`, true alpha, noise level, and metric.
It includes aggregate cards, per-configuration scatter plots, noise-grouped
means, and a table linking to every individual model/dataset dashboard. A
configuration selector embeds the corresponding:

- conductance multiplier map;
- correction distributions;
- delta map;
- pressure comparison;
- velocity parity plot; and
- training history.

For the vanilla pressure GCN, conductance-correction panels are explicitly
marked not applicable.

Use a nondefault comparison directory or output path:

```bash
python scripts/make_gnn_108_dashboard.py \
  --comparison-root outputs/figures/gnn_comparison \
  --output outputs/figures/gnn_comparison/custom_dashboard.html
```

### `make_classical_solver_comparison.py`

Aggregates all currently completed linear/Bayesian tile/mosaic results across
the 36 synthetic datasets. It compares unconditioned runs with K=0 and
physics-informed GNN pressure-prior runs and can be rerun safely while solver
arrays are still completing.

```bash
python scripts/make_classical_solver_comparison.py
```

Outputs are written to:

```text
outputs/figures/solver_comparison/
├── dashboard.html
├── all_solver_results.csv
├── configuration_summary.csv
└── manifest.json
```

The dashboard provides filters for method, pressure prior, alpha treatment,
harmonics, true \(D_0\), true alpha, and noise. It includes predicted-versus-
true \(D_0\) and alpha plots, overlays of the distensibility profiles, a
single-run profile explorer with the 3.84 and 5.99 references, a detailed run
table, and an across-dataset summary table. Tile estimates and profiles are
summarized by their median across valid tiles; mosaic runs use their single
whole-mosaic estimate.

CPU and GPU tile implementations are numerically equivalent and may both
exist for the same configuration. By default, the script de-duplicates these
and prefers the completed GPU result. Use `--include-engines` to retain both.
Missing or still-running configurations are simply absent, so rerun the
command after additional Slurm jobs complete to refresh the dashboard.

## Planned scripts

The following entry points are part of the intended workflow but have not yet
been implemented:

### `train_gnn.py`

Will train a physics-informed GNN, vanilla GCN, or edge-local MLP using
`configs/gnn_base.yaml`. It should use common dataset loaders, physics
functions, splits, and metrics from `src/`, rather than duplicating those
operations in the training script.

### `evaluate.py`

Will calculate the shared recovery and reconstruction metrics for one run or
prediction file. Both classical and learned methods should pass through this
entry point—or its underlying `src/metrics.py` functions—so comparisons use
identical definitions.

### `aggregate_results.py`

Will combine run-level metrics into summary CSV files, tables, and inputs for
final plots under `outputs/metrics/`, `outputs/tables/`, and
`outputs/figures/`.

## Dependency direction

The intended dependency flow is:

```text
configs + data
      ↓
scripts/*.py
      ↓
src/distensibility/*
      ↓
outputs
```

Source modules must not import command-line scripts. This keeps the scientific
logic reusable from tests, notebooks, batch jobs, and future real-data
workflows.

Update this README whenever a script is added, renamed, or changes its inputs
or outputs.
