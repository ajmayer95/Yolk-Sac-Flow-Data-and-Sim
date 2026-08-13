# DC Workflow

This directory is the preferred user-facing entrypoint for the DC workflow.

The current DC step order is:

`00 -> 01 -> 02 -> 03 -> 04 -> 99`

- `00`: ideal models
- `01`: boundary parameter calibration
- `02`: physics weight sweep
- `03`: pressure constraint sensitivity
- `04`: message passing sensitivity
- `99`: release packaging only

The existing `scripts/python` entrypoints remain supported temporarily for
backward compatibility with current rerun scripts and sbatch launchers.

## Layout

### `solver/`

- `00_poiseuille_only_baseline.py`
  Runs the DC Poiseuille-only baseline and writes step-00 run outputs.
- `01_boundary_parameter_calibration.py`
  Runs the boundary parameter calibration sweep and writes step-01 run outputs.
- `02_physics_weight_sweep.py`
  Runs the Step 02 GNN and Poiseuille sweep.
- `03_pressure_constraint_sensitivity.py`
  Runs the Step 03 pressure-constraint sensitivity experiments.
- `04_message_passing_sensitivity.py`
  Runs the Step 04 message-passing depth sweep.

### `analysis/`

- `02_physics_weight_sweep_analysis.py`
  Aggregates completed Step 02 runs into summary CSV/YAML outputs.
- `03_pressure_constraint_sensitivity_analysis.py`
  Aggregates completed Step 03 runs into summary CSV outputs.
- `99_package_dc_results_for_release.py`
  Packages completed DC outputs for repo-ready and release-ready publishing.

### `plotting/`

- `00_poiseuille_baseline_plots.py`
  Generates Step 00 baseline field plots.
- `01_boundary_parameter_calibration_plots.py`
  Generates Step 01 boundary calibration plots.
- `02_physics_weight_sweep_plots.py`
  Generates Step 02 sweep figures from aggregated CSV outputs.
- `03_pressure_constraint_sensitivity_plots.py`
  Generates Step 03 sensitivity figures from aggregated CSV outputs.
- `04_message_passing_sensitivity_plots.py`
  Generates Step 04 message-passing depth figures from aggregated run outputs.

### `sbatch_scripts/`

- `dc_step02_somite21_gpu.sbatch`
  GPU-backed Step 02 launcher for the Somite21 dataset.
- `dc_step02_canonical_gpu.sbatch`
  GPU-backed Step 02 launcher for the canonical harmonized dataset.
- `dc_step02_canonical_norm_gpu.sbatch`
  GPU-backed Step 02 launcher for the canonical normalized dataset.
- `dc_step03_canonical_gpu.sbatch`
  GPU-backed Step 03 launcher for the canonical harmonized dataset.
- `dc_step03_canonical_norm_gpu.sbatch`
  GPU-backed Step 03 launcher for the canonical normalized dataset.
- `dc_step03_somite21_gpu.sbatch`
  GPU-backed Step 03 launcher for the Somite21 dataset.
- `dc_step04_canonical_gpu.sbatch`
  GPU-backed Step 04 launcher for the canonical harmonized dataset.
- `dc_step04_canonical_norm_gpu.sbatch`
  GPU-backed Step 04 launcher for the canonical normalized dataset.
- `dc_step04_somite21_gpu.sbatch`
  GPU-backed Step 04 launcher for the Somite21 dataset.

## High-Level Inputs and Outputs

- Solver scripts consume a graph dataset plus step-specific options and write run
  directories under a DC output root such as `outputs/.../dc/`.
- Analysis scripts consume an existing step output root and write aggregated CSV
  and YAML summaries back into that step directory.
- Plotting scripts consume existing run or summary outputs and write figures into
  a `figures/` directory under the corresponding step root.
- Step `99` consumes completed DC outputs and stages publishable release bundles.

## Canonical Usage

From the repo root, the preferred command path is through `scripts/dc_workflow`.

### Step 00

This step runs the Poiseuille baseline solvers using a gauge pressure of 0 at one of the nodes. All commands for this step can be run on CPU. 

```bash
conda run -n yolk-sac python scripts/dc_workflow/solver/00_poiseuille_only_baseline.py \
  datasets/somite21_mosaic_cut_pipeline_ready.gpickle \
  --output-dir outputs/somite21/dc/00_ideal_models/poiseuille_only_baseline \
  --run-name default_partitioned
```

```bash
conda run -n yolk-sac python scripts/dc_workflow/plotting/00_poiseuille_baseline_plots.py \
  --input-dir outputs/somite21/dc/00_ideal_models/poiseuille_only_baseline/default_partitioned \
  --output-dir outputs/somite21/dc/00_ideal_models/poiseuille_only_baseline/default_partitioned/figures
```

### Step 01

This step sweeps over lambda_B = {0.1, 1, 10, 100} with lambda_Q = lambda_K = 1. This step also runs the least squares Poiseuille solver (no GNN) using equal pressures at the arterial (A) and venous (V) nodes. The gauge pressure at the venous nodes is 0. All commands for this step can be run on CPU in a few minutes. 

```bash
conda run -n yolk-sac python scripts/dc_workflow/solver/01_boundary_parameter_calibration.py \
  --graph datasets/somite21_mosaic_cut_pipeline_ready.gpickle \
  --output-root outputs/somite21/dc/01_boundary_parameter_calibration
```

```bash
conda run -n yolk-sac python scripts/dc_workflow/plotting/01_boundary_parameter_calibration_plots.py \
  --input-csv outputs/somite21/dc/01_boundary_parameter_calibration/boundary_weight_summary.csv \
  --input-root outputs/somite21/dc/01_boundary_parameter_calibration \
  --output-dir outputs/somite21/dc/01_boundary_parameter_calibration/figures \
  --lambda-b 100
```

### Step 02

This step sweeps over lambda_Q, lambda_K, lambda_delta = {0.1, 1, 10, 100} for a given lambda_B value (currently set at 100). This step runs the least squares Poiseuille solver with or without the GNN using equal pressures at the arterial (A) and venous (V) nodes. The gauge pressure at the venous nodes is 0.

Without the GNN, there are 4 x 4 (lambda_Q x lambda_K) combinations of parameters to evaluate.  

With the GNN, there are 4 x 4 x 4 (lambda_Q x lambda_K x lambda_delta) combinations of parameters to evaluate.

The total number of combinations to evalute is 4 x 4 + 4 x 4 x 4 = 16 + 64 = 80.

The following command performs the sweep on CPU, though it may be too slow:

```bash
conda run -n yolk-sac python scripts/dc_workflow/solver/02_physics_weight_sweep.py \
  --graph datasets/somite21_mosaic_cut_pipeline_ready.gpickle \
  --output-root outputs/somite21/dc/02_physics_weight_sweep \
  --aggregate-after
```

The following commands performs the sweep on GPU, which may be faster:
GPU-backed `sbatch` commands:

```bash
sbatch scripts/dc_workflow/sbatch_scripts/dc_step02_somite21_gpu.sbatch
```

```bash
sbatch scripts/dc_workflow/sbatch_scripts/dc_step02_canonical_gpu.sbatch
```

```bash
sbatch scripts/dc_workflow/sbatch_scripts/dc_step02_canonical_norm_gpu.sbatch
```

The following commands are for analysis and plotting. These scripts are run on CPU. 

```bash
conda run -n yolk-sac python scripts/dc_workflow/analysis/02_physics_weight_sweep_analysis.py \
  --input-root outputs/somite21/dc/02_physics_weight_sweep
```

```bash
conda run -n yolk-sac python scripts/dc_workflow/plotting/02_physics_weight_sweep_plots.py \
  --input-root outputs/somite21/dc/02_physics_weight_sweep
```

### Step 03

This step utilizes the best balanced Step 02 representative configuration or a custom configuration set by the user. The goal of this step is to explore how the pressure and flow fields vary under different pressure boundary conditions. 

1. Venous gauge pressure = 0 only.
2. Equal arterial pressures and equal venous pressures (set to 0). 
3. Equal arterial-to-venous pressure differences. For the somite21 dataset with synthetic arterial boundaries, the A nodes are selected based on highest SNR. 
4. Prescribed mean arterial-to-venous pressure differences. 

By default, this step uses the best balanced Step 02 representative configuration.
You can override that default by passing an explicit
`--lambda-q/--lambda-k/--lambda-delta` triple that matches a Step 02 representative.

The example lambda_q, lambda_k, and lambda_delta values come from the lowest flow error for the Somites21 dataset with synthetic arterial nodes. 

The following command performs the solves on CPU, though it may be too slow:

Default balanced configuration:

```bash
conda run -n yolk-sac python scripts/dc_workflow/solver/03_pressure_constraint_sensitivity.py \
  --graph datasets/somite21_mosaic_cut_pipeline_ready.gpickle \
  --output-root outputs/somite21/dc/03_pressure_constraint_sensitivity \
  --aggregate-after
```

Explicit Step 02 lambda override:

```bash
conda run -n yolk-sac python scripts/dc_workflow/solver/03_pressure_constraint_sensitivity.py \
  --graph datasets/somite21_mosaic_cut_pipeline_ready.gpickle \
  --output-root outputs/somite21/dc/03_pressure_constraint_sensitivity_q100_k0p1_delta0p1 \
  --lambda-q 100 \
  --lambda-k 0.1 \
  --lambda-delta 0.1 \
  --aggregate-after
```

The following commands performs the solves on GPU, which may be faster:
GPU-backed `sbatch` commands:

```bash
sbatch scripts/dc_workflow/sbatch_scripts/dc_step03_somite21_gpu.sbatch
```

```bash
sbatch scripts/dc_workflow/sbatch_scripts/dc_step03_canonical_gpu.sbatch
```

```bash
sbatch scripts/dc_workflow/sbatch_scripts/dc_step03_canonical_norm_gpu.sbatch
```

Example with an explicit Step 02 lambda override:

```bash
sbatch scripts/dc_workflow/sbatch_scripts/dc_step03_somite21_gpu.sbatch \
  --lambda-q 100 \
  --lambda-k 0.1 \
  --lambda-delta 0.1 \
  --output-root outputs/somite21/dc/03_pressure_constraint_sensitivity_q100_k0p1_delta0p1
```

```bash
sbatch scripts/dc_workflow/sbatch_scripts/dc_step03_canonical_gpu.sbatch \
  --lambda-q 100 \
  --lambda-k 0.1 \
  --lambda-delta 0.1 \
  --output-root outputs_canonical/dc/03_pressure_constraint_sensitivity_q100_k0p1_delta0p1
```
The following commands are for analysis and plotting. These scripts are run on CPU. 
You may have to adjust the file path. 

```bash
conda run -n yolk-sac python scripts/dc_workflow/analysis/03_pressure_constraint_sensitivity_analysis.py \
  --input-root outputs/somite21/dc/03_pressure_constraint_sensitivity_q100_k0p1_delta0p1
```

```bash
conda run -n yolk-sac python scripts/dc_workflow/plotting/03_pressure_constraint_sensitivity_plots.py \
  --input-root outputs/somite21/dc/03_pressure_constraint_sensitivity_q100_k0p1_delta0p1
```

### Step 04

This step utilizes the best balanced Step 02 representative configuration or a custom configuration set by the user. The goal of this step is to explore how varying the number of "hops" affects the solution.

By default, this step uses the best balanced Step 02 representative configuration.
You can override that default by passing an explicit
`--lambda-q/--lambda-k/--lambda-delta` triple that matches a Step 02 representative.

The following command performs the solves on CPU, though it may be too slow:

Default balanced configuration:

```bash
conda run -n yolk-sac python scripts/dc_workflow/solver/04_message_passing_sensitivity.py \
  --graph datasets/somite21_mosaic_cut_pipeline_ready.gpickle \
  --output-root outputs/somite21/dc/04_message_passing_sensitivity
```

Explicit Step 02 lambda override:

```bash
conda run -n yolk-sac python scripts/dc_workflow/solver/04_message_passing_sensitivity.py \
  --graph datasets/somite21_mosaic_cut_pipeline_ready.gpickle \
  --output-root outputs/somite21/dc/04_message_passing_sensitivity_q100_k0p1_delta0p1 \
  --lambda-q 100 \
  --lambda-k 0.1 \
  --lambda-delta 0.1
```

The following commands performs the sweep on GPU, which may be faster:
GPU-backed `sbatch` commands:

```bash
sbatch scripts/dc_workflow/sbatch_scripts/dc_step04_somite21_gpu.sbatch
```

```bash
sbatch scripts/dc_workflow/sbatch_scripts/dc_step04_canonical_gpu.sbatch
```

```bash
sbatch scripts/dc_workflow/sbatch_scripts/dc_step04_canonical_norm_gpu.sbatch
```

Example with an explicit Step 02 lambda override:

```bash
sbatch scripts/dc_workflow/sbatch_scripts/dc_step04_somite21_gpu.sbatch \
  --lambda-q 100 \
  --lambda-k 0.1 \
  --lambda-delta 0.1 \
  --output-root outputs/somite21/dc/04_message_passing_sensitivity_q100_k0p1_delta0p1
```

```bash
sbatch scripts/dc_workflow/sbatch_scripts/dc_step04_canonical_gpu.sbatch \
  --lambda-q 100 \
  --lambda-k 0.1 \
  --lambda-delta 0.1 \
  --output-root outputs_canonical/dc/04_message_passing_sensitivity_q100_k0p1_delta0p1
```

The following command generates the Step 04 figures on CPU after the solver has
written the run outputs and summary tables:

```bash
conda run -n yolk-sac python scripts/dc_workflow/plotting/04_message_passing_sensitivity_plots.py \
  --input-root outputs/somite21/dc/04_message_passing_sensitivity_q100_k0p1_delta0p1
```

### Step 99

```bash
conda run -n yolk-sac python scripts/dc_workflow/analysis/99_package_dc_results_for_release.py \
  --outputs-root outputs/somite21/dc \
  --output-root publish/somite21
```

Step `99` is packaging-only. It is not part of model solving.

## CPU vs GPU Consistency

For the DC solver steps that support both CPU and GPU execution, the intended
behavior is numerically identical results up to floating-point roundoff
differences. The GPU-backed `sbatch` launchers listed above now pass explicit
CUDA-required flags, so they fail fast instead of silently running on CPU.

CPU-only aggregation or plotting launchers such as `*_agg*.sbatch` are omitted
here on purpose. This section lists only the DC Step 01-04 commands where the
launch mode is relevant to solver execution.
