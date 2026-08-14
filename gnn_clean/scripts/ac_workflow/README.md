# AC Workflow

This directory is the preferred user-facing entrypoint for the AC workflow.

The current AC step order is:

`00 -> 01 -> 02 -> 03`

- `00`: ideal models
- `01`: boundary parameter calibration
- `02`: physics weight sweep
- `03`: representative distensibility alpha/D0 sweep

The existing `scripts/python` AC entrypoints remain supported temporarily for
backward compatibility.

## Layout

### `solver/`

- `00_ideal_models.py`
  Runs the AC ideal-model distensibility sweep.
- `01_boundary_parameter_calibration.py`
  Runs the AC boundary-parameter calibration sweep.
- `02_physics_weight_sweep.py`
  Runs the AC physics-weight sweep.
- `03_distensibility_alpha_profiles.py`
  Runs the representative AC distensibility alpha/D0 sweep.

### `analysis/`

- `01_boundary_parameter_calibration_analysis.py`
  Aggregates Step 01 runs into summary CSV outputs.
- `02_physics_weight_sweep_analysis.py`
  Aggregates Step 02 runs into summary CSV outputs.
- `03_distensibility_alpha_profiles_analysis.py`
  Aggregates Step 03 runs into summary CSV outputs.

### `plotting/`

- `00_ideal_models_plots.py`
  Generates Step 00 distensibility sweep figures.
- `01_boundary_parameter_calibration_plots.py`
  Generates Step 01 boundary-calibration figures.
- `02_physics_weight_sweep_plots.py`
  Generates Step 02 physics-weight sweep figures.
- `03_distensibility_alpha_profiles_plots.py`
  Generates Step 03 alpha/D0 sweep figures.

### `sbatch_scripts/`

- `ac_step00_*_gpu.sbatch`
  GPU-backed Step 00 launchers for Somite21, canonical, and canonical normalized datasets.
- `ac_step01_*_gpu.sbatch`
  GPU-backed Step 01 launchers for Somite21, canonical, and canonical normalized datasets.
- `ac_step02_*_gpu.sbatch`
  GPU-backed Step 02 launchers for Somite21, canonical, and canonical normalized datasets.
- `ac_step03_H{1,2}_*_gpu.sbatch`
  GPU-backed Step 03 H1/H2 launchers for Somite21, canonical, and canonical normalized datasets.

## High-Level Inputs and Outputs

- Solver scripts consume a compatible graph plus step-specific options and write
  run directories under an AC output root such as `outputs/.../ac/`.
- Analysis scripts consume an existing step output root and write aggregated CSV
  outputs back into that step directory.
- Plotting scripts consume existing metrics or summaries and write figures into
  a `figures/` directory under the corresponding step root.

## Canonical Usage

From the repo root, the preferred command path is through `scripts/ac_workflow`.

## Parameter Naming Note

AC Steps `00-02` may reference two different parameter families in the same
command:

- `--lambda-q`, `--lambda-k`, and `--lambda-delta` select which DC Step 2
  configuration to pull conductances from.
- AC sweep parameters such as `--lambda-b-values`, `--lambda-b-override`,
  `--lambda-q-values`, and `--lambda-k-values` control the AC solve itself.

These are intentionally distinct. For example, AC Step 2 can use the DC Step 2
representative chosen by `--lambda-q 100 --lambda-k 0.1 --lambda-delta 0.1`
while still sweeping AC weights with `--lambda-q-values ...`,
`--lambda-k-values ...`, and a fixed `--lambda-b-override 100`.

## Boundary Conditions

The current AC solver entrypoints use these script defaults unless the caller
overrides them:

- `--arterial-boundary-mode all`
- `--venous-boundary-mode observed`

For the Somite21 workflow, the recommended settings used in the example
commands in this README are:

- `--arterial-boundary-mode per_tip_highest_snr`
- `--venous-boundary-mode rebalance_to_sources`

In practice, this means:

- arterial boundary forcing is applied only at one modeled arterial boundary
  node per arterial tip, chosen by highest adjacent-edge SNR
- venous boundary phasors are rescaled together so the net harmonic boundary
  injection is exactly zero while preserving the selected arterial sources

### Step 00

This step runs the AC ideal-model distensibility sweep. It currently uses the
balanced DC Step 2 representative by default through the existing `b1` run
selection logic.

```bash
conda run -n yolk-sac python scripts/ac_workflow/solver/00_ideal_models.py \
  --graph-path datasets/somite21_mosaic_cut_pipeline_ready.gpickle \
  --dc-step2-root outputs/somite21/dc/02_physics_weight_sweep \
  --output-root outputs/somite21/ac/00_ideal_models/distensibility_sweep \
  --scratch-root outputs/somite21/ac/00_ideal_models/distensibility_sweep/_raw_runs \
  --harmonic-numbers 1 2
```

```bash
conda run -n yolk-sac python scripts/ac_workflow/plotting/00_ideal_models_plots.py \
  --input outputs/somite21/ac/00_ideal_models/distensibility_sweep/distensibility_sweep_metrics.csv \
  --output-dir outputs/somite21/ac/00_ideal_models/distensibility_sweep/figures
```

To use a specific DC Step 2 configuration instead of the balanced default,
pass `--lambda-q`, `--lambda-k`, and `--lambda-delta`.

```bash
conda run -n yolk-sac python scripts/ac_workflow/solver/00_ideal_models.py \
  --graph-path datasets/somite21_mosaic_cut_pipeline_ready.gpickle \
  --dc-step2-root outputs/somite21/dc/02_physics_weight_sweep \
  --output-root outputs/somite21/ac/00_ideal_models/distensibility_sweep_q100_k0p1_delta0p1 \
  --scratch-root outputs/somite21/ac/00_ideal_models/distensibility_sweep_q100_k0p1_delta0p1/_raw_runs \
  --device cuda \
  --lstsq-backend torch \
  --require-cuda \
  --lambda-q 100 \
  --lambda-k 0.1 \
  --lambda-delta 0.1
```

GPU-backed `sbatch` commands:

```bash
sbatch scripts/ac_workflow/sbatch_scripts/ac_step00_somite21_gpu.sbatch
```

```bash
sbatch scripts/ac_workflow/sbatch_scripts/ac_step00_canonical_gpu.sbatch
```

```bash
sbatch scripts/ac_workflow/sbatch_scripts/ac_step00_canonical_norm_gpu.sbatch
```

Example with an explicit DC Step 2 configuration:

```bash
sbatch scripts/ac_workflow/sbatch_scripts/ac_step00_somite21_gpu.sbatch \
  --harmonic-numbers 1 2 \
  --lambda-q 100 \
  --lambda-k 0.1 \
  --lambda-delta 0.1 \
  --output-root outputs/somite21/ac/00_ideal_models/distensibility_sweep_q100_k0p1_delta0p1 \
  --scratch-root outputs/somite21/ac/00_ideal_models/distensibility_sweep_q100_k0p1_delta0p1/_raw_runs
```

### Step 01

This step sweeps AC boundary weights while resolving the balanced DC Step 2
representative by default through the existing `b1` selection logic. For the
current Somite21 rerun configuration, use `lambda_B = 10` for `H1` and
`lambda_B = 1` for `H2`. Callers can still specify any harmonic and any
boundary-weight list they want with `--harmonic-numbers` and
`--lambda-b-values`.

```bash
conda run -n yolk-sac python scripts/ac_workflow/solver/01_boundary_parameter_calibration.py \
  --graph-path datasets/somite21_mosaic_cut_pipeline_ready.gpickle \
  --dc-step2-root outputs/somite21/dc/02_physics_weight_sweep \
  --output-root outputs/somite21/ac/01_boundary_parameter_calibration \
  --device cuda \
  --lstsq-backend torch \
  --require-cuda \
  --harmonic-numbers 1 \
  --lambda-b-values 100 \
  --arterial-boundary-mode per_tip_highest_snr \
  --venous-boundary-mode rebalance_to_sources
```

```bash
conda run -n yolk-sac python scripts/ac_workflow/analysis/01_boundary_parameter_calibration_analysis.py \
  --input-root outputs/somite21/ac/01_boundary_parameter_calibration \
  --harmonic-number 1 \
  --output-csv outputs/somite21/ac/01_boundary_parameter_calibration/boundary_parameter_calibration_summary_H1.csv
```

```bash
conda run -n yolk-sac python scripts/ac_workflow/plotting/01_boundary_parameter_calibration_plots.py \
  --input-csv outputs/somite21/ac/01_boundary_parameter_calibration/boundary_parameter_calibration_summary_H1.csv \
  --harmonic-number 1 \
  --output-dir outputs/somite21/ac/01_boundary_parameter_calibration/figures
```

To use a specific DC Step 2 configuration instead of the balanced default,
pass `--lambda-q`, `--lambda-k`, and `--lambda-delta`.

```bash
conda run -n yolk-sac python scripts/ac_workflow/solver/01_boundary_parameter_calibration.py \
  --graph-path datasets/somite21_mosaic_cut_pipeline_ready.gpickle \
  --dc-step2-root outputs/somite21/dc/02_physics_weight_sweep \
  --output-root outputs/somite21/ac/01_boundary_parameter_calibration_q100_k0p1_delta0p1 \
  --device cuda \
  --lstsq-backend torch \
  --require-cuda \
  --harmonic-numbers 1 \
  --lambda-b-values 100 \
  --arterial-boundary-mode per_tip_highest_snr \
  --venous-boundary-mode rebalance_to_sources \
  --lambda-q 100 \
  --lambda-k 0.1 \
  --lambda-delta 0.1
```

GPU-backed `sbatch` commands:

```bash
sbatch scripts/ac_workflow/sbatch_scripts/ac_step01_somite21_gpu.sbatch
```

```bash
sbatch scripts/ac_workflow/sbatch_scripts/ac_step01_canonical_gpu.sbatch
```

```bash
sbatch scripts/ac_workflow/sbatch_scripts/ac_step01_canonical_norm_gpu.sbatch
```

Example with an explicit DC Step 2 configuration:

```bash
sbatch scripts/ac_workflow/sbatch_scripts/ac_step01_somite21_gpu.sbatch \
  --harmonic-numbers 1 \
  --lambda-b-values 10 \
  --arterial-boundary-mode all \
  --venous-boundary-mode observed \
  --lambda-q 100 \
  --lambda-k 0.1 \
  --lambda-delta 0.1 \
  --output-root outputs/somite21/ac/01_boundary_parameter_calibration_q100_k0p1_delta0p1_all_observed
```

```bash
sbatch scripts/ac_workflow/sbatch_scripts/ac_step01_somite21_gpu.sbatch \
  --harmonic-numbers 2 \
  --lambda-b-values 1 \
  --arterial-boundary-mode all \
  --venous-boundary-mode observed \
  --lambda-q 100 \
  --lambda-k 0.1 \
  --lambda-delta 0.1 \
  --output-root outputs/somite21/ac/01_boundary_parameter_calibration_q100_k0p1_delta0p1_all_observed
```

Matching analysis and plotting commands for that explicit-output root:

```bash
conda run -n yolk-sac python scripts/ac_workflow/analysis/01_boundary_parameter_calibration_analysis.py \
  --input-root outputs/somite21/ac/01_boundary_parameter_calibration_q100_k0p1_delta0p1 \
  --harmonic-number 1 \
  --output-csv outputs/somite21/ac/01_boundary_parameter_calibration_q100_k0p1_delta0p1/boundary_parameter_calibration_summary_H1.csv
```

```bash
conda run -n yolk-sac python scripts/ac_workflow/plotting/01_boundary_parameter_calibration_plots.py \
  --input-csv outputs/somite21/ac/01_boundary_parameter_calibration_q100_k0p1_delta0p1/boundary_parameter_calibration_summary_H1.csv \
  --harmonic-number 1 \
  --output-dir outputs/somite21/ac/01_boundary_parameter_calibration_q100_k0p1_delta0p1/figures
```

```bash
conda run -n yolk-sac python scripts/ac_workflow/analysis/01_boundary_parameter_calibration_analysis.py \
  --input-root outputs/somite21/ac/01_boundary_parameter_calibration_q100_k0p1_delta0p1 \
  --harmonic-number 2 \
  --output-csv outputs/somite21/ac/01_boundary_parameter_calibration_q100_k0p1_delta0p1/boundary_parameter_calibration_summary_H2.csv
```

```bash
conda run -n yolk-sac python scripts/ac_workflow/plotting/01_boundary_parameter_calibration_plots.py \
  --input-csv outputs/somite21/ac/01_boundary_parameter_calibration_q100_k0p1_delta0p1/boundary_parameter_calibration_summary_H2.csv \
  --harmonic-number 2 \
  --output-dir outputs/somite21/ac/01_boundary_parameter_calibration_q100_k0p1_delta0p1/figures
```

### Step 02

This step sweeps AC physics weights. It also resolves the balanced DC Step 2
representative by default through the existing `b1` selection logic. When you
want to hold the boundary penalty fixed, the current Somite21 rerun
configuration uses `lambda_B = 10` for `H1` and `lambda_B = 1` for `H2`.
Harmonics are configurable with
`--harmonic-numbers`.

```bash
conda run -n yolk-sac python scripts/ac_workflow/solver/02_physics_weight_sweep.py \
  --graph-path datasets/somite21_mosaic_cut_pipeline_ready.gpickle \
  --dc-step2-root outputs/somite21/dc/02_physics_weight_sweep \
  --output-root outputs/somite21/ac/02_physics_weight_sweep \
  --device cuda \
  --lstsq-backend torch \
  --require-cuda \
  --harmonic-numbers 1 \
  --lambda-b-override 100
```

```bash
conda run -n yolk-sac python scripts/ac_workflow/analysis/02_physics_weight_sweep_analysis.py \
  --input-root outputs/somite21/ac/02_physics_weight_sweep/H1
```

```bash
conda run -n yolk-sac python scripts/ac_workflow/plotting/02_physics_weight_sweep_plots.py \
  --input-root outputs/somite21/ac/02_physics_weight_sweep/H1
```

To use a specific DC Step 2 configuration instead of the balanced default,
pass `--lambda-q`, `--lambda-k`, and `--lambda-delta`.

```bash
conda run -n yolk-sac python scripts/ac_workflow/solver/02_physics_weight_sweep.py \
  --graph-path datasets/somite21_mosaic_cut_pipeline_ready.gpickle \
  --dc-step2-root outputs/somite21/dc/02_physics_weight_sweep \
  --output-root outputs/somite21/ac/02_physics_weight_sweep_q100_k0p1_delta0p1 \
  --device cuda \
  --lstsq-backend torch \
  --require-cuda \
  --harmonic-numbers 1 \
  --lambda-b-override 100 \
  --lambda-q 100 \
  --lambda-k 0.1 \
  --lambda-delta 0.1
```

GPU-backed `sbatch` commands:

```bash
sbatch scripts/ac_workflow/sbatch_scripts/ac_step02_somite21_gpu.sbatch
```

```bash
sbatch scripts/ac_workflow/sbatch_scripts/ac_step02_canonical_gpu.sbatch
```

```bash
sbatch scripts/ac_workflow/sbatch_scripts/ac_step02_canonical_norm_gpu.sbatch
```

Example with an explicit DC Step 2 configuration:

```bash
sbatch scripts/ac_workflow/sbatch_scripts/ac_step02_somite21_gpu.sbatch \
  --harmonic-numbers 1 \
  --lambda-b-override 10 \
  --arterial-boundary-mode all \
  --venous-boundary-mode observed \
  --lambda-q 100 \
  --lambda-k 0.1 \
  --lambda-delta 0.1 \
  --output-root outputs/somite21/ac/02_physics_weight_sweep_q100_k0p1_delta0p1_all_observed
```

```bash
sbatch scripts/ac_workflow/sbatch_scripts/ac_step02_somite21_gpu.sbatch \
  --harmonic-numbers 2 \
  --lambda-b-override 1 \
  --arterial-boundary-mode all \
  --venous-boundary-mode observed \
  --lambda-q 100 \
  --lambda-k 0.1 \
  --lambda-delta 0.1 \
  --output-root outputs/somite21/ac/02_physics_weight_sweep_q100_k0p1_delta0p1_all_observed
```

Matching analysis and plotting commands for that explicit-output root:

```bash
conda run -n yolk-sac python scripts/ac_workflow/analysis/02_physics_weight_sweep_analysis.py \
  --input-root outputs/somite21/ac/02_physics_weight_sweep_q100_k0p1_delta0p1_all_observed/H1
```

```bash
conda run -n yolk-sac python scripts/ac_workflow/plotting/02_physics_weight_sweep_plots.py \
  --input-root outputs/somite21/ac/02_physics_weight_sweep_q100_k0p1_delta0p1_all_observed/H1
```

```bash
conda run -n yolk-sac python scripts/ac_workflow/analysis/02_physics_weight_sweep_analysis.py \
  --input-root outputs/somite21/ac/02_physics_weight_sweep_q100_k0p1_delta0p1_all_observed/H2
```

```bash
conda run -n yolk-sac python scripts/ac_workflow/plotting/02_physics_weight_sweep_plots.py \
  --input-root outputs/somite21/ac/02_physics_weight_sweep_q100_k0p1_delta0p1_all_observed/H2
```

### Step 03

This step sweeps representative AC distensibility alpha/D0 profiles. By
default, it uses representative labels `F1 B1 K1` unless the caller overrides
them. To run only the best balanced AC Step 2 representative, use
`--representative-labels B1`.

```bash
conda run -n yolk-sac python scripts/ac_workflow/solver/03_distensibility_alpha_profiles.py \
  --graph-path datasets/somite21_mosaic_cut_pipeline_ready.gpickle \
  --step2-root outputs/somite21/ac/02_physics_weight_sweep \
  --output-root outputs/somite21/ac/03_distensibility_alpha_profiles \
  --device cuda \
  --lstsq-backend torch \
  --require-cuda \
  --harmonic-numbers 1 2 \
  --representative-labels F1 B1 K1
```

```bash
conda run -n yolk-sac python scripts/ac_workflow/analysis/03_distensibility_alpha_profiles_analysis.py \
  --input-root outputs/somite21/ac/03_distensibility_alpha_profiles/H1
```

```bash
conda run -n yolk-sac python scripts/ac_workflow/plotting/03_distensibility_alpha_profiles_plots.py \
  --input-root outputs/somite21/ac/03_distensibility_alpha_profiles/H1
```

Example using a custom DC Step 2 configuration together with the best balanced
AC Step 2 representative:

```bash
conda run -n yolk-sac python scripts/ac_workflow/solver/03_distensibility_alpha_profiles.py \
  --graph-path datasets/somite21_mosaic_cut_pipeline_ready.gpickle \
  --step2-root outputs/somite21/ac/02_physics_weight_sweep_q100_k0p1_delta0p1_all_observed \
  --output-root outputs/somite21/ac/03_distensibility_alpha_profiles_q100_k0p1_delta0p1_all_observed_B1 \
  --device cuda \
  --lstsq-backend torch \
  --require-cuda \
  --harmonic-numbers 1 2 \
  --representative-labels B1 \
  --arterial-boundary-mode all \
  --venous-boundary-mode observed
```

Matching GPU `sbatch` commands:

```bash
sbatch scripts/ac_workflow/sbatch_scripts/ac_step03_H1_somite21_gpu.sbatch \
  --step2-root outputs/somite21/ac/02_physics_weight_sweep_q100_k0p1_delta0p1_all_observed \
  --output-root outputs/somite21/ac/03_distensibility_alpha_profiles_q100_k0p1_delta0p1_all_observed_B1 \
  --representative-labels B1 \
  --arterial-boundary-mode all \
  --venous-boundary-mode observed
```

```bash
sbatch scripts/ac_workflow/sbatch_scripts/ac_step03_H2_somite21_gpu.sbatch \
  --step2-root outputs/somite21/ac/02_physics_weight_sweep_q100_k0p1_delta0p1_all_observed \
  --output-root outputs/somite21/ac/03_distensibility_alpha_profiles_q100_k0p1_delta0p1_all_observed_B1 \
  --representative-labels B1 \
  --arterial-boundary-mode all \
  --venous-boundary-mode observed
```

Matching analysis and plotting commands:

```bash
conda run -n yolk-sac python scripts/ac_workflow/analysis/03_distensibility_alpha_profiles_analysis.py \
  --input-root outputs/somite21/ac/03_distensibility_alpha_profiles_q100_k0p1_delta0p1_all_observed_B1/H1
```

```bash
conda run -n yolk-sac python scripts/ac_workflow/plotting/03_distensibility_alpha_profiles_plots.py \
  --input-root outputs/somite21/ac/03_distensibility_alpha_profiles_q100_k0p1_delta0p1_all_observed_B1/H1
```

```bash
conda run -n yolk-sac python scripts/ac_workflow/analysis/03_distensibility_alpha_profiles_analysis.py \
  --input-root outputs/somite21/ac/03_distensibility_alpha_profiles_q100_k0p1_delta0p1_all_observed_B1/H2
```

```bash
conda run -n yolk-sac python scripts/ac_workflow/plotting/03_distensibility_alpha_profiles_plots.py \
  --input-root outputs/somite21/ac/03_distensibility_alpha_profiles_q100_k0p1_delta0p1_all_observed_B1/H2
```

GPU-backed `sbatch` commands:

```bash
sbatch scripts/ac_workflow/sbatch_scripts/ac_step03_H1_somite21_gpu.sbatch
```

```bash
sbatch scripts/ac_workflow/sbatch_scripts/ac_step03_H2_somite21_gpu.sbatch
```

```bash
sbatch scripts/ac_workflow/sbatch_scripts/ac_step03_H1_canonical_gpu.sbatch
```

```bash
sbatch scripts/ac_workflow/sbatch_scripts/ac_step03_H2_canonical_gpu.sbatch
```

```bash
sbatch scripts/ac_workflow/sbatch_scripts/ac_step03_H1_canonical_norm_gpu.sbatch
```

```bash
sbatch scripts/ac_workflow/sbatch_scripts/ac_step03_H2_canonical_norm_gpu.sbatch
```

## Dataset Notes

- Somite21 uses `datasets/somite21_mosaic_cut_pipeline_ready.gpickle` with
  repo-local outputs under `outputs/somite21/ac/...`.
- Canonical harmonized uses
  `datasets/mosaic_graph_canonical_harmonized/mosaic_graph_canonical_harmonized.gpickle`
  with outputs under `outputs_canonical/ac/...`.
- Canonical normalized uses `datasets/mosaic_graph_norm_canonical.gpickle`
  with outputs under `outputs_canonical_norm/ac/...`.

## AC 03p5

AC `03p5` is intentionally not migrated into `scripts/ac_workflow` in this
first pass. It remains in the legacy rerun folders:

- `scripts/somite21_rerun/sbatch`
- `scripts/canonical_rerun/sbatch`
- `scripts/canonical_norm_rerun/sbatch`

Use those legacy launchers for the sparse distensibility profile workflows
until the next AC organization pass.
