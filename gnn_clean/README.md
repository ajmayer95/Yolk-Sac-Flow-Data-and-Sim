# `gnn_clean`

Clean repository for DC and AC graph-based flow studies on any compatible
`.gpickle` graph.

The examples below use `datasets/harmonized_scaled_dataset.gpickle`, but the
workflow is meant to support other inputs such as
`datasets/emb1_mosaic_graph_analyzed.gpickle` or
`datasets/mosaic_graph_norm_canonical.gpickle` as long as the graph exposes the
schema expected by the DC and AC scripts.

In the example commands, replace the graph path with whichever compatible
`.gpickle` you want to analyze.

## Repository Layout

- `scripts/python/`: public workflow entrypoints and plotting/analysis scripts
- `src/`: shared GNN, solver, and harmonic utilities
- `pertile/`: lower-level analysis code used by the workflow
- `datasets/`: input graph data
- `outputs/`: generated workflow outputs

## DC Workflow

### DC Published Results

The DC results are packaged into two layers:

- **GitHub repo**: essential summary tables, representative-selection tables, and final figures.
- **GitHub Releases**: larger step-level DC raw archives for deeper reuse.

Packaging command:

```bash
python scripts/python/package_dc_results_for_release.py
```

Dry run:

```bash
python scripts/python/package_dc_results_for_release.py --dry-run
```

Custom publish root:

```bash
python scripts/python/package_dc_results_for_release.py \
  --output-root publish
```

The script stages:

- `publish/dc/repo_bundle/dc/`
- `publish/dc/release_bundle/dc/`
- `publish/dc/manifest.csv`
- `publish/dc/release_bundle/SHA256SUMS`

Artifact guide:

| Artifact | Contains | Location | Intended use | Approximate size |
| --- | --- | --- | --- | --- |
| `repo_bundle/dc/` | Essential DC summaries and final figures | Repo staging | Browse and cite the main findings | MB to low-GB scale |
| `dc_repo_bundle.tar.gz` | Tarball of the repo-ready DC bundle | Release staging | One-file download of repo-ready DC outputs | MB to low-GB scale |
| `dc_step00_raw.tar.gz` | Full raw outputs for DC Step 0 | Release staging | Reuse Poiseuille baseline artifacts | Small |
| `dc_step01_raw.tar.gz` | Full raw outputs for DC Step 1 | Release staging | Reuse boundary-parameter calibration artifacts | Small |
| `dc_step02_raw.tar.gz` | Full raw outputs for DC Step 2 | Release staging | Reuse physics-weight sweep artifacts | Moderate |
| `dc_step03_raw.tar.gz` | Full raw outputs for DC Step 3 | Release staging | Reuse pressure-constraint sensitivity artifacts | Moderate |
| `dc_step04_raw.tar.gz` | Full raw outputs for DC Step 4 | Release staging | Reuse message-passing sensitivity artifacts | Moderate |

Selection rules:

- The repo bundle keeps only final DC figures and top-level summary CSV/YAML artifacts.
- The release bundle archives each existing DC step directory as a whole:
  - `outputs/dc/00_ideal_models`
  - `outputs/dc/01_boundary_parameter_calibration`
  - `outputs/dc/02_physics_weight_sweep`
  - `outputs/dc/03_pressure_constraint_sensitivity`
  - `outputs/dc/04_message_passing_sensitivity`

### Step 0: Ideal Poiseuille Baseline

Main script:

- `scripts/python/poiseuille_only_baseline.py`
- `scripts/python/plot_poiseuille_baseline.py`

Example:

```bash
python scripts/python/poiseuille_only_baseline.py \
  datasets/harmonized_scaled_dataset.gpickle \
  --output-dir outputs/dc/00_ideal_models/poiseuille_only_baseline \
  --run-name default_partitioned
```

Plot example:

```bash
python scripts/python/plot_poiseuille_baseline.py \
  --input-dir outputs/dc/00_ideal_models/poiseuille_only_baseline/default_partitioned \
  --output-dir outputs/dc/00_ideal_models/poiseuille_only_baseline/default_partitioned/figures
```

Notes:

- The Step 0 plotting script writes `flow_field.png`, `flow_magnitude_field.png`, `pressure_field.png`, and `flow_kirchhoff_metrics.png`.
- `flow_field.png` preserves the sign of the predicted edge flow, while `flow_magnitude_field.png` shows only `|flow|` using a logarithmic `coolwarm` edge colormap like the poster-style flow-amplitude figures.
- A negative predicted flow means the physical flow runs opposite the stored source-to-target edge orientation in `edge_predictions.csv`.
- For interpretation, the sign is an orientation convention, while the magnitude reflects how much flow the model predicts through that vessel.
- `pressure_field.png` plots the solved nodal `pressure_pa` values directly from `node_predictions.csv`; it does not convert them into pressure differences or re-center them before plotting, aside from percentile-based color clipping for readability.

### Step 1: Boundary Parameter Calibration

Scripts:

- `scripts/python/run_boundary_weight_sweep.py`
- `scripts/python/plot_boundary_weight_sweep.py`

Example:

```bash
python scripts/python/run_boundary_weight_sweep.py \
  --graph datasets/harmonized_scaled_dataset.gpickle \
  --output-root outputs/dc/01_boundary_parameter_calibration
```

Plot example:

```bash
python scripts/python/plot_boundary_weight_sweep.py \
  --input-csv outputs/dc/01_boundary_parameter_calibration/boundary_weight_summary.csv \
  --output-dir outputs/dc/01_boundary_parameter_calibration/figures
```

Plot example for the selected `lambda_B` run fields:

```bash
python scripts/python/plot_boundary_weight_sweep.py \
  --input-csv outputs/dc/01_boundary_parameter_calibration/boundary_weight_summary.csv \
  --input-root outputs/dc/01_boundary_parameter_calibration \
  --output-dir outputs/dc/01_boundary_parameter_calibration/figures \
  --lambda-b 100
```

Notes:

- This step tests `\(\lambda_B\)` while holding `\(\lambda_Q = 1.0\)` and `\(\lambda_K = 1.0\)` fixed.
- The Step 1 calibration runs use the Poiseuille-only reduced soft constrained least-squares pressure solve, with `\(\lambda_B\)` mapped to the pressure-constraint weight while the Kirchhoff and flow-residual weights stay fixed at `1.0`.
- The default sweep is `\(\lambda_B \in \{0.1, 1, 10, 100\}\)`.
- For the example dataset and commands in this README, record `\(\lambda_B = 100\)` as the selected boundary-calibration setting for downstream example runs.
- With `--lambda-b`, the plotting script also writes the corresponding `flow_field`, `flow_magnitude_field`, and `pressure_field` figures for that specific calibration run.

### Step 2: Physics Weight Sweep

Scripts:

- `scripts/python/run_physics_weight_sweep.py`
- `scripts/python/analyze_physics_weight_sweep.py`
- `scripts/python/plot_physics_weight_sweep.py`

Run + aggregate example:

```bash
python scripts/python/run_physics_weight_sweep.py \
  --graph datasets/harmonized_scaled_dataset.gpickle \
  --output-root outputs/dc/02_physics_weight_sweep \
  --aggregate-after
```

Run only example:

```bash
python scripts/python/run_physics_weight_sweep.py \
  --graph datasets/harmonized_scaled_dataset.gpickle \
  --output-root outputs/dc/02_physics_weight_sweep
```

Aggregate only example:

```bash
python scripts/python/run_physics_weight_sweep.py \
  --graph datasets/harmonized_scaled_dataset.gpickle \
  --output-root outputs/dc/02_physics_weight_sweep \
  --aggregate-only
```

Plot after aggregation:

```bash
python scripts/python/plot_physics_weight_sweep.py \
  --input-root outputs/dc/02_physics_weight_sweep
```

Parallel shard example:

```bash
python scripts/python/run_physics_weight_sweep.py \
  --graph datasets/harmonized_scaled_dataset.gpickle \
  --output-root outputs/dc/02_physics_weight_sweep \
  --num-shards 80 \
  --shard-index 0
```

Mode-specific examples:

```bash
python scripts/python/run_physics_weight_sweep.py \
  --graph datasets/harmonized_scaled_dataset.gpickle \
  --output-root outputs/dc/02_physics_weight_sweep \
  --mode gnn-only
```

```bash
python scripts/python/run_physics_weight_sweep.py \
  --graph datasets/harmonized_scaled_dataset.gpickle \
  --output-root outputs/dc/02_physics_weight_sweep \
  --mode poiseuille-only
```

Notes:

- This step fixes `\(\lambda_B = 100\)` and sweeps `\(\lambda_Q\)`, `\(\lambda_K\)`, and, for GNN runs, `\(\lambda_\delta\)` over `\(\{0.1, 1, 10, 100\}\)`.
- The default `both` mode expands to `64` GNN runs plus `16` Poiseuille baseline runs.
- The Step 2 Poiseuille baseline runs use the same reduced soft constrained least-squares Poiseuille solve as Step 1, with `\(\lambda_B = 100\)` mapped to the pressure-constraint weight; if you generated Step 2 Poiseuille outputs before this change, rerun those baseline jobs to refresh them.
- Aggregation writes the main sweep summaries to `physics_weight_all_runs.csv`, `physics_weight_gnn_summary.csv`, `physics_weight_poiseuille_summary.csv`, `representative_configurations.csv`, and `physics_weight_analysis.yaml`.
- `representative_configurations.csv` records the selected example configurations for the main weighting regimes: flow-prioritized, balanced, conservation-prioritized, and correction-regularized.
- Representative labels use the prefixes `F`, `B`, `K`, and `C` for flow-prioritized, balanced, conservation-prioritized, and correction-regularized, with rank `1` meaning the best-scoring representative within that regime.
- The designation is based on the dominant loss weight pattern: `flow_prioritized` when `\(\lambda_Q\)` is at least `10x` larger than both `\(\lambda_K\)` and `\(\lambda_\delta\)`, `conservation_prioritized` when `\(\lambda_K\)` is dominant, `correction_regularized` when `\(\lambda_\delta\)` is dominant, and `balanced` otherwise.
- Regime scores are computed from physical metrics, not just the raw lambda values:
  `flow_prioritized = 0.75 * flow_rmse_nl_s + 0.25 * kirchhoff_rms_per_internal_node_nl_s`
  `balanced = 0.5 * flow_rmse_nl_s + 0.5 * kirchhoff_rms_per_internal_node_nl_s`
  `conservation_prioritized = 0.25 * flow_rmse_nl_s + 0.75 * kirchhoff_rms_per_internal_node_nl_s`
  `correction_regularized = 0.5 * flow_rmse_nl_s + 0.5 * kirchhoff_rms_per_internal_node_nl_s`
- For the example dataset, the top-ranked representatives are:
  `F1`: `q_100__k_0p1__delta_1`
  `B1`: `q_1__k_1__delta_1`
  `K1`: `q_10__k_100__delta_10`
  `C1`: `q_0p1__k_0p1__delta_10`
- The main Step 2 plots are the flow/Kirchhoff tradeoff figures in `figures/`, especially `flow_kirchhoff_pareto.png`, `flow_kirchhoff_pareto_labeled.png`, and `flow_kirchhoff_pareto_with_fit.png`.
- The main Step 2 sweep tradeoff plots are `flow_kirchhoff_pareto.png` and `flow_kirchhoff_pareto_with_fit.png`; these show the regime-colored GNN runs and the Poiseuille baseline without selected-representative markers or Pareto-front overlays.
- `flow_rmse_vs_delta_rms.png` and `kirchhoff_rms_vs_delta_rms.png` are useful for checking how strongly the learned conductance corrections track performance changes.
- The `supp_*` figures and the `*_vs_log_lambda_q_over_k_*` plots are supplementary diagnostics for understanding how the sweep responds to loss-weight ratios and `\(\lambda_\delta\)`; the `by_delta` plots now include legends for the regime colors.
- The representative field plots for `F1`, `B1`, `K1`, `C1`, and the Poiseuille baseline are written under `outputs/dc/02_physics_weight_sweep/figures/representative_fields/`, with flow, flow-amplitude, pressure, and correction-field views for each.


### Step 3: Pressure Constraint Sensitivity

Scripts:

- `scripts/python/run_pressure_constraint_sensitivity.py`
- `scripts/python/analyze_pressure_constraint_sensitivity.py`
- `scripts/python/plot_pressure_constraint_sensitivity.py`

Example:

```bash
python scripts/python/run_pressure_constraint_sensitivity.py \
  --graph datasets/harmonized_scaled_dataset.gpickle \
  --output-root outputs/dc/03_pressure_constraint_sensitivity \
  --aggregate-after
```

Analysis:

```bash
python scripts/python/analyze_pressure_constraint_sensitivity.py \
  --input-root outputs/dc/03_pressure_constraint_sensitivity
```

Plotting:

```bash
python scripts/python/plot_pressure_constraint_sensitivity.py \
  --input-root outputs/dc/03_pressure_constraint_sensitivity
```

### Step 4: Message-Passing Sensitivity

Scripts:

- `scripts/python/run_message_passing_depth_sweep.py`
- `scripts/python/compute_message_passing_field_similarity.py`

Example:

```bash
python scripts/python/run_message_passing_depth_sweep.py \
  --graph datasets/harmonized_scaled_dataset.gpickle \
  --output-root outputs/dc/04_message_passing_sensitivity
```

### Step 5: Radius Corrections

Scripts:

- `scripts/python/run_radius_correction_experiment.py`
- `scripts/python/analyze_radius_correction_experiment.py`
- `scripts/python/plot_radius_correction_experiment.py`

Example:

```bash
python scripts/python/run_radius_correction_experiment.py \
  --graph datasets/harmonized_scaled_dataset.gpickle \
  --output-root outputs/dc/05_radius_corrections \
  --aggregate-after \
  --plot-after
```

### Step 6: Scale Analysis

Script:

- `scripts/python/plot_scale_analysis.py`

Example:

```bash
python scripts/python/plot_scale_analysis.py
```

## AC Workflow

### AC Published Results

The AC results are packaged into two layers:

- **GitHub repo**: essential summary tables, representative-selection tables, and final figures.
- **GitHub Releases**: larger AC raw archives for representative runs that are useful for deeper reuse or figure regeneration.

DC is not part of this packaging pass; the existing `outputs_dc.tar.gz` is treated as acceptable as-is.

Packaging command:

```bash
python scripts/python/package_ac_results_for_release.py
```

Dry run:

```bash
python scripts/python/package_ac_results_for_release.py --dry-run
```

Custom publish root:

```bash
python scripts/python/package_ac_results_for_release.py \
  --output-root publish
```

The script stages:

- `publish/ac/repo_bundle/ac/`
- `publish/ac/release_bundle/ac/`
- `publish/ac/manifest.csv`
- `publish/ac/release_bundle/SHA256SUMS`

Artifact guide:

| Artifact | Contains | Location | Intended use | Approximate size |
| --- | --- | --- | --- | --- |
| `repo_bundle/ac/` | Essential AC summaries and final figures | Repo staging | Browse and cite the main findings | MB to low-GB scale |
| `ac_repo_bundle.tar.gz` | Tarball of the repo-ready AC bundle | Release staging | One-file download of repo-ready AC outputs | MB to low-GB scale |
| `ac_step00_representative_raw.tar.gz` | Representative AC Step 0 raw runs selected from distensibility-sweep minima | Release staging | Inspect ideal-model internals and regenerate representative/raw visualizations | GB scale |
| `ac_step03_H1_representative_raw.tar.gz` | Representative AC Step 3 raw runs for `H1` | Release staging | Reuse saved model outputs for `H1` representative runs | GB scale |
| `ac_step03_H2_representative_raw.tar.gz` | Representative AC Step 3 raw runs for `H2` | Release staging | Reuse saved model outputs for `H2` representative runs | GB scale |

Selection rules:

- The repo bundle keeps only final AC figures, summary CSV/YAML files, representative tables, and curated representative-field outputs.
- The Step 0 raw archive is selected from `distensibility_sweep_metrics.csv` by taking the minima run for each combination of harmonic, model, and `\(\alpha\)` over:
  - complex flow RMSE
  - Kirchhoff RMS
  - arterial pressure phase difference
- The Step 3 raw archives are selected from the unique `profile_run_dir` entries listed in:
  - `outputs/ac/03_distensibility_alpha_profiles/H1/representative_configurations.csv`
  - `outputs/ac/03_distensibility_alpha_profiles/H2/representative_configurations.csv`

### Step 0: Ideal Harmonic Model Comparison

Main script:

- `scripts/python/harmonic_stage1_admittance_model_comparison.py`

Example:

```bash
python scripts/python/harmonic_stage1_admittance_model_comparison.py \
  --graph-path datasets/harmonized_scaled_dataset.gpickle \
  --dc-step2-root outputs/dc/02_physics_weight_sweep \
  --harmonic-number 1 \
  --output-dir outputs/ac/00_ideal_models/harmonic_stage1_admittance_model_comparison/H1 \
  --overwrite
```

If `--b1-run-dir` is not provided, the script will infer the balanced DC Step 2
representative from `outputs/dc/02_physics_weight_sweep/representative_configurations.csv`.

### Step 0A: Distensibility Summary Sweep

Scripts:

- `scripts/python/run_ac_distensibility_sweep.py`
- `scripts/python/plot_distensibility_sweep.py`
- `scripts/python/plot_distensibility_field.py`

Example:

```bash
python scripts/python/run_ac_distensibility_sweep.py \
  --graph-path datasets/harmonized_scaled_dataset.gpickle \
  --scratch-root outputs/ac/00_ideal_models/distensibility_sweep/_raw_runs \
  --output-root outputs/ac/00_ideal_models/distensibility_sweep \
  --plot-after
```

Analysis:

```bash
python scripts/python/run_ac_distensibility_sweep.py \
  --graph-path datasets/harmonized_scaled_dataset.gpickle \
  --scratch-root outputs/ac/00_ideal_models/distensibility_sweep/_raw_runs \
  --output-root outputs/ac/00_ideal_models/distensibility_sweep \
  --harmonic-number 1 \
  --aggregate-only
```

```bash
python scripts/python/run_ac_distensibility_sweep.py \
  --graph-path datasets/harmonized_scaled_dataset.gpickle \
  --scratch-root outputs/ac/00_ideal_models/distensibility_sweep/_raw_runs \
  --output-root outputs/ac/00_ideal_models/distensibility_sweep \
  --harmonic-number 2 \
  --aggregate-only
```

Plotting:

```bash
python scripts/python/plot_distensibility_sweep.py \
  --input outputs/ac/00_ideal_models/distensibility_sweep/distensibility_sweep_metrics.csv \
  --output-dir outputs/ac/00_ideal_models/distensibility_sweep/figures
```

Notes:

- The Step 0A summary plots use smooth curves and label each distensibility profile by `\(\alpha\)` and `\(D_0\)`.

### Step 1: Boundary Parameter Calibration

Scripts:

- `scripts/python/run_ac_boundary_parameter_calibration.py`
- `scripts/python/analyze_ac_boundary_parameter_calibration.py`
- `scripts/python/plot_ac_boundary_parameter_calibration.py`

Example:

```bash
python scripts/python/run_ac_boundary_parameter_calibration.py \
  --graph-path datasets/harmonized_scaled_dataset.gpickle \
  --output-root outputs/ac/01_boundary_parameter_calibration \
  --aggregate-after \
  --plot-after
```

### Step 2: Physics Weight Sweep

Scripts:

- `scripts/python/run_ac_physics_weight_sweep.py`
- `scripts/python/analyze_ac_physics_weight_sweep.py`
- `scripts/python/plot_ac_physics_weight_sweep.py`

Example:

```bash
python scripts/python/run_ac_physics_weight_sweep.py \
  --graph-path datasets/harmonized_scaled_dataset.gpickle \
  --output-root outputs/ac/02_physics_weight_sweep \
  --aggregate-after \
  --plot-after
```

Analysis:

```bash
python scripts/python/analyze_ac_physics_weight_sweep.py \
  --input-root outputs/ac/02_physics_weight_sweep/H1
```

```bash
python scripts/python/analyze_ac_physics_weight_sweep.py \
  --input-root outputs/ac/02_physics_weight_sweep/H2
```

Plotting:

```bash
python scripts/python/plot_ac_physics_weight_sweep.py \
  --input-root outputs/ac/02_physics_weight_sweep/H1 \
  --output-dir outputs/ac/02_physics_weight_sweep/H1/figures
```

```bash
python scripts/python/plot_ac_physics_weight_sweep.py \
  --input-root outputs/ac/02_physics_weight_sweep/H2 \
  --output-dir outputs/ac/02_physics_weight_sweep/H2/figures
```

Notes:

- The example AC Step 2 sweep is typically run with `\(\lambda_B = 100\)` for both `H1` and `H2`.
- Based on the current example outputs, the best overall selected configuration for `H1` is `Taylor DC Transferred`, `B1`: `q_0p1__k_0p1`, with `\(\lambda_Q = 0.1\)`, `\(\lambda_K = 0.1\)`, and `\(\lambda_B = 100\)`.
- Based on the current example outputs, the best overall selected configuration for `H2` is also `Taylor DC Transferred`, `B1`: `q_0p1__k_0p1`, with `\(\lambda_Q = 0.1\)`, `\(\lambda_K = 0.1\)`, and `\(\lambda_B = 100\)`.

### Step 3: Distensibility-Alpha Profiles

Scripts:

- `scripts/python/run_ac_distensibility_alpha_profiles.py`
- `scripts/python/run_ac_distensibility_profile_task.py`
- `scripts/python/analyze_ac_distensibility_alpha_profiles.py`
- `scripts/python/plot_ac_distensibility_alpha_profiles.py`
- `scripts/python/plot_ac_representative_fields.py`

Example:

```bash
python scripts/python/run_ac_distensibility_alpha_profiles.py \
  --graph-path datasets/harmonized_scaled_dataset.gpickle \
  --output-root outputs/ac/03_distensibility_alpha_profiles \
  --aggregate-after \
  --plot-after
```

Example `sbatch` sweep for `H1` using the selected `F1`, `B1`, and `K1`
representatives with at most `8` concurrent GPUs:

```bash
sbatch \
  --job-name=ac_step3_H1_FBK_dist \
  --partition=gpu \
  --gpus=1 \
  --cpus-per-task=4 \
  --mem=24G \
  --time=24:00:00 \
  --array=0-458%8 \
  --output=logs/ac_step3_H1_FBK_dist.%A_%a.out \
  --wrap="bash -lc '
    cd /mnt/home/sswee/yolk/Yolk-Sac-Flow-Data-and-Sim/gnn_clean
    conda run -n yolk-sac python scripts/python/run_ac_distensibility_alpha_profiles.py \
      --graph-path /mnt/home/sswee/yolk/Yolk-Sac-Flow-Data-and-Sim/gnn_clean/datasets/harmonized_scaled_dataset.gpickle \
      --step2-root outputs/ac/02_physics_weight_sweep \
      --output-root outputs/ac/03_distensibility_alpha_profiles \
      --harmonic-numbers 1 \
      --representative-labels F1 B1 K1 \
      --num-shards 459 \
      --shard-index \${SLURM_ARRAY_TASK_ID} \
      --overwrite
  '"
```

Example `sbatch` sweep for `H2` using the selected `F1`, `B1`, and `K1`
representatives with at most `8` concurrent GPUs:

```bash
sbatch \
  --job-name=ac_step3_H2_FBK_dist \
  --partition=gpu \
  --gpus=1 \
  --cpus-per-task=4 \
  --mem=24G \
  --time=24:00:00 \
  --array=0-458%8 \
  --output=logs/ac_step3_H2_FBK_dist.%A_%a.out \
  --wrap="bash -lc '
    cd /mnt/home/sswee/yolk/Yolk-Sac-Flow-Data-and-Sim/gnn_clean
    conda run -n yolk-sac python scripts/python/run_ac_distensibility_alpha_profiles.py \
      --graph-path /mnt/home/sswee/yolk/Yolk-Sac-Flow-Data-and-Sim/gnn_clean/datasets/harmonized_scaled_dataset.gpickle \
      --step2-root outputs/ac/02_physics_weight_sweep \
      --output-root outputs/ac/03_distensibility_alpha_profiles \
      --harmonic-numbers 2 \
      --representative-labels F1 B1 K1 \
      --num-shards 459 \
      --shard-index \${SLURM_ARRAY_TASK_ID} \
      --overwrite
  '"
```

Analysis:

```bash
python scripts/python/analyze_ac_distensibility_alpha_profiles.py \
  --input-root outputs/ac/03_distensibility_alpha_profiles/H1
```

```bash
python scripts/python/analyze_ac_distensibility_alpha_profiles.py \
  --input-root outputs/ac/03_distensibility_alpha_profiles/H2
```

Plotting:

```bash
python scripts/python/plot_ac_distensibility_alpha_profiles.py \
  --input-root outputs/ac/03_distensibility_alpha_profiles/H1
```

```bash
python scripts/python/plot_ac_distensibility_alpha_profiles.py \
  --input-root outputs/ac/03_distensibility_alpha_profiles/H2
```

```bash
python scripts/python/plot_ac_representative_fields.py \
  --input-root outputs/ac/03_distensibility_alpha_profiles/H1
```

Notes:

- Step 3 sweeps the distensibility parameter `\(D_0\)` over `51` log-spaced values
  from `\(10^{-6}\)` through `\(10^{-1}\)`.
- For each `\(D_0\)` value, the sweep also evaluates `\(\alpha \in \{0, 1, 2\}\)`,
  so each representative contributes `153` runs.
- Using `F1`, `B1`, and `K1` therefore gives `459` runs per harmonic, which is why
  the example `sbatch` arrays use `0-458`.
- The selected representative label determines which Step 2 configuration supplies
  `\(\lambda_Q\)`, `\(\lambda_K\)`, and `\(\lambda_B\)` to each Step 3 run.
- Outputs are organized under
  `outputs/ac/03_distensibility_alpha_profiles/H*/<label>/alpha_<alpha>/D0_<token>/`.
- The representative-field plotting command above reads the `Taylor Ideal` rows from
  `outputs/ac/03_distensibility_alpha_profiles/H1/representative_configurations.csv`
  and writes flow-amplitude, flow-field, pressure-field, and pressure-phase-field
  panels under
  `outputs/ac/03_distensibility_alpha_profiles/H1/figures/representative_fields/taylor_ideal`.

## Notes

- Most sweep runners support `--num-shards` and `--shard-index` for parallel execution.
- Use `--dry-run` to inspect generated commands before launching long sweeps.
- AC Step 0 and the AC sweep runners can infer the balanced DC Step 2 representative from `outputs/dc/02_physics_weight_sweep/representative_configurations.csv`; use `--b1-run-dir` if you want to override that choice.
- Plotting and analysis scripts expect the same summary CSV/YAML structure produced by the paired run scripts in this repo.
