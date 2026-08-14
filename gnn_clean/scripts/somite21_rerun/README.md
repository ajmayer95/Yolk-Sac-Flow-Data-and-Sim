# Somite21 Rerun Commands

This directory packages the somite21 rerun command set into runnable repo
artifacts.

Inputs and outputs:

- Graph: `datasets/somite21_mosaic_cut_pipeline_ready.gpickle`
- DC outputs: `outputs/somite21/dc`
- AC outputs: `outputs/somite21/ac`
- Publish root: `publish/somite21`

Main entrypoints:

- Local Python workflow: `scripts/somite21_rerun/run_python_workflow.sh`
- Cluster jobs: `scripts/somite21_rerun/sbatch/*.sbatch`

Notes:

- Run the DC steps before the AC steps.
- AC jobs read the somite21 DC Step 2 representative from
  `outputs/somite21/dc/02_physics_weight_sweep`.
- The `sbatch` files assume the `yolk-sac` conda environment is available.
- The `sbatch` files write logs under `logs/`.
- GPU compute jobs now pass explicit CUDA requirements into the Python entrypoints.
- `_agg` jobs are intentionally CPU-only aggregation and plotting stages.
