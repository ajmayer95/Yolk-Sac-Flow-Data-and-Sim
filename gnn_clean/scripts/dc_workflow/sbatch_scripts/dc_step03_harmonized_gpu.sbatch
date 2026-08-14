#!/usr/bin/env bash
#SBATCH --job-name=dc_step03_harmonized_gpu
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=24:00:00
#SBATCH --array=0-7
#SBATCH --output=logs/dc_step03_harmonized_gpu.%A_%a.out

set -euo pipefail

cd /mnt/home/sswee/yolk/Yolk-Sac-Flow-Data-and-Sim/gnn_clean
mkdir -p logs

conda run -n yolk-sac python scripts/dc_workflow/solver/03_pressure_constraint_sensitivity.py \
  --graph datasets_upload/harmonized_scaled_dataset.gpickle \
  --representative-csv outputs_harmonized/dc/02_physics_weight_sweep/representative_configurations.csv \
  --representative-labels-csv outputs_harmonized/dc/02_physics_weight_sweep/figures/representative_plot_labels.csv \
  --output-root outputs_harmonized/dc/03_pressure_constraint_sensitivity \
  --device cuda \
  --require-cuda \
  --num-shards 8 \
  --shard-index "${SLURM_ARRAY_TASK_ID}" \
  "$@"
