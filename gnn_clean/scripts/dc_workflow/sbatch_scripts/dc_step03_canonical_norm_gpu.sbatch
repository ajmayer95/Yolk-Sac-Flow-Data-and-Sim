#!/usr/bin/env bash
#SBATCH --job-name=dc_step03_canonical_norm_gpu
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=24:00:00
#SBATCH --array=0-31
#SBATCH --output=logs/dc_step03_canonical_norm_gpu.%A_%a.out

set -euo pipefail

cd /mnt/home/sswee/yolk/Yolk-Sac-Flow-Data-and-Sim/gnn_clean
mkdir -p logs

conda run -n yolk-sac python scripts/dc_workflow/solver/03_pressure_constraint_sensitivity.py \
  --representative-csv outputs_canonical_norm/dc/02_physics_weight_sweep/representative_configurations.csv \
  --graph datasets/mosaic_graph_norm_canonical.gpickle \
  --output-root outputs_canonical_norm/dc/03_pressure_constraint_sensitivity \
  --device cuda \
  --require-cuda \
  --num-shards 32 \
  --shard-index "${SLURM_ARRAY_TASK_ID}" \
  "$@"
