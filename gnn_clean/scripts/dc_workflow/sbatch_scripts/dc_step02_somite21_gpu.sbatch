#!/usr/bin/env bash
#SBATCH --job-name=dc_step02_somite21_gpu
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=24:00:00
#SBATCH --array=0-79
#SBATCH --output=logs/dc_step02_somite21_gpu.%A_%a.out

# GPU compute job: GNN runs must fail fast if CUDA is unavailable.

set -euo pipefail

cd /mnt/home/sswee/yolk/Yolk-Sac-Flow-Data-and-Sim/gnn_clean
mkdir -p logs

conda run -n yolk-sac python scripts/dc_workflow/solver/02_physics_weight_sweep.py \
  --graph datasets/somite21_mosaic_cut_pipeline_ready.gpickle \
  --output-root outputs/somite21/dc/02_physics_weight_sweep \
  --device cuda \
  --require-cuda \
  --num-shards 80 \
  --shard-index "${SLURM_ARRAY_TASK_ID}"
