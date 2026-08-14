#!/usr/bin/env bash
#SBATCH --job-name=ac_step00_somite21_gpu
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=24:00:00
#SBATCH --array=0-152%8
#SBATCH --output=logs/ac_step00_somite21_gpu.%A_%a.out

set -euo pipefail

cd /mnt/home/sswee/yolk/Yolk-Sac-Flow-Data-and-Sim/gnn_clean
mkdir -p logs

conda run -n yolk-sac python scripts/ac_workflow/solver/00_ideal_models.py \
  --graph-path datasets/somite21_mosaic_cut_pipeline_ready.gpickle \
  --dc-step2-root outputs/somite21/dc/02_physics_weight_sweep \
  --output-root outputs/somite21/ac/00_ideal_models/distensibility_sweep \
  --scratch-root outputs/somite21/ac/00_ideal_models/distensibility_sweep/_raw_runs \
  --device cuda \
  --lstsq-backend torch \
  --require-cuda \
  --num-shards 153 \
  --shard-index "${SLURM_ARRAY_TASK_ID}" \
  "$@"
