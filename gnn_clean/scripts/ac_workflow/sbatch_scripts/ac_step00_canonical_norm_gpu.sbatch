#!/usr/bin/env bash
#SBATCH --job-name=ac_step00_canonical_norm_gpu
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=24:00:00
#SBATCH --array=0-152%8
#SBATCH --output=logs/ac_step00_canonical_norm_gpu.%A_%a.out

set -euo pipefail

cd /mnt/home/sswee/yolk/Yolk-Sac-Flow-Data-and-Sim/gnn_clean
mkdir -p logs

conda run -n yolk-sac python scripts/ac_workflow/solver/00_ideal_models.py \
  --graph-path datasets/mosaic_graph_norm_canonical.gpickle \
  --dc-step2-root outputs_canonical_norm/dc/02_physics_weight_sweep \
  --output-root outputs_canonical_norm/ac/00_ideal_models/distensibility_sweep \
  --scratch-root outputs_canonical_norm/ac/00_ideal_models/distensibility_sweep/_raw_runs \
  --device cuda \
  --lstsq-backend torch \
  --require-cuda \
  --arterial-boundary-mode per_tip_highest_snr \
  --venous-boundary-mode rebalance_to_sources \
  --num-shards 153 \
  --shard-index "${SLURM_ARRAY_TASK_ID}" \
  "$@"
