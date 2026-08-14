#!/usr/bin/env bash
#SBATCH --job-name=ac_step03_H2_canonical_gpu
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=24:00:00
#SBATCH --array=0-458%8
#SBATCH --output=logs/ac_step03_H2_canonical_gpu.%A_%a.out

set -euo pipefail

cd /mnt/home/sswee/yolk/Yolk-Sac-Flow-Data-and-Sim/gnn_clean
mkdir -p logs

conda run -n yolk-sac python scripts/ac_workflow/solver/03_distensibility_alpha_profiles.py \
  --graph-path datasets/mosaic_graph_canonical_harmonized/mosaic_graph_canonical_harmonized.gpickle \
  --step2-root outputs_canonical/ac/02_physics_weight_sweep \
  --output-root outputs_canonical/ac/03_distensibility_alpha_profiles \
  --device cuda \
  --lstsq-backend torch \
  --require-cuda \
  --harmonic-numbers 2 \
  --representative-labels F1 B1 K1 \
  --arterial-boundary-mode per_tip_highest_snr \
  --venous-boundary-mode rebalance_to_sources \
  --num-shards 459 \
  --shard-index "${SLURM_ARRAY_TASK_ID}" \
  --overwrite \
  "$@"
