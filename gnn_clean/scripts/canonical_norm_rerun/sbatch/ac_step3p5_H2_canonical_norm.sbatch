#!/usr/bin/env bash
#SBATCH --job-name=ac_step3p5_H2_canonical
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=24:00:00
#SBATCH --array=0-215%8
#SBATCH --output=logs/ac_step3p5_H2_canonical_norm.%A_%a.out

set -euo pipefail

cd /mnt/home/sswee/yolk/Yolk-Sac-Flow-Data-and-Sim/gnn_clean
mkdir -p logs

conda run -n yolk-sac python scripts/python/run_ac_distensibility_alpha_profiles.py \
  --graph-path datasets/mosaic_graph_norm_canonical.gpickle \
  --step2-root outputs_canonical_norm/ac/02_physics_weight_sweep \
  --output-root outputs_canonical_norm/ac/03p5_sparse_distensibility_alpha_profiles \
  --harmonic-numbers 2 \
  --representative-labels F1 F2 F3 F4 B1 B2 B3 B4 K1 K2 K3 K4 \
  --alpha-values 0 1 2 \
  --d0-values 1e-6 1e-5 1e-4 1e-3 1e-2 1e-1 \
  --arterial-boundary-mode per_tip_highest_snr \
  --venous-boundary-mode rebalance_to_sources \
  --num-shards 216 \
  --shard-index "${SLURM_ARRAY_TASK_ID}" \
  --overwrite
