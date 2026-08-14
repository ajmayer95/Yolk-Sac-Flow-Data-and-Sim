#!/usr/bin/env bash
#SBATCH --job-name=ac_step3p5_B234_agg
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=08:00:00
#SBATCH --output=logs/ac_step3p5_B2_B3_B4_canonical_agg.%j.out

set -euo pipefail

cd /mnt/home/sswee/yolk/Yolk-Sac-Flow-Data-and-Sim/gnn_clean
mkdir -p logs

conda run -n yolk-sac python scripts/python/run_ac_distensibility_alpha_profiles.py \
  --graph-path datasets/mosaic_graph_canonical_harmonized/mosaic_graph_canonical_harmonized.gpickle \
  --step2-root outputs_canonical/ac/02_physics_weight_sweep \
  --output-root outputs_canonical/ac/03p5_sparse_distensibility_alpha_profiles_B2_B3_B4 \
  --harmonic-numbers 1 2 \
  --representative-labels B2 B3 B4 \
  --alpha-values 0 1 2 \
  --d0-values 1e-6 1e-5 1e-4 1e-3 1e-2 1e-1 \
  --arterial-boundary-mode per_tip_highest_snr \
  --venous-boundary-mode rebalance_to_sources \
  --aggregate-only \
  --plot-after
