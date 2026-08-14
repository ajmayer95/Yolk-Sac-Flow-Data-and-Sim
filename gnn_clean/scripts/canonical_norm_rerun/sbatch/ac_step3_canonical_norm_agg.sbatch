#!/usr/bin/env bash
#SBATCH --job-name=ac_step3_canonical_norm_agg
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=08:00:00
#SBATCH --output=logs/ac_step3_canonical_norm_agg.%j.out

set -euo pipefail

cd /mnt/home/sswee/yolk/Yolk-Sac-Flow-Data-and-Sim/gnn_clean
mkdir -p logs

conda run -n yolk-sac python scripts/python/run_ac_distensibility_alpha_profiles.py \
  --graph-path datasets/mosaic_graph_norm_canonical.gpickle \
  --step2-root outputs_canonical_norm/ac/02_physics_weight_sweep \
  --output-root outputs_canonical_norm/ac/03_distensibility_alpha_profiles \
  --arterial-boundary-mode per_tip_highest_snr \
  --venous-boundary-mode rebalance_to_sources \
  --aggregate-only \
  --plot-after

conda run -n yolk-sac python scripts/python/plot_ac_representative_fields.py \
  --input-root outputs_canonical_norm/ac/03_distensibility_alpha_profiles/H1

conda run -n yolk-sac python scripts/python/plot_ac_representative_fields.py \
  --input-root outputs_canonical_norm/ac/03_distensibility_alpha_profiles/H2
