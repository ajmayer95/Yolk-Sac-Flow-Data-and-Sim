#!/usr/bin/env bash
#SBATCH --job-name=ac_step3_somite21_agg_cpu
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=08:00:00
#SBATCH --output=logs/ac_step3_somite21_agg_cpu.%j.out

# CPU aggregation job: summarize and plot existing sweep outputs only.

set -euo pipefail

cd /mnt/home/sswee/yolk/Yolk-Sac-Flow-Data-and-Sim/gnn_clean
mkdir -p logs

conda run -n yolk-sac python scripts/python/run_ac_distensibility_alpha_profiles.py \
  --graph-path datasets/somite21_mosaic_cut_pipeline_ready.gpickle \
  --step2-root outputs/somite21/ac/02_physics_weight_sweep \
  --output-root outputs/somite21/ac/03_distensibility_alpha_profiles \
  --aggregate-only \
  --plot-after

conda run -n yolk-sac python scripts/python/plot_ac_representative_fields.py \
  --input-root outputs/somite21/ac/03_distensibility_alpha_profiles/H1

conda run -n yolk-sac python scripts/python/plot_ac_representative_fields.py \
  --input-root outputs/somite21/ac/03_distensibility_alpha_profiles/H2
