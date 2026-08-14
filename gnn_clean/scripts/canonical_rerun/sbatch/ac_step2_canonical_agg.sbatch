#!/usr/bin/env bash
#SBATCH --job-name=ac_step2_canonical_agg
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --output=logs/ac_step2_canonical_agg.%j.out

set -euo pipefail

cd /mnt/home/sswee/yolk/Yolk-Sac-Flow-Data-and-Sim/gnn_clean
mkdir -p logs

LAMBDA_B=100.0
echo "[info] AC Step 2 canonical aggregation expects runs generated with fixed lambda_B=${LAMBDA_B} for both H1 and H2."

conda run -n yolk-sac python scripts/python/run_ac_physics_weight_sweep.py \
  --graph-path datasets/mosaic_graph_canonical_harmonized/mosaic_graph_canonical_harmonized.gpickle \
  --dc-step2-root outputs_canonical/dc/02_physics_weight_sweep \
  --output-root outputs_canonical/ac/02_physics_weight_sweep \
  --lambda-b-override "${LAMBDA_B}" \
  --arterial-boundary-mode per_tip_highest_snr \
  --venous-boundary-mode rebalance_to_sources \
  --aggregate-only \
  --plot-after
