#!/usr/bin/env bash
#SBATCH --job-name=ac_step2_canonical_norm
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=12:00:00
#SBATCH --array=0-31
#SBATCH --output=logs/ac_step2_canonical_norm.%A_%a.out

set -euo pipefail

cd /mnt/home/sswee/yolk/Yolk-Sac-Flow-Data-and-Sim/gnn_clean
mkdir -p logs

LAMBDA_B=100.0
echo "[info] AC Step 2 canonical sweep uses fixed lambda_B=${LAMBDA_B} for both H1 and H2."

conda run -n yolk-sac python scripts/python/run_ac_physics_weight_sweep.py \
  --graph-path datasets/mosaic_graph_norm_canonical.gpickle \
  --dc-step2-root outputs_canonical_norm/dc/02_physics_weight_sweep \
  --output-root outputs_canonical_norm/ac/02_physics_weight_sweep \
  --lambda-b-override "${LAMBDA_B}" \
  --arterial-boundary-mode per_tip_highest_snr \
  --venous-boundary-mode rebalance_to_sources \
  --num-shards 32 \
  --shard-index "${SLURM_ARRAY_TASK_ID}"
