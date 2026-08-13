#!/usr/bin/env bash
#SBATCH --job-name=dc_step02_canonical_norm_gpu
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=24:00:00
#SBATCH --array=0-79
#SBATCH --output=logs/dc_step02_canonical_norm_gpu.%A_%a.out

set -euo pipefail

cd /mnt/home/sswee/yolk/Yolk-Sac-Flow-Data-and-Sim/gnn_clean
mkdir -p logs

LAMBDA_B=100.0
echo "[info] DC Step 02 canonical-norm sweep uses fixed lambda_B=${LAMBDA_B} for both the outer boundary loss and the reduced pressure-solver boundary constraint."

conda run -n yolk-sac python scripts/dc_workflow/solver/02_physics_weight_sweep.py \
  --graph datasets/mosaic_graph_norm_canonical.gpickle \
  --output-root outputs_canonical_norm/dc/02_physics_weight_sweep \
  --device cuda \
  --require-cuda \
  --num-shards 80 \
  --shard-index "${SLURM_ARRAY_TASK_ID}"
