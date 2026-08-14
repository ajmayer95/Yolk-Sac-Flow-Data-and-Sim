#!/usr/bin/env bash
#SBATCH --job-name=ac_step02_harmonized_gpu
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=12:00:00
#SBATCH --array=0-31
#SBATCH --output=logs/ac_step02_harmonized_gpu.%A_%a.out

set -euo pipefail

cd /mnt/home/sswee/yolk/Yolk-Sac-Flow-Data-and-Sim/gnn_clean
mkdir -p logs

LAMBDA_B=100.0
echo "[info] AC Step 2 harmonized sweep uses fixed lambda_B=${LAMBDA_B} for both H1 and H2."

conda run -n yolk-sac python scripts/ac_workflow/solver/02_physics_weight_sweep.py \
  --graph-path datasets_upload/harmonized_scaled_dataset.gpickle \
  --dc-step2-root outputs_harmonized/dc/02_physics_weight_sweep \
  --output-root outputs_harmonized/ac/02_physics_weight_sweep_q10_k10_delta0p1 \
  --lambda-b-override "${LAMBDA_B}" \
  --device cuda \
  --lstsq-backend torch \
  --require-cuda \
  --arterial-boundary-mode all \
  --venous-boundary-mode observed \
  --num-shards 32 \
  --shard-index "${SLURM_ARRAY_TASK_ID}" \
  "$@"
