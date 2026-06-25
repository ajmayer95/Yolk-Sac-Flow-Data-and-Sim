#!/bin/bash
#SBATCH --job-name=synthetic_gnn
#SBATCH --partition=gpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
# Request two GPUs since per-run GPU memory is very small (~1% or less)
#SBATCH --gres=gpu:2
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=/mnt/home/sswee/yolk/Yolk-Sac-Flow-Data-and-Sim/Somites21_demo/PerTileFlow/synthetic/scripts/logs/synthetic_gnn-%j.out
#SBATCH --error=/mnt/home/sswee/yolk/Yolk-Sac-Flow-Data-and-Sim/Somites21_demo/PerTileFlow/synthetic/scripts/logs/synthetic_gnn-%j.err

set -euo pipefail

SYN_DIR="/mnt/home/sswee/yolk/Yolk-Sac-Flow-Data-and-Sim/Somites21_demo/PerTileFlow/synthetic"
PROJECT_ROOT="${SYN_DIR}"
SCRIPT="${PROJECT_ROOT}/scripts/run_gnn_grid.py"
DEFAULT_CONFIG="${PROJECT_ROOT}/configs/gnn_experiments.yaml"

# Overrideable environment variables:
#  CONFIG: path to experiment config yaml (default: configs/gnn_experiments.yaml)
#  DATASETS: space-separated list of dataset paths (optional)
#  DEVICE: training device (auto, cpu, cuda, mps)
#  EPOCHS: override number of epochs
#  SKIP_EXISTING: set to "true" to skip experiments with existing metrics
#  DRY_RUN: set to "true" to print commands without running

CONFIG="${CONFIG:-${DEFAULT_CONFIG}}"
DEVICE="${DEVICE:-auto}"
EPOCHS="${EPOCHS:-}"
SKIP_EXISTING="${SKIP_EXISTING:-false}"
DRY_RUN="${DRY_RUN:-false}"

source /mnt/home/sswee/miniforge3/etc/profile.d/conda.sh
conda activate yolk-sac
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"

cd "${PROJECT_ROOT}"
export PYTHONPATH="${PROJECT_ROOT}/src:${PYTHONPATH:-}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"

echo "Running on host: $(hostname)"
echo "Start time: $(date)"
echo "Config: ${CONFIG}"
echo "Device: ${DEVICE}"
echo "Dry run: ${DRY_RUN}"
echo "Python: $(which python)"
nvidia-smi || true

python - <<'PY'
import torch
print("torch:", torch.__version__)
print("cuda available:", torch.cuda.is_available())
print("cuda device count:", torch.cuda.device_count())
if torch.cuda.is_available():
    print("cuda device 0:", torch.cuda.get_device_name(0))
print("mps available:", getattr(torch.backends, 'mps', None) is not None and torch.backends.mps.is_available())
PY

# Build a list of per-experiment commands by running the grid script in dry-run mode
TMP_CMDS=$(mktemp)
DRYRUN_CMD=("$(which python)" "${SCRIPT}" "--config" "${CONFIG}" "--dry-run")
if [[ -n "${DEVICE}" ]]; then
  DRYRUN_CMD+=("--device" "${DEVICE}")
fi
if [[ -n "${EPOCHS}" ]]; then
  DRYRUN_CMD+=("--epochs" "${EPOCHS}")
fi
if [[ "${SKIP_EXISTING}" == "true" ]]; then
  DRYRUN_CMD+=("--skip-existing")
fi
if [[ -n "${DATASETS:-}" ]]; then
  for d in ${DATASETS}; do
    DRYRUN_CMD+=("--datasets" "$d")
  done
fi

echo "Generating command list with: ${DRYRUN_CMD[*]}"
"${DRYRUN_CMD[@]}" | sed -n 's/^\[[0-9]*\/\([0-9]*\)\] //p' > "${TMP_CMDS}"
NUM_CMDS=$(wc -l < "${TMP_CMDS}" || echo 0)
echo "Commands generated: ${NUM_CMDS}"

if [[ "${DRY_RUN}" == "true" ]]; then
  echo "Dry-run requested; command list saved to ${TMP_CMDS}" && exit 0
fi

# Determine number of GPUs available to this job
if [[ -n "${SLURM_GPUS_ON_NODE:-}" ]]; then
  NUM_GPUS=${SLURM_GPUS_ON_NODE}
elif [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  NUM_GPUS=$(echo "${CUDA_VISIBLE_DEVICES}" | awk -F, '{print NF}')
else
  NUM_GPUS=$(nvidia-smi -L 2>/dev/null | wc -l || echo 1)
fi
NUM_GPUS=${NUM_GPUS:-1}
if [[ "${NUM_GPUS}" -lt 1 ]]; then NUM_GPUS=1; fi

# Workers per GPU (tuneable)
WORKERS_PER_GPU=${WORKERS_PER_GPU:-8}
TOTAL_WORKERS=$((NUM_GPUS * WORKERS_PER_GPU))

echo "Dispatching up to ${TOTAL_WORKERS} concurrent workers across ${NUM_GPUS} GPUs"

# Ensure light-weight workers do not oversubscribe CPU threads
export OMP_NUM_THREADS=1

LOGDIR="${PROJECT_ROOT}/scripts/logs/parallel"
mkdir -p "${LOGDIR}"

i=0
while IFS= read -r LINE || [[ -n "$LINE" ]]; do
  gpu=$(( i % NUM_GPUS ))
  out="${LOGDIR}/run_${i}.out"
  err="${LOGDIR}/run_${i}.err"
  echo "[task ${i}] GPU=${gpu} -> ${LINE}"
  bash -c "CUDA_VISIBLE_DEVICES=${gpu} OMP_NUM_THREADS=1 ${LINE}" >"${out}" 2>"${err}" &
  ((i++))
  # throttle total concurrency
  while [ "$(jobs -p | wc -l)" -ge "${TOTAL_WORKERS}" ]; do
    sleep 1
  done
done < "${TMP_CMDS}"

wait
echo "All tasks finished. Logs in ${LOGDIR}."
echo "End time: $(date)"
