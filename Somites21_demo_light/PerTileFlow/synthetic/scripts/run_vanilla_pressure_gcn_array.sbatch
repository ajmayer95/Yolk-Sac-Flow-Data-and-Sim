#!/bin/bash
# Temporary rerun of only the pressure-decoding vanilla GCN grid.
#
# One array task owns one dataset and trains:
#   K = 1, 2, 3, 4
#   dc_only, dc_h1, dc_h1_h2
# for 12 runs per dataset. Existing pressure-decoder runs are skipped, making
# resubmission safe at the completed-run level.
#
# Submit:
#   sbatch scripts/run_vanilla_pressure_gcn_array.sbatch
#
# Preview task zero:
#   SLURM_ARRAY_TASK_ID=0 DRY_RUN=true \
#     bash scripts/run_vanilla_pressure_gcn_array.sbatch

#SBATCH --job-name=pressure_gcn
#SBATCH --partition=gpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --array=0-35%10
#SBATCH --output=/mnt/home/sswee/yolk/Yolk-Sac-Flow-Data-and-Sim/Somites21_demo/PerTileFlow/synthetic/scripts/logs/pressure_gcn-%A_%a.out
#SBATCH --error=/mnt/home/sswee/yolk/Yolk-Sac-Flow-Data-and-Sim/Somites21_demo/PerTileFlow/synthetic/scripts/logs/pressure_gcn-%A_%a.err

set -euo pipefail

SYN_DIR="/mnt/home/sswee/yolk/Yolk-Sac-Flow-Data-and-Sim/Somites21_demo/PerTileFlow/synthetic"
MANIFEST="${SYN_DIR}/data/synthetic/manifest.csv"
SCRIPT="${SYN_DIR}/scripts/run_gnn_grid.py"
CONFIG="${CONFIG:-${SYN_DIR}/configs/gnn_experiments.yaml}"
DEVICE="${DEVICE:-auto}"
EPOCHS="${EPOCHS:-}"
SKIP_EXISTING="${SKIP_EXISTING:-true}"
DRY_RUN="${DRY_RUN:-false}"

source /mnt/home/sswee/miniforge3/etc/profile.d/conda.sh
conda activate yolk-sac
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"

cd "${SYN_DIR}"
export PYTHONPATH="${SYN_DIR}/src:${PYTHONPATH:-}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"

if [[ -z "${SLURM_ARRAY_TASK_ID:-}" ]]; then
  echo "SLURM_ARRAY_TASK_ID not set; exiting" >&2
  exit 1
fi

LINE="$(awk -v idx="${SLURM_ARRAY_TASK_ID}" 'NR==idx+2 {print; exit}' "${MANIFEST}")"
if [[ -z "${LINE}" ]]; then
  echo "No dataset for array task ${SLURM_ARRAY_TASK_ID}" >&2
  exit 1
fi
FILENAME="$(echo "${LINE}" | cut -d, -f1)"
DATASET="${SYN_DIR}/data/synthetic/${FILENAME}"

COMMAND=(
  "$(which python)"
  "${SCRIPT}"
  --config "${CONFIG}"
  --datasets "${DATASET}"
  --model vanilla_gcn
  --K 1 2 3 4
  --device "${DEVICE}"
)
if [[ -n "${EPOCHS}" ]]; then
  COMMAND+=(--epochs "${EPOCHS}")
fi
if [[ "${SKIP_EXISTING}" == "true" ]]; then
  COMMAND+=(--skip-existing)
fi
if [[ "${DRY_RUN}" == "true" ]]; then
  COMMAND+=(--dry-run)
fi

echo "Host: $(hostname)"
echo "Start: $(date --iso-8601=seconds)"
echo "Dataset: ${DATASET}"
echo "Command: ${COMMAND[*]}"
nvidia-smi || true
"${COMMAND[@]}"
echo "End: $(date --iso-8601=seconds)"
