#!/bin/bash
#SBATCH --job-name=synthetic_gnn_array
#SBATCH --partition=gpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --array=0-35%10
#SBATCH --output=/mnt/home/sswee/yolk/Yolk-Sac-Flow-Data-and-Sim/Somites21_demo/PerTileFlow/synthetic/scripts/logs/synthetic_gnn_array-%A_%a.out
#SBATCH --error=/mnt/home/sswee/yolk/Yolk-Sac-Flow-Data-and-Sim/Somites21_demo/PerTileFlow/synthetic/scripts/logs/synthetic_gnn_array-%A_%a.err

set -euo pipefail

SYN_DIR="/mnt/home/sswee/yolk/Yolk-Sac-Flow-Data-and-Sim/Somites21_demo/PerTileFlow/synthetic"
MANIFEST="${SYN_DIR}/data/synthetic/manifest.csv"
SCRIPT="${SYN_DIR}/scripts/run_gnn_grid.py"
DEFAULT_CONFIG="${SYN_DIR}/configs/gnn_experiments.yaml"

CONFIG="${CONFIG:-${DEFAULT_CONFIG}}"
DEVICE="${DEVICE:-auto}"
EPOCHS="${EPOCHS:-}"
# Existing completed physics-GNN/MLP runs are retained by default. The new
# pressure-decoder GCN has a distinct run suffix and will therefore still run.
SKIP_EXISTING="${SKIP_EXISTING:-true}"
MODEL="${MODEL:-}"
K_VALUES="${K_VALUES:-}"
HARMONIC_MODE="${HARMONIC_MODE:-}"
DRY_RUN="${DRY_RUN:-false}"

source /mnt/home/sswee/miniforge3/etc/profile.d/conda.sh
conda activate yolk-sac
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"

cd "${SYN_DIR}"
export PYTHONPATH="${SYN_DIR}/src:${PYTHONPATH:-}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"

if [[ -z "${SLURM_ARRAY_TASK_ID:-}" ]]; then
  echo "SLURM_ARRAY_TASK_ID not set; exiting"
  exit 1
fi

# Select dataset filename from manifest (skip header). Array index 0 -> first data line.
LINE=$(awk -v idx=${SLURM_ARRAY_TASK_ID} 'NR==idx+2 {print; exit}' "${MANIFEST}")
if [[ -z "${LINE}" ]]; then
  echo "No line found for task ${SLURM_ARRAY_TASK_ID} in ${MANIFEST}" >&2
  exit 1
fi
FILENAME=$(echo "${LINE}" | cut -d, -f1)
DATASET="${SYN_DIR}/data/synthetic/${FILENAME}"

echo "Running on host: $(hostname)"
echo "Start time: $(date)"
echo "Dataset: ${DATASET}"
echo "Config: ${CONFIG}"
echo "Device: ${DEVICE}"
echo "Model filter: ${MODEL:-all}"
echo "K filter: ${K_VALUES:-all}"
echo "Harmonic filter: ${HARMONIC_MODE:-all}"
echo "Skip existing: ${SKIP_EXISTING}"
echo "Dry run: ${DRY_RUN}"
echo "Python: $(which python)"
nvidia-smi || true

cmd=("$(which python)" "${SCRIPT}" "--config" "${CONFIG}" "--datasets" "${DATASET}")
if [[ -n "${DEVICE}" ]]; then
  cmd+=("--device" "${DEVICE}")
fi
if [[ -n "${EPOCHS}" ]]; then
  cmd+=("--epochs" "${EPOCHS}")
fi
if [[ "${SKIP_EXISTING}" == "true" ]]; then
  cmd+=("--skip-existing")
fi
if [[ -n "${MODEL}" ]]; then
  cmd+=("--model" "${MODEL}")
fi
if [[ -n "${K_VALUES}" ]]; then
  read -r -a K_ARRAY <<< "${K_VALUES}"
  cmd+=("--K" "${K_ARRAY[@]}")
fi
if [[ -n "${HARMONIC_MODE}" ]]; then
  cmd+=("--harmonic-mode" "${HARMONIC_MODE}")
fi
if [[ "${DRY_RUN}" == "true" ]]; then
  cmd+=("--dry-run")
fi

echo "Command: ${cmd[*]}"
"${cmd[@]}"

echo "End time: $(date)"
