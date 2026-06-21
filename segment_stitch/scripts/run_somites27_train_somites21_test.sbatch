#!/bin/bash
#SBATCH --job-name=seg_27_to_21
#SBATCH --partition=gpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=/mnt/home/sswee/yolk/Yolk-Sac-Flow-Data-and-Sim/segment_stitch/scripts/seg_27_to_21-%j.out
#SBATCH --error=/mnt/home/sswee/yolk/Yolk-Sac-Flow-Data-and-Sim/segment_stitch/scripts/seg_27_to_21-%j.err

set -euo pipefail

source /mnt/home/sswee/miniforge3/etc/profile.d/conda.sh
conda activate yolk-sac
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"

bash /mnt/home/sswee/yolk/Yolk-Sac-Flow-Data-and-Sim/segment_stitch/scripts/run_somites27_train_somites21_test.sh
