#!/bin/bash
#SBATCH --job-name=seg_21_to_27
#SBATCH --partition=gpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=/mnt/home/sswee/yolk/Yolk-Sac-Flow-Data-and-Sim/segment_stitch/scripts/seg_21_to_27-%j.out
#SBATCH --error=/mnt/home/sswee/yolk/Yolk-Sac-Flow-Data-and-Sim/segment_stitch/scripts/seg_21_to_27-%j.err

set -euo pipefail

source /mnt/home/sswee/miniforge3/etc/profile.d/conda.sh
conda activate yolk-sac
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"

bash /mnt/home/sswee/yolk/Yolk-Sac-Flow-Data-and-Sim/segment_stitch/scripts/run_somites21_train_somites27_test.sh
