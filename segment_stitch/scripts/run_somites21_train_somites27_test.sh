#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="/mnt/home/sswee/yolk/Yolk-Sac-Flow-Data-and-Sim"
TRAIN_ROOT="${TRAIN_ROOT:-/mnt/home/sswee/ceph/Somites21_demo}"
TEST_ROOT="${TEST_ROOT:-/mnt/home/sswee/ceph/Somites27_demo}"
TRAIN_LABELS="${TRAIN_LABELS:-}"
TEST_LABELS="${TEST_LABELS:-}"
WORK_DIR="${WORK_DIR:-${PROJECT_DIR}/segment_stitch/work}"
REPORT_DIR="${REPORT_DIR:-${PROJECT_DIR}/segment_stitch/outputs}"
EPOCHS="${EPOCHS:-50}"
BATCH_SIZE="${BATCH_SIZE:-8}"
NUM_WORKERS="${NUM_WORKERS:-4}"
LR="${LR:-1e-3}"
DEVICE="${DEVICE:-cuda}"
SEED="${SEED:-7}"
OVERWRITE="${OVERWRITE:-0}"

source "${PROJECT_DIR}/segment_stitch/scripts/run_workflow_common.sh"
