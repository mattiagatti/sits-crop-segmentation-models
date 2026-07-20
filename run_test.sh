#!/usr/bin/env bash
set -euo pipefail

DELAY="${1:-60}"

ARCHS=("deeplabv3" "fpn" "swin_unetr" "tsvit" "tsvit_lookup" "unet" "vistaformer")

mkdir -p logs/bash

run_test_queue() {
  local dataset="$1"
  local gpu="$2"
  local test_id="${3:-}"

  for arch in "${ARCHS[@]}"; do
    local ckpt
    local job_name
    local log_file

    if [[ -n "$test_id" ]]; then
      ckpt="./exp/${arch}/${dataset}/train/weights/best.pt"
      job_name="${dataset}_${test_id}_${arch}"
    else
      ckpt="./exp/${arch}/${dataset}/train/weights/best.pt"
      job_name="${dataset}_${arch}"
    fi

    log_file="logs/bash/${job_name}_test.log"

    echo "[test] ${job_name} -> GPU ${gpu}"
    echo "[test] log: ${log_file}"

    if [[ ! -f "$ckpt" ]]; then
      echo "[ERROR] ${job_name}: checkpoint not found -> $ckpt"
      return 1
    fi

    if [[ -n "$test_id" ]]; then
      CUDA_VISIBLE_DEVICES="${gpu}" \
        python test.py \
          --arch "${arch}" \
          --dataset "${dataset}" \
          --test_id "${test_id}" \
          --weights_path "${ckpt}" \
          > "${log_file}" 2>&1
    else
      CUDA_VISIBLE_DEVICES="${gpu}" \
        python test.py \
          --arch "${arch}" \
          --dataset "${dataset}" \
          --weights_path "${ckpt}" \
          > "${log_file}" 2>&1
    fi

    sleep "$DELAY"
  done

  echo "[done] ${dataset}${test_id:+_${test_id}} queue finished"
}

# -------------------------
# Launch queues (parallel)
# -------------------------

run_test_queue munich 0 &
pid_munich=$!

run_test_queue lombardia 1 A &
pid_lombardia_A=$!

run_test_queue lombardia 2 Y &
pid_lombardia_Y=$!

fail=0

wait "$pid_munich" || fail=1
wait "$pid_lombardia_A" || fail=1
wait "$pid_lombardia_Y" || fail=1

if [[ "$fail" -ne 0 ]]; then
  echo "One or more test queues failed."
  exit 1
fi

echo "All test queues finished."