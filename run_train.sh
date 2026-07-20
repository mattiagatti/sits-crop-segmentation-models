#!/usr/bin/env bash
set -euo pipefail

MODE="${1:-start}"

ARCHS=("deeplabv3" "fpn" "swin_unetr" "tsvit" "tsvit_lookup" "unet" "vistaformer")

mkdir -p logs/bash

run_queue() {
  local dataset="$1"
  local gpu="$2"

  for arch in "${ARCHS[@]}"; do
    log_file="logs/bash/${dataset}_${arch}.log"

    echo "[train] ${dataset}_${arch} -> GPU ${gpu}"

    if [[ "$MODE" == "resume" ]]; then
      ckpt="./logs/${arch}/${dataset}/train/checkpoints/last.pt"
      CUDA_VISIBLE_DEVICES=${gpu} \
        python train.py --arch "$arch" --dataset "$dataset" --ckpt_path "$ckpt" \
        2>&1 | tee "$log_file"
    else
      CUDA_VISIBLE_DEVICES=${gpu} \
        python train.py --arch "$arch" --dataset "$dataset" \
        2>&1 | tee "$log_file"
    fi
  done

  echo "[done] ${dataset} queue finished"
}

# launch both queues in parallel
run_queue munich 0 &
run_queue lombardia 1 &

# wait for both to finish
wait

echo "All training finished."