#!/usr/bin/env bash
set -euo pipefail

MODEL_ID="Qwen/Qwen2-VL-2B-Instruct"
DATASET="linesdetailed"
SIZES=(50 20  10 100 300)
SEEDS=(44 45 46 47 48)

for SEED in "${SEEDS[@]}"; do
  for S in "${SIZES[@]}"; do
    echo "=== Running --train_size=${S} --seed=${SEED} ==="
    python train_sft.py \
      --model_id "$MODEL_ID" \
      --dataset_name "$DATASET" \
      --epochs 50 \
      --batch_size 4 \
      --lora_r 16 \
      --lora_alpha 64 \
      --lora_dropout 0.3 \
      --lr 1e-4 \
      --use_wandb \
      --device cuda:0 \
      --seed "$SEED" \
      --num_workers 8 \
      --user_system_message "" \
      --user_query_message "" \
      --train_size "$S" \
      --test_size 300 \
      --test_subset_size 300
  done
done

