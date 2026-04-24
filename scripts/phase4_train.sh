#!/bin/bash
# Phase 4 — Fine-tuning 실행 스크립트
# 실행 전: source ~/venv/vision/bin/activate

POLICY=${1:-"diffusion"}   # "diffusion" 또는 "act"
TIMESTAMP=$(date +%Y%m%d_%H%M)
OUTPUT_DIR="outputs/${POLICY}_pusht_${TIMESTAMP}"

echo "=========================================="
echo "Phase 4 — Fine-tuning"
echo "policy    : $POLICY"
echo "output    : $OUTPUT_DIR"
echo "=========================================="

python -m lerobot.scripts.lerobot_train \
  --policy.type=$POLICY \
  --env.type=pusht \
  --dataset.repo_id=lerobot/pusht_image \
  --output_dir=$OUTPUT_DIR \
  --training.num_epochs=100 \
  --wandb.enable=true \
  --wandb.project=vla_pusht

# 실행 방법:
#   bash phase4_train.sh diffusion   # Diffusion Policy (thesis 연결)
#   bash phase4_train.sh act         # ACT (더 빠름)
