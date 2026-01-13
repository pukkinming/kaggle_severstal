#!/bin/bash

# Train a single fold
# Usage: bash train_single_fold.sh <fold_number> [checkpoint_path]
# Example: bash train_single_fold.sh 0
# Example with checkpoint: bash train_single_fold.sh 0 outputs_20260110_220612/fold_0/best_model.pth

# Activate conda environment
source ~/miniconda3/etc/profile.d/conda.sh
conda activate compute_gpu_3_10

# Get fold number from argument (default to 0)
FOLD=${1:-0}

# Get checkpoint path from second argument (optional)
CHECKPOINT=$2

echo "=========================================="
echo "Training Fold $FOLD"
if [ -n "$CHECKPOINT" ]; then
    echo "Resuming from checkpoint: $CHECKPOINT"
fi
echo "=========================================="

# Run training
if [ -n "$CHECKPOINT" ]; then
    python train.py --fold $FOLD --checkpoint "$CHECKPOINT"
else
python train.py --fold $FOLD
fi

echo "=========================================="
echo "Training Fold $FOLD completed!"
echo "=========================================="



