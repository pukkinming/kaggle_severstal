#!/bin/bash

# Train all folds
# Usage: bash train_all_folds.sh

# Activate conda environment
source ~/miniconda3/etc/profile.d/conda.sh
conda activate compute_gpu_3_10

echo "=========================================="
echo "Training All Folds"
echo "=========================================="

# Run training for all folds
python train.py --all_folds

echo "=========================================="
echo "All folds training completed!"
echo "=========================================="




