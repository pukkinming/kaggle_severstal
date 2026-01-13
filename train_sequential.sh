#!/bin/bash

# Train all folds sequentially (one by one)
# Usage: bash train_sequential.sh

# Activate conda environment
source ~/miniconda3/etc/profile.d/conda.sh
conda activate compute_gpu_3_10

echo "=========================================="
echo "Training All Folds Sequentially"
echo "=========================================="

# Train each fold
for FOLD in 0 1 2 3 4
do
    echo ""
    echo "=========================================="
    echo "Training Fold $FOLD"
    echo "=========================================="
    python train.py --fold $FOLD
    
    if [ $? -ne 0 ]; then
        echo "Error training fold $FOLD"
        exit 1
    fi
done

echo ""
echo "=========================================="
echo "All folds training completed!"
echo "=========================================="




