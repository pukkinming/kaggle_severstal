#!/bin/bash

# Generate test predictions
# Usage: bash inference.sh <output_dir> <checkpoint1> [checkpoint2] [checkpoint3] ...

# Activate conda environment
source ~/miniconda3/etc/profile.d/conda.sh
conda activate compute_gpu_3_10

# Get output directory from argument
OUTPUT_DIR=$1
shift

# Get checkpoint paths from remaining arguments
CHECKPOINTS="$@"

if [ -z "$CHECKPOINTS" ]; then
    echo "Error: Please provide at least one checkpoint path"
    echo "Usage: bash inference.sh <output_dir> <checkpoint1> [checkpoint2] [checkpoint3] ..."
    exit 1
fi

echo "=========================================="
echo "Generating Test Predictions"
echo "=========================================="
echo "Output directory: $OUTPUT_DIR"
echo "Checkpoints: $CHECKPOINTS"
echo "=========================================="

# Run inference
python inference.py --output_dir $OUTPUT_DIR --checkpoints $CHECKPOINTS

echo "=========================================="
echo "Inference completed!"
echo "=========================================="




