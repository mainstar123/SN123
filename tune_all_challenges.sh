#!/bin/bash
# Script to run hyperparameter tuning for all challenges

set -e

DATA_DIR="data/raw"
TUNING_DIR="models/tuning"
EPOCHS=100
# GPU-optimized batch size (increased from 64 to 128 for GPU)
BATCH_SIZE=128
MIN_ACCURACY=0.55

echo "=========================================="
echo "MANTIS - Hyperparameter Tuning for All Challenges (GPU Optimized)"
echo "=========================================="
echo ""
echo "This will tune hyperparameters for all challenges."
echo "This may take several hours depending on the number of configurations."
echo ""
echo "Configuration:"
echo "  Epochs per config: $EPOCHS"
echo "  Batch size: $BATCH_SIZE (GPU optimized)"
echo "  Min accuracy: $MIN_ACCURACY"
echo ""

# Check for GPU
if command -v nvidia-smi &> /dev/null && nvidia-smi &> /dev/null; then
    echo "GPU Status:"
    nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader
    echo ""
else
    echo "⚠️  Warning: No GPU detected. Training will be slower on CPU."
    BATCH_SIZE=64  # Reduce batch size for CPU
    echo "  Batch size reduced to $BATCH_SIZE for CPU"
    echo ""
fi

read -p "Continue? (y/n) " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Cancelled."
    exit 1
fi

# Activate virtual environment
if [ -f "venv/bin/activate" ]; then
    source venv/bin/activate
    echo "✓ Virtual environment activated"
fi

# Create tuning directory
mkdir -p "$TUNING_DIR"

# Set TensorFlow environment variables for GPU optimization
export TF_FORCE_GPU_ALLOW_GROWTH=true
export TF_GPU_THREAD_MODE=gpu_private
export TF_GPU_THREAD_COUNT=2
export CUDA_VISIBLE_DEVICES=0

# Run tuning
echo ""
echo "Starting hyperparameter tuning..."
echo ""

python scripts/training/tune_all_challenges.py \
    --data-dir "$DATA_DIR" \
    --tuning-dir "$TUNING_DIR" \
    --epochs "$EPOCHS" \
    --batch-size "$BATCH_SIZE" \
    --min-accuracy "$MIN_ACCURACY"

echo ""
echo "=========================================="
echo "Tuning complete!"
echo "=========================================="
echo ""
echo "Results saved in: $TUNING_DIR"
echo ""
echo "To view results:"
echo "  cat $TUNING_DIR/tuning_results_*.json"
echo "  cat $TUNING_DIR/best_configs_*.json"
echo ""

