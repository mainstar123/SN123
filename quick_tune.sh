#!/bin/bash
# Quick Hyperparameter Tuning Script
# Tests a few key configurations to find best settings

TICKER="ETH"
EPOCHS=100
# GPU-optimized batch size (increased from 64 to 128)
BATCH_SIZE=128

# Check for GPU and adjust batch size
if command -v nvidia-smi &> /dev/null && nvidia-smi &> /dev/null; then
    echo "✓ GPU detected, using batch size: $BATCH_SIZE"
    export TF_FORCE_GPU_ALLOW_GROWTH=true
    export TF_GPU_THREAD_MODE=gpu_private
    export CUDA_VISIBLE_DEVICES=0
else
    BATCH_SIZE=64
    echo "⚠️  No GPU detected, using CPU batch size: $BATCH_SIZE"
fi

echo ""
echo "Quick Hyperparameter Tuning for $TICKER (GPU Optimized)"
echo "========================================"

# Configuration 1: Current (baseline)
echo ""
echo "Config 1: Baseline (256 hidden, 20 features, 0.3 dropout, 0.0005 LR)"
python scripts/training/train_model.py \
    --ticker $TICKER \
    --epochs $EPOCHS \
    --batch-size $BATCH_SIZE \
    --model-dir models/tuning/config1

# Configuration 2: More features
echo ""
echo "Config 2: More features (256 hidden, 25 features, 0.3 dropout, 0.0005 LR)"
# Edit train_model.py: tmfg_n_features=25
python scripts/training/train_model.py \
    --ticker $TICKER \
    --epochs $EPOCHS \
    --batch-size $BATCH_SIZE \
    --model-dir models/tuning/config2

# Configuration 3: More regularization
echo ""
echo "Config 3: More regularization (256 hidden, 20 features, 0.4 dropout, 0.0005 LR)"
# Edit train_model.py: dropout=0.4
python scripts/training/train_model.py \
    --ticker $TICKER \
    --epochs $EPOCHS \
    --batch-size $BATCH_SIZE \
    --model-dir models/tuning/config3

# Configuration 4: Larger model
echo ""
echo "Config 4: Larger model (512 hidden, 20 features, 0.3 dropout, 0.0005 LR)"
# Edit train_model.py: lstm_hidden=512
python scripts/training/train_model.py \
    --ticker $TICKER \
    --epochs $EPOCHS \
    --batch-size $BATCH_SIZE \
    --model-dir models/tuning/config4

echo ""
echo "Tuning complete! Check results:"
echo "python scripts/training/check_training_results.py --ticker $TICKER --model-dir models/tuning/config1"
echo "python scripts/training/check_training_results.py --ticker $TICKER --model-dir models/tuning/config2"
echo "python scripts/training/check_training_results.py --ticker $TICKER --model-dir models/tuning/config3"
echo "python scripts/training/check_training_results.py --ticker $TICKER --model-dir models/tuning/config4"

