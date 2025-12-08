#!/bin/bash
# Script to run hyperparameter tuning for all challenges

set -e

DATA_DIR="data/raw"
TUNING_DIR="models/tuning"
EPOCHS=100
BATCH_SIZE=64
MIN_ACCURACY=0.55

echo "=========================================="
echo "MANTIS - Hyperparameter Tuning for All Challenges"
echo "=========================================="
echo ""
echo "This will tune hyperparameters for all challenges."
echo "This may take several hours depending on the number of configurations."
echo ""
echo "Configuration:"
echo "  Epochs per config: $EPOCHS"
echo "  Batch size: $BATCH_SIZE"
echo "  Min accuracy: $MIN_ACCURACY"
echo ""

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

