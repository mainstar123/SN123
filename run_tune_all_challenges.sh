#!/bin/bash
# Script to run hyperparameter tuning for ALL challenges
# This will tune all 11 challenges defined in config.CHALLENGES

set -e  # Exit on error

echo "=================================================================================="
echo "MANTIS - Hyperparameter Tuning for ALL Challenges"
echo "=================================================================================="
echo ""
echo "This will tune all challenges:"
echo "  - Binary challenges: ETH, EURUSD, GBPUSD, CADUSD, NZDUSD, CHFUSD, AUDUSD, JPYUSD, XAUUSD, XAGUSD"
echo "  - LBFGS challenges: ETHLBFGS, BTCLBFGS"
echo "  - HitFirst challenges: ETHHITFIRST"
echo ""
echo "Estimated time: 30-40 hours (depending on GPU)"
echo ""
read -p "Continue? (y/n) " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]
then
    exit 1
fi

# Default parameters
DATA_DIR="${DATA_DIR:-data/raw}"
TUNING_DIR="${TUNING_DIR:-models/tuning}"
EPOCHS="${EPOCHS:-100}"
BATCH_SIZE="${BATCH_SIZE:-64}"
MIN_ACCURACY="${MIN_ACCURACY:-0.55}"

echo ""
echo "Configuration:"
echo "  Data directory: $DATA_DIR"
echo "  Tuning directory: $TUNING_DIR"
echo "  Epochs per config: $EPOCHS"
echo "  Batch size: $BATCH_SIZE"
echo "  Min accuracy: $MIN_ACCURACY"
echo ""

# Create directories
mkdir -p "$DATA_DIR"
mkdir -p "$TUNING_DIR"

# Run tuning
echo "Starting hyperparameter tuning for all challenges..."
echo ""

python scripts/training/tune_all_challenges.py \
    --data-dir "$DATA_DIR" \
    --tuning-dir "$TUNING_DIR" \
    --epochs "$EPOCHS" \
    --batch-size "$BATCH_SIZE" \
    --min-accuracy "$MIN_ACCURACY"

echo ""
echo "=================================================================================="
echo "Tuning completed!"
echo "=================================================================================="
echo ""
echo "Results saved to: $TUNING_DIR"
echo ""
echo "To test salience for all models, run:"
echo "  ./test_all_salience.sh"
echo ""


