#!/bin/bash
# Quick script to check training results for all models

echo "=================================================================================="
echo "MANTIS Training Results Summary"
echo "=================================================================================="
echo ""

MODEL_DIR="models/checkpoints"
DATA_DIR="data/raw"

# Activate virtual environment if available
if [ -f "venv/bin/activate" ]; then
    source venv/bin/activate
fi

# Run the results check script
python scripts/training/check_training_results.py \
    --model-dir "$MODEL_DIR" \
    --data-dir "$DATA_DIR" \
    --min-accuracy 0.55

echo ""
echo "=================================================================================="

