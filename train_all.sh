#!/bin/bash
#
# MANTIS Training Pipeline - One Command to Train All
#
# Usage:
#   ./train_all.sh              # Full training with hyperparameter tuning
#   ./train_all.sh --quick      # Quick training (no tuning)
#   ./train_all.sh --trials 100 # More thorough tuning (100 trials)
#

set -e  # Exit on error

echo "=============================================="
echo "🚀 MANTIS Multi-Challenge Training Pipeline"
echo "=============================================="

# Activate virtual environment
if [ -d "venv" ]; then
    echo "✓ Activating virtual environment..."
    source venv/bin/activate
else
    echo "⚠️  No virtual environment found at ./venv"
    echo "   Create one with: python -m venv venv"
    exit 1
fi

# Check if data directory exists
if [ ! -d "data" ]; then
    echo "⚠️  Data directory not found!"
    echo "   Creating data directory..."
    mkdir -p data
    echo "   Please add your CSV files to the data/ directory"
fi

# Run training
echo ""
echo "🏃 Starting training pipeline..."
echo ""

python scripts/training/train_all_challenges.py "$@"

echo ""
echo "✅ Training pipeline completed!"
echo ""
echo "📁 Check results in: models/tuned/"
echo "📊 View training log: models/tuned/training_results.json"

