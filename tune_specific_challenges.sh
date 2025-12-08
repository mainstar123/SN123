#!/bin/bash
# Script to tune specific challenge types or individual tickers

set -e

DATA_DIR="data/raw"
TUNING_DIR="models/tuning"
EPOCHS=100
BATCH_SIZE=64
MIN_ACCURACY=0.55

echo "=========================================="
echo "MANTIS - Hyperparameter Tuning (Selective)"
echo "=========================================="
echo ""
echo "Options:"
echo "  1) Tune all binary challenges (forex pairs + ETH)"
echo "  2) Tune all LBFGS challenges"
echo "  3) Tune all HITFIRST challenges"
echo "  4) Tune specific ticker"
echo "  5) Tune problematic models only (forex pairs)"
echo ""

read -p "Select option (1-5): " -n 1 -r
echo

# Activate virtual environment
if [ -f "venv/bin/activate" ]; then
    source venv/bin/activate
    echo "✓ Virtual environment activated"
fi

mkdir -p "$TUNING_DIR"

case $REPLY in
    1)
        echo ""
        echo "Tuning all binary challenges..."
        python scripts/training/tune_all_challenges.py \
            --data-dir "$DATA_DIR" \
            --tuning-dir "$TUNING_DIR" \
            --epochs "$EPOCHS" \
            --batch-size "$BATCH_SIZE" \
            --min-accuracy "$MIN_ACCURACY" \
            --challenge-type binary
        ;;
    2)
        echo ""
        echo "Tuning all LBFGS challenges..."
        python scripts/training/tune_all_challenges.py \
            --data-dir "$DATA_DIR" \
            --tuning-dir "$TUNING_DIR" \
            --epochs "$EPOCHS" \
            --batch-size "$BATCH_SIZE" \
            --min-accuracy "$MIN_ACCURACY" \
            --challenge-type lbfgs
        ;;
    3)
        echo ""
        echo "Tuning all HITFIRST challenges..."
        python scripts/training/tune_all_challenges.py \
            --data-dir "$DATA_DIR" \
            --tuning-dir "$TUNING_DIR" \
            --epochs "$EPOCHS" \
            --batch-size "$BATCH_SIZE" \
            --min-accuracy "$MIN_ACCURACY" \
            --challenge-type hitfirst
        ;;
    4)
        read -p "Enter ticker to tune: " TICKER
        echo ""
        echo "Tuning $TICKER..."
        python scripts/training/tune_all_challenges.py \
            --ticker "$TICKER" \
            --data-dir "$DATA_DIR" \
            --tuning-dir "$TUNING_DIR" \
            --epochs "$EPOCHS" \
            --batch-size "$BATCH_SIZE" \
            --min-accuracy "$MIN_ACCURACY"
        ;;
    5)
        echo ""
        echo "Tuning problematic forex models..."
        for ticker in CHFUSD NZDUSD CADUSD GBPUSD EURUSD; do
            echo ""
            echo "Tuning $ticker..."
            python scripts/training/tune_all_challenges.py \
                --ticker "$ticker" \
                --data-dir "$DATA_DIR" \
                --tuning-dir "$TUNING_DIR" \
                --epochs "$EPOCHS" \
                --batch-size "$BATCH_SIZE" \
                --min-accuracy "$MIN_ACCURACY"
        done
        ;;
    *)
        echo "Invalid option"
        exit 1
        ;;
esac

echo ""
echo "=========================================="
echo "Tuning complete!"
echo "=========================================="

