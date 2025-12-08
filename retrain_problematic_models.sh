#!/bin/bash
# Script to retrain models with issues

set -e

DATA_DIR="data/raw"
MODEL_DIR="models/checkpoints"
EPOCHS=200
BATCH_SIZE=64
LOG_DIR="logs/training"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

# Create log directory
mkdir -p "$LOG_DIR"

echo "=========================================="
echo "MANTIS - Retrain Problematic Models"
echo "=========================================="
echo ""

# Activate virtual environment
if [ -f "venv/bin/activate" ]; then
    source venv/bin/activate
    echo "✓ Virtual environment activated"
fi

# Models with single-class prediction issue (forex pairs)
FOREX_TICKERS=("CHFUSD" "NZDUSD" "CADUSD" "GBPUSD" "EURUSD")

# Models that need investigation (may need different approach)
INVESTIGATE_TICKERS=("BTCLBFGS" "ETHLBFGS" "ETHHITFIRST")

echo "Models to retrain:"
echo "  Forex pairs (single-class prediction): ${FOREX_TICKERS[*]}"
echo "  LBFGS/HITFIRST (below threshold): ${INVESTIGATE_TICKERS[*]}"
echo ""

read -p "Do you want to retrain forex models? (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo ""
    echo "Retraining forex models..."
    for ticker in "${FOREX_TICKERS[@]}"; do
        echo ""
        echo "=========================================="
        echo "Training $ticker"
        echo "=========================================="
        
        LOG_FILE="$LOG_DIR/retrain_${ticker}_${TIMESTAMP}.log"
        
        python scripts/training/train_model.py \
            --ticker "$ticker" \
            --data-dir "$DATA_DIR" \
            --model-dir "$MODEL_DIR" \
            --epochs "$EPOCHS" \
            --batch-size "$BATCH_SIZE" \
            --use-gpu \
            > "$LOG_FILE" 2>&1
        
        if [ $? -eq 0 ]; then
            echo "✓ $ticker training completed"
        else
            echo "✗ $ticker training failed (check $LOG_FILE)"
        fi
    done
fi

echo ""
read -p "Do you want to retrain LBFGS/HITFIRST models? (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo ""
    echo "Retraining LBFGS/HITFIRST models..."
    echo "⚠ Note: These may need different evaluation metrics"
    for ticker in "${INVESTIGATE_TICKERS[@]}"; do
        echo ""
        echo "=========================================="
        echo "Training $ticker"
        echo "=========================================="
        
        LOG_FILE="$LOG_DIR/retrain_${ticker}_${TIMESTAMP}.log"
        
        python scripts/training/train_model.py \
            --ticker "$ticker" \
            --data-dir "$DATA_DIR" \
            --model-dir "$MODEL_DIR" \
            --epochs "$EPOCHS" \
            --batch-size "$BATCH_SIZE" \
            --use-gpu \
            > "$LOG_FILE" 2>&1
        
        if [ $? -eq 0 ]; then
            echo "✓ $ticker training completed"
        else
            echo "✗ $ticker training failed (check $LOG_FILE)"
        fi
    done
fi

echo ""
echo "=========================================="
echo "Retraining complete!"
echo "=========================================="
echo ""
echo "Check results with:"
echo "  ./check_training_results.sh"
echo ""

