#!/bin/bash
# Script to test salience for all tuned models

set -e

TUNING_DIR="${TUNING_DIR:-models/tuning}"
DATA_DIR="${DATA_DIR:-data/raw}"

echo "=================================================================================="
echo "Testing Salience for All Tuned Models"
echo "=================================================================================="
echo ""

# Get all tickers from config
TICKERS=(
    "ETH"
    "EURUSD"
    "GBPUSD"
    "CADUSD"
    "NZDUSD"
    "CHFUSD"
    "AUDUSD"
    "JPYUSD"
    "XAUUSD"
    "XAGUSD"
    "ETHLBFGS"
    "BTCLBFGS"
    "ETHHITFIRST"
)

RESULTS_DIR="$TUNING_DIR/salience_tests"
mkdir -p "$RESULTS_DIR"

for ticker in "${TICKERS[@]}"; do
    echo "Testing $ticker..."
    
    # Find the best model directory for this ticker
    TICKER_DIR="$TUNING_DIR/$ticker"
    
    if [ ! -d "$TICKER_DIR" ]; then
        echo "  ⚠ Skipping $ticker: No tuning directory found"
        continue
    fi
    
    # Find the most recent config directory
    LATEST_CONFIG=$(ls -td "$TICKER_DIR"/*/ 2>/dev/null | head -1)
    
    if [ -z "$LATEST_CONFIG" ]; then
        echo "  ⚠ Skipping $ticker: No model found"
        continue
    fi
    
    MODEL_DIR=$(dirname "$LATEST_CONFIG")
    CONFIG_NAME=$(basename "$LATEST_CONFIG")
    
    echo "  Using model: $CONFIG_NAME"
    
    # Run salience test
    OUTPUT_FILE="$RESULTS_DIR/${ticker}_salience.json"
    
    python scripts/testing/test_local_salience.py \
        --ticker "$ticker" \
        --model-dir "$MODEL_DIR" \
        --data-dir "$DATA_DIR" \
        --output "$OUTPUT_FILE" 2>&1 | tee "$RESULTS_DIR/${ticker}_output.log"
    
    echo ""
done

echo "=================================================================================="
echo "Salience testing completed!"
echo "=================================================================================="
echo ""
echo "Results saved to: $RESULTS_DIR"
echo ""
echo "Summary:"
for ticker in "${TICKERS[@]}"; do
    RESULT_FILE="$RESULTS_DIR/${ticker}_salience.json"
    if [ -f "$RESULT_FILE" ]; then
        SALIENCE=$(python -c "import json; print(json.load(open('$RESULT_FILE')).get('salience_score', 'N/A'))" 2>/dev/null || echo "N/A")
        echo "  $ticker: Salience = $SALIENCE"
    fi
done
echo ""


