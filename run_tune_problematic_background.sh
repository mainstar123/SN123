#!/bin/bash
# Script to tune all problematic models (forex pairs) in background

set -e

# Configuration
DATA_DIR="data/raw"
TUNING_DIR="models/tuning"
EPOCHS=100
BATCH_SIZE=64
MIN_ACCURACY=0.55
LOG_DIR="logs/tuning"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

# Problematic models (forex pairs with single-class prediction)
TICKERS=("CHFUSD" "NZDUSD" "CADUSD" "GBPUSD" "EURUSD")

# Create directories
mkdir -p "$LOG_DIR"
mkdir -p "$TUNING_DIR"

echo "=========================================="
echo "MANTIS - Background Tuning: Problematic Models"
echo "=========================================="
echo ""
echo "This will tune the following models:"
for ticker in "${TICKERS[@]}"; do
    echo "  - $ticker"
done
echo ""
echo "Estimated time: ~30-60 hours (CPU) or ~10-20 hours (GPU)"
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

# Create master log file
MASTER_LOG="$LOG_DIR/tune_problematic_${TIMESTAMP}.log"
PID_FILE="$LOG_DIR/tune_problematic.pid"
STATUS_FILE="$LOG_DIR/tune_problematic.status"

echo "" >> "$MASTER_LOG"
echo "==========================================" >> "$MASTER_LOG"
echo "Starting tuning for problematic models" >> "$MASTER_LOG"
echo "Started: $(date)" >> "$MASTER_LOG"
echo "==========================================" >> "$MASTER_LOG"
echo ""

# Function to tune a single ticker
tune_ticker() {
    local ticker=$1
    local ticker_log="$LOG_DIR/tune_${ticker}_${TIMESTAMP}.log"
    
    echo "[$(date)] Starting tuning for $ticker" | tee -a "$MASTER_LOG"
    
    python scripts/training/tune_all_challenges.py \
        --ticker "$ticker" \
        --data-dir "$DATA_DIR" \
        --tuning-dir "$TUNING_DIR" \
        --epochs "$EPOCHS" \
        --batch-size "$BATCH_SIZE" \
        --min-accuracy "$MIN_ACCURACY" \
        > "$ticker_log" 2>&1
    
    local exit_code=$?
    
    if [ $exit_code -eq 0 ]; then
        echo "[$(date)] ✓ Completed tuning for $ticker" | tee -a "$MASTER_LOG"
    else
        echo "[$(date)] ✗ Failed tuning for $ticker (exit code: $exit_code)" | tee -a "$MASTER_LOG"
    fi
    
    return $exit_code
}

# Start tuning in background
(
    echo "IN_PROGRESS" > "$STATUS_FILE"
    echo "Started: $(date)" >> "$STATUS_FILE"
    echo "PID: $$" >> "$STATUS_FILE"
    echo "Tickers: ${TICKERS[*]}" >> "$STATUS_FILE"
    echo "Log: $MASTER_LOG" >> "$STATUS_FILE"
    
    echo "$$" > "$PID_FILE"
    
    # Tune each ticker sequentially
    for ticker in "${TICKERS[@]}"; do
        tune_ticker "$ticker"
    done
    
    # Final status
    echo "COMPLETED" > "$STATUS_FILE"
    echo "Completed: $(date)" >> "$STATUS_FILE"
    echo "[$(date)] All tuning completed" >> "$MASTER_LOG"
    
    rm -f "$PID_FILE"
) > "$MASTER_LOG" 2>&1 &

TUNING_PID=$!

echo "✓ Tuning started in background"
echo "  Master PID: $TUNING_PID"
echo "  Master log: $MASTER_LOG"
echo "  PID file: $PID_FILE"
echo "  Status file: $STATUS_FILE"
echo ""
echo "To monitor progress:"
echo "  ./check_tuning_status.sh"
echo ""
echo "To view live logs:"
echo "  tail -f $MASTER_LOG"
echo ""
echo "To view specific ticker logs:"
echo "  tail -f $LOG_DIR/tune_*_${TIMESTAMP}.log"
echo ""
echo "To stop tuning:"
echo "  kill $TUNING_PID"
echo ""

