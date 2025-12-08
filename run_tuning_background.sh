#!/bin/bash
# Script to run hyperparameter tuning in the background with logging and status tracking

set -e

# Configuration
DATA_DIR="data/raw"
TUNING_DIR="models/tuning"
EPOCHS=100
BATCH_SIZE=64
MIN_ACCURACY=0.55
LOG_DIR="logs/tuning"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

# Create directories
mkdir -p "$LOG_DIR"
mkdir -p "$TUNING_DIR"

# Log files
LOG_FILE="$LOG_DIR/tuning_${TIMESTAMP}.log"
PID_FILE="$LOG_DIR/tuning.pid"
STATUS_FILE="$LOG_DIR/tuning.status"

echo "=========================================="
echo "MANTIS - Background Hyperparameter Tuning"
echo "=========================================="
echo ""

# Activate virtual environment
if [ -f "venv/bin/activate" ]; then
    source venv/bin/activate
    echo "✓ Virtual environment activated"
else
    echo "⚠ Warning: Virtual environment not found"
fi

# Check if tuning is already running
if [ -f "$PID_FILE" ]; then
    OLD_PID=$(cat "$PID_FILE")
    if ps -p "$OLD_PID" > /dev/null 2>&1; then
        echo "⚠ Tuning is already running (PID: $OLD_PID)"
        echo "  To check status: ./check_tuning_status.sh"
        echo "  To stop: kill $OLD_PID"
        exit 1
    else
        echo "⚠ Removing stale PID file"
        rm -f "$PID_FILE"
    fi
fi

# Menu for tuning options
echo "Select tuning option:"
echo "  1) Tune all challenges (takes ~66-132 hours)"
echo "  2) Tune problematic models only (forex pairs - recommended)"
echo "  3) Tune all binary challenges"
echo "  4) Tune all LBFGS challenges"
echo "  5) Tune all HITFIRST challenges"
echo "  6) Tune specific ticker"
echo ""

read -p "Select option (1-6): " -n 1 -r
echo ""

case $REPLY in
    1)
        TUNING_CMD="python scripts/training/tune_all_challenges.py \
            --data-dir $DATA_DIR \
            --tuning-dir $TUNING_DIR \
            --epochs $EPOCHS \
            --batch-size $BATCH_SIZE \
            --min-accuracy $MIN_ACCURACY"
        TUNING_TYPE="all_challenges"
        ;;
    2)
        TUNING_CMD="python scripts/training/tune_all_challenges.py \
            --ticker CHFUSD \
            --data-dir $DATA_DIR \
            --tuning-dir $TUNING_DIR \
            --epochs $EPOCHS \
            --batch-size $BATCH_SIZE \
            --min-accuracy $MIN_ACCURACY"
        TUNING_TYPE="problematic_models"
        # Note: This will start with CHFUSD, you can add others in sequence
        echo "⚠ Note: This starts with CHFUSD. To tune all problematic models,"
        echo "  run this script multiple times or modify to loop through tickers."
        ;;
    3)
        TUNING_CMD="python scripts/training/tune_all_challenges.py \
            --challenge-type binary \
            --data-dir $DATA_DIR \
            --tuning-dir $TUNING_DIR \
            --epochs $EPOCHS \
            --batch-size $BATCH_SIZE \
            --min-accuracy $MIN_ACCURACY"
        TUNING_TYPE="binary_challenges"
        ;;
    4)
        TUNING_CMD="python scripts/training/tune_all_challenges.py \
            --challenge-type lbfgs \
            --data-dir $DATA_DIR \
            --tuning-dir $TUNING_DIR \
            --epochs $EPOCHS \
            --batch-size $BATCH_SIZE \
            --min-accuracy $MIN_ACCURACY"
        TUNING_TYPE="lbfgs_challenges"
        ;;
    5)
        TUNING_CMD="python scripts/training/tune_all_challenges.py \
            --challenge-type hitfirst \
            --data-dir $DATA_DIR \
            --tuning-dir $TUNING_DIR \
            --epochs $EPOCHS \
            --batch-size $BATCH_SIZE \
            --min-accuracy $MIN_ACCURACY"
        TUNING_TYPE="hitfirst_challenges"
        ;;
    6)
        read -p "Enter ticker to tune: " TICKER
        TUNING_CMD="python scripts/training/tune_all_challenges.py \
            --ticker $TICKER \
            --data-dir $DATA_DIR \
            --tuning-dir $TUNING_DIR \
            --epochs $EPOCHS \
            --batch-size $BATCH_SIZE \
            --min-accuracy $MIN_ACCURACY"
        TUNING_TYPE="ticker_${TICKER}"
        ;;
    *)
        echo "Invalid option"
        exit 1
        ;;
esac

echo ""
echo "Starting tuning in background..."
echo "  Type: $TUNING_TYPE"
echo "  Log file: $LOG_FILE"
echo "  PID file: $PID_FILE"
echo ""

# Start tuning in background
nohup bash -c "$TUNING_CMD" > "$LOG_FILE" 2>&1 &

TUNING_PID=$!
echo "$TUNING_PID" > "$PID_FILE"

# Initial status
echo "IN_PROGRESS" > "$STATUS_FILE"
echo "Started: $(date)" >> "$STATUS_FILE"
echo "PID: $TUNING_PID" >> "$STATUS_FILE"
echo "Type: $TUNING_TYPE" >> "$STATUS_FILE"
echo "Log: $LOG_FILE" >> "$STATUS_FILE"

echo "✓ Tuning started in background"
echo "  PID: $TUNING_PID"
echo "  Log file: $LOG_FILE"
echo "  PID file: $PID_FILE"
echo ""
echo "To monitor progress:"
echo "  ./check_tuning_status.sh"
echo ""
echo "To view live logs:"
echo "  tail -f $LOG_FILE"
echo ""
echo "To stop tuning:"
echo "  kill $TUNING_PID"
echo ""

