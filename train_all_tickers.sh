#!/bin/bash
# Script to train all tickers in background with logging

set -e

# Configuration
DATA_DIR="data/raw"
MODEL_DIR="models/checkpoints"
EPOCHS=200
BATCH_SIZE=64
LOG_DIR="logs/training"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

# Create log directory
mkdir -p "$LOG_DIR"

# Log file
LOG_FILE="$LOG_DIR/train_all_${TIMESTAMP}.log"
PID_FILE="$LOG_DIR/train_all.pid"
STATUS_FILE="$LOG_DIR/train_all.status"

echo "=========================================="
echo "MANTIS - Train All Tickers"
echo "=========================================="
echo "Data directory: $DATA_DIR"
echo "Model directory: $MODEL_DIR"
echo "Epochs: $EPOCHS"
echo "Batch size: $BATCH_SIZE"
echo "Log file: $LOG_FILE"
echo "=========================================="
echo ""

# Activate virtual environment
if [ -f "venv/bin/activate" ]; then
    source venv/bin/activate
    echo "✓ Virtual environment activated"
else
    echo "⚠ Warning: Virtual environment not found"
fi

# Check if training is already running
if [ -f "$PID_FILE" ]; then
    OLD_PID=$(cat "$PID_FILE")
    if ps -p "$OLD_PID" > /dev/null 2>&1; then
        echo "⚠ Training is already running (PID: $OLD_PID)"
        echo "  To check status: ./check_training_status.sh"
        echo "  To stop: kill $OLD_PID"
        exit 1
    else
        echo "⚠ Removing stale PID file"
        rm -f "$PID_FILE"
    fi
fi

# Start training in background
echo "Starting training for all tickers..."
echo ""

nohup python scripts/training/train_model.py \
    --data-dir "$DATA_DIR" \
    --model-dir "$MODEL_DIR" \
    --epochs "$EPOCHS" \
    --batch-size "$BATCH_SIZE" \
    --use-gpu \
    > "$LOG_FILE" 2>&1 &

TRAIN_PID=$!
echo "$TRAIN_PID" > "$PID_FILE"

echo "✓ Training started in background"
echo "  PID: $TRAIN_PID"
echo "  Log file: $LOG_FILE"
echo "  PID file: $PID_FILE"
echo ""
echo "To monitor progress:"
echo "  ./check_training_status.sh"
echo ""
echo "To view live logs:"
echo "  tail -f $LOG_FILE"
echo ""
echo "To stop training:"
echo "  kill $TRAIN_PID"
echo ""

# Initial status
echo "IN_PROGRESS" > "$STATUS_FILE"
echo "Started: $(date)" >> "$STATUS_FILE"
echo "PID: $TRAIN_PID" >> "$STATUS_FILE"

