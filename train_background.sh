#!/bin/bash
#
# MANTIS Training - Background Service Runner
#
# This script runs training in the background with logging
# Usage:
#   ./train_background.sh [options]
#

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_DIR="$SCRIPT_DIR/logs/training"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="$LOG_DIR/training_${TIMESTAMP}.log"
PID_FILE="$LOG_DIR/training.pid"

# Create log directory
mkdir -p "$LOG_DIR"

# Check if training is already running
if [ -f "$PID_FILE" ]; then
    OLD_PID=$(cat "$PID_FILE")
    if ps -p "$OLD_PID" > /dev/null 2>&1; then
        echo "⚠️  Training is already running (PID: $OLD_PID)"
        echo "   Log file: $(readlink -f $LOG_DIR/training_current.log)"
        echo ""
        echo "To stop it: kill $OLD_PID"
        echo "To monitor: tail -f $LOG_DIR/training_current.log"
        exit 1
    else
        echo "Cleaning up stale PID file..."
        rm -f "$PID_FILE"
    fi
fi

echo "=============================================="
echo "🚀 Starting MANTIS Training in Background"
echo "=============================================="
echo "Log file: $LOG_FILE"
echo "PID file: $PID_FILE"
echo ""

# Activate virtual environment
if [ -d "$SCRIPT_DIR/venv" ]; then
    source "$SCRIPT_DIR/venv/bin/activate"
else
    echo "❌ Virtual environment not found!"
    exit 1
fi

# Create symlink to current log
ln -sf "$(basename "$LOG_FILE")" "$LOG_DIR/training_current.log"

# Run training in background with nohup
nohup bash -c "
    cd '$SCRIPT_DIR'
    source venv/bin/activate
    
    echo '=============================================='
    echo 'MANTIS Training Started'
    echo 'Time: $(date)'
    echo 'PID: $$'
    echo '=============================================='
    echo ''
    
    # Run training
    python scripts/training/train_all_challenges.py $@ 2>&1
    
    EXIT_CODE=\$?
    
    echo ''
    echo '=============================================='
    echo 'Training Completed'
    echo 'Time: $(date)'
    echo 'Exit Code: '\$EXIT_CODE
    echo '=============================================='
    
    # Remove PID file on completion
    rm -f '$PID_FILE'
    
    exit \$EXIT_CODE
" > "$LOG_FILE" 2>&1 &

TRAINING_PID=$!
echo $TRAINING_PID > "$PID_FILE"

echo "✅ Training started in background!"
echo "   PID: $TRAINING_PID"
echo ""
echo "📊 Monitor progress:"
echo "   tail -f $LOG_DIR/training_current.log"
echo ""
echo "🛑 Stop training:"
echo "   kill $TRAINING_PID"
echo "   # or: ./train_stop.sh"
echo ""
echo "📁 View results when done:"
echo "   cat models/tuned/training_results.json"
echo ""
echo "=============================================="

