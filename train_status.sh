#!/bin/bash
#
# Check training status
#

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PID_FILE="$SCRIPT_DIR/logs/training/training.pid"
LOG_DIR="$SCRIPT_DIR/logs/training"

echo "=============================================="
echo "📊 MANTIS Training Status"
echo "=============================================="
echo ""

if [ ! -f "$PID_FILE" ]; then
    echo "Status: ⚪ Not running"
    echo ""
    
    # Check if there are any logs
    if [ -d "$LOG_DIR" ] && [ -n "$(ls -A $LOG_DIR/training_*.log 2>/dev/null)" ]; then
        LATEST_LOG=$(ls -t $LOG_DIR/training_*.log 2>/dev/null | head -1)
        echo "📁 Latest log: $LATEST_LOG"
        echo ""
        echo "Last 20 lines:"
        echo "---"
        tail -20 "$LATEST_LOG"
    fi
    
    exit 0
fi

TRAINING_PID=$(cat "$PID_FILE")

if ! ps -p "$TRAINING_PID" > /dev/null 2>&1; then
    echo "Status: ⚠️  PID file exists but process not running"
    echo "   (Training may have crashed)"
    echo "   Cleaning up..."
    rm -f "$PID_FILE"
    exit 1
fi

echo "Status: ✅ Running"
echo "PID: $TRAINING_PID"
echo ""

# Show process info
echo "Process Info:"
ps -p "$TRAINING_PID" -o pid,ppid,%cpu,%mem,etime,cmd --no-headers | awk '{
    printf "  PID: %s\n", $1
    printf "  CPU: %s%%\n", $3
    printf "  Memory: %s%%\n", $4
    printf "  Running Time: %s\n", $5
}'
echo ""

# Check GPU usage if nvidia-smi is available
if command -v nvidia-smi &> /dev/null; then
    echo "GPU Info:"
    nvidia-smi --query-gpu=index,utilization.gpu,memory.used,memory.total --format=csv,noheader,nounits | while read line; do
        echo "  GPU $line"
    done
    echo ""
fi

# Show current log
if [ -f "$LOG_DIR/training_current.log" ]; then
    CURRENT_LOG=$(readlink -f "$LOG_DIR/training_current.log")
    echo "📁 Current log: $CURRENT_LOG"
    echo ""
    echo "Last 30 lines:"
    echo "---"
    tail -30 "$CURRENT_LOG"
    echo "---"
    echo ""
    echo "💡 Monitor live: tail -f $CURRENT_LOG"
fi

echo ""
echo "🛑 To stop: ./train_stop.sh or kill $TRAINING_PID"
echo "=============================================="

