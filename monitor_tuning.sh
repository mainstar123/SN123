#!/bin/bash
# Monitor background tuning process

LOG_DIR="logs/tuning"

# Find the most recent tuning process
LATEST_LOG=$(ls -t "$LOG_DIR"/tuning_background_*.log 2>/dev/null | head -1)
LATEST_STATUS=$(ls -t "$LOG_DIR"/tuning_background_*.status 2>/dev/null | head -1)
LATEST_PID=$(ls -t "$LOG_DIR"/tuning_background_*.pid 2>/dev/null | head -1)

if [ -z "$LATEST_LOG" ]; then
    echo "No tuning process found"
    exit 1
fi

echo "=================================================================================="
echo "MANTIS Tuning Monitor"
echo "=================================================================================="
echo ""

# Show status
if [ -f "$LATEST_STATUS" ]; then
    STATUS=$(cat "$LATEST_STATUS")
    echo "Status: $STATUS"
else
    echo "Status: UNKNOWN"
fi

# Check if process is running
if [ -f "$LATEST_PID" ]; then
    PID=$(cat "$LATEST_PID")
    if ps -p "$PID" > /dev/null 2>&1; then
        echo "Process: RUNNING (PID: $PID)"
        echo ""
        echo "Resource usage:"
        ps -p "$PID" -o pid,pcpu,pmem,etime,cmd --no-headers | awk '{printf "  CPU: %s%%, Memory: %s%%, Runtime: %s\n", $2, $3, $4}'
    else
        echo "Process: NOT RUNNING (PID file exists but process dead)"
    fi
else
    echo "Process: NO PID FILE"
fi

echo ""
echo "Log file: $LATEST_LOG"
echo ""

# Show last 20 lines of log
echo "Recent log output:"
echo "--------------------------------------------------------------------------------"
tail -20 "$LATEST_LOG" 2>/dev/null || echo "No log output yet"
echo "--------------------------------------------------------------------------------"
echo ""

# Show progress if available
if grep -q "Tuning:" "$LATEST_LOG" 2>/dev/null; then
    echo "Current progress:"
    grep "Tuning:" "$LATEST_LOG" | tail -1
    echo ""
fi

# Show completed challenges
if grep -q "Best Configuration for" "$LATEST_LOG" 2>/dev/null; then
    echo "Completed challenges:"
    grep "Best Configuration for" "$LATEST_LOG" | awk '{print "  ✓ " $4}'
    echo ""
fi

# Show summary if available
if grep -q "Tuning Summary" "$LATEST_LOG" 2>/dev/null; then
    echo "Summary:"
    grep -A 20 "Tuning Summary" "$LATEST_LOG" | tail -15
    echo ""
fi

echo "To follow log in real-time:"
echo "  tail -f $LATEST_LOG"
echo ""
echo "To stop tuning:"
if [ -f "$LATEST_PID" ]; then
    PID=$(cat "$LATEST_PID")
    echo "  kill $PID"
fi
echo ""


