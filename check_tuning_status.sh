#!/bin/bash
# Quick status check for background tuning

LOG_DIR="logs/tuning"
LATEST_LOG=$(ls -t "$LOG_DIR"/tuning_background_*.log 2>/dev/null | head -1)
LATEST_STATUS=$(ls -t "$LOG_DIR"/tuning_background_*.status 2>/dev/null | head -1)

if [ -z "$LATEST_LOG" ]; then
    echo "No tuning process found"
    exit 1
fi

echo "=================================================================================="
echo "Quick Tuning Status Check"
echo "=================================================================================="
echo ""

# Status
if [ -f "$LATEST_STATUS" ]; then
    STATUS=$(cat "$LATEST_STATUS")
    echo "Status: $STATUS"
else
    echo "Status: UNKNOWN"
fi

# Check if running
LATEST_PID=$(ls -t "$LOG_DIR"/tuning_background_*.pid 2>/dev/null | head -1)
if [ -f "$LATEST_PID" ]; then
    PID=$(cat "$LATEST_PID")
    if ps -p "$PID" > /dev/null 2>&1; then
        echo "Process: RUNNING (PID: $PID)"
        RUNTIME=$(ps -p "$PID" -o etime= | tr -d ' ')
        echo "Runtime: $RUNTIME"
    else
        echo "Process: NOT RUNNING"
    fi
fi

echo ""

# Completed challenges
COMPLETED=$(grep "Best Configuration for" "$LATEST_LOG" 2>/dev/null | wc -l)
echo "Completed challenges: $COMPLETED / 13"

if [ "$COMPLETED" -gt 0 ]; then
    echo ""
    echo "Completed:"
    grep "Best Configuration for" "$LATEST_LOG" 2>/dev/null | awk '{print "  ✓ " $4}'
fi

echo ""

# Current challenge
CURRENT=$(grep "Tuning:" "$LATEST_LOG" 2>/dev/null | tail -1)
if [ -n "$CURRENT" ]; then
    echo "Current: $CURRENT"
fi

echo ""

# Recent errors
ERRORS=$(grep -i "error\|failed\|exception" "$LATEST_LOG" 2>/dev/null | tail -3)
if [ -n "$ERRORS" ]; then
    echo "Recent errors:"
    echo "$ERRORS"
else
    echo "No recent errors ✓"
fi

echo ""
echo "Full log: $LATEST_LOG"
echo "Monitor: tail -f $LATEST_LOG"
