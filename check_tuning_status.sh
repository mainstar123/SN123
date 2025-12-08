#!/bin/bash
# Script to check status of background hyperparameter tuning

echo "=================================================================================="
echo "MANTIS Hyperparameter Tuning Status"
echo "=================================================================================="
echo ""

LOG_DIR="logs/tuning"
PID_FILE="$LOG_DIR/tuning.pid"
STATUS_FILE="$LOG_DIR/tuning.status"

# Check if tuning process is running
echo "=== Tuning Process Status ==="
if [ -f "$PID_FILE" ]; then
    TUNING_PID=$(cat "$PID_FILE")
    if ps -p "$TUNING_PID" > /dev/null 2>&1; then
        echo "✓ Tuning process is RUNNING"
        echo "  PID: $TUNING_PID"
        
        # Get process info
        if command -v ps > /dev/null; then
            CPU=$(ps -p "$TUNING_PID" -o %cpu --no-headers 2>/dev/null | tr -d ' ')
            MEM=$(ps -p "$TUNING_PID" -o %mem --no-headers 2>/dev/null | tr -d ' ')
            if [ -n "$CPU" ]; then
                echo "  CPU: ${CPU}% | Memory: ${MEM}%"
            fi
        fi
        
        # Check status file
        if [ -f "$STATUS_FILE" ]; then
            if grep -q "Type:" "$STATUS_FILE"; then
                TYPE=$(grep "Type:" "$STATUS_FILE" | cut -d: -f2- | xargs)
                echo "  Type: $TYPE"
            fi
            if grep -q "Started:" "$STATUS_FILE"; then
                STARTED=$(grep "Started:" "$STATUS_FILE" | cut -d: -f2- | xargs)
                echo "  Started: $STARTED"
            fi
        fi
    else
        echo "✗ Tuning process is NOT running (PID file exists but process not found)"
        echo "  Checking if tuning completed..."
        rm -f "$PID_FILE"
    fi
else
    echo "✗ Tuning process is NOT running (no PID file)"
fi

echo ""
echo "=== Recent Log Activity ==="
echo ""

# Find most recent log file
LATEST_LOG=$(ls -t "$LOG_DIR"/tuning_*.log 2>/dev/null | head -1)

if [ -n "$LATEST_LOG" ] && [ -f "$LATEST_LOG" ]; then
    echo "  Latest log: $LATEST_LOG"
    echo ""
    
    # Get log file size
    LOG_SIZE=$(du -h "$LATEST_LOG" | cut -f1)
    echo "  Log size: $LOG_SIZE"
    
    # Count lines
    LOG_LINES=$(wc -l < "$LATEST_LOG" 2>/dev/null || echo "0")
    echo "  Log lines: $LOG_LINES"
    
    echo ""
    echo "  Last 10 lines:"
    tail -10 "$LATEST_LOG" | sed 's/^/    /'
    
    # Check for errors
    ERROR_COUNT=$(grep -i "error\|failed\|exception" "$LATEST_LOG" 2>/dev/null | wc -l || echo "0")
    if [ "$ERROR_COUNT" -gt 0 ]; then
        echo ""
        echo "  ⚠ Found $ERROR_COUNT error(s) in log"
        echo "  Recent errors:"
        grep -i "error\|failed\|exception" "$LATEST_LOG" 2>/dev/null | tail -3 | sed 's/^/    /'
    fi
    
    # Check for progress indicators
    if grep -q "Testing:" "$LATEST_LOG" 2>/dev/null; then
        echo ""
        echo "  Current progress:"
        grep "Testing:" "$LATEST_LOG" 2>/dev/null | tail -1 | sed 's/^/    /'
    fi
    
    if grep -q "Best Configuration" "$LATEST_LOG" 2>/dev/null; then
        COMPLETED=$(grep -c "Best Configuration" "$LATEST_LOG" 2>/dev/null || echo "0")
        echo ""
        echo "  ✓ Completed challenges: $COMPLETED"
    fi
    
else
    echo "  No log files found in $LOG_DIR"
fi

echo ""
echo "=== Tuning Results ==="
echo ""

TUNING_DIR="models/tuning"

# Check for results files
if [ -d "$TUNING_DIR" ]; then
    RESULTS_FILES=$(ls -t "$TUNING_DIR"/tuning_results_*.json 2>/dev/null | head -5)
    BEST_CONFIGS=$(ls -t "$TUNING_DIR"/best_configs_*.json 2>/dev/null | head -1)
    
    if [ -n "$RESULTS_FILES" ]; then
        echo "  Results files found:"
        echo "$RESULTS_FILES" | while read -r file; do
            if [ -n "$file" ]; then
                SIZE=$(du -h "$file" | cut -f1)
                echo "    - $(basename $file) ($SIZE)"
            fi
        done
    fi
    
    if [ -n "$BEST_CONFIGS" ] && [ -f "$BEST_CONFIGS" ]; then
        echo ""
        echo "  Best configurations file: $(basename $BEST_CONFIGS)"
        if command -v jq > /dev/null 2>&1; then
            echo ""
            echo "  Best configs summary:"
            jq -r 'to_entries[] | "    \(.key): Accuracy=\(.value.accuracy // "N/A"), CI=[\(.value.ci_lower // "N/A"), \(.value.ci_upper // "N/A")]"' "$BEST_CONFIGS" 2>/dev/null | head -10
        fi
    fi
else
    echo "  No tuning directory found: $TUNING_DIR"
fi

echo ""
echo "=== Quick Commands ==="
echo ""
echo "  View live logs:"
if [ -n "$LATEST_LOG" ]; then
    echo "    tail -f $LATEST_LOG"
else
    echo "    tail -f $LOG_DIR/tuning_*.log"
fi
echo ""
echo "  Stop tuning:"
if [ -f "$PID_FILE" ]; then
    TUNING_PID=$(cat "$PID_FILE")
    if ps -p "$TUNING_PID" > /dev/null 2>&1; then
        echo "    kill $TUNING_PID"
    else
        echo "    (No tuning process running)"
    fi
else
    echo "    (No tuning process running)"
fi
echo ""
echo "  View results:"
echo "    cat $TUNING_DIR/best_configs_*.json"
echo ""

