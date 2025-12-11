#!/bin/bash
# Check if tuning process is actually working (even if log appears stuck)

PID_FILE=$(ls -t logs/tuning/tuning_background_*.pid 2>/dev/null | head -1)

if [ -z "$PID_FILE" ] || [ ! -f "$PID_FILE" ]; then
    echo "No tuning process found"
    exit 1
fi

PID=$(cat "$PID_FILE")

if ! ps -p "$PID" > /dev/null 2>&1; then
    echo "Process $PID is not running"
    exit 1
fi

echo "=================================================================================="
echo "Tuning Process Status Check"
echo "=================================================================================="
echo ""
echo "PID: $PID"
echo ""

# Check CPU usage
CPU=$(ps -p "$PID" -o %cpu --no-headers | tr -d ' ')
echo "CPU Usage: ${CPU}%"
if (( $(echo "$CPU > 50" | bc -l) )); then
    echo "  ✓ Process is actively working"
else
    echo "  ⚠ Process may be idle or stuck"
fi
echo ""

# Check memory
MEM=$(ps -p "$PID" -o %mem --no-headers | tr -d ' ')
echo "Memory Usage: ${MEM}%"
echo ""

# Check runtime
RUNTIME=$(ps -p "$PID" -o etime --no-headers | tr -d ' ')
echo "Runtime: $RUNTIME"
echo ""

# Check log file
LOG_FILE=$(ls -t logs/tuning/tuning_background_*.log 2>/dev/null | head -1)
if [ -n "$LOG_FILE" ]; then
    LOG_SIZE=$(stat -c "%s" "$LOG_FILE" 2>/dev/null || echo "0")
    LOG_MTIME=$(stat -c "%y" "$LOG_FILE" 2>/dev/null | cut -d'.' -f1)
    NOW=$(date +%s)
    LOG_AGE=$(($NOW - $(date -d "$LOG_MTIME" +%s 2>/dev/null || echo $NOW)))
    
    echo "Log File: $LOG_FILE"
    echo "Log Size: $(numfmt --to=iec-i --suffix=B $LOG_SIZE 2>/dev/null || echo "${LOG_SIZE} bytes")"
    echo "Last Update: $LOG_MTIME ($(($LOG_AGE / 60)) minutes ago)"
    
    if [ $LOG_AGE -gt 3600 ]; then
        echo "  ⚠ WARNING: Log hasn't updated in over 1 hour"
        echo "  This is likely due to Python output buffering"
        echo "  The process may still be working (check CPU usage above)"
    else
        echo "  ✓ Log is being updated"
    fi
    echo ""
    
    # Show last few lines
    echo "Last 5 lines of log:"
    tail -5 "$LOG_FILE" | sed 's/^/  /'
    echo ""
fi

# Check for GPU usage (if nvidia-smi available)
if command -v nvidia-smi &> /dev/null; then
    echo "GPU Usage:"
    nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader,nounits | grep "$PID" || echo "  No GPU usage detected for this PID"
    echo ""
fi

echo "=================================================================================="
echo "Summary:"
echo "=================================================================================="
if (( $(echo "$CPU > 50" | bc -l) )); then
    echo "✓ Process is WORKING (high CPU usage)"
    echo "  The log may appear stuck due to Python output buffering"
    echo "  This is normal - the process will continue and complete"
else
    echo "⚠ Process may be IDLE or STUCK (low CPU usage)"
    echo "  Check the log file for errors"
fi
echo ""

