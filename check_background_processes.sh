#!/bin/bash
# Script to check all background processes related to MANTIS training/tuning

echo "=================================================================================="
echo "MANTIS Background Processes"
echo "=================================================================================="
echo ""

# Check for training processes
echo "=== Training Processes ==="
TRAIN_PIDS=$(ps aux | grep -E "train_model\.py|train_all_tickers" | grep -v grep | awk '{print $2}')
if [ -n "$TRAIN_PIDS" ]; then
    ps aux | grep -E "train_model\.py|train_all_tickers" | grep -v grep | while read -r line; do
        PID=$(echo "$line" | awk '{print $2}')
        CPU=$(echo "$line" | awk '{print $3}')
        MEM=$(echo "$line" | awk '{print $4}')
        CMD=$(echo "$line" | awk '{for(i=11;i<=NF;i++) printf "%s ", $i; print ""}')
        echo "  PID: $PID | CPU: ${CPU}% | MEM: ${MEM}%"
        echo "    Command: $CMD"
    done
else
    echo "  No training processes running"
fi

echo ""
echo "=== Hyperparameter Tuning Processes ==="
TUNE_PIDS=$(ps aux | grep -E "tune_all_challenges\.py|tune.*challenges" | grep -v grep | awk '{print $2}')
if [ -n "$TUNE_PIDS" ]; then
    ps aux | grep -E "tune_all_challenges\.py|tune.*challenges" | grep -v grep | while read -r line; do
        PID=$(echo "$line" | awk '{print $2}')
        CPU=$(echo "$line" | awk '{print $3}')
        MEM=$(echo "$line" | awk '{print $4}')
        CMD=$(echo "$line" | awk '{for(i=11;i<=NF;i++) printf "%s ", $i; print ""}')
        echo "  PID: $PID | CPU: ${CPU}% | MEM: ${MEM}%"
        echo "    Command: $CMD"
    done
else
    echo "  No tuning processes running"
fi

echo ""
echo "=== Python Processes (MANTIS related) ==="
PYTHON_PROCS=$(ps aux | grep python | grep -E "MANTIS|training|tuning|mining" | grep -v grep)
if [ -n "$PYTHON_PROCS" ]; then
    echo "$PYTHON_PROCS" | while read -r line; do
        PID=$(echo "$line" | awk '{print $2}')
        CPU=$(echo "$line" | awk '{print $3}')
        MEM=$(echo "$line" | awk '{print $4}')
        CMD=$(echo "$line" | awk '{for(i=11;i<=NF;i++) printf "%s ", $i; print ""}')
        echo "  PID: $PID | CPU: ${CPU}% | MEM: ${MEM}%"
        echo "    Command: $CMD"
    done
else
    echo "  No MANTIS-related Python processes found"
fi

echo ""
echo "=== PID Files ==="
if [ -f "logs/training/train_all.pid" ]; then
    TRAIN_PID=$(cat logs/training/train_all.pid 2>/dev/null)
    if ps -p "$TRAIN_PID" > /dev/null 2>&1; then
        echo "  ✓ Training PID file: $TRAIN_PID (running)"
    else
        echo "  ✗ Training PID file: $TRAIN_PID (not running - stale)"
    fi
fi

if [ -f "logs/tuning/tuning.pid" ]; then
    TUNE_PID=$(cat logs/tuning/tuning.pid 2>/dev/null)
    if ps -p "$TUNE_PID" > /dev/null 2>&1; then
        echo "  ✓ Tuning PID file: $TUNE_PID (running)"
    else
        echo "  ✗ Tuning PID file: $TUNE_PID (not running - stale)"
    fi
fi

if [ -f "logs/tuning/tune_problematic.pid" ]; then
    TUNE_PID=$(cat logs/tuning/tune_problematic.pid 2>/dev/null)
    if ps -p "$TUNE_PID" > /dev/null 2>&1; then
        echo "  ✓ Problematic tuning PID file: $TUNE_PID (running)"
    else
        echo "  ✗ Problematic tuning PID file: $TUNE_PID (not running - stale)"
    fi
fi

echo ""
echo "=== Quick Commands ==="
echo ""
echo "  View all Python processes:"
echo "    ps aux | grep python | grep -v grep"
echo ""
echo "  View all background jobs:"
echo "    jobs -l"
echo ""
echo "  Kill a process:"
echo "    kill <PID>"
echo ""
echo "  Force kill a process:"
echo "    kill -9 <PID>"
echo ""

