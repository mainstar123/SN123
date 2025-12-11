#!/bin/bash
# Safely stop current tuning and restart with GPU optimizations

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "=================================================================================="
echo "Restarting Tuning with GPU Optimizations"
echo "=================================================================================="
echo ""

# Find current tuning process
PID_FILE=$(ls -t logs/tuning/tuning_background_*.pid 2>/dev/null | head -1)

if [ -z "$PID_FILE" ] || [ ! -f "$PID_FILE" ]; then
    echo "No active tuning process found"
    echo ""
else
    PID=$(cat "$PID_FILE")
    echo "Found active tuning process:"
    echo "  PID: $PID"
    echo "  PID file: $PID_FILE"
    echo ""
    
    # Check if process is still running
    if ps -p "$PID" > /dev/null 2>&1; then
        echo "Stopping process $PID..."
        
        # Find the actual Python process (child of the bash script)
        PYTHON_PID=$(ps --ppid "$PID" -o pid --no-headers 2>/dev/null | head -1 | tr -d ' ')
        
        if [ -n "$PYTHON_PID" ] && ps -p "$PYTHON_PID" > /dev/null 2>&1; then
            echo "  Found Python process: $PYTHON_PID"
            echo "  Sending SIGTERM to Python process..."
            kill -TERM "$PYTHON_PID" 2>/dev/null || true
            
            # Wait up to 30 seconds for graceful shutdown
            for i in {1..30}; do
                if ! ps -p "$PYTHON_PID" > /dev/null 2>&1; then
                    echo "  ✓ Process stopped gracefully"
                    break
                fi
                sleep 1
            done
            
            # Force kill if still running
            if ps -p "$PYTHON_PID" > /dev/null 2>&1; then
                echo "  Process didn't stop, sending SIGKILL..."
                kill -KILL "$PYTHON_PID" 2>/dev/null || true
                sleep 2
            fi
        fi
        
        # Stop the bash script
        if ps -p "$PID" > /dev/null 2>&1; then
            echo "  Stopping bash script..."
            kill -TERM "$PID" 2>/dev/null || true
            sleep 2
            
            if ps -p "$PID" > /dev/null 2>&1; then
                kill -KILL "$PID" 2>/dev/null || true
            fi
        fi
        
        # Clean up PID file
        rm -f "$PID_FILE"
        echo "  ✓ Cleaned up PID file"
        echo ""
    else
        echo "  Process is not running (stale PID file)"
        rm -f "$PID_FILE"
        echo ""
    fi
fi

# Check for any remaining Python tuning processes
REMAINING=$(ps aux | grep "tune_all_challenges.py" | grep -v grep | awk '{print $2}' || true)
if [ -n "$REMAINING" ]; then
    echo "Found remaining Python processes: $REMAINING"
    echo "Stopping them..."
    echo "$REMAINING" | xargs kill -TERM 2>/dev/null || true
    sleep 3
    echo "$REMAINING" | xargs kill -KILL 2>/dev/null || true
    echo "  ✓ Cleaned up remaining processes"
    echo ""
fi

# Wait a moment for cleanup
sleep 2

echo "=================================================================================="
echo "Starting New Tuning with GPU Optimizations"
echo "=================================================================================="
echo ""
echo "New optimizations:"
echo "  ✓ Batch size: 128 (GPU optimized, was 64)"
echo "  ✓ Mixed precision training: Enabled"
echo "  ✓ GPU memory growth: Enabled"
echo ""
echo "Expected improvements:"
echo "  • ~2x faster training (15-20ms/step vs 27-30ms/step)"
echo "  • ~40-50 hours total (vs ~72 hours)"
echo ""

# Start new tuning process
echo "Starting tuning..."
echo ""

./run_tuning_background_improved.sh

echo ""
echo "=================================================================================="
echo "Tuning restarted successfully!"
echo "=================================================================================="
echo ""
echo "Monitor progress:"
echo "  tail -f logs/tuning/tuning_background_*.log"
echo "  ./check_tuning_realtime.sh"
echo "  watch -n 1 nvidia-smi  # Monitor GPU usage"
echo ""

