#!/bin/bash
# Safe restart script for tuning - kills stuck process and restarts with resume support

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

LOG_DIR="logs/tuning"
PID_FILE=$(ls -t "$LOG_DIR"/tuning_background_*.pid 2>/dev/null | head -1)

if [ -z "$PID_FILE" ]; then
    echo "No PID file found. Starting fresh tuning..."
else
    OLD_PID=$(cat "$PID_FILE")
    if ps -p "$OLD_PID" > /dev/null 2>&1; then
        echo "Found running process PID: $OLD_PID"
        echo "Checking if it's stuck..."
        
        # Check last log update time
        LOG_FILE="${PID_FILE%.pid}.log"
        if [ -f "$LOG_FILE" ]; then
            LAST_MODIFIED=$(stat -c %Y "$LOG_FILE")
            NOW=$(date +%s)
            AGE=$((NOW - LAST_MODIFIED))
            
            if [ $AGE -gt 3600 ]; then  # Stuck for more than 1 hour
                echo "Process appears stuck (log not updated for $((AGE/60)) minutes)"
                echo "Killing stuck process..."
                kill "$OLD_PID" 2>/dev/null || true
                sleep 2
                # Force kill if still running
                if ps -p "$OLD_PID" > /dev/null 2>&1; then
                    echo "Force killing..."
                    kill -9 "$OLD_PID" 2>/dev/null || true
                fi
                echo "Process killed."
            else
                echo "Process seems active (log updated $((AGE/60)) minutes ago)"
                echo "If you want to restart anyway, kill it manually: kill $OLD_PID"
                exit 1
            fi
        else
            echo "Log file not found, killing process anyway..."
            kill "$OLD_PID" 2>/dev/null || true
            sleep 2
            kill -9 "$OLD_PID" 2>/dev/null || true
        fi
        
        # Clean up PID file
        rm -f "$PID_FILE"
    else
        echo "PID file exists but process is not running. Cleaning up..."
        rm -f "$PID_FILE"
    fi
fi

echo ""
echo "Starting tuning with resume support..."
echo "The script will skip already-completed configurations."
echo ""

# Start the tuning script
exec ./run_tuning_background_improved.sh

