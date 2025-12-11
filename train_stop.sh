#!/bin/bash
#
# Stop background training
#

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PID_FILE="$SCRIPT_DIR/logs/training/training.pid"

if [ ! -f "$PID_FILE" ]; then
    echo "❌ No training process found (PID file doesn't exist)"
    exit 1
fi

TRAINING_PID=$(cat "$PID_FILE")

if ! ps -p "$TRAINING_PID" > /dev/null 2>&1; then
    echo "⚠️  Process $TRAINING_PID is not running"
    echo "   Cleaning up PID file..."
    rm -f "$PID_FILE"
    exit 1
fi

echo "🛑 Stopping training process (PID: $TRAINING_PID)..."
kill "$TRAINING_PID"

# Wait for process to stop
for i in {1..10}; do
    if ! ps -p "$TRAINING_PID" > /dev/null 2>&1; then
        echo "✅ Training stopped successfully"
        rm -f "$PID_FILE"
        exit 0
    fi
    sleep 1
done

# Force kill if still running
if ps -p "$TRAINING_PID" > /dev/null 2>&1; then
    echo "⚠️  Process didn't stop gracefully, forcing..."
    kill -9 "$TRAINING_PID"
    sleep 1
    if ! ps -p "$TRAINING_PID" > /dev/null 2>&1; then
        echo "✅ Training stopped (forced)"
        rm -f "$PID_FILE"
        exit 0
    else
        echo "❌ Failed to stop training"
        exit 1
    fi
fi

