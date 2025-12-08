#!/bin/bash
# Enhanced script to check training status for all tickers

echo "=================================================================================="
echo "MANTIS Training Status Check"
echo "=================================================================================="
echo ""

# Configuration
MODEL_DIR="models/checkpoints"
LOG_DIR="logs/training"
PID_FILE="$LOG_DIR/train_all.pid"
STATUS_FILE="$LOG_DIR/train_all.status"

# Check if training process is running
echo "=== Training Process Status ==="
if [ -f "$PID_FILE" ]; then
    TRAIN_PID=$(cat "$PID_FILE")
    if ps -p "$TRAIN_PID" > /dev/null 2>&1; then
        echo "✓ Training process is RUNNING"
        echo "  PID: $TRAIN_PID"
        
        # Get process info
        if command -v ps > /dev/null; then
            CPU=$(ps -p "$TRAIN_PID" -o %cpu --no-headers 2>/dev/null | tr -d ' ')
            MEM=$(ps -p "$TRAIN_PID" -o %mem --no-headers 2>/dev/null | tr -d ' ')
            if [ -n "$CPU" ]; then
                echo "  CPU: ${CPU}% | Memory: ${MEM}%"
            fi
        fi
        
        # Check status file
        if [ -f "$STATUS_FILE" ]; then
            echo "  Status: $(grep -v "^Started:\|^PID:" "$STATUS_FILE" | head -1)"
            if grep -q "Started:" "$STATUS_FILE"; then
                STARTED=$(grep "Started:" "$STATUS_FILE" | cut -d: -f2-)
                echo "  Started: $STARTED"
            fi
        fi
    else
        echo "✗ Training process is NOT running (PID file exists but process not found)"
        rm -f "$PID_FILE"
    fi
else
    echo "✗ Training process is NOT running (no PID file)"
fi

echo ""
echo "=== Model Training Progress ==="
echo ""

# Get all challenges from config
if [ -f "config.py" ]; then
    # Extract tickers from config.py
    TICKERS=$(python3 -c "
import sys
sys.path.insert(0, '.')
import config
tickers = [c['ticker'] for c in config.CHALLENGES]
print(' '.join(tickers))
" 2>/dev/null)
    
    if [ -z "$TICKERS" ]; then
        # Fallback: check what models exist
        echo "  Checking existing models..."
        TICKERS=$(ls -1 "$MODEL_DIR" 2>/dev/null | head -20)
    fi
else
    # Fallback: check what models exist
    TICKERS=$(ls -1 "$MODEL_DIR" 2>/dev/null | head -20)
fi

if [ -z "$TICKERS" ]; then
    echo "  No tickers found. Check config.py or model directory."
    echo ""
else
    TOTAL=$(echo "$TICKERS" | wc -w)
    COMPLETED=0
    IN_PROGRESS=0
    FAILED=0
    
    echo "  Total tickers to train: $TOTAL"
    echo ""
    
    for ticker in $TICKERS; do
        TICKER_DIR="$MODEL_DIR/$ticker"
        
        if [ -d "$TICKER_DIR" ]; then
            # Check if model is complete
            if [ -f "$TICKER_DIR/lstm_model.h5" ] && \
               [ -f "$TICKER_DIR/xgb_model.json" ] && \
               [ -f "$TICKER_DIR/config.json" ]; then
                # Check file modification time
                LAST_MOD=$(stat -c %Y "$TICKER_DIR/lstm_model.h5" 2>/dev/null || echo 0)
                NOW=$(date +%s)
                AGE=$((NOW - LAST_MOD))
                
                if [ $AGE -lt 3600 ]; then
                    # Modified in last hour - might be in progress
                    echo "  ⏳ $ticker: Training in progress (modified $((AGE/60)) min ago)"
                    IN_PROGRESS=$((IN_PROGRESS + 1))
                else
                    # Older - likely completed
                    SIZE=$(du -sh "$TICKER_DIR" 2>/dev/null | cut -f1)
                    echo "  ✓ $ticker: Completed (size: $SIZE)"
                    COMPLETED=$((COMPLETED + 1))
                fi
            else
                echo "  ⏳ $ticker: In progress (partial files)"
                IN_PROGRESS=$((IN_PROGRESS + 1))
            fi
        else
            echo "  ⏸  $ticker: Not started"
        fi
    done
    
    echo ""
    echo "  Summary:"
    echo "    ✓ Completed: $COMPLETED/$TOTAL"
    echo "    ⏳ In Progress: $IN_PROGRESS"
    echo "    ⏸  Not Started: $((TOTAL - COMPLETED - IN_PROGRESS))"
fi

echo ""
echo "=== Recent Log Activity ==="
echo ""

# Find most recent log file
LATEST_LOG=$(ls -t "$LOG_DIR"/train_all_*.log 2>/dev/null | head -1)

if [ -n "$LATEST_LOG" ] && [ -f "$LATEST_LOG" ]; then
    echo "  Latest log: $LATEST_LOG"
    echo ""
    echo "  Last 5 lines:"
    tail -5 "$LATEST_LOG" | sed 's/^/    /'
    echo ""
    echo "  To view full log:"
    echo "    tail -f $LATEST_LOG"
else
    echo "  No log files found in $LOG_DIR"
fi

echo ""
echo "=== Quick Commands ==="
echo ""
echo "  View live logs:"
echo "    tail -f $LATEST_LOG"
echo ""
echo "  Check specific ticker:"
echo "    ls -lh $MODEL_DIR/<TICKER>/"
echo ""
echo "  Stop training:"
if [ -f "$PID_FILE" ]; then
    TRAIN_PID=$(cat "$PID_FILE")
    echo "    kill $TRAIN_PID"
else
    echo "    (No training process running)"
fi
echo ""
echo "  Check results:"
echo "    python scripts/training/check_training_results.py --model-dir $MODEL_DIR"
echo ""
