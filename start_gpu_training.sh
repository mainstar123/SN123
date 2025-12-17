#!/bin/bash

echo "🚀 MANTIS GPU Training - Background Mode"
echo "========================================"
echo "Starting hyperparameter tuning on RTX4090"
echo ""

# Check if already running (exclude grep process itself)
if ps aux | grep "tune_all_challenges" | grep -v grep > /dev/null; then
    echo "❌ Training already running!"
    echo "Check: ps aux | grep tune_all_challenges | grep -v grep"
    echo ""
    echo "If you want to start anyway, run:"
    echo "pkill -f tune_all_challenges"
    echo "Then try again."
    exit 1
fi

# Clean up any stale PID files
if [ -f "logs/training/training.pid" ]; then
    OLD_PID=$(cat logs/training/training.pid 2>/dev/null)
    if ! ps -p $OLD_PID > /dev/null 2>&1; then
        echo "🧹 Removing stale PID file (process $OLD_PID not running)"
        rm -f logs/training/training.pid
    fi
fi

# Activate virtual environment and start training
source .venv/bin/activate

echo "🎯 Starting training in background..."
nohup python scripts/training/tune_all_challenges.py \
    --data-dir data \
    --tuning-dir models/tuned \
    --epochs 100 \
    --batch-size 128 \
    > logs/training/gpu_training_$(date +%Y%m%d_%H%M%S).log 2>&1 &

PID=$!
echo $PID > logs/training/training.pid

echo "✅ Training started successfully!"
echo "Process ID: $PID"
echo "PID saved to: logs/training/training.pid"
echo ""
echo "📊 MONITORING:"
echo "  Check status:    ps aux | grep tune_all_challenges"
echo "  View logs:       tail -f logs/training/gpu_training_*.log"
echo "  Monitor GPU:     watch -n 5 nvidia-smi"
echo "  Stop training:   kill $PID"
echo ""
echo "⏱️ Expected: 4-8 hours for remaining 6 challenges"
echo "🎉 Background training active!"
