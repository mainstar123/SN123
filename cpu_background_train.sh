#!/bin/bash
echo "🚀 Starting MANTIS CPU Training (Background Mode)"
echo "=================================================="
echo "TensorFlow CPU + Resume from stopped point"
echo "Will skip completed challenges automatically"
echo ""

# Check if training is already running
if ps aux | grep -v grep | grep -q "tune_all_challenges.py"; then
    echo "⚠️  Training is already running!"
    echo "Check: ps aux | grep tune_all_challenges.py"
    exit 1
fi

# Activate virtual environment and start training
echo "🎯 Starting hyperparameter tuning..."
source venv/bin/activate

# Force CPU-only to avoid CUDA issues
export CUDA_VISIBLE_DEVICES=""

nohup python scripts/training/tune_all_challenges.py \
    --data-dir data \
    --tuning-dir models/tuned \
    --epochs 100 \
    --batch-size 64 \
    > logs/training/cpu_background_training_$(date +%Y%m%d_%H%M%S).log 2>&1 &

PID=$!
echo $PID > logs/training/cpu_training.pid

echo "✅ CPU Training started in background!"
echo "Process ID: $PID"
echo "PID saved to: logs/training/cpu_training.pid"
echo ""
echo "=================================================="
echo "📊 MONITORING COMMANDS"
echo "=================================================="
echo ""
echo "Check if running:"
echo "  ps aux | grep tune_all_challenges.py | grep -v grep"
echo ""
echo "Watch live progress:"
echo "  tail -f logs/training/cpu_background_training_*.log"
echo ""
echo "Check system resources:"
echo "  htop  # or: top"
echo ""
echo "View recent progress:"
echo "  tail -50 logs/training/cpu_background_training_*.log | grep -E '(Training|Epoch|complete|Progress)'"
echo ""
echo "Stop training:"
echo "  kill $PID  # or: kill \$(cat logs/training/cpu_training.pid)"
echo ""
echo "=================================================="
echo "⏱️ EXPECTED TIMELINE (CPU)"
echo "=================================================="
echo "• Initialization: 1-2 minutes"
echo "• Challenge scanning: 2-5 minutes (skips completed)"
echo "• Training per challenge: 2-4 hours (slower on CPU)"
echo "• Total: 6-12 hours for remaining challenges"
echo ""
echo "💡 CPU Training Notes:"
echo "• Slower than GPU but reliable"
echo "• Uses ~4-8 CPU cores"
echo "• Memory usage: ~2-4GB RAM"
echo "• Will complete successfully"
echo ""
echo "🎉 Training is running in background!"
echo "=================================================="
