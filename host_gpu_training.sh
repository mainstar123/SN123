#!/bin/bash
echo "🚀 MANTIS GPU Training on RTX4090 Host"
echo "========================================"
echo "Run this script directly on your RTX4090 host machine"
echo "NOT in the container environment"
echo ""

# This script should be run on the host RTX4090, not in container
if [ -f "/.dockerenv" ] || grep -q "docker\|container" /proc/1/cgroup 2>/dev/null; then
    echo "❌ ERROR: You are running this inside a container!"
    echo ""
    echo "📋 To run on your RTX4090 host:"
    echo "1. SSH to your RTX4090 host: ssh user@your-host-ip"
    echo "2. Navigate to project: cd /path/to/your/Nereus/SN123"
    echo "3. Run this script: ./host_gpu_training.sh"
    echo ""
    echo "Or copy this script to your host and run it there."
    exit 1
fi

echo "🎯 Checking GPU availability..."
if ! command -v nvidia-smi &> /dev/null; then
    echo "❌ NVIDIA drivers not found!"
    exit 1
fi

nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader,nounits
echo ""

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "📦 Creating virtual environment..."
    python3 -m venv venv
fi

echo "🔧 Activating environment..."
source venv/bin/activate

echo "📦 Installing/updating dependencies..."
pip install -r requirements.txt --quiet

echo ""
echo "🎯 Starting hyperparameter tuning with GPU..."
echo "Will automatically resume from stopped point"
echo ""

# Force GPU usage and start training
export TF_FORCE_GPU_ALLOW_GROWTH=true
export CUDA_VISIBLE_DEVICES=0

nohup python scripts/training/tune_all_challenges.py \
    --data-dir data \
    --tuning-dir models/tuned \
    --epochs 100 \
    --batch-size 128 \
    > logs/training/host_gpu_training_$(date +%Y%m%d_%H%M%S).log 2>&1 &

PID=$!
echo $PID > logs/training/host_training.pid

echo "✅ GPU Training started on host!"
echo "Process ID: $PID"
echo "PID saved to: logs/training/host_training.pid"
echo ""
echo "========================================"
echo "📊 MONITORING COMMANDS"
echo "========================================"
echo ""
echo "Check if running:"
echo "  ps aux | grep tune_all_challenges.py | grep -v grep"
echo ""
echo "Watch live progress:"
echo "  tail -f logs/training/host_gpu_training_*.log"
echo ""
echo "Monitor GPU usage:"
echo "  watch -n 5 nvidia-smi"
echo ""
echo "Check system resources:"
echo "  htop"
echo ""
echo "Stop training:"
echo "  kill $PID"
echo ""
echo "========================================"
echo "⏱️ EXPECTED TIMELINE"
echo "========================================"
echo "• GPU Detection: 30 seconds"
echo "• Challenge scanning: 2-5 minutes (skips completed)"
echo "• Training per challenge: 45-90 minutes"
echo "• Total: 2-4 hours for remaining challenges"
echo ""
echo "🎉 GPU training active on RTX4090!"
echo "========================================"
