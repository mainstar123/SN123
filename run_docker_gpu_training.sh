#!/bin/bash
echo "🚀 MANTIS GPU Training (Container Mode)"
echo "========================================"
echo "Modified for container environments with GPU access"
echo ""

# Check Docker availability
if ! command -v docker &> /dev/null; then
    echo "❌ Docker not found!"
    echo "Please install Docker first using the commands above"
    exit 1
fi

# Check NVIDIA GPU support (container environment)
if ! nvidia-smi &> /dev/null; then
    echo "❌ NVIDIA GPU not available!"
    echo "Please ensure NVIDIA drivers are installed and GPU is accessible"
    exit 1
fi

echo "✅ GPU detected:"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader,nounits
echo ""

# Check if container is already running
if docker ps -a | grep -q mantis_training; then
    echo "🧹 Cleaning up old containers..."
    docker stop mantis_training 2>/dev/null || true
    docker rm mantis_training 2>/dev/null || true
fi

echo "🎯 Starting GPU training in background (Container Mode)..."

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "📦 Creating virtual environment..."
    python3 -m venv venv
fi

echo "🔧 Activating environment..."
source venv/bin/activate

echo "📦 Installing/updating dependencies..."
pip install -r requirements.txt --quiet

echo "✅ Dependencies installed"
echo ""

# Force GPU usage
export TF_FORCE_GPU_ALLOW_GROWTH=true
export CUDA_VISIBLE_DEVICES=0

echo "🎯 Starting hyperparameter tuning..."
nohup python scripts/training/tune_all_challenges.py \
        --data-dir data \
        --tuning-dir models/tuned \
        --epochs 100 \
        --batch-size 128 \
    > logs/training/container_gpu_training_$(date +%Y%m%d_%H%M%S).log 2>&1 &

PID=$!
echo $PID > logs/training/container_training.pid

if [ $? -eq 0 ]; then
    echo "✅ Training started successfully!"
    echo "Process ID: $PID"
    echo "PID saved to: logs/training/container_training.pid"
    echo ""
    echo "==============================================="
    echo "📊 MONITORING COMMANDS"
    echo "==============================================="
    echo "Check status:     ps aux | grep tune_all_challenges.py | grep -v grep"
    echo "Watch logs:       tail -f logs/training/container_gpu_training_*.log"
    echo "View progress:    tail -f logs/training/container_gpu_training_*.log"
    echo "Check GPU:        watch -n 5 nvidia-smi"
    echo "Stop training:    kill $PID"
    echo ""
    echo "⏱️ Expected: 2-4 hours total"
    echo "🎉 Background training active!"
else
    echo "❌ Failed to start training"
    exit 1
fi
