#!/bin/bash
echo "🚀 Starting MANTIS GPU Training in Docker (Background Mode)"
echo "=========================================================="
echo "TensorFlow 2.17.0 + CUDA 12.3 + RTX4090 GPU"
echo "Will resume from stopped point automatically"
echo ""

# Check if container is already running
if docker ps | grep -q mantis_training; then
    echo "⚠️  Training container already running!"
    echo "Stop it first: docker stop mantis_training"
    exit 1
fi

# Start training in background
echo "🎯 Starting hyperparameter tuning..."
docker run --gpus all --rm -d \
    --name mantis_training \
    -v /home/ocean/Nereus/SN123:/workspace \
    -w /workspace \
    tensorflow/tensorflow:2.17.0-gpu \
    bash -c "
    echo '📦 Installing dependencies...' && \
    pip install -r requirements.txt --quiet && \
    echo '✅ Dependencies installed' && \
    echo '' && \
    echo '🎯 Starting hyperparameter tuning (resume mode)...' && \
    python scripts/training/tune_all_challenges.py \
        --data-dir data \
        --tuning-dir models/tuned \
        --epochs 100 \
        --batch-size 128 \
        2>&1 | tee logs/training/docker_gpu_training_$(date +%Y%m%d_%H%M%S).log
    "

echo "✅ Training started in background!"
echo "Container ID: $(docker ps | grep mantis_training | awk '{print $1}')"
echo ""
echo "=========================================================="
echo "📊 MONITORING COMMANDS"
echo "=========================================================="
echo ""
echo "Check if running:"
echo "  docker ps | grep mantis_training"
echo ""
echo "Watch live progress:"
echo "  tail -f logs/training/docker_gpu_training_*.log"
echo ""
echo "Check GPU usage:"
echo "  watch -n 5 nvidia-smi"
echo ""
echo "View training logs:"
echo "  docker logs mantis_training"
echo ""
echo "Stop training:"
echo "  docker stop mantis_training"
echo ""
echo "=========================================================="
echo "⏱️ EXPECTED TIMELINE"
echo "=========================================================="
echo "• Initialization: 2-5 minutes"
echo "• Challenge scanning: 5-10 minutes (skips completed)"
echo "• Training per challenge: 45-90 minutes"
echo "• Total: 2-4 hours for remaining challenges"
echo ""
echo "🎉 Training is running in background!"
echo "=========================================================="
