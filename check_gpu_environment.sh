#!/bin/bash

echo "🔍 GPU Environment Check"
echo "========================"
echo ""

echo "📊 System Information:"
echo "  Hostname: $(hostname)"
echo "  OS: $(uname -s) $(uname -r)"
echo ""

echo "🎮 GPU Check:"
if command -v nvidia-smi &> /dev/null; then
    echo "  ✅ NVIDIA drivers found"
    nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader,nounits
else
    echo "  ❌ NVIDIA drivers not found"
fi
echo ""

echo "🐳 Container Check:"
if [ -f /.dockerenv ] || grep -q "docker\|container" /proc/1/cgroup 2>/dev/null; then
    echo "  ⚠️  WARNING: Running in container environment"
    echo "     GPU access may not be available"
    echo "     Consider running on host machine"
else
    echo "  ✅ Running on host machine"
fi
echo ""

echo "🐍 Python Environment:"
if [ -d ".venv" ]; then
    echo "  ✅ Virtual environment found"
    source .venv/bin/activate
    which python
    python --version
    python -c "import tensorflow as tf; print('  TensorFlow version:', tf.__version__)" 2>/dev/null || echo "  ❌ TensorFlow not available"
else
    echo "  ❌ Virtual environment not found"
fi
echo ""

if [ -f /.dockerenv ] || grep -q "docker\|container" /proc/1/cgroup 2>/dev/null; then
    echo "💡 RECOMMENDATION:"
    echo "   Run training on your GPU host machine, not in container"
    echo "   Use: ssh user@your-host-ip"
    echo "   Then: cd /path/to/project && ./start_gpu_training.sh"
else
    echo "✅ READY TO START TRAINING:"
    echo "   Run: ./start_gpu_training.sh"
fi
