# GPU-Accelerated Hyperparameter Tuning Guide

## Overview

This guide explains how to enable GPU acceleration for MANTIS hyperparameter tuning, which can reduce tuning time from **~90-100 hours (CPU)** to **~30-40 hours (GPU)** — a **2-3x speedup**.

---

## What Was Changed

### 1. **Shell Scripts Updated for GPU**

The following scripts have been updated to use GPU-optimized batch sizes and environment variables:

- ✅ `tune_all_challenges.sh` - Batch size increased from 64 → 128
- ✅ `quick_tune.sh` - GPU detection and batch size optimization
- ✅ `run_tuning_background_improved.sh` - TensorFlow GPU environment variables

### 2. **Python Scripts Enhanced**

- ✅ `scripts/training/hyperparameter_tuning.py` - Sets TF environment variables before import
- ✅ `scripts/training/tune_all_challenges.py` - GPU verification at startup, batch size 128 default

### 3. **Automatic GPU Configuration**

The following are already configured in `scripts/training/model_architecture.py`:

- ✅ **GPU memory growth** - Prevents TensorFlow from allocating all GPU memory at once
- ✅ **Mixed precision training** (FP16) - ~2x faster training on modern GPUs (RTX 20xx/30xx/40xx, A100, etc.)
- ✅ **XGBoost GPU support** - Uses `gpu_hist` tree method when available

---

## Quick Start

### 1. **Check GPU Status**

First, verify your GPU is properly configured:

```bash
./check_gpu_status.sh
```

This will show:
- ✅ NVIDIA driver and CUDA version
- ✅ TensorFlow GPU detection
- ✅ XGBoost CUDA support
- ✅ Environment variables
- 📊 Recommendations

### 2. **Start GPU-Optimized Tuning**

**Option A: Interactive tuning (single challenge)**
```bash
./quick_tune.sh
```

**Option B: All challenges tuning**
```bash
./tune_all_challenges.sh
```

**Option C: Background tuning (recommended for long runs)**
```bash
./run_tuning_background_improved.sh

# Monitor progress
./check_tuning_realtime.sh

# Or watch GPU usage
watch -n 1 nvidia-smi
```

**Option D: Restart existing tuning with GPU optimizations**
```bash
./restart_tuning_with_gpu.sh
```

---

## GPU Configuration Details

### Environment Variables

The scripts now automatically set these TensorFlow environment variables:

```bash
export TF_FORCE_GPU_ALLOW_GROWTH=true      # Prevents allocating all GPU memory
export TF_GPU_THREAD_MODE=gpu_private      # Optimizes GPU threading
export TF_GPU_THREAD_COUNT=2               # Number of GPU threads
export CUDA_VISIBLE_DEVICES=0              # Use first GPU (GPU:0)
```

### Batch Size Optimization

| Hardware | Old Batch Size | New Batch Size | Speedup |
|----------|----------------|----------------|---------|
| **CPU**  | 64             | 64 (unchanged) | 1x      |
| **GPU**  | 64             | 128            | ~2-3x   |

**Why larger batch size on GPU?**
- GPUs have thousands of cores and excel at parallel processing
- Larger batches = better GPU utilization
- Batch size 128-256 is optimal for RTX 3090 with 24GB VRAM

### Mixed Precision Training (FP16)

Automatically enabled in `model_architecture.py`:

```python
policy = tf.keras.mixed_precision.Policy('mixed_float16')
tf.keras.mixed_precision.set_global_policy(policy)
```

**Benefits:**
- ✅ ~2x faster training (tensor cores on modern GPUs)
- ✅ ~50% lower memory usage
- ✅ Same accuracy (mixed precision uses FP32 for sensitive operations)

**Supported GPUs:**
- NVIDIA RTX 20xx, 30xx, 40xx series
- NVIDIA Tesla V100, A100, H100
- NVIDIA Titan RTX

---

## Troubleshooting

### GPU Not Detected

**Symptom:**
```
⚠️ WARNING: No GPU detected - training will be SLOW on CPU!
```

**Solutions:**

1. **Install NVIDIA drivers**
   ```bash
   # Check current driver
   nvidia-smi
   
   # If not found, install drivers:
   # Ubuntu/Debian:
   sudo apt update
   sudo apt install nvidia-driver-535  # or latest version
   sudo reboot
   ```

2. **Install CUDA toolkit** (if needed)
   ```bash
   # TensorFlow 2.13+ includes CUDA libraries
   # But you may need CUDA toolkit for XGBoost
   wget https://developer.download.nvidia.com/compute/cuda/12.2.0/local_installers/cuda_12.2.0_535.54.03_linux.run
   sudo sh cuda_12.2.0_535.54.03_linux.run
   ```

3. **Install TensorFlow with GPU support**
   ```bash
   # Modern TensorFlow (2.13+) includes CUDA
   pip install --upgrade tensorflow[and-cuda]
   
   # Verify
   python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
   ```

4. **Install XGBoost with GPU support** (optional, for faster XGBoost)
   ```bash
   pip install xgboost
   # GPU support is automatically detected if CUDA is available
   ```

### Out of Memory Error

**Symptom:**
```
ResourceExhaustedError: OOM when allocating tensor
```

**Solutions:**

1. **Reduce batch size** (automatic fallback in scripts)
   ```bash
   # Edit the script or use environment variable
   export BATCH_SIZE=64
   ./tune_all_challenges.sh
   ```

2. **Enable memory growth** (already done automatically)
   ```python
   for gpu in gpus:
       tf.config.experimental.set_memory_growth(gpu, True)
   ```

3. **Reduce model size** (if still OOM)
   ```python
   # In hyperparameter configs, use:
   lstm_hidden: 128  # instead of 256 or 512
   ```

### Slow GPU Performance

**Check:**

1. **GPU utilization**
   ```bash
   watch -n 1 nvidia-smi
   # Should show 80-100% GPU utilization during training
   ```

2. **Mixed precision enabled**
   ```bash
   python check_gpu_status.py
   # Should show: "Mixed precision: mixed_float16"
   ```

3. **Batch size is large enough**
   ```bash
   # For RTX 3090 (24GB), try batch size 256
   python scripts/training/tune_all_challenges.py --batch-size 256
   ```

4. **Data loading bottleneck**
   - The bottleneck might be CPU data preprocessing, not GPU
   - This is normal and already optimized in the code

---

## Performance Comparison

### Expected Training Times

| Configuration | Time per Epoch | Total Time (13 challenges × 9 configs) |
|---------------|----------------|----------------------------------------|
| **CPU** (64 batch) | ~30ms/step | ~90-100 hours |
| **GPU** (128 batch, FP32) | ~20ms/step | ~50-60 hours |
| **GPU** (128 batch, FP16) | ~15ms/step | **~30-40 hours** ✅ |

### GPU Utilization Targets

During training, you should see:

```bash
$ nvidia-smi

+-----------------------------------------------------------------------------+
| NVIDIA-SMI 535.54.03    Driver Version: 535.54.03    CUDA Version: 12.2     |
|-------------------------------+----------------------+----------------------+
| GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
|===============================+======================+======================|
|   0  NVIDIA GeForce ...  Off  | 00000000:01:00.0 Off |                  N/A |
| 40%   65C    P2   280W / 350W |  18000MiB / 24576MiB |    95%      Default |
+-------------------------------+----------------------+----------------------+
```

**Good indicators:**
- ✅ GPU-Util: 80-100%
- ✅ Power: Close to max TDP
- ✅ Memory: 10-20GB used (for batch size 128)
- ✅ Temp: 60-80°C is normal

---

## Advanced: Multi-GPU Setup

If you have multiple GPUs, you can parallelize tuning:

### Option 1: Run multiple tuning processes (recommended)

```bash
# Terminal 1: Tune challenges 1-4 on GPU 0
CUDA_VISIBLE_DEVICES=0 python scripts/training/tune_all_challenges.py --ticker ETH &

# Terminal 2: Tune challenges 5-8 on GPU 1
CUDA_VISIBLE_DEVICES=1 python scripts/training/tune_all_challenges.py --ticker EURUSD &

# Monitor both
watch -n 1 'nvidia-smi'
```

### Option 2: Data parallelism (requires code changes)

Not currently implemented. Would require:
- TensorFlow `tf.distribute.MirroredStrategy`
- Model refactoring to support distributed training

---

## Verification Checklist

Before starting a long tuning run, verify:

- [ ] ✅ `./check_gpu_status.sh` shows GPU detected
- [ ] ✅ TensorFlow can see GPU devices
- [ ] ✅ Mixed precision is enabled (FP16)
- [ ] ✅ Memory growth is enabled
- [ ] ✅ Batch size is set to 128 (not 64)
- [ ] ✅ `nvidia-smi` shows GPU utilization during training
- [ ] ✅ Virtual environment is activated

---

## Summary of Changes

### Files Modified

1. **Shell Scripts:**
   - `tune_all_challenges.sh` - GPU batch size (128), env vars, GPU detection
   - `quick_tune.sh` - GPU detection and optimization
   - `run_tuning_background_improved.sh` - TensorFlow GPU env vars

2. **Python Scripts:**
   - `scripts/training/hyperparameter_tuning.py` - Set TF env vars before import
   - `scripts/training/tune_all_challenges.py` - GPU verification, batch size default 128

3. **New Files:**
   - `check_gpu_status.py` - Comprehensive GPU diagnostics
   - `check_gpu_status.sh` - Shell wrapper for GPU diagnostics
   - `GPU_TUNING_GUIDE.md` - This guide

### What's Already Configured (No Changes Needed)

- `scripts/training/model_architecture.py` - GPU configuration (lines 19-41)
  - Memory growth
  - Mixed precision (FP16)
  - GPU device selection
  - XGBoost GPU support

---

## Next Steps

1. **Verify GPU setup:**
   ```bash
   ./check_gpu_status.sh
   ```

2. **Start GPU-optimized tuning:**
   ```bash
   # Background mode (recommended)
   ./run_tuning_background_improved.sh
   
   # Monitor
   ./check_tuning_realtime.sh
   watch -n 1 nvidia-smi
   ```

3. **If already running tuning, restart with GPU optimizations:**
   ```bash
   ./restart_tuning_with_gpu.sh
   ```

---

## Support

If you encounter issues:

1. Run diagnostics: `./check_gpu_status.sh`
2. Check logs: `tail -f logs/tuning/tuning_background_*.log`
3. Monitor GPU: `watch -n 1 nvidia-smi`
4. Check this guide for troubleshooting steps

---

**Estimated Time Savings: 50-60 hours (GPU vs CPU) 🚀**

