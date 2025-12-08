# GPU Setup Guide for MANTIS Training

## Current Status

✅ **GPU support is now enabled** in the codebase!

The training code will automatically:
- Detect and use GPU for TensorFlow/Keras LSTM training
- Use GPU for XGBoost if available
- Fall back to CPU if GPU is not available

## GPU Detection

When you run training, you'll see:
```
✓ GPU detected: 1 device(s) available
  Using GPU: /physical_device:GPU:0
  Model will use GPU: /physical_device:GPU:0
  XGBoost will use GPU
```

Or if no GPU:
```
ℹ No GPU detected, using CPU
  Model will use CPU
  XGBoost will use CPU
```

## Requirements

### For TensorFlow GPU:
```bash
# Install TensorFlow with GPU support
pip install tensorflow[and-cuda]

# Or if you have CUDA installed separately:
pip install tensorflow-gpu
```

### For XGBoost GPU:
```bash
# XGBoost with GPU support (requires CUDA)
pip install xgboost[gpu]

# Or build from source with GPU support
```

## Usage

### Default (Auto-detect GPU):
```bash
python scripts/training/train_model.py --ticker BTC --data-dir data/raw
```

### Force GPU:
```bash
python scripts/training/train_model.py --ticker BTC --use-gpu
```

### Force CPU:
```bash
python scripts/training/train_model.py --ticker BTC --use-cpu
```

## Performance

### Expected Speedup with GPU:

| Component | CPU Time | GPU Time | Speedup |
|-----------|----------|----------|---------|
| LSTM Training | 30-60 min | 10-20 min | 2-3x |
| XGBoost Training | 5-10 min | 1-2 min | 3-5x |
| **Total** | **35-70 min** | **11-22 min** | **~3x** |

*Times are per ticker, for 100 epochs*

## GPU Memory Management

The code automatically:
- Enables memory growth (doesn't allocate all GPU memory at once)
- Prevents OOM errors
- Allows multiple processes to share GPU

## Troubleshooting

### Issue: "No GPU detected"

**Check:**
```bash
# Check if GPU is visible to TensorFlow
python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"

# Check NVIDIA driver
nvidia-smi
```

**Solution:**
- Install NVIDIA drivers
- Install CUDA toolkit
- Install cuDNN
- Install tensorflow[and-cuda] or tensorflow-gpu

### Issue: "XGBoost GPU not working"

**Check:**
```bash
# Check if XGBoost GPU is available
python -c "import xgboost as xgb; print(xgb.get_config())"
```

**Solution:**
```bash
# Reinstall XGBoost with GPU support
pip uninstall xgboost
pip install xgboost[gpu]
```

### Issue: "Out of Memory (OOM)"

**Solution:**
- Reduce batch size: `--batch-size 32` (default: 64)
- The code already uses memory growth, but you can reduce batch size further

## Multi-GPU Support

Currently, the code uses a single GPU. For multi-GPU training, you would need to:
1. Use TensorFlow's `tf.distribute.MirroredStrategy()`
2. Modify the model training code

This is not currently implemented but can be added if needed.

## Verification

Test GPU availability:
```bash
python -c "
import tensorflow as tf
print('TensorFlow version:', tf.__version__)
print('GPU devices:', tf.config.list_physical_devices('GPU'))
print('CUDA available:', tf.test.is_built_with_cuda())
"
```

---

**Last Updated**: 2025-01-15
**Status**: ✅ GPU support enabled and tested

