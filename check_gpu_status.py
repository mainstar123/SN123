#!/usr/bin/env python3
"""
GPU Status Checker for MANTIS Hyperparameter Tuning
Verifies GPU configuration and provides recommendations
"""

import os
import sys

def check_gpu_status():
    """Check GPU availability and configuration"""
    print("=" * 80)
    print("MANTIS GPU Configuration Check")
    print("=" * 80)
    print()
    
    # Check CUDA
    print("1. Checking CUDA Installation...")
    try:
        import subprocess
        result = subprocess.run(['nvidia-smi'], capture_output=True, text=True)
        if result.returncode == 0:
            print("   ✓ nvidia-smi found")
            # Parse GPU info
            lines = result.stdout.split('\n')
            for line in lines:
                if 'NVIDIA' in line or 'CUDA Version' in line:
                    print(f"   {line.strip()}")
        else:
            print("   ✗ nvidia-smi not found or failed")
            print("   → Install NVIDIA drivers: https://www.nvidia.com/Download/index.aspx")
    except FileNotFoundError:
        print("   ✗ nvidia-smi not found")
        print("   → Install NVIDIA drivers: https://www.nvidia.com/Download/index.aspx")
    except Exception as e:
        print(f"   ⚠️  Error checking nvidia-smi: {e}")
    
    print()
    
    # Check TensorFlow
    print("2. Checking TensorFlow GPU Support...")
    try:
        import tensorflow as tf
        print(f"   ✓ TensorFlow version: {tf.__version__}")
        
        # Check GPU devices
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            print(f"   ✓ TensorFlow can see {len(gpus)} GPU(s):")
            for i, gpu in enumerate(gpus):
                print(f"     {i+1}. {gpu.name}")
                try:
                    # Check memory growth
                    memory_growth = tf.config.experimental.get_memory_growth(gpu)
                    print(f"        Memory growth: {'Enabled' if memory_growth else 'Disabled'}")
                except:
                    pass
            
            # Check mixed precision
            try:
                policy = tf.keras.mixed_precision.global_policy()
                print(f"   ✓ Mixed precision policy: {policy.name}")
                if policy.name == 'mixed_float16':
                    print("     → ~2x faster training on modern GPUs!")
                elif policy.name == 'float32':
                    print("     → Consider enabling mixed precision for faster training")
            except Exception as e:
                print(f"   ⚠️  Mixed precision check failed: {e}")
            
            # Test GPU computation
            print()
            print("   Testing GPU computation...")
            try:
                with tf.device('/GPU:0'):
                    a = tf.constant([[1.0, 2.0], [3.0, 4.0]])
                    b = tf.constant([[1.0, 1.0], [0.0, 1.0]])
                    c = tf.matmul(a, b)
                print("   ✓ GPU computation test PASSED")
            except Exception as e:
                print(f"   ✗ GPU computation test FAILED: {e}")
        else:
            print("   ✗ TensorFlow cannot see any GPUs")
            print("   → Install tensorflow-gpu or tensorflow[and-cuda]")
            print("   → Verify CUDA and cuDNN are installed correctly")
            
    except ImportError:
        print("   ✗ TensorFlow not installed")
        print("   → Install: pip install tensorflow[and-cuda]")
    except Exception as e:
        print(f"   ✗ Error checking TensorFlow: {e}")
    
    print()
    
    # Check XGBoost GPU
    print("3. Checking XGBoost GPU Support...")
    try:
        import xgboost as xgb
        print(f"   ✓ XGBoost version: {xgb.__version__}")
        
        # Check CUDA support
        try:
            from xgboost.core import _has_cuda_support
            has_cuda = _has_cuda_support()
            if has_cuda:
                print("   ✓ XGBoost CUDA support: Enabled")
                print("     → XGBoost will use GPU (gpu_hist)")
            else:
                print("   ⚠️  XGBoost CUDA support: Not available")
                print("     → Install xgboost with GPU: pip install xgboost[gpu]")
                print("     → XGBoost will fall back to CPU (hist method)")
        except Exception as e:
            print(f"   ⚠️  Cannot determine XGBoost CUDA support: {e}")
            
    except ImportError:
        print("   ✗ XGBoost not installed")
        print("   → Install: pip install xgboost")
    except Exception as e:
        print(f"   ✗ Error checking XGBoost: {e}")
    
    print()
    
    # Check environment variables
    print("4. Checking Environment Variables...")
    env_vars = {
        'CUDA_VISIBLE_DEVICES': 'Controls which GPU(s) to use',
        'TF_FORCE_GPU_ALLOW_GROWTH': 'Prevents TensorFlow from allocating all GPU memory',
        'TF_GPU_THREAD_MODE': 'GPU threading mode',
        'TF_GPU_THREAD_COUNT': 'Number of GPU threads'
    }
    
    for var, description in env_vars.items():
        value = os.environ.get(var)
        if value:
            print(f"   ✓ {var}={value}")
            print(f"     ({description})")
        else:
            print(f"   ○ {var} not set")
    
    print()
    
    # Recommendations
    print("=" * 80)
    print("Recommendations for GPU-Accelerated Hyperparameter Tuning")
    print("=" * 80)
    
    try:
        import tensorflow as tf
        gpus = tf.config.list_physical_devices('GPU')
        
        if gpus:
            print("\n✓ Your system is GPU-ready!")
            print("\nOptimal settings for hyperparameter tuning:")
            print("  • Batch size: 128-256 (vs 64 for CPU)")
            print("  • Mixed precision: Enabled (automatic in model_architecture.py)")
            print("  • Memory growth: Enabled (automatic in model_architecture.py)")
            print("\nExpected speedup:")
            print("  • ~2-3x faster per epoch compared to CPU")
            print("  • Total tuning time: ~30-40 hours (vs ~90-100 hours on CPU)")
            print("\nTo start GPU-optimized tuning:")
            print("  ./restart_tuning_with_gpu.sh")
            print("  or")
            print("  ./run_tuning_background_improved.sh")
            
        else:
            print("\n⚠️  No GPU detected!")
            print("\nYour options:")
            print("  1. Install NVIDIA GPU drivers and CUDA toolkit")
            print("  2. Install tensorflow[and-cuda]: pip install tensorflow[and-cuda]")
            print("  3. Use a cloud GPU instance (Google Colab, AWS, GCP, etc.)")
            print("  4. Continue with CPU (will be much slower)")
            print("\nCPU tuning time estimate: ~90-100 hours")
            
    except:
        print("\nCannot determine GPU status.")
        print("Please ensure TensorFlow is installed: pip install tensorflow[and-cuda]")
    
    print()
    print("=" * 80)


if __name__ == "__main__":
    check_gpu_status()

