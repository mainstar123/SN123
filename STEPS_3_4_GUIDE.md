# Steps 3 & 4 Implementation Guide

This guide shows you how to complete **Step 3** (Verify Training Process) and **Step 4** (Check Training Results) from the Implementation Guide.

## Quick Start

### Step 3: Verify Training Process

This step verifies that your training completed successfully by checking:
- Model files exist and are valid
- Configuration is correct
- Data loading works
- Data splitting is correct
- Feature extraction works
- Model architecture is valid

**Command:**
```bash
# Verify a single model
python scripts/training/verify_training_process.py \
    --ticker ETH \
    --model-dir models/checkpoints \
    --data-dir data/raw

# Verify all trained models
python scripts/training/verify_training_process.py \
    --model-dir models/checkpoints \
    --data-dir data/raw
```

**Expected Output:**
```
✓ PASS: Model Files
✓ PASS: Configuration
✓ PASS: Data Loading
✓ PASS: Data Splitting
✓ PASS: Feature Extraction
✓ PASS: Model Architecture

✓ All steps passed! Training process verified successfully.
```

### Step 4: Check Training Results

This step evaluates your models on the test set and shows:
- Test accuracy
- AUC score
- 95% confidence interval
- Pass/fail status (>55% required)

**Command:**
```bash
# Check results for a single model
python scripts/training/check_training_results.py \
    --ticker ETH \
    --model-dir models/checkpoints \
    --data-dir data/raw \
    --min-accuracy 0.55

# Check all trained models
python scripts/training/check_training_results.py \
    --model-dir models/checkpoints \
    --data-dir data/raw \
    --min-accuracy 0.55
```

**Expected Output:**
```
Test Set Results:
  Accuracy: 0.5678 (56.78%)
  AUC: 0.6234
  95% CI: [0.5512, 0.5845]
  Test samples: 8135
  ✓ PASS: Lower CI (0.5512) >= 0.55
```

## What Was Implemented

### New Scripts Created

1. **`scripts/training/verify_training_process.py`**
   - Verifies all 6 steps of the training process
   - Checks model files, configuration, data, features, and architecture
   - Provides detailed output for each verification step

2. **`scripts/training/check_training_results.py`**
   - Evaluates models on test set
   - Calculates accuracy, AUC, and confidence intervals
   - Determines pass/fail based on minimum accuracy threshold

### Bug Fixes

- Fixed Keras model loading compatibility issue by using `compile=False` when loading models
- This resolves the "Could not deserialize 'keras.metrics.mse'" error

## Usage Examples

### Example 1: Verify ETH Model
```bash
cd /home/ocean/MANTIS
source venv/bin/activate
python scripts/training/verify_training_process.py --ticker ETH
```

### Example 2: Check All Models
```bash
python scripts/training/check_training_results.py \
    --model-dir models/checkpoints \
    --data-dir data/raw \
    --min-accuracy 0.55
```

### Example 3: Custom Date Ranges
```bash
python scripts/training/check_training_results.py \
    --ticker BTC \
    --train-end 2023-12-31 \
    --val-end 2024-12-31 \
    --min-accuracy 0.55
```

## Next Steps

After completing Steps 3 and 4:

1. **If all models pass**: Proceed to Local Testing (Step 5 in Implementation Guide)
2. **If models fail**: Review training parameters, data quality, or feature engineering
3. **If verification fails**: Check model files, data availability, or configuration

## Troubleshooting

### Issue: Model loading fails
- **Solution**: The fix has been applied. If you still see errors, try retraining the model with the current TensorFlow/Keras version.

### Issue: No test data available
- **Solution**: Adjust `--val-end` to ensure test period has data, or use a later date range.

### Issue: Low accuracy
- **Solution**: Review the troubleshooting section in IMPLEMENTATION_GUIDE.md for suggestions on improving model performance.

## Files Modified

- `scripts/training/model_architecture.py`: Fixed model loading compatibility
- `IMPLEMENTATION_GUIDE.md`: Updated with clear commands for Steps 3 and 4

## Files Created

- `scripts/training/verify_training_process.py`: Step 3 implementation
- `scripts/training/check_training_results.py`: Step 4 implementation
- `STEPS_3_4_GUIDE.md`: This guide


