# Next Steps After Training Fix

## Current Status

✅ **Fix Verified**: The price change fix is working correctly!
- Model predicts both classes (0: 2351, 1: 5784)
- Training loss is reasonable (~339, not millions)
- Training accuracy: 56.11% (above 55%)

❌ **Issue**: Model not generalizing well
- Test accuracy: 50.10% (essentially random)
- Large gap between train (56%) and test (50%)
- Model is overfitting to training data

## Immediate Next Steps

### Step 1: Verify Model Behavior ✅ (Done)
The model is now correctly:
- Using price changes (not absolute prices)
- Predicting both classes
- Learning on training data

### Step 2: Improve Model Performance

The model needs improvement to reach >55% test accuracy. Here are the options:

#### Option A: Increase Model Capacity (Recommended First)

Try training with more capacity:

```bash
# Edit model_architecture.py or create a new training script
# Increase LSTM hidden units from 128 to 256
# Increase feature selection from 10 to 15-20
```

#### Option B: Adjust Hyperparameters

```bash
# Try different hyperparameters
python scripts/training/train_model.py \
    --ticker ETH \
    --epochs 200 \
    --batch-size 32 \
    --train-end 2023-12-31 \
    --val-end 2024-12-31
```

#### Option C: Add More Features

Increase the number of selected features:
- Current: 10 features
- Try: 15-20 features
- This might capture more patterns

#### Option D: Improve Regularization

- Add more dropout
- Add L2 regularization
- Use different early stopping patience

### Step 3: Test Different Approaches

1. **Try Different Time Periods**
   ```bash
   # Test on different validation/test splits
   python scripts/training/train_model.py \
       --ticker ETH \
       --train-end 2023-06-30 \
       --val-end 2024-06-30
   ```

2. **Train Multiple Models**
   - Train with different random seeds
   - Ensemble multiple models
   - Average predictions

3. **Feature Engineering**
   - Add more technical indicators
   - Try different VMD parameters
   - Add interaction features

## Recommended Action Plan

### Phase 1: Quick Wins (Try First)

1. **Increase Feature Selection**
   - Change `tmfg_n_features` from 10 to 15
   - Retrain and test

2. **Train Longer**
   - Increase epochs to 200
   - Model stopped early at epoch 16
   - May need more training

3. **Adjust Learning Rate**
   - Current learning rate might be too high
   - Try lower initial learning rate

### Phase 2: Model Improvements

1. **Increase Model Capacity**
   - LSTM hidden units: 128 → 256
   - Add more layers if needed

2. **Better Regularization**
   - Increase dropout
   - Add L2 regularization
   - Adjust early stopping

### Phase 3: Advanced Techniques

1. **Ensemble Methods**
   - Train multiple models
   - Combine predictions

2. **Feature Engineering**
   - Add more features
   - Try different feature combinations

3. **Hyperparameter Tuning**
   - Grid search or random search
   - Optimize for validation accuracy

## Quick Test Script

Create a script to test different configurations:

```python
# test_configs.py
configs = [
    {'tmfg_n_features': 15, 'lstm_hidden': 128},
    {'tmfg_n_features': 20, 'lstm_hidden': 128},
    {'tmfg_n_features': 10, 'lstm_hidden': 256},
    {'tmfg_n_features': 15, 'lstm_hidden': 256},
]

for config in configs:
    # Train with config
    # Test accuracy
    # Save best model
```

## Expected Timeline

- **Quick fixes** (Phase 1): 1-2 hours
- **Model improvements** (Phase 2): 2-4 hours
- **Advanced techniques** (Phase 3): 4-8 hours

## Success Criteria

Model should achieve:
- ✅ Test accuracy > 55%
- ✅ 95% CI lower bound > 55%
- ✅ AUC > 0.55
- ✅ Balanced predictions (not biased to one class)

---

**Current Status**: Fix working, model needs improvement for better generalization.

**Next Action**: Try increasing feature selection and training longer.

