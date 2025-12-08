# Model Improvement Guide

## Current Status

✅ **Fix Working**: Price change calculation is correct
- Training accuracy: 56.11%
- Test accuracy: 50.10% (needs improvement)

## Quick Improvements Applied

I've increased `tmfg_n_features` from 10 to 15 in the training script. This will:
- Select more features (15 instead of 10)
- Capture more patterns in the data
- Potentially improve generalization

## Next Steps

### 1. Retrain with More Features

```bash
python scripts/training/train_model.py \
    --ticker ETH \
    --epochs 200 \
    --batch-size 64
```

**Why 200 epochs?** The model stopped early at epoch 16. With more features, it might need more training.

### 2. If Still Not Good Enough, Try:

#### A. Increase Model Capacity
Edit `train_model.py` line 148:
```python
lstm_hidden=256,  # Instead of 128
```

#### B. Adjust Learning Rate
The model might benefit from a lower learning rate. Edit `model_architecture.py`:
```python
optimizer=Adam(learning_rate=0.0005),  # Instead of 0.001
```

#### C. Add More Regularization
Edit `train_model.py`:
```python
dropout=0.3,  # Instead of 0.2
```

### 3. Test Results

After retraining, check results:
```bash
python scripts/training/check_training_results.py --ticker ETH
```

## Expected Improvements

With 15 features instead of 10:
- More information for the model
- Better pattern recognition
- Potentially 1-3% accuracy improvement

If still not enough:
- Try 20 features
- Increase LSTM capacity
- Adjust hyperparameters

## Success Criteria

Target metrics:
- Test accuracy > 55%
- 95% CI lower bound > 55%
- AUC > 0.55
- Balanced predictions

---

**Action**: Retrain with 15 features and 200 epochs, then evaluate.

