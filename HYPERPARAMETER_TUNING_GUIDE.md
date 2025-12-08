# Hyperparameter Tuning Guide

## Should You Do Hyperparameter Tuning?

### Current Status
- ✅ Fix applied: Price changes working correctly
- ✅ Improvements made: Increased capacity, regularization, features
- ❌ Still not reaching >55% test accuracy
- ⚠️ Overfitting issue persists

## Recommendation: **Try Retraining First, Then Tune**

### Step 1: Retrain with Current Improvements (Do This First)

Before tuning, retrain with the improvements we just made:
- LSTM hidden: 256 (was 128)
- Features: 20 (was 15)
- Dropout: 0.3 (was 0.2)
- Learning rate: 0.0005 (was 0.001)

```bash
python scripts/training/train_model.py \
    --ticker ETH \
    --epochs 200 \
    --batch-size 64
```

**Why?** These changes might be enough. Test first before spending time on tuning.

### Step 2: If Still Not >55%, Then Do Hyperparameter Tuning

If retraining doesn't reach >55%, then systematic tuning is worth it.

## Hyperparameter Tuning Options

### Option A: Manual Grid Search (Recommended for Start)

Test key hyperparameters one at a time:

```python
# Test different configurations
configs = [
    {'lstm_hidden': 256, 'tmfg_n_features': 20, 'dropout': 0.3, 'lr': 0.0005},  # Current
    {'lstm_hidden': 256, 'tmfg_n_features': 25, 'dropout': 0.3, 'lr': 0.0005},  # More features
    {'lstm_hidden': 512, 'tmfg_n_features': 20, 'dropout': 0.3, 'lr': 0.0005},  # Larger model
    {'lstm_hidden': 256, 'tmfg_n_features': 20, 'dropout': 0.4, 'lr': 0.0005},  # More dropout
    {'lstm_hidden': 256, 'tmfg_n_features': 20, 'dropout': 0.3, 'lr': 0.0003},  # Lower LR
]
```

### Option B: Automated Tuning (More Thorough)

Use tools like:
- **Optuna** (recommended)
- **Ray Tune**
- **Keras Tuner**

## Key Hyperparameters to Tune

### Priority 1: Most Impact
1. **LSTM Hidden Units**: 128, 256, 512
2. **Number of Features**: 15, 20, 25, 30
3. **Dropout Rate**: 0.2, 0.3, 0.4, 0.5
4. **Learning Rate**: 0.0003, 0.0005, 0.001

### Priority 2: Moderate Impact
5. **LSTM Layers**: 2, 3
6. **Time Steps**: 15, 20, 25
7. **Batch Size**: 32, 64, 128
8. **XGBoost Parameters**: max_depth, learning_rate

### Priority 3: Fine-tuning
9. **VMD Components**: 6, 8, 10
10. **Early Stopping Patience**: 10, 15, 20

## Quick Tuning Script

I can create a simple tuning script that tests different configurations automatically.

## Alternative: Check Other Issues First

Before tuning, consider:

1. **Data Quality**: Is the test period (2025) too different from training?
2. **Feature Engineering**: Are the features actually predictive?
3. **Model Architecture**: Is the hybrid LSTM+XGBoost approach optimal?
4. **Evaluation Method**: Is the evaluation correct?

## My Recommendation

### Do This Now:
1. ✅ **Retrain with current improvements** (already done in code)
2. ✅ **Check results** - see if it reaches >55%
3. ⚠️ **If not, then do hyperparameter tuning**

### Tuning Strategy:
- Start with **manual testing** of 3-5 key configs
- If that doesn't work, use **automated tuning** (Optuna)
- Focus on: features, dropout, learning rate first

---

**Bottom Line**: Retrain first, then tune if needed. The improvements we made might be enough!

