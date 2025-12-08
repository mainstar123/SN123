# Hyperparameter Tuning Recommendation

## My Recommendation: **Retrain First, Then Tune**

### ✅ Step 1: Retrain with Current Improvements (DO THIS FIRST)

You haven't retrained yet with the improvements we just made:
- LSTM hidden: 256 (was 128)
- Features: 20 (was 15)  
- Dropout: 0.3 (was 0.2)
- Learning rate: 0.0005 (was 0.001)

**Action:**
```bash
python scripts/training/train_model.py \
    --ticker ETH \
    --epochs 200 \
    --batch-size 64
```

**Why?** These changes might be enough to reach >55%. Test first before spending hours on tuning.

---

### ⚠️ Step 2: If Still Not >55%, Then Tune

If retraining doesn't reach >55%, then hyperparameter tuning is worth it.

## Tuning Strategy

### Option A: Quick Manual Testing (Recommended First)

Test 3-4 key configurations manually:

1. **More Features** (25 instead of 20)
2. **More Regularization** (dropout 0.4 instead of 0.3)
3. **Larger Model** (512 hidden instead of 256)
4. **Lower Learning Rate** (0.0003 instead of 0.0005)

**Time**: ~4-6 hours (1-1.5 hours per config)

### Option B: Automated Tuning (If Manual Doesn't Work)

Use Optuna or similar for systematic search:

**Time**: 8-24 hours (depending on search space)

**When to use**: If manual testing doesn't find good configs

## Key Hyperparameters to Tune

### Priority 1 (Biggest Impact):
1. **Number of Features** (15, 20, 25, 30)
2. **Dropout Rate** (0.2, 0.3, 0.4, 0.5)
3. **LSTM Hidden Units** (128, 256, 512)
4. **Learning Rate** (0.0001, 0.0003, 0.0005, 0.001)

### Priority 2 (Moderate Impact):
5. **LSTM Layers** (2, 3)
6. **Time Steps** (15, 20, 25)
7. **Batch Size** (32, 64, 128)

## Quick Manual Tuning Guide

### Test Config 1: More Features
```bash
# Edit train_model.py line 152: tmfg_n_features=25
python scripts/training/train_model.py --ticker ETH --epochs 200
python scripts/training/check_training_results.py --ticker ETH
```

### Test Config 2: More Regularization
```bash
# Edit train_model.py line 153: dropout=0.4
python scripts/training/train_model.py --ticker ETH --epochs 200
python scripts/training/check_training_results.py --ticker ETH
```

### Test Config 3: Larger Model
```bash
# Edit train_model.py line 148: lstm_hidden=512
python scripts/training/train_model.py --ticker ETH --epochs 200
python scripts/training/check_training_results.py --ticker ETH
```

## When NOT to Tune

Don't tune if:
- ❌ You haven't retrained with current improvements yet
- ❌ The issue is data quality (not model)
- ❌ The test period is fundamentally different from training
- ❌ Features are not predictive

## Expected Results from Tuning

**Realistic expectations:**
- Manual tuning: +1-3% accuracy improvement
- Automated tuning: +2-5% accuracy improvement
- May not reach >55% if underlying issue is data/features

## My Final Recommendation

1. **NOW**: Retrain with current improvements (already in code)
2. **IF <55%**: Test 2-3 manual configs (features, dropout, model size)
3. **IF STILL <55%**: Consider if the problem is:
   - Data quality/period differences
   - Feature engineering
   - Model architecture
   - Not just hyperparameters

**Bottom Line**: Retrain first! The improvements we made might be enough. Only tune if needed.

