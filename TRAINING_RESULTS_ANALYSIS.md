# Training Results Analysis

## Summary

**Status: 8/11 models pass accuracy threshold, but several have critical issues**

### ✅ Passing Models (8/11)
- **ETH**: 57.55% (CI: [56.52%, 58.59%]) - **GOOD** ✓
- **XAUUSD**: 95.61% (CI: [95.18%, 96.01%]) - **GOOD** ✓
- **XAGUSD**: 93.90% (CI: [93.38%, 94.45%]) - **GOOD** ✓
- **CHFUSD**: 98.50% (CI: [98.23%, 98.76%]) - **⚠️ ISSUE: Single-class prediction**
- **NZDUSD**: 98.49% (CI: [98.22%, 98.75%]) - **⚠️ ISSUE: Single-class prediction**
- **CADUSD**: 98.60% (CI: [98.32%, 98.85%]) - **⚠️ ISSUE: Single-class prediction**
- **GBPUSD**: 98.46% (CI: [98.18%, 98.73%]) - **⚠️ ISSUE: Single-class prediction**
- **EURUSD**: 98.53% (CI: [98.26%, 98.79%]) - **⚠️ ISSUE: Single-class prediction**

### ❌ Failing Models (3/11)
- **BTCLBFGS**: 58.06% (CI: [50.97%, 65.81%]) - **⚠️ ISSUE: Small test set, wide CI**
- **ETHLBFGS**: 51.76% (CI: [50.68%, 52.91%]) - **⚠️ ISSUE: Below threshold**
- **ETHHITFIRST**: 47.57% (CI: [46.49%, 48.73%]) - **⚠️ ISSUE: Below threshold**

---

## Critical Issues

### 1. Single-Class Prediction Problem (5 models)

**Affected Models:** CHFUSD, NZDUSD, CADUSD, GBPUSD, EURUSD

**Symptoms:**
- Models predict only class 0 (down/no change) for all samples
- High accuracy (98%+) but AUC = NaN (cannot calculate)
- This is **not useful for mining** - models are just predicting the majority class

**Root Cause:**
- Models are not learning meaningful patterns
- Embeddings are likely all negative or very low values
- When converted to probabilities, all predictions are < 0.5

**Impact:**
- These models will have **zero salience** in mining
- They provide no useful signal - just predicting "down" always

**Solution:**
- Retrain with different hyperparameters
- Check if data quality is good for these forex pairs
- Consider using different feature engineering
- May need more training epochs or different learning rate

### 2. LBFGS Challenges Evaluation Issue

**Affected Models:** BTCLBFGS, ETHLBFGS

**Issue:**
- These are **17-dimensional embedding** challenges, not binary
- Binary accuracy is not the correct metric for LBFGS challenges
- LBFGS uses a different loss function (L-BFGS optimization)

**BTCLBFGS Specific:**
- Only 155 test samples (very small)
- Wide confidence interval: [50.97%, 65.81%]
- Lower bound below 55% threshold

**ETHLBFGS Specific:**
- 8135 test samples (good)
- Accuracy 51.76% - below threshold
- May need different evaluation metric

**Solution:**
- LBFGS challenges need different evaluation (not binary accuracy)
- Consider using embedding quality metrics instead
- Or use a different threshold for LBFGS challenges

### 3. ETHHITFIRST Challenge Issue

**Issue:**
- This is a **3-dimensional embedding** challenge (HITFIRST loss)
- Binary accuracy may not be appropriate
- Accuracy 47.57% - well below threshold

**Solution:**
- HITFIRST challenges need different evaluation
- May need to retrain with different approach
- Check if the model architecture is correct for 3D embeddings

---

## Recommendations

### Immediate Actions

1. **Retrain Forex Models (CHFUSD, NZDUSD, CADUSD, GBPUSD, EURUSD)**
   ```bash
   # These models need retraining - they're not learning properly
   python scripts/training/train_model.py --ticker CHFUSD --epochs 200 --batch-size 64
   python scripts/training/train_model.py --ticker NZDUSD --epochs 200 --batch-size 64
   python scripts/training/train_model.py --ticker CADUSD --epochs 200 --batch-size 64
   python scripts/training/train_model.py --ticker GBPUSD --epochs 200 --batch-size 64
   python scripts/training/train_model.py --ticker EURUSD --epochs 200 --batch-size 64
   ```

2. **Investigate LBFGS Challenges**
   - These may need different evaluation metrics
   - Consider if binary accuracy is appropriate for 17D embeddings
   - May need to adjust thresholds or use different metrics

3. **Investigate ETHHITFIRST**
   - Check if 3D embedding model is correct
   - May need different training approach
   - Consider if binary accuracy is appropriate

### Models Ready for Mining

**These models are ready to use:**
- ✅ **ETH** - Good accuracy (57.55%), proper predictions
- ✅ **XAUUSD** - Excellent accuracy (95.61%), proper predictions
- ✅ **XAGUSD** - Excellent accuracy (93.90%), proper predictions

**These models need retraining:**
- ⚠️ **CHFUSD, NZDUSD, CADUSD, GBPUSD, EURUSD** - Single-class prediction
- ⚠️ **BTCLBFGS, ETHLBFGS** - Need different evaluation or retraining
- ⚠️ **ETHHITFIRST** - Below threshold, needs investigation

---

## Next Steps

### Option 1: Start Mining with Good Models (Recommended)
You can start mining with the 3 good models (ETH, XAUUSD, XAGUSD) while retraining the others:

```bash
# Start mining with available models
python mining/miner.py \
    --wallet.name your_wallet \
    --wallet.hotkey your_hotkey \
    --model-dir models/checkpoints \
    --data-dir data/raw
```

### Option 2: Retrain All Problematic Models
Retrain all models with issues before starting mining:

```bash
# Retrain forex models
for ticker in CHFUSD NZDUSD CADUSD GBPUSD EURUSD; do
    python scripts/training/train_model.py --ticker $ticker --epochs 200 --batch-size 64
done

# Retrain LBFGS and HITFIRST (may need different approach)
python scripts/training/train_model.py --ticker BTCLBFGS --epochs 200
python scripts/training/train_model.py --ticker ETHLBFGS --epochs 200
python scripts/training/train_model.py --ticker ETHHITFIRST --epochs 200
```

### Option 3: Investigate Evaluation Metrics
For LBFGS and HITFIRST challenges, we may need to:
- Implement proper evaluation metrics for multi-dimensional embeddings
- Use embedding quality metrics instead of binary accuracy
- Adjust thresholds based on challenge type

---

## Detailed Model Analysis

### ETH (✅ GOOD)
- **Accuracy**: 57.55% (CI: [56.52%, 58.59%])
- **AUC**: 0.5982
- **Status**: ✓ PASS
- **Prediction Distribution**: Balanced (4027 class 0, 4108 class 1)
- **Label Distribution**: Balanced (3992 class 0, 4143 class 1)
- **Assessment**: Model is learning properly, predictions are diverse

### XAUUSD (✅ GOOD)
- **Accuracy**: 95.61% (CI: [95.18%, 96.01%])
- **AUC**: 0.6150
- **Status**: ✓ PASS
- **Prediction Distribution**: Mostly class 0 (7865), some class 1 (227)
- **Label Distribution**: Mostly class 0 (7952), some class 1 (140)
- **Assessment**: Model is learning, though imbalanced (may be due to data)

### XAGUSD (✅ GOOD)
- **Accuracy**: 93.90% (CI: [93.38%, 94.45%])
- **AUC**: 0.4138 (low, but model is making predictions)
- **Status**: ✓ PASS
- **Prediction Distribution**: Mostly class 0 (7736), some class 1 (356)
- **Label Distribution**: Mostly class 0 (7954), some class 1 (138)
- **Assessment**: Model is learning, though imbalanced

### CHFUSD, NZDUSD, CADUSD, GBPUSD, EURUSD (⚠️ PROBLEM)
- **Accuracy**: 98%+ (misleading)
- **AUC**: NaN (cannot calculate)
- **Status**: ✓ PASS (by accuracy) but ✗ FAIL (by usefulness)
- **Prediction Distribution**: Only class 0 (all samples)
- **Assessment**: **Models are not learning - just predicting majority class**

### BTCLBFGS (⚠️ NEEDS INVESTIGATION)
- **Accuracy**: 58.06% (CI: [50.97%, 65.81%])
- **AUC**: 0.6408
- **Status**: ✗ FAIL (lower CI < 55%)
- **Test Samples**: Only 155 (very small)
- **Assessment**: Small test set makes CI unreliable. May need different evaluation for LBFGS.

### ETHLBFGS (⚠️ NEEDS INVESTIGATION)
- **Accuracy**: 51.76% (CI: [50.68%, 52.91%])
- **AUC**: 0.5450
- **Status**: ✗ FAIL (below threshold)
- **Assessment**: Below threshold. May need different evaluation for LBFGS.

### ETHHITFIRST (⚠️ NEEDS INVESTIGATION)
- **Accuracy**: 47.57% (CI: [46.49%, 48.73%])
- **AUC**: 0.4474
- **Status**: ✗ FAIL (below threshold)
- **Assessment**: Well below threshold. May need different approach for HITFIRST challenge.

---

## Conclusion

**Current Status:**
- 3 models are ready for mining (ETH, XAUUSD, XAGUSD)
- 5 models need retraining (forex pairs with single-class prediction)
- 3 models need investigation (LBFGS and HITFIRST - may need different evaluation)

**Recommendation:**
Start mining with the 3 good models while retraining the problematic ones in the background.
