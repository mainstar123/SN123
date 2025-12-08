# Training Process Investigation Report

## Executive Summary

**Status**: ⚠️ **CRITICAL ISSUE FOUND**

All binary models are failing because they are being trained on **absolute prices** instead of **price changes**. This causes:
- All binary labels to be 1 (since prices are always positive)
- Models to learn to always predict class 1
- Very low accuracy (1-2% for forex, 50% for ETH due to slight imbalance)

## Test Results Summary

| Model | Accuracy | Status | Issue |
|-------|----------|--------|-------|
| ETH | 50.93% | ✗ FAIL | Predicts only class 1 |
| EURUSD | 1.47% | ✗ FAIL | Predicts only class 0 |
| GBPUSD | 1.54% | ✗ FAIL | Predicts only class 0 |
| CADUSD | 1.40% | ✗ FAIL | Predicts only class 0 |
| BTCLBFGS | 0.00% | ✗ FAIL | LBFGS challenge (different evaluation) |

## Root Cause Analysis

### Problem 1: Training on Absolute Prices

**Location**: `scripts/training/model_architecture.py`, line 264

```python
# Current (WRONG):
y_train_delta = y_train  # y_train contains absolute prices!

# Should be:
# Calculate price changes from the data
```

**Impact**: 
- Model learns to predict absolute prices (e.g., ETH ~$3000)
- Loss values are in millions (predicting prices in thousands)
- Validation loss is in billions

### Problem 2: Binary Labels from Absolute Prices

**Location**: `scripts/training/model_architecture.py`, line 323

```python
# Current (WRONG):
y_train_binary = (y_train_delta > 0).astype(int)
# Since y_train_delta contains absolute prices (always > 0),
# all labels become 1!

# Should be:
# Calculate price changes first, then create binary labels
y_train_binary = (price_changes > 0).astype(int)
```

**Impact**:
- All training labels are 1 (price always > 0)
- Model learns to always predict 1
- XGBoost gets no signal to learn from

### Problem 3: Data Preparation Issue

**Location**: `scripts/training/model_architecture.py`, `prepare_feature_matrix()`

The `prepare_feature_matrix()` function with `target_col='close'` returns absolute close prices, not price changes. The training code assumes these are price changes.

## Evidence from Training Logs

```
Epoch 17/100
loss: 35184876.0000 - mae: 4489.0225
val_loss: 1811894528.0000 - val_mae: 38247.5234
```

- **Loss**: 35 million (predicting absolute prices)
- **MAE**: 4,489 (average error in price prediction)
- **Val Loss**: 1.8 billion (extremely high)
- **Val MAE**: 38,247 (huge validation error)

This confirms the model is predicting absolute prices, not price changes.

## Training Process Flow (Current - WRONG)

1. **Data Loading**: Load OHLCV data ✓
2. **Feature Extraction**: Extract features ✓
3. **Target Preparation**: `y = close prices` ✗ (should be price changes)
4. **Sequence Creation**: Create sequences with absolute prices ✗
5. **LSTM Training**: Train to predict absolute prices ✗
6. **Binary Labels**: `(absolute_prices > 0)` → all 1 ✗
7. **XGBoost Training**: Train on all-1 labels ✗
8. **Result**: Model always predicts 1

## What Should Happen (CORRECT)

1. **Data Loading**: Load OHLCV data ✓
2. **Feature Extraction**: Extract features ✓
3. **Target Preparation**: `y = price_changes` (next_price - current_price) ✓
4. **Sequence Creation**: Create sequences with price changes ✓
5. **LSTM Training**: Train to predict price changes ✓
6. **Binary Labels**: `(price_changes > 0)` → balanced 0/1 ✓
7. **XGBoost Training**: Train on balanced binary labels ✓
8. **Result**: Model learns to predict up/down direction

## Fix Required

### Option 1: Fix in `prepare_feature_matrix()` (Recommended)

Modify the feature extraction to return price changes instead of absolute prices:

```python
# In prepare_feature_matrix()
# Instead of returning close prices, return price changes
df['price_change'] = df['close'].diff().shift(-1)  # Forward-looking change
y = df['price_change'].values
```

### Option 2: Fix in Training Code

Modify the training code to calculate price changes:

```python
# In train() method
# Calculate price changes from y_train (which contains absolute prices)
# This requires access to the original dataframe or calculating from sequences
```

### Option 3: Fix in `prepare_data()`

Modify `prepare_data()` to return price changes:

```python
# In prepare_data()
# After getting y from prepare_feature_matrix()
# Calculate price changes
y_changes = np.diff(y)  # Or use forward-looking changes
```

## Recommended Action Plan

1. **Immediate**: Fix the target preparation to use price changes
2. **Verify**: Retrain one model (e.g., ETH) and verify it works
3. **Retrain**: Retrain all binary models with the fix
4. **Test**: Run evaluation on all models
5. **Document**: Update training documentation

## Files That Need Modification

1. `scripts/training/model_architecture.py`
   - `prepare_feature_matrix()` - Return price changes
   - `train()` - Ensure binary labels use price changes
   - `prepare_data()` - Calculate price changes correctly

2. `scripts/training/train_model.py`
   - Verify data preparation flow

## Testing Checklist

After fix:
- [ ] Verify y_train contains price changes (positive and negative)
- [ ] Verify binary labels are balanced (roughly 50/50)
- [ ] Verify training loss is reasonable (< 1.0)
- [ ] Verify validation accuracy > 55%
- [ ] Verify test accuracy > 55%
- [ ] Verify model predicts both classes (0 and 1)

## Current Status

- ✅ **Evaluation scripts working correctly** (fixed price change calculation)
- ✅ **Model loading working** (fixed Keras compatibility)
- ✅ **Data loading working**
- ❌ **Training process broken** (using absolute prices instead of changes)
- ❌ **All models need retraining** after fix

---

**Next Steps**: Fix the training code to use price changes, then retrain all models.

