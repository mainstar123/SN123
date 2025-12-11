# Salience Optimization Guide

## Overview

This guide explains the improvements made to maximize your salience score and achieve first place in the MANTIS mainnet.

## What is Salience?

**Salience** measures how much unique, non-redundant information your embeddings provide to the validator's ensemble model. High salience means:
- Your predictions complement other miners' predictions
- You capture information others miss
- Removing your embeddings significantly hurts the ensemble's performance

## Key Improvements Implemented

### 1. Class Imbalance Handling ✅

**Problem**: Your model was predicting only class 0 (98.5% accuracy but useless for salience)

**Solution**:
- Added class weights to XGBoost training
- Implemented `scale_pos_weight` for imbalanced binary classification
- Sample weights based on class frequency

**Impact**: Model now learns to predict minority class patterns, essential for high salience

### 2. Unique Feature Extraction ✅

**Problem**: Standard features are redundant with other miners

**Solution**: Added 7 categories of unique features:

1. **Microstructure Features**
   - Price position within bar
   - Upper/lower shadow ratios
   - Body-to-range ratio
   - Captures intra-bar dynamics others miss

2. **Regime Detection**
   - Volatility regime identification
   - Trend strength measurement
   - Market state classification

3. **Cross-Timeframe Features**
   - Momentum divergence (short vs long-term)
   - Acceleration (rate of change of momentum)
   - Distance from recent extremes

4. **Volume-Price Divergence**
   - VWAP-based features
   - Accumulation/Distribution line
   - Smart money flow indicators

5. **Statistical Features**
   - Returns skewness and kurtosis
   - Z-scores vs recent distribution
   - Tail risk indicators

6. **Pattern Recognition**
   - Higher highs / lower lows detection
   - Support/resistance proximity
   - Chart pattern indicators

7. **Time-Based Features**
   - Cyclical encoding (sine/cosine) for hour/day
   - Intraday/seasonal patterns

**Impact**: These features capture information that standard technical indicators miss, improving your uniqueness

### 3. Enhanced Model Architecture ✅

**Improvements**:
- Increased embedding layer capacity (64→128→64→32)
- Better dropout scheduling
- More capacity for diverse representations

**Impact**: Model can learn more complex, unique patterns

## How Salience is Calculated

### Binary Challenges (2D embeddings)

1. **Top-K Selection**: Validators select top 20 miners by individual AUC
2. **Ensemble Training**: XGBoost ensemble on selected miners
3. **Permutation Importance**: Your salience = how much removing your embedding hurts ensemble AUC
4. **Normalization**: Scores normalized across all miners

**Key Insight**: You need to:
- Be in the top 20 by individual AUC (get selected)
- Provide unique information (high permutation importance)
- Not be redundant with other top miners

### LBFGS Challenges (17D embeddings)

1. **Classifier Salience (50%)**: 5-bucket probability prediction
2. **Q Salience (50%)**: Opposite-move probabilities
3. **Final**: Average of both, normalized

## Testing Your Salience Locally

Use the new testing script:

```bash
python scripts/testing/test_local_salience.py --ticker CHFUSD --model-dir models/tuning/CHFUSD/h256_f20_d0.4_lr0.0005
```

This will show:
- Embedding diversity score
- Prediction diversity (should predict both classes)
- Estimated salience score
- Recommendations for improvement

## Best Practices for High Salience

### 1. Ensure Prediction Diversity

**Critical**: Your model MUST predict both classes, not just the majority class.

**Check**:
```python
# After training, verify:
unique_predictions = len(np.unique(y_pred))
assert unique_predictions == 2, "Model predicts only one class!"
```

### 2. Maximize Embedding Diversity

Your embeddings should:
- Have non-zero variance across samples
- Capture different patterns for different market conditions
- Not collapse to a single point

**Check**: Embedding std should be > 0.1 for each dimension

### 3. Focus on AUC, Not Just Accuracy

**Why**: Validators use AUC for top-K selection. High accuracy with low AUC = low salience.

**Target**: AUC > 0.55 (better than random) to get selected

### 4. Use Unique Features

The new unique features help you:
- Capture microstructure others miss
- Identify regime changes
- Detect volume-price divergences
- Recognize patterns others don't

### 5. Avoid Overfitting

**Problem**: Overfitting to training data reduces generalization

**Solution**:
- Use proper train/val/test splits
- Monitor validation AUC
- Use early stopping
- Regularization (dropout, L1/L2)

## Training Workflow

1. **Train with class weights** (already implemented)
2. **Use unique features** (already implemented)
3. **Verify prediction diversity**:
   ```bash
   python scripts/testing/test_local_salience.py --ticker YOUR_TICKER
   ```
4. **Check embedding diversity** (should see non-zero std)
5. **Monitor AUC** (should be > 0.55)

## Expected Results

After these improvements:

- ✅ Model predicts both classes (not just majority)
- ✅ Embeddings are diverse (std > 0.1)
- ✅ AUC > 0.55 (gets selected by validators)
- ✅ High permutation importance (unique information)
- ✅ High salience score (top rankings)

## Troubleshooting

### Problem: Model still predicts only one class

**Solutions**:
1. Increase class weights further
2. Use focal loss (future improvement)
3. Increase model capacity
4. Add more minority class samples (SMOTE)

### Problem: Low embedding diversity

**Solutions**:
1. Increase embedding layer size
2. Reduce dropout in embedding layers
3. Add more unique features
4. Train longer with lower learning rate

### Problem: Low AUC

**Solutions**:
1. Add more features (use unique features)
2. Increase model capacity
3. Better hyperparameter tuning
4. More training data

## Next Steps

1. **Retrain models** with the new improvements
2. **Test locally** using the salience testing script
3. **Verify** prediction diversity and embedding diversity
4. **Deploy** to mainnet
5. **Monitor** salience scores on-chain

## Summary

The key to high salience is **uniqueness**:
- Unique features → capture information others miss
- Diverse embeddings → provide non-redundant signals
- Balanced predictions → learn both classes
- High AUC → get selected by validators

With these improvements, you should achieve significantly higher salience scores and rank in the top positions!



