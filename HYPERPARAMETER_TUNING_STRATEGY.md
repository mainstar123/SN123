# Hyperparameter Tuning Strategy for All Challenges

## Overview

Different challenge types may require different hyperparameters. This guide explains the tuning strategy and how to use it.

## Challenge Types

### 1. Binary Challenges (2D embeddings)
- **Examples**: ETH, XAUUSD, XAGUSD, EURUSD, GBPUSD, CADUSD, NZDUSD, CHFUSD
- **Issue**: Some forex models show single-class prediction (not learning)
- **Focus**: Prevent overfitting, improve generalization, ensure diverse predictions

### 2. LBFGS Challenges (17D embeddings)
- **Examples**: BTCLBFGS, ETHLBFGS
- **Issue**: Below accuracy threshold, may need different evaluation
- **Focus**: Larger models, more features for higher dimensionality

### 3. HITFIRST Challenges (3D embeddings)
- **Examples**: ETHHITFIRST
- **Issue**: Below accuracy threshold
- **Focus**: Balance between model capacity and regularization

## Hyperparameter Search Space

### Binary Challenges
- **LSTM Hidden**: 128, 256, 512
- **Features**: 20, 25, 30, 35
- **Dropout**: 0.3, 0.4, 0.5
- **Learning Rate**: 0.0001, 0.0003, 0.0005, 0.001

**Key Configurations:**
- More regularization (for overfitting): Higher dropout, fewer features
- More features (for better signal): 30-35 features
- Larger model (for capacity): 512 hidden units
- Lower learning rate (for stability): 0.0001-0.0003

### LBFGS Challenges
- **LSTM Hidden**: 256, 512
- **Features**: 25, 30, 35
- **Dropout**: 0.3, 0.4
- **Learning Rate**: 0.0003, 0.0005

**Key Configurations:**
- Larger models for 17D embeddings
- More features to capture complex patterns

### HITFIRST Challenges
- **LSTM Hidden**: 256, 512
- **Features**: 20, 25, 30
- **Dropout**: 0.3, 0.4
- **Learning Rate**: 0.0003, 0.0005

**Key Configurations:**
- Balanced approach
- Moderate regularization

## Usage

### Option 1: Tune All Challenges
```bash
# This will tune all 11 challenges
# Warning: This takes a long time (several hours)
./tune_all_challenges.sh
```

### Option 2: Tune Specific Challenge Types
```bash
# Interactive menu
./tune_specific_challenges.sh

# Options:
# 1) Tune all binary challenges (forex pairs + ETH)
# 2) Tune all LBFGS challenges
# 3) Tune all HITFIRST challenges
# 4) Tune specific ticker
# 5) Tune problematic models only (forex pairs)
```

### Option 3: Tune Individual Ticker
```bash
# Tune a specific ticker
python scripts/training/tune_all_challenges.py \
    --ticker CHFUSD \
    --data-dir data/raw \
    --tuning-dir models/tuning \
    --epochs 100 \
    --batch-size 64
```

### Option 4: Tune Only Problematic Models
```bash
# Tune forex models with single-class prediction issue
for ticker in CHFUSD NZDUSD CADUSD GBPUSD EURUSD; do
    python scripts/training/tune_all_challenges.py \
        --ticker $ticker \
        --data-dir data/raw \
        --tuning-dir models/tuning \
        --epochs 100
done
```

## Recommended Approach

### Step 1: Tune Problematic Models First (Recommended)
Focus on fixing the models with issues:

```bash
# Tune forex models (single-class prediction)
./tune_specific_challenges.sh
# Select option 5

# Or tune individually
python scripts/training/tune_all_challenges.py --ticker CHFUSD --epochs 100
python scripts/training/tune_all_challenges.py --ticker NZDUSD --epochs 100
python scripts/training/tune_all_challenges.py --ticker CADUSD --epochs 100
python scripts/training/tune_all_challenges.py --ticker GBPUSD --epochs 100
python scripts/training/tune_all_challenges.py --ticker EURUSD --epochs 100
```

### Step 2: Tune LBFGS and HITFIRST
```bash
# Tune LBFGS challenges
python scripts/training/tune_all_challenges.py \
    --challenge-type lbfgs \
    --epochs 100

# Tune HITFIRST challenges
python scripts/training/tune_all_challenges.py \
    --challenge-type hitfirst \
    --epochs 100
```

### Step 3: Tune All (Optional)
If you want to optimize all models:
```bash
./tune_all_challenges.sh
```

## Understanding Results

### Output Files
- `models/tuning/tuning_results_YYYYMMDD_HHMMSS.json`: Full results for all configurations
- `models/tuning/best_configs_YYYYMMDD_HHMMSS.json`: Best configuration for each challenge

### Best Configuration Selection
Configurations are ranked by:
1. **CI Lower Bound** (primary): Ensures >55% threshold
2. **Test Accuracy** (secondary): Higher is better
3. **AUC** (tertiary): Model discrimination ability

### Key Metrics
- **Test Accuracy**: Overall prediction accuracy
- **CI Lower Bound**: 95% confidence interval lower bound (must be >0.55)
- **AUC**: Area under ROC curve (higher is better, NaN if single-class)
- **Passed**: Whether CI lower bound >= 0.55

## After Tuning

### Step 1: Review Best Configurations
```bash
# View best configs
cat models/tuning/best_configs_*.json | jq .
```

### Step 2: Retrain with Best Configs
After finding best hyperparameters, retrain models in the main directory:

```bash
# Example: Retrain CHFUSD with best config
python scripts/training/train_model.py \
    --ticker CHFUSD \
    --epochs 200 \
    --batch-size 64 \
    --lstm-hidden 256 \
    --tmfg-n-features 30 \
    --dropout 0.4 \
    --learning-rate 0.0003
```

**Note**: The `train_model.py` script needs to be updated to accept these parameters via command line, or you can modify the defaults in the script.

### Step 3: Verify Results
```bash
# Check results
./check_training_results.sh
```

## Time Estimates

- **Per configuration**: ~30-60 minutes (CPU), ~10-20 minutes (GPU)
- **Binary challenge** (12 configs): ~6-12 hours (CPU), ~2-4 hours (GPU)
- **All challenges** (11 challenges × 12 configs avg): ~66-132 hours (CPU), ~22-44 hours (GPU)

**Recommendation**: 
- Start with problematic models only (5 forex pairs)
- Use GPU if available
- Run in background: `nohup ./tune_specific_challenges.sh > tuning.log 2>&1 &`

## Troubleshooting

### Out of Memory
- Reduce batch size: `--batch-size 32`
- Reduce LSTM hidden units in search space
- Train one ticker at a time

### Single-Class Predictions Persist
- Try more regularization (higher dropout, fewer features)
- Try lower learning rate
- Check data quality
- May need different feature engineering

### Low Accuracy
- Try more features (30-35)
- Try larger model (512 hidden)
- Try different learning rates
- May need more training data

## Next Steps

After tuning:
1. Review best configurations
2. Retrain models with best hyperparameters
3. Verify results with `check_training_results.sh`
4. Start mining with optimized models

