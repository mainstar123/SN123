# Hyperparameter Tuning - Quick Start Guide

## Yes, You Should Do Hyperparameter Tuning!

Based on your results, hyperparameter tuning is **highly recommended** because:

1. **5 forex models** (CHFUSD, NZDUSD, CADUSD, GBPUSD, EURUSD) have single-class prediction issues
2. **3 models** (BTCLBFGS, ETHLBFGS, ETHHITFIRST) are below the 55% threshold
3. Different challenge types may need different hyperparameters

## Quick Start

### Option 1: Tune Problematic Models First (Recommended)

Focus on fixing the models with issues:

```bash
# Interactive menu - select option 5
./tune_specific_challenges.sh

# Or tune forex models individually
for ticker in CHFUSD NZDUSD CADUSD GBPUSD EURUSD; do
    python scripts/training/tune_all_challenges.py \
        --ticker $ticker \
        --data-dir data/raw \
        --tuning-dir models/tuning \
        --epochs 100
done
```

### Option 2: Tune All Challenges

```bash
# This takes several hours - run in background
nohup ./tune_all_challenges.sh > tuning.log 2>&1 &
```

### Option 3: Tune by Challenge Type

```bash
# Tune only binary challenges
python scripts/training/tune_all_challenges.py \
    --challenge-type binary \
    --epochs 100

# Tune only LBFGS challenges
python scripts/training/tune_all_challenges.py \
    --challenge-type lbfgs \
    --epochs 100
```

## After Tuning

### Step 1: Review Best Configurations

```bash
# View best configs
cat models/tuning/best_configs_*.json
```

### Step 2: Retrain with Best Configs

```bash
# Example: Retrain CHFUSD with best config from tuning
python scripts/training/train_model.py \
    --ticker CHFUSD \
    --epochs 200 \
    --batch-size 64 \
    --lstm-hidden 256 \
    --tmfg-n-features 30 \
    --dropout 0.4 \
    --learning-rate 0.0003
```

### Step 3: Verify Results

```bash
./check_training_results.sh
```

## Expected Time

- **Per configuration**: ~30-60 min (CPU), ~10-20 min (GPU)
- **Per challenge** (12 configs): ~6-12 hours (CPU), ~2-4 hours (GPU)
- **5 forex models**: ~30-60 hours (CPU), ~10-20 hours (GPU)

**Recommendation**: Start with problematic models only, use GPU, run in background.

## What Gets Tuned

For each challenge, the script tests:
- **LSTM Hidden Units**: 128, 256, 512
- **TMFG Features**: 20, 25, 30, 35
- **Dropout**: 0.3, 0.4, 0.5
- **Learning Rate**: 0.0001, 0.0003, 0.0005, 0.001

Different challenge types have different search spaces optimized for their needs.

## Output

After tuning, you'll get:
- `models/tuning/tuning_results_*.json`: Full results for all configurations
- `models/tuning/best_configs_*.json`: Best configuration for each challenge

The best configuration is selected based on:
1. CI Lower Bound (must be >0.55)
2. Test Accuracy
3. AUC score

## Next Steps

1. **Start tuning** problematic models (forex pairs)
2. **Review results** and identify best configs
3. **Retrain** models with best hyperparameters
4. **Verify** results meet requirements
5. **Start mining** with optimized models

See `HYPERPARAMETER_TUNING_STRATEGY.md` for detailed information.

