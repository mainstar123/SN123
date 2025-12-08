# Training All Tickers - Quick Guide

## Overview

This guide shows you how to train all tickers in the background and monitor progress.

**Total Tickers to Train:** 11
- ETH (already trained ✓)
- ETHHITFIRST
- ETHLBFGS
- BTCLBFGS
- EURUSD
- GBPUSD
- CADUSD
- NZDUSD
- CHFUSD
- XAUUSD
- XAGUSD

## Quick Start

### Step 1: Start Training All Tickers

```bash
./train_all_tickers.sh
```

This will:
- Train all 11 tickers sequentially in the background
- Save logs to `logs/training/train_all_YYYYMMDD_HHMMSS.log`
- Use best hyperparameters (256 hidden, 25 features, 0.3 dropout, 0.0005 LR)
- Run for 200 epochs per ticker

**Expected Time:** 5-10 hours (depending on GPU/CPU)

### Step 2: Check Training Status

```bash
./check_training_status.sh
```

This shows:
- ✓ Which tickers are completed
- ⏳ Which tickers are in progress
- ⏸ Which tickers haven't started
- Process CPU/Memory usage
- Recent log activity

### Step 3: View Live Logs

```bash
# Find latest log file
tail -f logs/training/train_all_*.log

# Or check specific log
tail -f logs/training/train_all_20251207_*.log
```

### Step 4: Check Results (After Training)

```bash
./check_training_results.sh
```

This evaluates all trained models and shows:
- Test accuracy for each ticker
- AUC scores
- 95% confidence intervals
- Pass/fail status (>55% required)

## Detailed Commands

### Start Training

```bash
# Train all tickers (background)
./train_all_tickers.sh

# Or train manually with custom settings
python scripts/training/train_model.py \
    --data-dir data/raw \
    --model-dir models/checkpoints \
    --epochs 200 \
    --batch-size 64 \
    --use-gpu
```

### Monitor Progress

```bash
# Check status
./check_training_status.sh

# View logs
tail -f logs/training/train_all_*.log

# Check specific ticker files
ls -lh models/checkpoints/ETH/
ls -lh models/checkpoints/BTCLBFGS/
```

### Stop Training

```bash
# Find PID
cat logs/training/train_all.pid

# Stop training
kill $(cat logs/training/train_all.pid)

# Or force kill
pkill -f "train_model.py"
```

### Check Results

```bash
# Check all models
./check_training_results.sh

# Check specific ticker
python scripts/training/check_training_results.py \
    --ticker ETH \
    --model-dir models/checkpoints \
    --min-accuracy 0.55
```

## Training Progress Indicators

### Model Files (per ticker)
- `lstm_model.h5` - LSTM model (9-10 MB)
- `xgb_model.json` - XGBoost model (100-200 KB)
- `config.json` - Model configuration
- `scaler.pkl` - Feature scaler
- `feature_indices.pkl` - Selected features

### Status Indicators
- **✓ Completed**: All files exist and are older than 1 hour
- **⏳ In Progress**: Files exist but recently modified (< 1 hour)
- **⏸ Not Started**: No model directory exists

## Expected Results

After training completes, you should see:

```
✓ Passed (11/11):
  ETH: 0.5657 (CI: [0.5561, 0.5757])
  ETHHITFIRST: ...
  ETHLBFGS: ...
  BTCLBFGS: ...
  ...
```

All models should achieve >55% accuracy (95% CI lower bound).

## Troubleshooting

### Training Stuck
```bash
# Check if process is running
ps aux | grep train_model

# Check logs for errors
tail -100 logs/training/train_all_*.log | grep -i error
```

### Out of Memory
- Reduce batch size: `--batch-size 32`
- Train one ticker at a time
- Use CPU instead: `--use-cpu`

### Missing Data
```bash
# Fetch data for specific ticker
python scripts/data_collection/data_fetcher.py
```

## Next Steps

After all models are trained:

1. **Verify Results**: Run `./check_training_results.sh`
2. **Test Locally**: Run backtest and salience tests
3. **Start Mining**: Configure R2 and start miner

See `IMPLEMENTATION_GUIDE.md` for complete workflow.

