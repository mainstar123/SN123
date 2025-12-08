# Background Hyperparameter Tuning Guide

## Quick Start

### Option 1: Tune Problematic Models (Recommended)

Tune the 5 forex models with single-class prediction issues:

```bash
./run_tune_problematic_background.sh
```

This will:
- Tune CHFUSD, NZDUSD, CADUSD, GBPUSD, EURUSD sequentially
- Run in background
- Save logs to `logs/tuning/`
- Estimated time: ~30-60 hours (CPU) or ~10-20 hours (GPU)

### Option 2: Interactive Menu

Choose what to tune:

```bash
./run_tuning_background.sh
```

Options:
1. Tune all challenges (~66-132 hours)
2. Tune problematic models only
3. Tune all binary challenges
4. Tune all LBFGS challenges
5. Tune all HITFIRST challenges
6. Tune specific ticker

## Monitor Progress

### Check Status

```bash
./check_tuning_status.sh
```

This shows:
- ✓ If tuning is running
- CPU/Memory usage
- Recent log activity
- Progress indicators
- Completed challenges
- Any errors

### View Live Logs

```bash
# View master log
tail -f logs/tuning/tune_problematic_*.log

# View specific ticker log
tail -f logs/tuning/tune_CHFUSD_*.log

# View latest log (any type)
tail -f logs/tuning/tuning_*.log
```

### Check Results

```bash
# View best configurations (when available)
cat models/tuning/best_configs_*.json

# View full results
cat models/tuning/tuning_results_*.json
```

## Stop Tuning

```bash
# Find PID
cat logs/tuning/tuning.pid
# or
cat logs/tuning/tune_problematic.pid

# Stop process
kill $(cat logs/tuning/tuning.pid)
```

## After Tuning Completes

### Step 1: Review Best Configurations

```bash
# View best configs
cat models/tuning/best_configs_*.json

# Pretty print (if jq is installed)
cat models/tuning/best_configs_*.json | jq .
```

### Step 2: Retrain with Best Configs

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

### Step 3: Verify Results

```bash
./check_training_results.sh
```

## File Locations

- **Logs**: `logs/tuning/`
  - `tune_problematic_*.log` - Master log for problematic models
  - `tune_<TICKER>_*.log` - Individual ticker logs
  - `tuning_*.log` - General tuning logs
  - `tuning.pid` - Process ID file
  - `tuning.status` - Status file

- **Results**: `models/tuning/`
  - `tuning_results_*.json` - Full results for all configurations
  - `best_configs_*.json` - Best configuration for each challenge
  - `<TICKER>/<config_name>/` - Individual model directories

## Troubleshooting

### Tuning Stuck

```bash
# Check if process is running
ps aux | grep tune_all_challenges

# Check logs for errors
tail -100 logs/tuning/tune_problematic_*.log | grep -i error
```

### Out of Memory

- Reduce batch size in the script
- Tune one ticker at a time
- Use CPU instead of GPU

### Want to Resume

If tuning stops, you can resume by running the script again. It will skip already completed configurations (models already exist).

## Example Workflow

```bash
# 1. Start tuning problematic models
./run_tune_problematic_background.sh

# 2. Check status (in another terminal)
./check_tuning_status.sh

# 3. Monitor logs
tail -f logs/tuning/tune_problematic_*.log

# 4. After completion, review results
cat models/tuning/best_configs_*.json

# 5. Retrain with best configs
# (Use the best configs from step 4)

# 6. Verify results
./check_training_results.sh
```

## Time Estimates

- **Per configuration**: ~30-60 min (CPU), ~10-20 min (GPU)
- **Per challenge** (12 configs): ~6-12 hours (CPU), ~2-4 hours (GPU)
- **5 problematic models**: ~30-60 hours (CPU), ~10-20 hours (GPU)
- **All challenges**: ~66-132 hours (CPU), ~22-44 hours (GPU)

**Recommendation**: Start with problematic models only, use GPU if available.

