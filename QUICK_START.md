# MANTIS Training Quick Start Guide

**Complete Step-by-Step Instructions**

---

## Step 1: Stop Any Running Training

First, stop any old training processes that might be running:

```bash
cd /home/ocean/MANTIS
./run_training.sh --stop
```

Wait for confirmation that training is stopped.

---

## Step 2: Start Hyperparameter Tuning

Run the training script in the background:

```bash
./run_training.sh
```

This will:
- Train all 11 challenges
- Run 50 hyperparameter tuning trials per challenge
- Use GPU acceleration automatically
- Run in the background
- Log everything to `logs/training/training_current.log`

**Alternative Options:**

```bash
# Quick test (10 trials only)
./run_training.sh --quick

# More thorough search (100 trials)
./run_training.sh --trials 100

# Train only one challenge
./run_training.sh --challenge ETH-LBFGS
```

---

## Step 3: Monitor Progress

### Watch the log in real-time:

```bash
tail -f logs/training/training_current.log
```

Press `Ctrl+C` to stop watching (training continues in background).

### Check status anytime:

```bash
./run_training.sh --status
```

This shows:
- Whether training is running
- Current progress
- CPU/Memory usage
- Log file location

---

## Step 4: Wait for Completion

Training will take **12-24 hours** depending on your GPU.

The script will:
1. Train ETH-LBFGS (highest weight) ← Starting here
2. Train BTC-LBFGS-6H
3. Train ETH-HITFIRST-100M
4. Train 8 binary challenges
5. Save best models to `models/tuned/`
6. Generate training report

You can safely close your terminal - training continues in background.

---

## Step 5: Check Results

When training completes, check the results:

```bash
# View training summary
cat logs/training/training_current.log | grep -A 20 "Training Summary"

# List trained models
ls -lh models/tuned/
```

Expected output:
```
models/tuned/
├── ETH-LBFGS/
├── BTC-LBFGS-6H/
├── ETH-HITFIRST-100M/
├── ETH-1H-BINARY/
├── EURUSD-1H-BINARY/
├── GBPUSD-1H-BINARY/
├── CADUSD-1H-BINARY/
├── NZDUSD-1H-BINARY/
├── CHFUSD-1H-BINARY/
├── XAUUSD-1H-BINARY/
└── XAGUSD-1H-BINARY/
```

---

## Useful Commands

### Stop training if needed:
```bash
./run_training.sh --stop
```

### Check what's happening:
```bash
./run_training.sh --status
```

### View recent trial results:
```bash
grep "Trial [0-9]*:" logs/training/training_current.log | tail -20
```

### Check for errors:
```bash
grep -i "error\|failed\|exception" logs/training/training_current.log | tail -20
```

### View GPU usage:
```bash
nvidia-smi
```

---

## Expected Behavior

### ✅ GOOD - What You Should See:

```
Trial 0: 0.127469
Trial 1: 0.156234
Trial 2: 0.143891
...
Best trial: 23. Best value: 0.055448
```

- Trials completing successfully (~95% success rate)
- Best value improving over time
- All tmfg_n_features values working (8-15)

### ⚠️ ISSUES - What to Watch For:

```
Trial failed: Input 0 of layer "functional_X" is incompatible...
```

- If you see many (>10%) shape errors, the fix didn't apply
- If training stops unexpectedly, check `--status`
- If GPU runs out of memory, use `--trials 25` for smaller batches

---

## Troubleshooting

### Training won't start:
```bash
# Check if already running
./run_training.sh --status

# Stop old process
./run_training.sh --stop

# Check log for errors
tail -50 logs/training/training_current.log
```

### Out of GPU memory:
```bash
# Reduce trials to free up memory
./run_training.sh --stop
./run_training.sh --trials 25
```

### Script not found:
```bash
# Make sure you're in the right directory
cd /home/ocean/MANTIS

# Make script executable
chmod +x run_training.sh
```

---

## What Happens Next?

After training completes:

1. **Models are saved** to `models/tuned/[CHALLENGE]/`
2. **You can use them immediately** for mining or validation
3. **Training report shows** which hyperparameters work best
4. **You're ready** to deploy on testnet/mainnet

---

## Summary

1. `./run_training.sh` - Start training
2. `./run_training.sh --status` - Check status
3. Wait 12-24 hours
4. Check results in `models/tuned/`
5. Done! ✅

That's it! Training runs completely automatically in the background.

