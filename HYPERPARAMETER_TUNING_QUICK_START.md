# ⚡ Hyperparameter Tuning Quick Start

**Quick reference for starting and monitoring hyperparameter tuning**

---

## 🚀 START TRAINING

### Option 1: Easy Way (Recommended)

```bash
cd /home/ocean/Nereus/SN123
source venv/bin/activate
./run_training.sh
```

The script will:
- Check all prerequisites automatically
- Ask for confirmation
- Start training in background
- Give you monitoring commands

### Option 2: Manual Way

```bash
cd /home/ocean/Nereus/SN123
source venv/bin/activate

# Train all 11 challenges (18-24 hours)
nohup python scripts/training/tune_all_challenges.py \
    --data-dir data \
    --tuning-dir models/tuned \
    > logs/training/training_$(date +%Y%m%d_%H%M%S).log 2>&1 &
```

### Option 3: Specific Challenge Type

```bash
# Only binary challenges (8 challenges, ~12 hours)
./run_training.sh --challenge-type binary

# Only LBFGS challenges (2 challenges, ~4 hours)  
./run_training.sh --challenge-type lbfgs

# Only HITFIRST challenges (1 challenge, ~2 hours)
./run_training.sh --challenge-type hitfirst
```

### Option 4: Single Challenge

```bash
./run_training.sh --ticker ETH-LBFGS
```

### Option 5: Custom Parameters

```bash
# More epochs for better convergence
./run_training.sh --epochs 150

# Smaller batch size if memory issues
./run_training.sh --batch-size 64
```

---

## 📊 MONITOR PROGRESS

### Check if training is running

```bash
ps aux | grep tune_all_challenges.py | grep -v grep
```

**Expected:** Shows running process with PID

### Watch progress in real-time

```bash
tail -f logs/training/training_*.log
```

**Press Ctrl+C to exit** (training continues)

### Check progress summary

```bash
grep -i "progress\|training challenge" logs/training/training_*.log | tail -20
```

**Expected output:**
```
Progress: 1/11
Training Challenge: ETH-LBFGS
Progress: 2/11
Training Challenge: BTC-LBFGS-6H
...
```

### Count completed models

```bash
ls -d models/tuned/*/ 2>/dev/null | wc -l
```

**Expected:** 
- 0 = just started
- 5-6 = halfway
- 11 = complete!

---

## ⏱️ TIME ESTIMATES

| Configuration | Time Estimate |
|--------------|---------------|
| Single challenge | 1.5-3 hours |
| Binary challenges only | 12-16 hours |
| LBFGS challenges only | 4-6 hours |
| HITFIRST challenges only | 2-3 hours |
| **All 11 challenges** | **18-24 hours** |
| With custom epochs (150) | 25-35 hours |

---

## 🎯 WHAT'S BEING TUNED

The training script tests **predefined configurations** for each challenge type:

### Binary Challenges (13 configs each)
Testing combinations of:
1. **LSTM Hidden:** 128, 256, 512
2. **TMFG Features:** 20, 25, 30, 35
3. **Dropout:** 0.3, 0.4, 0.5
4. **Learning Rate:** 0.0001, 0.0003, 0.0005, 0.001

### LBFGS Challenges (7 configs each)
Testing combinations of:
1. **LSTM Hidden:** 256, 512, 768, 1024
2. **TMFG Features:** 25, 30, 35, 40, 50
3. **Dropout:** 0.2, 0.3, 0.4
4. **Learning Rate:** 0.0003, 0.0005, 0.001

### HITFIRST Challenges (5 configs each)
Testing combinations of:
1. **LSTM Hidden:** 256, 512
2. **TMFG Features:** 25, 30
3. **Dropout:** 0.3, 0.4
4. **Learning Rate:** 0.0003, 0.0005

**Total:** Each challenge tests its predefined search space automatically!

---

## ✅ VERIFY COMPLETION

When training finishes, verify everything:

```bash
# 1. Check completion message
grep -i "complete\|summary" logs/training/training_*.log | tail -20

# 2. Verify all 11 models exist
ls -d models/tuned/*/
# Should show:
# - ETH-LBFGS
# - BTC-LBFGS-6H
# - ETH-HITFIRST-100M
# - ETH-1H-BINARY
# - EURUSD-1H-BINARY
# - GBPUSD-1H-BINARY
# - CADUSD-1H-BINARY
# - NZDUSD-1H-BINARY
# - CHFUSD-1H-BINARY
# - XAUUSD-1H-BINARY
# - XAGUSD-1H-BINARY

# 3. Check each model has all files
for model in models/tuned/*/; do
    echo "$(basename $model): $(ls $model | wc -l) files"
done
# Each should have 4-5 files

# 4. Check for any errors
grep -i error logs/training/training_*.log | wc -l
# Should be 0 or very few
```

**All checks passed?** → Proceed to local testing!

---

## 🔧 COMMON SCENARIOS

### Start training now (standard)
```bash
source venv/bin/activate
./run_training.sh
```

### Train only binary challenges
```bash
./run_training.sh --challenge-type binary
```

### Retrain specific challenge (after mainnet deployment)
```bash
./run_training.sh --ticker ETH-LBFGS
```

### Extended training (more epochs for difficult challenges)
```bash
./run_training.sh --ticker BTC-LBFGS-6H --epochs 150
```

### Check status anytime
```bash
tail -20 logs/training/training_*.log
```

### Stop training (if needed)
```bash
# Find PID
ps aux | grep tune_all_challenges.py | grep -v grep

# Kill it
kill <PID>
```

---

## ⚠️ TROUBLESHOOTING

### Problem: "Data directory not found"
```bash
# Check data exists
ls -lh data/
# Should show CSV files for each ticker
```

### Problem: "Out of memory"
```bash
# Check GPU memory
nvidia-smi

# If needed, reduce batch size in script
# Or train one challenge at a time
./run_training.sh --challenge ETH-LBFGS
```

### Problem: "CUDA error"
```bash
# Check GPU
nvidia-smi

# Restart if needed
sudo reboot
```

### Problem: Training seems stuck
```bash
# Check if process is actually running
ps aux | grep tune_all_challenges.py | grep -v grep

# Check recent log activity
ls -lh logs/training/*.log

# View latest logs
tail -50 logs/training/training_*.log
```

---

## 📋 PRE-FLIGHT CHECKLIST

Before starting training, verify:

```bash
# ✓ In correct directory
cd /home/ocean/Nereus/SN123

# ✓ Data exists
ls data/*.csv | wc -l  # Should be 11+

# ✓ GPU available
nvidia-smi

# ✓ Sufficient disk space (need 10GB+)
df -h

# ✓ Script exists
ls -lh scripts/training/tune_all_challenges.py

# ✓ Python packages installed
python -c "import torch, xgboost, pandas; print('OK')"
```

All checks passed? **Start training!**

```bash
./run_training.sh
```

---

## 🎯 NEXT STEPS

**After training completes:**

1. ✅ Verify all 11 models exist
2. ✅ Run local testing: `./test_locally.sh`
3. ✅ Review results: `cat results/backtest_results_latest.txt`
4. ✅ Deploy to mainnet (if results are good)

**See:** `MASTER_GUIDE_TO_FIRST_PLACE.md` - Phase 1.5 (Local Testing)

---

## 📞 NEED HELP?

**Training takes too long?** Use `--quick` flag (20 trials)  
**Out of memory?** Train one challenge at a time  
**Not sure what to do?** Read `MASTER_GUIDE_TO_FIRST_PLACE.md`  
**Training complete?** Run `./test_locally.sh`

---

**Remember:** Training is mostly waiting. Start it and let it run overnight!

**Good luck! 🚀**

