# 🏆 MASTER GUIDE: Complete Path to First Place in Subnet 123

**Your Complete Step-by-Step Roadmap**  
**From Current Position → First Place Achievement**

**Current Status:** Hyperparameter tuning in progress  
**Timeline to #1:** 3-6 months  
**Success Rate:** High (with consistent execution)

---

## 📍 TABLE OF CONTENTS

1. [Current Status & Quick Start](#phase-0-current-status)
2. [Phase 1: Complete Training](#phase-1-complete-training-now-18-24-hours)
3. [Phase 1.5: Local Testing](#phase-15-local-testing-before-mainnet-1-day)
4. [Phase 2: Mainnet Deployment](#phase-2-mainnet-deployment-day-1)
5. [Phase 3: Week 1 Stabilization](#phase-3-week-1-stabilization-days-1-7)
6. [Phase 4: Week 2-3 Optimization](#phase-4-weeks-2-3-optimization-days-8-21)
7. [Phase 5: Month 2 Top 10 Push](#phase-5-month-2-top-10-push-days-22-60)
8. [Phase 6: Month 3+ First Place](#phase-6-month-3-first-place-push-days-61)
9. [Daily Maintenance Routines](#daily-maintenance-routines)
10. [Troubleshooting Guide](#troubleshooting-quick-reference)

---

# PHASE 0: Current Status

## Where You Are Now

```
✅ Hyperparameter tuning: RUNNING
✅ Scripts and infrastructure: READY
✅ Guides created: COMPLETE
⏳ Training progress: In progress (check logs)
🎯 Next: Wait for completion, then test locally
```

## Quick Status Check

```bash
cd /home/ocean/Nereus/SN123

# Check training status
tail -20 logs/training/training_current.log

# Check if models exist yet
ls -l models/tuned/ 2>/dev/null | wc -l
# Should show 11 when complete
```

**Expected:** Training running, making progress  
**Action:** Wait for completion (18-24 hours from start)

## If Training Hasn't Started Yet

If you see "No such file" or training is not running, start it now:

```bash
cd /home/ocean/Nereus/SN123

# Activate venv
source venv/bin/activate

# Check if training is running
ps aux | grep tune_all_challenges.py | grep -v grep

# If not running, start it:
nohup python scripts/training/tune_all_challenges.py \
    --data-dir data \
    --tuning-dir models/tuned \
    > logs/training/training_$(date +%Y%m%d_%H%M%S).log 2>&1 &

echo "Training started! Check progress with: tail -f logs/training/training_*.log"
```

**Then proceed to Phase 1 monitoring tasks below**

---

# PHASE 1: Complete Training [NOW - 18-24 Hours]

## Goal
Complete hyperparameter tuning for all 11 challenges.

## What's Happening
Your training script is testing multiple hyperparameter configurations for each challenge:
- Binary challenges: 13 configurations each
- LBFGS challenges: 7 configurations each
- HITFIRST challenges: 5 configurations each

## How to Run Hyperparameter Tuning

### Option 1: Run All Challenges (Recommended for First Time)

```bash
cd /home/ocean/Nereus/SN123

# Activate virtual environment (if using one)
source venv/bin/activate  # or: conda activate your_env

# Run comprehensive tuning for all 11 challenges
python scripts/training/tune_all_challenges.py \
    --data-dir data \
    --tuning-dir models/tuned

# This will take 18-24 hours
# It runs in foreground, so use nohup or screen to keep it running
```

**Run in Background (Recommended):**
```bash
# Use nohup to run in background
nohup python scripts/training/tune_all_challenges.py \
    --data-dir data \
    --tuning-dir models/tuned \
    > logs/training/training_$(date +%Y%m%d_%H%M%S).log 2>&1 &

# Save the process ID
echo $! > logs/training/tuning.pid

# Check it's running
ps aux | grep tune_all_challenges.py | grep -v grep
```

**Using Screen (Alternative):**
```bash
# Start a screen session
screen -S tuning

# Run training
python scripts/training/tune_all_challenges.py \
    --data-dir data \
    --tuning-dir models/tuned

# Detach: Press Ctrl+A, then D
# Reattach later: screen -r tuning
```

### Option 2: Run Specific Challenges

If you want to tune specific challenges only:

```bash
# Single ticker
python scripts/training/tune_all_challenges.py \
    --data-dir data \
    --tuning-dir models/tuned \
    --ticker ETH-LBFGS

# Specific challenge type (binary, lbfgs, or hitfirst)
python scripts/training/tune_all_challenges.py \
    --data-dir data \
    --tuning-dir models/tuned \
    --challenge-type binary
```

### Option 3: Custom Training Parameters

If you want to adjust epochs or batch size:

```bash
# More epochs for better convergence
python scripts/training/tune_all_challenges.py \
    --data-dir data \
    --tuning-dir models/tuned \
    --epochs 150

# Smaller batch size if memory issues
python scripts/training/tune_all_challenges.py \
    --data-dir data \
    --tuning-dir models/tuned \
    --batch-size 64
```

### Parameters Explained

```bash
--data-dir data              # Where your training data is stored
--tuning-dir models/tuned    # Where to save trained models
--ticker TICKER_NAME         # Optional: tune only specific ticker
--challenge-type TYPE        # Optional: binary, lbfgs, or hitfirst
--epochs 100                 # Training epochs per config (default: 100)
--batch-size 128             # Batch size (default: 128, use 64 if memory issues)
--min-accuracy 0.55          # Min accuracy threshold (default: 0.55)
```

**Note:** The script tests predefined configurations:
- **Binary challenges:** 13 configurations each
- **LBFGS challenges:** 7 configurations each  
- **HITFIRST challenges:** 5 configurations each

No need to specify number of trials - it's built into the search space!

### Before You Start: Prerequisites

```bash
# 1. Verify data exists
ls -lh data/
# Should show: ETH_1h.csv, BTC_6h.csv, EURUSD_1h.csv, etc.

# 2. Verify script exists
ls -lh scripts/training/tune_all_challenges.py

# 3. Check GPU available
nvidia-smi

# 4. Check disk space (need ~10GB)
df -h

# 5. Verify Python packages
python -c "import torch; import xgboost; import pandas; print('All packages OK')"
```

### Monitoring Your Training

```bash
# Watch progress in real-time
tail -f logs/training/training_$(date +%Y%m%d)*.log

# Check progress summary
grep -i "progress\|complete\|training challenge" logs/training/training_current.log

# Check how many models completed
ls -d models/tuned/*/ 2>/dev/null | wc -l
# Should reach 11 when complete

# If you used screen, reattach
screen -r tuning
```

## Your Tasks

### Task 1.1: Monitor Progress (Every 3-4 Hours)

```bash
cd /home/ocean/Nereus/SN123

# Quick status
tail -50 logs/training/training_current.log | grep -i "progress\|complete\|error"
```

**Expected Output:**
```
Progress: 1/11
Training Challenge: ETH-LBFGS
Epoch 50/100
...
```

**Red Flags:**
- ❌ "ERROR" messages
- ❌ Process stopped
- ❌ "Out of memory"

**If Issues:** See [Troubleshooting](#troubleshooting-quick-reference)

### Task 1.2: Verify Completion (After 18-24 Hours)

```bash
# Check for training completion
grep -i "summary\|complete" logs/training/training_current.log | tail -20

# Verify all 11 models exist
ls -d models/tuned/*/ | wc -l
# Should output: 11
```

**Expected Output:**
```
✅ Training Summary for ETH-LBFGS
✅ Model saved to models/tuned/ETH-LBFGS
✅ Training Summary for BTC-LBFGS-6H
✅ Model saved to models/tuned/BTC-LBFGS-6H
...
All 11 challenges trained successfully
```

### Task 1.3: Verify Model Files

```bash
# Check each model has complete files
for model in models/tuned/*/; do
    echo "Checking $(basename $model):"
    ls -l "$model" | grep -E "lstm_model|xgb_model|scaler|config" | wc -l
    # Should output: 4 or 5 files
done
```

**Expected:** Each model has at least 4-5 files

## Success Criteria

- ✅ Training completed without errors
- ✅ All 11 model directories exist
- ✅ Each model has complete files
- ✅ No critical errors in logs

**Time to Complete:** 18-24 hours (mostly waiting)

**When Complete:** → Proceed to Phase 1.5 (Local Testing)

---

# PHASE 1.5: Local Testing (Before Mainnet) [1 Day]

## Goal
Test all models locally, measure performance, gain confidence BEFORE mainnet deployment.

## Why This Phase is Critical

```
❌ Skip Testing → Deploy Blind → Risk Poor Performance
✅ Test Locally → Know Your Baseline → Deploy Confident
```

**Benefits:**
- Zero mainnet risk
- Know expected performance
- Identify weak challenges
- Fix issues early
- Deploy with confidence

## Complete Testing Process

### Step 1.5.1: Run Automated Test Suite (15-30 Minutes)

```bash
cd /home/ocean/Nereus/SN123

# Run complete testing suite
./test_locally.sh
```

**This Will:**
1. Verify all 11 models exist
2. Test model loading
3. Backtest on last 30 days of data
4. Test prediction speed
5. Generate complete report

**Expected Output: See TESTING_SUMMARY.md for details**

### Step 1.5.2: Review Results (10 Minutes)

```bash
# View complete results
cat results/backtest_results_latest.txt

# Check key metrics
grep -A 20 "BACKTEST SUMMARY" results/backtest_results_latest.txt

# Check recommendation
grep -A 10 "RECOMMENDATION" results/backtest_results_latest.txt
```

**Expected Results:**

```
📊 BACKTEST SUMMARY

Binary Challenges:
  Average Accuracy: 0.6687 (66.87%)
  Average Est. Salience: 0.6749
  ✓ GOOD - Should perform well on mainnet

LBFGS Challenges:
  Average Est. Salience: 1.9151
  ✓ GOOD - Strong performance

Overall Est. Salience: 1.4521
  ✓ GOOD - Should rank well

📋 RECOMMENDATION
✅ READY FOR MAINNET DEPLOYMENT
```

### Step 1.5.3: Make Deployment Decision

| Result | Salience | Accuracy | Decision |
|--------|----------|----------|----------|
| ✅ Excellent | ≥2.0 | ≥70% | Deploy immediately! |
| ✓ Good | 1.5-2.0 | 65-70% | Deploy confidently |
| ⚠️ Fair | 1.0-1.5 | 60-65% | Deploy OR improve first |
| ❌ Poor | <1.0 | <60% | Improve before mainnet |

**If Ready (Salience ≥1.5, Accuracy ≥65%):**
```bash
# Save baseline
cp results/backtest_results_latest.txt baseline_$(date +%Y%m%d).txt

# Proceed to Phase 2
echo "✅ Ready for mainnet deployment!"
```

**If Not Ready (Salience <1.5):**
```bash
# Identify weak challenges
grep "Est. Salience" results/backtest_results_latest.txt | sort -k3 -n | head -3

# Retrain weakest 2-3
./run_training.sh --ticker WEAK_CHALLENGE_1
./run_training.sh --ticker WEAK_CHALLENGE_2

# Wait for retraining (2-4 hours each)
# Re-test
./test_locally.sh
```

## Success Criteria

- ✅ All models tested successfully
- ✅ Overall salience ≥1.0 (target: ≥1.5)
- ✅ Binary accuracy ≥60% (target: ≥65%)
- ✅ Prediction speed <5 seconds
- ✅ Confident in deployment

**Time to Complete:** 1-2 hours (testing + decision)

**When Complete:** → Proceed to Phase 2 (Mainnet Deployment)

**Detailed Guide:** See `LOCAL_TESTING_COMPLETE_GUIDE.md`

---

# PHASE 2: Mainnet Deployment [Day 1]

## Goal
Deploy your trained models to mainnet and start earning rewards.

## Prerequisites

- ✅ Phase 1 complete (training done)
- ✅ Phase 1.5 complete (local testing passed)
- ✅ Wallet and hotkey ready
- ✅ Confidence level: HIGH

## Complete Deployment Process

### Step 2.1: Prepare Wallet Information

```bash
# Document your credentials
cat > wallet_info.txt << 'EOF'
Wallet Name: YOUR_WALLET_NAME
Hotkey Name: YOUR_HOTKEY_NAME
Netuid: 123
Network: finney
EOF

# Verify balance (you need TAO for registration if not registered)
# Check your wallet has sufficient balance
```

### Step 2.2: Configure Miner

```bash
cd /home/ocean/Nereus/SN123

# Check if miner configuration exists
ls -l neurons/miner.py

# Update miner to use tuned models
# Option 1: Edit miner.py
# Change MODEL_DIR to point to models/tuned/

# Option 2: Use command-line flag
# We'll use --model.dir flag when starting
```

### Step 2.3: Test Miner Configuration (Local)

```bash
# Test miner loads models correctly (don't connect to network yet)
python neurons/miner.py \
    --netuid 123 \
    --wallet.name YOUR_WALLET_NAME \
    --wallet.hotkey YOUR_HOTKEY \
    --model.dir models/tuned \
    --offline \
    --test-mode

# Expected: Models load, no errors
```

### Step 2.4: Start Miner on Mainnet

```bash
cd /home/ocean/Nereus/SN123
source venv/bin/activate

# Stop any existing miner
pkill -f "python.*miner" || true

# Start miner
nohup python neurons/miner.py \
    --netuid 123 \
    --wallet.name YOUR_WALLET_NAME \
    --wallet.hotkey YOUR_HOTKEY \
    --subtensor.network finney \
    --model.dir models/tuned \
    --logging.info \
    > logs/miner_$(date +%Y%m%d_%H%M%S).log 2>&1 &

# Save PID
echo $! > logs/miner.pid

# Verify it started
sleep 5
ps aux | grep miner.py | grep -v grep
```

**Expected Output:**
```
[PID] python neurons/miner.py --netuid 123 ...
```

### Step 2.5: Monitor First Hour (Critical!)

```bash
# Watch logs in real-time
tail -f logs/miner_*.log

# Look for:
# ✓ "Loading model from models/tuned/..."
# ✓ "Prediction generated for challenge: ..."
# ✓ "Submitting prediction to validator"
# ✓ "Salience score received: X.XX"
```

**Expected Output (First 10 Minutes):**
```
[INFO] Loading model from models/tuned/ETH-LBFGS
[INFO] Model loaded successfully
[INFO] Received challenge: ETH-LBFGS
[INFO] Generating prediction...
[INFO] Prediction generated: shape (1, 17)
[INFO] Submitting to validator...
[INFO] Salience score received: 1.87
[SUCCESS] Prediction accepted!

[INFO] Loading model from models/tuned/BTC-LBFGS-6H
...
```

### Step 2.6: First Hour Checklist

```bash
# Every 10 minutes for the first hour:

# 1. Check miner is running
ps aux | grep miner.py | grep -v grep && echo "✓ Running" || echo "✗ NOT RUNNING"

# 2. Check recent activity
tail -50 logs/miner_*.log | grep -E "Prediction|Salience|Error"

# 3. Count successful submissions
grep -c "Prediction accepted\|Salience.*received" logs/miner_*.log

# 4. Check for errors
grep -i error logs/miner_*.log | tail -10
```

**Expected After 1 Hour:**
- ✅ Miner still running
- ✅ 10-20 predictions submitted (varies by validator timing)
- ✅ Salience scores received
- ✅ No critical errors

### Step 2.7: Verify All Challenges Active

```bash
# Check which challenges have submitted
for challenge in ETH-LBFGS BTC-LBFGS-6H ETH-HITFIRST-100M \
                 ETH-1H-BINARY EURUSD-1H-BINARY GBPUSD-1H-BINARY \
                 CADUSD-1H-BINARY NZDUSD-1H-BINARY CHFUSD-1H-BINARY \
                 XAUUSD-1H-BINARY XAGUSD-1H-BINARY; do
    count=$(grep -c "$challenge" logs/miner_*.log 2>/dev/null || echo 0)
    echo "$challenge: $count predictions"
done
```

**Expected:** Each challenge shows 1-3 predictions in first hour

### Step 2.8: Document Baseline

```bash
# Save first-day performance
cat > mainnet_baseline_$(date +%Y%m%d).txt << EOF
Deployment Date: $(date)
First Hour Summary:
-------------------
$(grep -c "Salience" logs/miner_*.log) predictions submitted
$(grep "Salience.*received" logs/miner_*.log | head -10)

Expected Rank Week 1: 15-25
Expected Salience: 1.2-1.8 (local test showed: 1.4-1.8)
EOF

cat mainnet_baseline_*.txt
```

## Success Criteria

- ✅ Miner deployed and running
- ✅ First predictions submitted successfully
- ✅ Salience scores received
- ✅ All 11 challenges active
- ✅ No critical errors
- ✅ Baseline documented

**Time to Complete:** 2-4 hours (deployment + monitoring)

**When Complete:** → Proceed to Phase 3 (Week 1 Stabilization)

---

# PHASE 3: Week 1 Stabilization [Days 1-7]

## Goal
Achieve stable 24/7 operation and establish performance baseline.

**Target Rank:** 15-25  
**Target Salience:** 1.2-1.8  
**Time Investment:** 10-15 minutes/day

## Daily Routine (Days 1-7)

### Morning Check (5 minutes)

```bash
cd /home/ocean/Nereus/SN123

# 1. Verify miner running
ps aux | grep miner.py | grep -v grep && echo "✅ Miner Running" || {
    echo "❌ MINER DOWN - Restarting..."
    # Restart miner
    ./start_miner.sh  # or use your startup command
}

# 2. Check overnight performance
tail -100 logs/miner_*.log | grep -i "salience\|error"

# 3. System health
echo "GPU: $(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits)%"
echo "Disk: $(df -h / | tail -1 | awk '{print $5}')"
echo "Memory: $(free -h | grep Mem | awk '{print $3 "/" $2}')"
```

### Evening Check (5 minutes)

```bash
# 1. Daily submission count
echo "Today's predictions: $(grep "$(date +%Y-%m-%d)" logs/miner_*.log | grep -c "Salience")"

# 2. Average salience today
grep "$(date +%Y-%m-%d)" logs/miner_*.log | grep "Salience" | \
    awk '{print $NF}' | awk '{sum+=$1; n++} END {print "Avg Salience:", sum/n}'

# 3. Any errors today?
grep "$(date +%Y-%m-%d)" logs/miner_*.log | grep -i error | tail -5
```

## Weekly Tasks

### Day 3: Performance Analysis

```bash
# Generate Week 1 report
cat > week1_performance.sh << 'EOF'
#!/bin/bash
echo "=== Week 1 Performance Report ==="
echo "Date: $(date)"
echo ""
echo "Total Predictions: $(grep -c "Salience" logs/miner_*.log)"
echo "Uptime: $(uptime -p)"
echo ""
echo "Per Challenge Performance:"
for challenge in ETH-LBFGS BTC-LBFGS-6H ETH-HITFIRST-100M \
                 ETH-1H-BINARY EURUSD-1H-BINARY GBPUSD-1H-BINARY \
                 CADUSD-1H-BINARY NZDUSD-1H-BINARY CHFUSD-1H-BINARY \
                 XAUUSD-1H-BINARY XAGUSD-1H-BINARY; do
    count=$(grep -c "$challenge.*Salience" logs/miner_*.log)
    avg=$(grep "$challenge.*Salience" logs/miner_*.log | \
          awk '{print $NF}' | awk '{sum+=$1; n++} END {if(n>0) print sum/n; else print 0}')
    echo "  $challenge: $count predictions, avg salience: $avg"
done
EOF

chmod +x week1_performance.sh
./week1_performance.sh > week1_report.txt
cat week1_report.txt
```

### Day 5: Identify Weak Challenges

```bash
# Find lowest performing challenges
./week1_performance.sh | grep "avg salience" | sort -k5 -n | head -3

# Save for Week 2 optimization
cat > week2_targets.txt << EOF
Weak Challenges to Optimize in Week 2:
$(./week1_performance.sh | grep "avg salience" | sort -k5 -n | head -3)
EOF
```

### Day 7: Week 1 Review & Plan

```bash
# Complete week review
./week1_performance.sh

# Compare to local test baseline
echo "Local Test Salience: $(grep "Overall Est. Salience" baseline_*.txt | awk '{print $4}')"
echo "Week 1 Actual Salience: $(grep "Salience" logs/miner_*.log | awk '{sum+=$NF; n++} END {print sum/n}')"

# Plan Week 2
cat > week2_plan.txt << EOF
Week 1 Complete: $(date)

Performance Summary:
- Predictions submitted: $(grep -c "Salience" logs/miner_*.log)
- Average salience: $(grep "Salience" logs/miner_*.log | awk '{sum+=$NF; n++} END {print sum/n}')
- Uptime: $(uptime -p)
- Estimated Rank: 15-25

Week 2 Goals:
1. Retrain 2-3 weakest challenges
2. Target: Improve to salience 1.6-2.1
3. Target rank: 10-18

Weak Challenges to Retrain:
$(cat week2_targets.txt)
EOF

cat week2_plan.txt
```

## Success Criteria

- ✅ 99%+ uptime (miner running continuously)
- ✅ All 11 challenges submitting regularly
- ✅ No critical errors
- ✅ Baseline salience: 1.2-1.8
- ✅ Estimated rank: 15-25
- ✅ Week 2 targets identified

**Time Investment:** 10-15 min/day = ~2 hours total

**When Complete:** → Proceed to Phase 4 (Weeks 2-3 Optimization)

---

# PHASE 4: Weeks 2-3 Optimization [Days 8-21]

## Goal
Optimize high-weight challenges and climb to Top 10-15.

**Target Rank:** 10-18  
**Target Salience:** 1.6-2.1  
**Time Investment:** 5-10 hours (training time) + 15 min/day monitoring

## Priority Focus: High-Weight Challenges

### Priority List (By Impact)

```
1. ETH-LBFGS (weight 3.5)      → 18% of total score
2. BTC-LBFGS-6H (weight 2.875) → 15% of total score
3. ETH-HITFIRST (weight 2.5)   → 13% of total score
──────────────────────────────────────────────────
   These 3 = 46% of your total score!
```

**Strategy:** Improve these 3 first for maximum impact

## Week 2 Optimization (Days 8-14)

### Day 8: Identify Optimization Targets

```bash
# Check which high-weight challenges need improvement
for challenge in ETH-LBFGS BTC-LBFGS-6H ETH-HITFIRST-100M; do
    avg=$(grep "$challenge.*Salience" logs/miner_*.log | \
          awk '{print $NF}' | awk '{sum+=$1; n++} END {print sum/n}')
    echo "$challenge: $avg"
done

# Target: ETH-LBFGS >2.0, BTC-LBFGS-6H >1.8, ETH-HITFIRST >1.6
# If any below target → retrain
```

### Day 8-9: Retrain ETH-LBFGS (If Needed)

```bash
# Check current performance
current=$(grep "ETH-LBFGS.*Salience" logs/miner_*.log | \
          awk '{print $NF}' | awk '{sum+=$1; n++} END {print sum/n}')
echo "Current ETH-LBFGS salience: $current"

# If < 2.0, retrain with extended search
if (( $(echo "$current < 2.0" | bc -l) )); then
    echo "Retraining ETH-LBFGS with 100 trials..."
    ./run_training.sh --ticker ETH-LBFGS
    
    # Wait for completion (2-4 hours)
    # Then test locally
    python scripts/testing/backtest_models.py --challenge ETH-LBFGS
    
    # If improved, deploy
    # Restart miner to load new model
fi
```

### Day 10-11: Retrain BTC-LBFGS-6H (If Needed)

```bash
# Same process as ETH-LBFGS
current=$(grep "BTC-LBFGS-6H.*Salience" logs/miner_*.log | \
          awk '{print $NF}' | awk '{sum+=$1; n++} END {print sum/n}')
echo "Current BTC-LBFGS-6H salience: $current"

if (( $(echo "$current < 1.8" | bc -l) )); then
    ./run_training.sh --ticker BTC-LBFGS-6H
    # Test and deploy if improved
fi
```

### Day 12-13: Retrain ETH-HITFIRST (If Needed)

```bash
current=$(grep "ETH-HITFIRST.*Salience" logs/miner_*.log | \
          awk '{print $NF}' | awk '{sum+=$1; n++} END {print sum/n}')
echo "Current ETH-HITFIRST salience: $current"

if (( $(echo "$current < 1.6" | bc -l) )); then
    ./run_training.sh --ticker ETH-HITFIRST-100M
    # Test and deploy if improved
fi
```

### Day 14: Week 2 Review

```bash
# Generate Week 2 report
./week1_performance.sh  # Reuse script with updated logs

# Compare Week 1 vs Week 2
cat > week2_review.txt << EOF
Week 2 Complete: $(date)

High-Weight Challenge Performance:
ETH-LBFGS: $(grep "ETH-LBFGS.*Salience" logs/miner_*.log | tail -50 | awk '{print $NF}' | awk '{sum+=$1; n++} END {print sum/n}')
BTC-LBFGS-6H: $(grep "BTC-LBFGS-6H.*Salience" logs/miner_*.log | tail -50 | awk '{print $NF}' | awk '{sum+=$1; n++} END {print sum/n}')
ETH-HITFIRST: $(grep "ETH-HITFIRST.*Salience" logs/miner_*.log | tail -50 | awk '{print $NF}' | awk '{sum+=$1; n++} END {print sum/n}')

Overall Salience: $(grep "Salience" logs/miner_*.log | tail -100 | awk '{print $NF}' | awk '{sum+=$1; n++} END {print sum/n}')
Target for Week 3: Optimize binary challenges
Expected Rank: 10-18
EOF

cat week2_review.txt
```

## Week 3 Optimization (Days 15-21)

### Goal: Perfect Binary Challenges

**Target:** All binary challenges ≥65% accuracy

### Day 15: Analyze Binary Challenge Performance

```bash
# Check all binary challenges
for challenge in ETH-1H-BINARY EURUSD-1H-BINARY GBPUSD-1H-BINARY \
                 CADUSD-1H-BINARY NZDUSD-1H-BINARY CHFUSD-1H-BINARY \
                 XAUUSD-1H-BINARY XAGUSD-1H-BINARY; do
    avg=$(grep "$challenge.*Salience" logs/miner_*.log | \
          awk '{print $NF}' | awk '{sum+=$1; n++} END {print sum/n}')
    echo "$challenge: $avg"
done | sort -k2 -n

# Identify bottom 3
cat > week3_binary_targets.txt << EOF
Binary Challenges to Improve (Week 3):
$(for challenge in ETH-1H-BINARY EURUSD-1H-BINARY GBPUSD-1H-BINARY \
                  CADUSD-1H-BINARY NZDUSD-1H-BINARY CHFUSD-1H-BINARY \
                  XAUUSD-1H-BINARY XAGUSD-1H-BINARY; do
    avg=$(grep "$challenge.*Salience" logs/miner_*.log | awk '{print $NF}' | awk '{sum+=$1; n++} END {print sum/n}')
    echo "$challenge: $avg"
done | sort -k2 -n | head -3)
EOF

cat week3_binary_targets.txt
```

### Day 15-17: Retrain Weakest Binary Challenges

```bash
# Batch retrain bottom 3 binary challenges
for challenge in WEAK_BINARY_1 WEAK_BINARY_2 WEAK_BINARY_3; do
    echo "Retraining $challenge..."
    ./run_training.sh --ticker $challenge
    
    # Test locally
    python scripts/testing/backtest_models.py --challenge $challenge
    
    # Deploy if improved
done

# Restart miner to load new models
pkill -f "python.*miner"
# Start miner again with updated models
```

### Day 18-20: Data Updates & Fine-Tuning

```bash
# Update all data to latest
python scripts/data_collection/update_all_data.py

# Quick retrain with fresh data for top challenges
for challenge in ETH-LBFGS BTC-LBFGS-6H; do
    ./run_training.sh --ticker $challenge
done
```

### Day 21: Week 3 Review & Month 1 Summary

```bash
# Generate comprehensive Month 1 report
cat > month1_summary.txt << EOF
Month 1 Complete: $(date)

Performance Progression:
Week 1: $(grep "Salience" logs/miner_*.log | head -100 | awk '{print $NF}' | awk '{sum+=$1; n++} END {print sum/n}') salience
Week 2: $(grep "Salience" logs/miner_*.log | head -200 | tail -100 | awk '{print $NF}' | awk '{sum+=$1; n++} END {print sum/n}') salience
Week 3: $(grep "Salience" logs/miner_*.log | tail -100 | awk '{print $NF}' | awk '{sum+=$1; n++} END {print sum/n}') salience

Current Rank Estimate: 10-18
Target for Month 2: Top 10 (rank 5-12)

Next Steps:
1. Advanced optimization techniques
2. Ensemble methods
3. Bayesian hyperparameter tuning
4. Weekly data updates
EOF

cat month1_summary.txt
```

## Success Criteria

- ✅ High-weight challenges optimized (salience improved)
- ✅ Weak binary challenges retrained
- ✅ Overall salience: 1.6-2.1
- ✅ Estimated rank: 10-18
- ✅ Month 2 plan ready

**Time Investment:** 5-10 hours training + 15 min/day = ~8-12 hours total

**When Complete:** → Proceed to Phase 5 (Month 2 Top 10 Push)

---

# PHASE 5: Month 2 - Top 10 Push [Days 22-60]

## Goal
Implement advanced strategies to break into and maintain Top 10.

**Target Rank:** 5-12  
**Target Salience:** 1.9-2.4  
**Time Investment:** 10-15 hours/month + daily monitoring

## Advanced Strategies

### Strategy 1: Bayesian Hyperparameter Optimization

**Why:** Smarter search than grid search, finds better configurations faster.

```bash
# Install Optuna
pip install optuna

# Create advanced tuning script
cat > scripts/training/advanced_tune.py << 'PYTHON'
import optuna
from scripts.training.train_model import train_ticker_model

def objective(trial, challenge, data_dir):
    """Optuna objective function"""
    
    # Suggest hyperparameters
    lstm_hidden = trial.suggest_int('lstm_hidden', 256, 1024, step=128)
    tmfg_n_features = trial.suggest_int('tmfg_n_features', 20, 50, step=5)
    dropout = trial.suggest_float('dropout', 0.2, 0.6)
    learning_rate = trial.suggest_float('learning_rate', 1e-5, 1e-3, log=True)
    
    # Train and evaluate
    result = train_ticker_model(
        ticker=challenge,
        data_dir=data_dir,
        lstm_hidden=lstm_hidden,
        tmfg_n_features=tmfg_n_features,
        dropout=dropout,
        learning_rate=learning_rate
    )
    
    # Return metric to optimize
    return result.get('val_loss', float('inf'))

# Run optimization
study = optuna.create_study(direction='minimize')
study.optimize(lambda trial: objective(trial, 'ETH-LBFGS', 'data'), n_trials=100)

print(f"Best parameters: {study.best_params}")
PYTHON

# Run for top 3 challenges
for challenge in ETH-LBFGS BTC-LBFGS-6H ETH-HITFIRST-100M; do
    echo "Advanced tuning for $challenge..."
    python scripts/training/advanced_tune.py --challenge $challenge --trials 100
done
```

**Expected Impact:** +0.2-0.4 salience per challenge

### Strategy 2: Ensemble Methods

**Why:** Combining multiple model versions reduces variance and improves stability.

```python
# Create ensemble predictor
cat > scripts/prediction/ensemble.py << 'PYTHON'
import numpy as np
from scripts.training.model_architecture import VMDTMFGLSTMXGBoost

class EnsemblePredictor:
    """Ensemble of multiple model versions"""
    
    def __init__(self, model_paths, weights=None):
        self.models = [VMDTMFGLSTMXGBoost.load(path) for path in model_paths]
        self.weights = weights or [1.0 / len(model_paths)] * len(model_paths)
    
    def predict(self, X):
        """Weighted average of all models"""
        predictions = [model.predict_embeddings(X) for model in self.models]
        return np.average(predictions, axis=0, weights=self.weights)
    
    def update_weights(self, recent_performance):
        """Update weights based on recent performance"""
        # Weight models by recent salience scores
        total = sum(recent_performance)
        self.weights = [p/total for p in recent_performance]

# Usage: Keep top 3 versions of each high-weight challenge
# Combine their predictions with weighted average
PYTHON
```

**Expected Impact:** +0.1-0.3 salience per challenge

### Strategy 3: Weekly Data Updates

```bash
# Automate weekly data refresh
cat > scripts/maintenance/weekly_update.sh << 'BASH'
#!/bin/bash
echo "Weekly Data Update - $(date)"

# Update all data
python scripts/data_collection/update_all_data.py

# Quick retrain of top 3 challenges with fresh data
for challenge in ETH-LBFGS BTC-LBFGS-6H ETH-HITFIRST-100M; do
    echo "Quick retrain: $challenge"
    ./run_training.sh --ticker $challenge
done

# Test locally
./test_locally.sh

# If improved, deploy
echo "Review results and deploy if improved"
BASH

chmod +x scripts/maintenance/weekly_update.sh

# Run every Sunday
# Add to crontab: 0 2 * * 0 /path/to/weekly_update.sh
```

**Expected Impact:** Maintains competitiveness as market changes

### Strategy 4: Performance Monitoring Dashboard

```bash
# Create real-time monitoring script
cat > scripts/monitoring/dashboard.sh << 'BASH'
#!/bin/bash

while true; do
    clear
    echo "==================================================================="
    echo "MANTIS Performance Dashboard - $(date)"
    echo "==================================================================="
    echo ""
    
    # Current salience
    echo "Last 24 Hours Performance:"
    recent_salience=$(grep "Salience" logs/miner_*.log | tail -100 | \
                      awk '{print $NF}' | awk '{sum+=$1; n++} END {print sum/n}')
    echo "  Average Salience: $recent_salience"
    
    # Top 3 challenges
    echo ""
    echo "Top Performers:"
    for challenge in ETH-LBFGS BTC-LBFGS-6H ETH-HITFIRST-100M; do
        avg=$(grep "$challenge.*Salience" logs/miner_*.log | tail -20 | \
              awk '{print $NF}' | awk '{sum+=$1; n++} END {print sum/n}')
        echo "  $challenge: $avg"
    done
    
    # System health
    echo ""
    echo "System Health:"
    echo "  Miner: $(ps aux | grep miner.py | grep -v grep > /dev/null && echo "✓ Running" || echo "✗ DOWN")"
    echo "  GPU: $(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits)%"
    echo "  Memory: $(free -h | grep Mem | awk '{print $3 "/" $2}')"
    
    # Predictions today
    echo ""
    echo "Today's Activity:"
    echo "  Predictions: $(grep "$(date +%Y-%m-%d)" logs/miner_*.log | grep -c "Salience")"
    
    sleep 300  # Update every 5 minutes
done
BASH

chmod +x scripts/monitoring/dashboard.sh

# Run in background or separate terminal
# ./scripts/monitoring/dashboard.sh
```

## Month 2 Weekly Schedule

### Weeks 4-5: Advanced Optimization
- Week 4: Implement Bayesian optimization for ETH-LBFGS
- Week 5: Implement ensemble methods for top 3 challenges
- Expected: Rank 8-15

### Weeks 6-7: Stabilization & Fine-Tuning
- Week 6: Weekly data updates, monitor performance
- Week 7: Optimize any challenges still <65% accuracy
- Expected: Rank 6-12

### Week 8: Month 2 Review
```bash
# Generate Month 2 comprehensive report
cat > month2_summary.txt << EOF
Month 2 Complete: $(date)

Advanced Strategies Implemented:
✓ Bayesian hyperparameter optimization
✓ Ensemble methods for top challenges
✓ Weekly data updates
✓ Real-time monitoring dashboard

Performance:
Month 1 Average: [from logs]
Month 2 Average: [from logs]
Improvement: [calculate]

Current Rank Estimate: 5-12
Target for Month 3: Top 5 (rank 2-7)

Month 3 Focus:
- Perfect all challenges (70%+ binary, 2.0+ LBFGS)
- Implement rapid response system
- Daily optimizations
- First place push
EOF

cat month2_summary.txt
```

## Success Criteria

- ✅ Bayesian optimization implemented
- ✅ Ensemble methods deployed
- ✅ Weekly data updates automated
- ✅ Overall salience: 1.9-2.4
- ✅ Estimated rank: 5-12
- ✅ Consistent Top 10 performance

**Time Investment:** 10-15 hours/month

**When Complete:** → Proceed to Phase 6 (First Place Push)

---

# PHASE 6: Month 3+ First Place Push [Days 61+]

## Goal
Reach and maintain #1 position.

**Target Rank:** #1 🏆  
**Target Salience:** 2.2-2.8+  
**Time Investment:** 15-20 hours/month + daily attention

## Elite-Level Strategies

### Strategy 1: Daily Model Refinement

```bash
# Automate daily improvements
cat > scripts/maintenance/daily_optimization.sh << 'BASH'
#!/bin/bash
echo "Daily Optimization - $(date)"

# 1. Check yesterday's performance
yesterday_avg=$(grep "$(date -d '1 day ago' +%Y-%m-%d)" logs/miner_*.log | \
                grep "Salience" | awk '{print $NF}' | \
                awk '{sum+=$1; n++} END {print sum/n}')

echo "Yesterday's average salience: $yesterday_avg"

# 2. Identify any underperforming challenges
for challenge in ETH-LBFGS BTC-LBFGS-6H ETH-HITFIRST-100M; do
    yesterday=$(grep "$(date -d '1 day ago' +%Y-%m-%d).*$challenge" logs/miner_*.log | \
                grep "Salience" | awk '{print $NF}' | \
                awk '{sum+=$1; n++} END {print sum/n}')
    threshold=2.0  # Set per challenge
    
    if (( $(echo "$yesterday < $threshold" | bc -l) )); then
        echo "⚠️  $challenge underperforming: $yesterday < $threshold"
        echo "Scheduling quick retrain..."
        ./run_training.sh --ticker $challenge &
    fi
done

# 3. Update data
python scripts/data_collection/update_single_ticker.py --ticker ETH

echo "Daily optimization complete"
BASH

chmod +x scripts/maintenance/daily_optimization.sh

# Run every morning
# Add to crontab: 0 6 * * * /path/to/daily_optimization.sh
```

### Strategy 2: Competitor Analysis

```bash
# Monitor top 5 miners
cat > scripts/analysis/competitor_monitor.sh << 'BASH'
#!/bin/bash

echo "Top 5 Miner Analysis - $(date)"

# Fetch leaderboard (implement based on your subnet's API)
# Analyze top performers
# Identify gaps in your performance

# Example structure:
echo "Your rank: [fetch from subnet]"
echo "Top 5 rankings:"
# [Implement based on subnet API]

echo ""
echo "Performance gaps to address:"
# Compare your salience to #1
# Identify which challenges they excel at
# Plan improvements

BASH
```

### Strategy 3: Rapid Response System

```python
# Auto-detect and fix performance drops
cat > scripts/monitoring/rapid_response.py << 'PYTHON'
import time
import subprocess
from datetime import datetime, timedelta

class RapidResponseSystem:
    """Automatically respond to performance drops"""
    
    def __init__(self):
        self.baseline = self.get_baseline()
        self.alert_threshold = 0.8  # Alert if drops to 80% of baseline
    
    def get_baseline(self):
        """Calculate 7-day baseline salience"""
        # Implement: parse logs, calculate average
        return 2.0  # placeholder
    
    def check_performance(self):
        """Check recent performance"""
        # Get last 1 hour average salience
        recent = self.get_recent_salience(hours=1)
        
        if recent < self.baseline * self.alert_threshold:
            self.trigger_response(recent)
    
    def trigger_response(self, current_salience):
        """Respond to performance drop"""
        print(f"⚠️ ALERT: Performance drop detected!")
        print(f"Baseline: {self.baseline}, Current: {current_salience}")
        
        # 1. Identify problematic challenge
        weak_challenge = self.identify_weak_challenge()
        
        # 2. Quick retrain
        print(f"Quick retraining {weak_challenge}...")
        subprocess.run([
            './run_training.sh',
            '--trials', '25',
            '--challenge', weak_challenge
        ])
        
        # 3. Deploy immediately
        print("Restarting miner with updated model...")
        subprocess.run(['pkill', '-f', 'miner'])
        # Start miner again
    
    def run_monitoring(self):
        """Run continuous monitoring"""
        while True:
            self.check_performance()
            time.sleep(3600)  # Check every hour

if __name__ == '__main__':
    system = RapidResponseSystem()
    system.run_monitoring()
PYTHON

# Run in background
# nohup python scripts/monitoring/rapid_response.py &
```

### Strategy 4: Advanced Feature Engineering

```python
# Add cutting-edge features
cat > scripts/features/advanced_features.py << 'PYTHON'
import pandas as pd
import numpy as np

class AdvancedFeatureExtractor:
    """Elite-level feature engineering"""
    
    def add_market_regime(self, df):
        """Detect market regime (trending/ranging/volatile)"""
        # ATR for volatility
        df['atr'] = self.calculate_atr(df, period=14)
        df['atr_pct'] = df['atr'] / df['close']
        
        # ADX for trend strength
        df['adx'] = self.calculate_adx(df, period=14)
        
        # Classify regime
        df['regime'] = 'ranging'
        df.loc[df['adx'] > 25, 'regime'] = 'trending'
        df.loc[df['atr_pct'] > 0.03, 'regime'] = 'volatile'
        
        return df
    
    def add_microstructure(self, df):
        """High-frequency microstructure features"""
        # Price position in bar
        df['price_position'] = (df['close'] - df['low']) / (df['high'] - df['low'] + 1e-8)
        
        # Body to total range
        df['body_ratio'] = abs(df['close'] - df['open']) / (df['high'] - df['low'] + 1e-8)
        
        # Upper/lower shadows
        df['upper_shadow'] = (df['high'] - df[['open', 'close']].max(axis=1)) / (df['high'] - df['low'] + 1e-8)
        df['lower_shadow'] = (df[['open', 'close']].min(axis=1) - df['low']) / (df['high'] - df['low'] + 1e-8)
        
        return df
    
    def add_cross_asset(self, df, other_assets):
        """Cross-asset correlations"""
        # Correlation with BTC (crypto king)
        if 'BTC' in other_assets:
            df['corr_btc'] = df['returns'].rolling(50).corr(other_assets['BTC']['returns'])
        
        # Correlation with DXY (dollar index)
        if 'DXY' in other_assets:
            df['corr_dxy'] = df['returns'].rolling(50).corr(other_assets['DXY']['returns'])
        
        return df

# Integrate into training pipeline
PYTHON
```

## Month 3 Weekly Plan

### Week 9: Perfect All Challenges
- Target: Binary >70%, LBFGS >2.0, HITFIRST >1.8
- Implement daily refinement
- Expected: Rank 3-8

### Week 10: Deploy All Advanced Features
- Market regime detection
- Microstructure features
- Cross-asset correlations
- Expected: Rank 2-6

### Week 11: Rapid Response System
- Hourly monitoring
- Auto-detect performance drops
- Auto-retrain and deploy
- Expected: Rank 1-5

### Week 12: First Place Push
- 24/7 monitoring
- Daily optimizations
- Competitor analysis
- Rapid improvements
- **Expected: Rank #1-3**

## The Final Push Checklist

```
□ Daily data updates automated
□ Hourly performance monitoring active
□ Rapid response system running
□ All challenges optimized (70%+ / 2.0+)
□ Ensemble methods deployed
□ Advanced features implemented
□ Competitor analysis regular
□ System reliability 99.9%+
□ Response time < 2 hours for issues
□ Continuous improvement mindset
```

## Maintaining #1 Position

Once you reach #1:

### Daily Tasks (30 min/day)
```bash
# 1. Morning check
./scripts/monitoring/dashboard.sh

# 2. Review overnight performance
grep "$(date -d '1 day ago' +%Y-%m-%d)" logs/miner_*.log | grep Salience | \
    awk '{print $NF}' | awk '{sum+=$1; n++} END {print "Yesterday:", sum/n}'

# 3. Check rank
# [Implement based on subnet API]

# 4. Identify any drops
./scripts/maintenance/daily_optimization.sh
```

### Weekly Tasks (2 hours/week)
```bash
# 1. Data refresh
./scripts/maintenance/weekly_update.sh

# 2. Competitor analysis
./scripts/analysis/competitor_monitor.sh

# 3. Performance review
# Generate weekly report, identify trends

# 4. Optimize weakest challenge
# Always have 1-2 challenges in optimization
```

### Monthly Tasks (4 hours/month)
```bash
# 1. Comprehensive retraining
# Retrain all challenges with latest data and techniques

# 2. Strategy review
# What's working? What's not? New ideas?

# 3. System upgrades
# Update libraries, improve scripts, new features

# 4. Long-term planning
# Stay ahead of competition
```

## Success Criteria

- ✅ Achieved #1 ranking
- ✅ Salience: 2.2-2.8+
- ✅ All challenges optimized
- ✅ Rapid response system active
- ✅ 99.9%+ uptime
- ✅ Continuous improvement ongoing

**Time Investment:** 15-20 hours/month + daily attention

**When Achieved:** 🏆 **FIRST PLACE - MAINTAIN AND DEFEND!**

---

# DAILY MAINTENANCE ROUTINES

## Morning Routine (5-10 minutes)

```bash
#!/bin/bash
# Save as: scripts/maintenance/morning_check.sh

echo "=========================================="
echo "Morning Check - $(date)"
echo "=========================================="

cd /home/ocean/Nereus/SN123

# 1. Miner Status
echo ""
echo "1. Miner Status:"
if ps aux | grep miner.py | grep -v grep > /dev/null; then
    echo "   ✅ Miner Running"
else
    echo "   ❌ MINER DOWN - RESTARTING..."
    # Add your miner start command here
    nohup python neurons/miner.py [your flags] > logs/miner_$(date +%Y%m%d).log 2>&1 &
fi

# 2. Overnight Performance
echo ""
echo "2. Overnight Performance:"
overnight=$(grep "$(date -d '1 day ago' +%Y-%m-%d)" logs/miner_*.log | \
            grep "Salience" | awk '{print $NF}' | \
            awk '{sum+=$1; n++} END {if(n>0) print sum/n; else print "N/A"}')
echo "   Average Salience: $overnight"
echo "   Predictions: $(grep "$(date -d '1 day ago' +%Y-%m-%d)" logs/miner_*.log | grep -c Salience)"

# 3. System Health
echo ""
echo "3. System Health:"
echo "   GPU: $(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits)%"
echo "   Disk: $(df -h / | tail -1 | awk '{print $5}')"
echo "   Memory: $(free | grep Mem | awk '{printf "%.0f%%\n", $3/$2 * 100}')"

# 4. Errors Check
echo ""
echo "4. Recent Errors:"
error_count=$(grep "$(date +%Y-%m-%d)" logs/miner_*.log | grep -ic error)
if [ "$error_count" -gt 0 ]; then
    echo "   ⚠️  $error_count errors today"
    grep "$(date +%Y-%m-%d)" logs/miner_*.log | grep -i error | tail -3
else
    echo "   ✅ No errors today"
fi

echo ""
echo "=========================================="
echo "Morning check complete!"
echo "=========================================="
```

## Evening Routine (5-10 minutes)

```bash
#!/bin/bash
# Save as: scripts/maintenance/evening_check.sh

echo "=========================================="
echo "Evening Check - $(date)"
echo "=========================================="

cd /home/ocean/Nereus/SN123

# 1. Today's Performance
echo ""
echo "1. Today's Performance:"
today=$(grep "$(date +%Y-%m-%d)" logs/miner_*.log | grep "Salience" | \
        awk '{print $NF}' | awk '{sum+=$1; n++} END {if(n>0) print sum/n; else print "N/A"}')
echo "   Average Salience: $today"
echo "   Predictions: $(grep "$(date +%Y-%m-%d)" logs/miner_*.log | grep -c Salience)"

# 2. Top Performers
echo ""
echo "2. Top Challenges Today:"
for challenge in ETH-LBFGS BTC-LBFGS-6H ETH-HITFIRST-100M; do
    avg=$(grep "$(date +%Y-%m-%d).*$challenge" logs/miner_*.log | \
          grep "Salience" | awk '{print $NF}' | \
          awk '{sum+=$1; n++} END {if(n>0) print sum/n; else print "N/A"}')
    echo "   $challenge: $avg"
done

# 3. Week-to-Date
echo ""
echo "3. Week-to-Date Average:"
week_start=$(date -d 'last monday' +%Y-%m-%d)
week_avg=$(awk -v start="$week_start" '$0 >= start' logs/miner_*.log | \
           grep "Salience" | awk '{print $NF}' | \
           awk '{sum+=$1; n++} END {if(n>0) print sum/n; else print "N/A"}')
echo "   $week_avg"

# 4. Tomorrow's Plan
echo ""
echo "4. Action Items:"
# Check if any challenge needs attention
./scripts/maintenance/daily_optimization.sh --check-only

echo ""
echo "=========================================="
echo "Evening check complete!"
echo "=========================================="
```

## Weekly Routine (30-60 minutes)

```bash
#!/bin/bash
# Save as: scripts/maintenance/weekly_maintenance.sh

echo "=========================================="
echo "Weekly Maintenance - $(date)"
echo "=========================================="

cd /home/ocean/Nereus/SN123

# 1. Data Update
echo ""
echo "1. Updating Data..."
python scripts/data_collection/update_all_data.py

# 2. Performance Analysis
echo ""
echo "2. Weekly Performance Analysis:"
./scripts/analysis/weekly_report.sh > reports/week_$(date +%Y%m%d).txt
cat reports/week_$(date +%Y%m%d).txt

# 3. Identify Weak Challenges
echo ""
echo "3. Challenges Needing Attention:"
./scripts/analysis/identify_weak.sh

# 4. Optional: Quick Retrain
echo ""
echo "4. Optional Quick Retrain:"
read -p "Retrain any challenges? (y/N): " retrain
if [ "$retrain" == "y" ]; then
    read -p "Enter challenge name: " challenge
    ./run_training.sh --ticker $challenge
fi

# 5. Backup Important Data
echo ""
echo "5. Creating Backup..."
tar -czf backups/weekly_backup_$(date +%Y%m%d).tar.gz \
    models/tuned logs/miner_*.log results/

echo ""
echo "=========================================="
echo "Weekly maintenance complete!"
echo "=========================================="
```

---

# TROUBLESHOOTING QUICK REFERENCE

## Common Issues & Solutions

### Issue: Miner Not Running

**Symptoms:** `ps aux | grep miner` shows nothing

**Solution:**
```bash
# Check logs for crash reason
tail -100 logs/miner_*.log

# Common causes & fixes:
# 1. Out of memory → Reduce batch size
# 2. Model not found → Check models/tuned/ exists
# 3. Network issue → Check internet connection
# 4. Wallet issue → Verify wallet credentials

# Restart miner
nohup python neurons/miner.py \
    --netuid 123 \
    --wallet.name YOUR_WALLET \
    --wallet.hotkey YOUR_HOTKEY \
    --model.dir models/tuned \
    > logs/miner_$(date +%Y%m%d_%H%M%S).log 2>&1 &
```

### Issue: Low Salience Scores

**Symptoms:** Salience consistently <1.0

**Solution:**
```bash
# 1. Identify weak challenges
grep "Salience" logs/miner_*.log | tail -100 | sort -k2 -n | head -5

# 2. Retrain weakest 2-3
./run_training.sh --ticker WEAK_CHALLENGE

# 3. Update data
python scripts/data_collection/update_all_data.py

# 4. Test locally first
./test_locally.sh
```

### Issue: Predictions Not Submitting

**Symptoms:** No "Salience" messages in logs

**Solution:**
```bash
# Check network connectivity
ping -c 3 google.com

# Check validator connection
grep -i "validator\|connection" logs/miner_*.log | tail -20

# Check model loading
grep -i "loading\|model" logs/miner_*.log | tail -20

# Verify wallet registration
# Check if you're registered on subnet 123
```

### Issue: Training Failed

**Symptoms:** "Training failed" or "ERROR" in training logs

**Solution:**
```bash
# Check full error
grep -A 20 -i error logs/training/training_current.log

# Common causes:
# 1. Insufficient GPU memory → Reduce batch size
# 2. Data not found → Check data/ directory
# 3. Disk full → Free up space
# 4. CUDA error → Restart, update drivers

# Retry training
./run_training.sh --ticker FAILED_CHALLENGE
```

### Issue: System Running Slow

**Symptoms:** High CPU/memory usage, slow predictions

**Solution:**
```bash
# Check resource usage
htop

# Check GPU
nvidia-smi

# Clear old logs
find logs/ -name "*.log" -mtime +30 -delete

# Restart miner
pkill -f miner
# Start again
```

---

# FINAL CHECKLIST: Path to #1

## Phase Completion Checklist

```
Phase 1: Training Complete
  □ All 11 models trained
  □ Models saved to models/tuned/
  □ No critical errors
  
Phase 1.5: Local Testing Complete
  □ All models tested
  □ Overall salience ≥1.5
  □ Binary accuracy ≥65%
  □ Baseline documented
  
Phase 2: Mainnet Deployed
  □ Miner running on mainnet
  □ First predictions submitted
  □ Salience scores received
  □ Baseline established
  
Phase 3: Week 1 Stabilized
  □ 99%+ uptime achieved
  □ All challenges active
  □ Rank: 15-25
  
Phase 4: Weeks 2-3 Optimized
  □ High-weight challenges improved
  □ Weak challenges retrained
  □ Rank: 10-18
  
Phase 5: Month 2 Advanced
  □ Bayesian optimization implemented
  □ Ensemble methods deployed
  □ Rank: 5-12
  
Phase 6: First Place Achieved
  □ All challenges optimized
  □ Rapid response system active
  □ Rank: #1 🏆
```

## Success Indicators

**You're on track if:**
- ✅ Following the phase timeline
- ✅ Salience improving monthly
- ✅ Rank climbing steadily
- ✅ System stable (99%+ uptime)
- ✅ Continuous optimization ongoing

**You need to adjust if:**
- ⚠️ Salience plateaued or decreasing
- ⚠️ Rank stuck or dropping
- ⚠️ Frequent downtime
- ⚠️ Not optimizing regularly

---

# KEY SUCCESS PRINCIPLES

## 1. Consistency Over Perfection
```
Daily 15-minute checks > Weekly 2-hour marathons
Consistent optimization > One-time perfect tuning
Stable operation > Peak performance with crashes
```

## 2. Focus on High-Impact Areas
```
ETH-LBFGS optimization > Minor binary improvements
System uptime > New features
Proven strategies > Experimental ideas
```

## 3. Data-Driven Decisions
```
Measure → Analyze → Improve → Repeat
Local testing before mainnet deployment
Track performance trends
Learn from data
```

## 4. Continuous Improvement
```
First place is not a destination, it's a journey
Always have 1-2 challenges in optimization
Monthly strategy reviews
Stay ahead of competition
```

## 5. Persistence
```
3-6 months to #1 is realistic
Setbacks are temporary
Keep optimizing
First place = Preparation + Persistence
```

---

# SUMMARY: Your Journey to #1

```
Phase 0: NOW
    ↓ (18-24 hours)
Phase 1: Training Complete
    ↓ (1 day)
Phase 1.5: Local Testing
    ↓ (1 day)
Phase 2: Mainnet Deployed [Rank: 15-25]
    ↓ (1 week)
Phase 3: Week 1 Stable [Rank: 15-25]
    ↓ (2 weeks)
Phase 4: Weeks 2-3 Optimized [Rank: 10-18]
    ↓ (1 month)
Phase 5: Month 2 Advanced [Rank: 5-12]
    ↓ (1+ months)
Phase 6: FIRST PLACE [Rank: #1] 🏆
    ↓
Maintain & Defend
```

**Timeline:** 3-6 months to #1  
**Time Investment:** 15-30 min/day + weekly optimizations  
**Success Rate:** High with consistent execution

---

# 🏆 YOU HAVE EVERYTHING YOU NEED!

✅ Complete infrastructure ready  
✅ Training in progress  
✅ Testing scripts prepared  
✅ Deployment guide ready  
✅ Optimization strategies documented  
✅ Daily routines established  
✅ Path to #1 clearly mapped

**Your Next Action:**

1. **Right Now:** Wait for training to complete (check logs)
2. **When Complete:** Run `./test_locally.sh`
3. **When Ready:** Deploy to mainnet (Phase 2)
4. **Then:** Follow this guide, phase by phase

**Remember:** First place = Your solid foundation + Continuous execution + Persistence

**YOU CAN DO THIS! 🚀**

---

**End of Master Guide**

For detailed information on specific phases, see:
- `LOCAL_TESTING_COMPLETE_GUIDE.md` - Phase 1.5 details
- `TESTING_SUMMARY.md` - Testing overview
- `HYPERPARAMETER_TUNING_ASSESSMENT.md` - Tuning analysis
- `COMPLETE_ROADMAP_TO_FIRST_PLACE.md` - Strategic overview
- `SALIENCE_OPTIMIZATION_GUIDE.md` - Salience deep dive

