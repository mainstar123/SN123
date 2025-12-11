# 🏆 Complete Roadmap to First Place in Subnet 123

**Your Definitive Step-by-Step Guide**

**Last Updated:** December 10, 2025  
**Current Status:** Hyperparameter tuning in progress (60% on Challenge 1/11)  
**Goal:** Reach #1 position in Subnet 123

---

## 📍 WHERE YOU ARE NOW

```
✅ Hyperparameter tuning: RUNNING
✅ Progress: Trial 30/50 on Challenge 1/11 (ETH-LBFGS)
✅ GPU optimization: ACTIVE
✅ Best configuration found: Trial 20 (score 1.42)
✅ Scripts: Ready and working
✅ Guides: Complete
⏳ ETA to completion: 18-24 hours
```

---

## 🗺️ THE COMPLETE ROADMAP

### **Phase 1: Training Completion** [CURRENT - Next 18-24 hours]
### **Phase 1.5: Local Testing** [1-3 days] ⭐ NEW - Test before mainnet!
### **Phase 2: Mainnet Deployment** [Day 1]
### **Phase 3: Week 1 Stabilization** [Days 1-7]
### **Phase 4: Week 2 Optimization** [Days 8-14]
### **Phase 5: Week 3 Refinement** [Days 15-21]
### **Phase 6: Week 4 First Place Push** [Days 22-30]
### **Phase 7: Defend Position** [Month 2+]

---

# PHASE 1: Training Completion ⏳

**Timeline:** NOW → Next 18-24 hours  
**Status:** IN PROGRESS (60% on Challenge 1/11)  
**Goal:** Complete hyperparameter tuning for all 11 challenges

## What's Happening Right Now

```
Current Process:
├─ Challenge 1/11: ETH-LBFGS (60% complete)
│   ├─ Completed trials: 30/50
│   ├─ Best configuration: Trial 20 (score 1.42)
│   ├─ Remaining trials: 20 (~2-3 hours)
│   └─ Status: ✅ Working optimally
├─ Challenge 2/11: BTC-LBFGS-6H (pending)
├─ Challenge 3/11: ETH-HITFIRST-100M (pending)
└─ Challenges 4-11: Binary challenges (pending)

GPU: ✅ Active (LSTM training)
CPU: ✅ Active (XGBoost training)
Mixed Precision: ✅ Enabled
Optimization: ✅ Optimal
```

## Your Actions for Phase 1

### ✅ Action 1.1: Monitor Progress (Every 3-4 hours)

```bash
cd /home/ocean/MANTIS

# Quick status check
./run_training.sh --status

# Expected output:
# ✓ Training is RUNNING (PID: 3804567)
# ℹ Current Progress: Trial XX/50
# ℹ Resource Usage: CPU: XXX% | Memory: X.X%
```

**What to look for:**
- ✅ Process running
- ✅ Progress increasing
- ✅ No errors
- ✅ CPU/Memory healthy

**If issues:** See `TRAINING_TROUBLESHOOTING_GUIDE.md`

### ✅ Action 1.2: Prepare Deployment (While Waiting)

```bash
# 1. Verify wallet information
echo "Wallet Name: _________________"
echo "Hotkey: _________________"
echo "Netuid: 123"

# 2. Check data freshness
ls -lh data/*.csv
# Should show recent dates

# 3. Verify miner setup
ls -l neurons/miner.py

# 4. Check system resources
nvidia-smi
df -h
free -h
```

### ✅ Action 1.3: Read Optimization Guides (Optional)

While training runs, familiarize yourself with:
- ✅ `SALIENCE_OPTIMIZATION_GUIDE.md` (20 min read)
- ✅ `FIRST_PLACE_GUIDE.md` (30 min read)
- ✅ `lbfgs_guide.md` (15 min read)

### ✅ Action 1.4: Verify Completion

**When to check:** After 18-24 hours

```bash
# Check for completion
grep "Training Summary" logs/training/training_current.log

# If you see "Training Summary" → Training is COMPLETE!
# If not → Wait longer and check again
```

**Success Criteria:**
- ✅ All 50 trials completed for each challenge
- ✅ Best hyperparameters saved
- ✅ Models saved to `models/tuned/`
- ✅ No critical errors

### 📋 Phase 1 Checklist

- [ ] Training is running (verified)
- [ ] Checked progress 2-3 times
- [ ] Wallet/hotkey information ready
- [ ] System resources adequate
- [ ] Read optimization guides (optional)
- [ ] Training completed (wait for this)
- [ ] All 11 models created in `models/tuned/`

**Next Phase:** When training completes → Phase 1.5 (Local Testing)

---

# PHASE 1.5: Local Testing 🧪

**Timeline:** After training completes, BEFORE mainnet  
**Duration:** 1-3 days  
**Goal:** Test models locally, gain confidence, THEN deploy to mainnet

## ⭐ **SMART APPROACH: Test Locally First!**

### Why Test Locally?

1. ✅ **Zero Risk** - No mainnet exposure during testing
2. ✅ **Fast Feedback** - Immediate results without waiting for validators
3. ✅ **Confidence** - Know your models work before deploying
4. ✅ **Baseline** - Establish expected performance
5. ✅ **Quality Check** - Catch problems before mainnet

### What You'll Test:

- ✅ All 11 models load and run correctly
- ✅ Predictions generate successfully
- ✅ Accuracy on historical data (binary challenges)
- ✅ Embedding quality (LBFGS challenges)
- ✅ Estimated salience scores
- ✅ System performance (latency, resources)

## Complete Local Testing Guide

**📖 See:** `LOCAL_TESTING_BEFORE_MAINNET.md`

This guide includes:
- Complete backtesting scripts (ready to run)
- Performance analysis tools
- Decision criteria (when you're ready for mainnet)
- Troubleshooting steps

### Quick Local Testing

```bash
cd /home/ocean/MANTIS
source venv/bin/activate

# Run automated backtest (from LOCAL_TESTING_BEFORE_MAINNET.md)
python scripts/testing/backtest_models.py

# View results
cat backtest_results.txt

# Make decision
grep "READINESS ASSESSMENT" backtest_results.txt
```

### Decision Point

**If backtest shows:**
- ✅ **Excellent** (binary acc >70%, salience >2.0) → Deploy to mainnet immediately!
- ✓ **Good** (binary acc 60-70%, salience 1.5-2.0) → Deploy to mainnet
- ⚠️ **Fair** (binary acc 55-60%, salience 1.0-1.5) → Improve weak challenges first
- ❌ **Poor** (binary acc <55%, salience <1.0) → Retrain before mainnet

### 📋 Phase 1.5 Checklist

- [ ] All models loaded successfully
- [ ] Backtesting script created and run
- [ ] Results analyzed
- [ ] Performance meets criteria
- [ ] Confidence level: HIGH
- [ ] Decision: READY for mainnet
- [ ] Saved baseline results for comparison

**Next Phase:** When confident → Phase 2 (Mainnet Deployment)

**Complete Guide:** See `LOCAL_TESTING_BEFORE_MAINNET.md` for full details

---

# PHASE 2: Mainnet Deployment 🚀

**Timeline:** After local testing passes  
**Duration:** 2-4 hours  
**Goal:** Deploy optimized models and start mining on Subnet 123

**Prerequisites:**
- ✅ Phase 1 completed (training done)
- ✅ Phase 1.5 completed (local testing passed)
- ✅ Confidence level: HIGH

## Prerequisites

✅ Phase 1 completed  
✅ All 11 models in `models/tuned/`  
✅ Wallet and hotkey ready  
✅ System resources available

## Step 2.1: Verify Training Results

```bash
cd /home/ocean/MANTIS

# 1. Check training summary
grep -A 30 "Training Summary" logs/training/training_current.log

# 2. List all trained models
ls -lR models/tuned/

# Expected: 11 directories (one per challenge)
# ETH-LBFGS/
# BTC-LBFGS-6H/
# ETH-HITFIRST-100M/
# ETH-1H-BINARY/
# EURUSD-1H-BINARY/
# GBPUSD-1H-BINARY/
# CADUSD-1H-BINARY/
# NZDUSD-1H-BINARY/
# CHFUSD-1H-BINARY/
# XAUUSD-1H-BINARY/
# XAGUSD-1H-BINARY/

# 3. Verify each model has required files
for challenge in models/tuned/*/; do
    echo "Checking $challenge"
    ls -l "$challenge"
done

# Each should contain:
# - LSTM model files
# - XGBoost model
# - Scaler
# - Config
```

**Success Criteria:**
- ✅ 11 challenge directories exist
- ✅ Each has model files
- ✅ No error messages in summary

## Step 2.2: Test One Model Locally

```bash
cd /home/ocean/MANTIS
source venv/bin/activate

# Test ETH-LBFGS (highest priority)
python << 'EOF'
import pandas as pd
import numpy as np
from scripts.training.model_architecture import VMDTMFGLSTMXGBoost

print("Loading model...")
model = VMDTMFGLSTMXGBoost.load('models/tuned/ETH-LBFGS')
print("✓ Model loaded successfully")

print("\nLoading test data...")
df = pd.read_csv('data/ETH_1h.csv').tail(100)
print(f"✓ Loaded {len(df)} rows")

print("\nPreparing data...")
X, y, features = model.prepare_data(df)
print(f"✓ Created {len(X)} sequences with {X.shape[2]} features")

print("\nGenerating prediction...")
pred = model.predict_embeddings(X[-1:])
print(f"✓ Prediction shape: {pred.shape}")
print(f"✓ Prediction (first 5 values): {pred[0][:5]}")

print("\n✅ Model test PASSED - Ready for deployment!")
EOF
```

**Expected Output:**
```
✓ Model loaded successfully
✓ Loaded 100 rows
✓ Created XX sequences with XX features
✓ Prediction shape: (1, 17)
✓ Prediction (first 5 values): [0.XX 0.XX 0.XX 0.XX 0.XX]
✅ Model test PASSED
```

**If errors:** Check logs, verify model files exist

## Step 2.3: Configure Miner for Tuned Models

```bash
cd /home/ocean/MANTIS

# Backup current config (if exists)
cp neurons/miner.py neurons/miner.py.backup

# Update miner configuration
# Edit neurons/miner.py to use models/tuned/
# Change: MODEL_DIR = "models/checkpoints"
# To:     MODEL_DIR = "models/tuned"
```

**Or create config file:**

```bash
cat > mining_config.json << 'EOF'
{
  "model_directory": "models/tuned",
  "wallet_name": "YOUR_WALLET_NAME",
  "wallet_hotkey": "YOUR_HOTKEY",
  "netuid": 123,
  "subtensor_network": "finney",
  "logging_level": "INFO",
  "challenges_enabled": [
    "ETH-LBFGS",
    "BTC-LBFGS-6H",
    "ETH-HITFIRST-100M",
    "ETH-1H-BINARY",
    "EURUSD-1H-BINARY",
    "GBPUSD-1H-BINARY",
    "CADUSD-1H-BINARY",
    "NZDUSD-1H-BINARY",
    "CHFUSD-1H-BINARY",
    "XAUUSD-1H-BINARY",
    "XAGUSD-1H-BINARY"
  ]
}
EOF
```

## Step 2.4: Start Miner with Optimized Models

```bash
cd /home/ocean/MANTIS
source venv/bin/activate

# Stop any existing miner
pkill -f "python.*miner"

# Start miner with new models
python neurons/miner.py \
    --netuid 123 \
    --wallet.name YOUR_WALLET_NAME \
    --wallet.hotkey YOUR_HOTKEY \
    --subtensor.network finney \
    --logging.debug \
    --model.dir models/tuned \
    > logs/miner.log 2>&1 &

# Save PID
echo $! > logs/miner.pid

echo "Miner started! PID: $(cat logs/miner.pid)"
```

## Step 2.5: Monitor Initial Performance

```bash
# Watch logs for 5-10 minutes
tail -f logs/miner.log

# Look for:
# ✓ "Loading model from models/tuned/ETH-LBFGS"
# ✓ "Prediction generated for challenge: ETH-LBFGS"
# ✓ "Submitting prediction to validator"
# ✓ "Salience score received: X.XX"
```

**First Hour Monitoring:**

```bash
# Every 10 minutes for first hour:
echo "=== Miner Status Check ==="
echo "Time: $(date)"
echo ""

# Check if running
ps aux | grep miner.py | grep -v grep && echo "✓ Miner running" || echo "✗ Miner NOT running"

# Check recent activity
tail -20 logs/miner.log | grep -E "Prediction|Salience|Error"

# Count submissions
grep -c "Prediction submitted" logs/miner.log
```

**Success Criteria (First Hour):**
- ✅ Miner stays running
- ✅ No critical errors
- ✅ Predictions generated for multiple challenges
- ✅ Submissions successful
- ✅ Salience scores received

## Step 2.6: Verify All Challenges Active

```bash
# Check which challenges are working
for challenge in ETH-LBFGS BTC-LBFGS-6H ETH-HITFIRST-100M ETH-1H-BINARY EURUSD-1H-BINARY GBPUSD-1H-BINARY CADUSD-1H-BINARY NZDUSD-1H-BINARY CHFUSD-1H-BINARY XAUUSD-1H-BINARY XAGUSD-1H-BINARY; do
    count=$(grep -c "$challenge" logs/miner.log)
    echo "$challenge: $count predictions"
done

# All should show > 0
```

### 📋 Phase 2 Checklist

- [ ] Training results verified
- [ ] Model test passed
- [ ] Miner configured for tuned models
- [ ] Miner started successfully
- [ ] Monitored for 1 hour
- [ ] All 11 challenges active
- [ ] Salience scores received
- [ ] No critical errors

**Next Phase:** After successful deployment → Phase 3 (Stabilization)

---

# PHASE 3: Week 1 Stabilization 🎯

**Timeline:** Days 1-7  
**Goal:** Stable 24/7 operation with baseline performance
**Target Rank:** Top 20

## Daily Tasks (5 minutes/day)

### Morning Check (3 minutes)

```bash
cd /home/ocean/MANTIS

# 1. Miner status
ps aux | grep miner.py | grep -v grep || echo "⚠️ MINER DOWN!"

# 2. Recent performance
tail -50 logs/miner.log | grep -i "salience\|error"

# 3. System health
echo "GPU: $(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits)%"
echo "Disk: $(df -h / | tail -1 | awk '{print $5}')"
echo "Memory: $(free | grep Mem | awk '{printf "%.0f%%\n", $3/$2 * 100}')"
```

### Evening Check (2 minutes)

```bash
# 1. Submission count
echo "Today's submissions: $(grep "$(date +%Y-%m-%d)" logs/miner.log | grep -c "submitted")"

# 2. Error count
echo "Today's errors: $(grep "$(date +%Y-%m-%d)" logs/miner.log | grep -ic error)"

# 3. Best challenge
grep "salience" logs/miner.log | tail -20 | sort -k3 -n | tail -5
```

## Week 1 Goals

### Day 1: Deploy & Monitor
- ✅ Miner running 24 hours
- ✅ All challenges submitting
- ✅ Baseline salience scores established

### Day 2: Stability
- ✅ No crashes
- ✅ 100% uptime
- ✅ Consistent submissions

### Day 3: Performance Analysis
```bash
# Generate performance report
cat > weekly_analysis.sh << 'EOF'
#!/bin/bash
echo "=== Week 1 Performance Report ==="
echo ""
echo "Uptime: $(uptime -p)"
echo "Total submissions: $(grep -c "submitted" logs/miner.log)"
echo ""
echo "Per Challenge:"
for challenge in ETH-LBFGS BTC-LBFGS-6H ETH-HITFIRST-100M ETH-1H-BINARY EURUSD-1H-BINARY GBPUSD-1H-BINARY CADUSD-1H-BINARY NZDUSD-1H-BINARY CHFUSD-1H-BINARY XAUUSD-1H-BINARY XAGUSD-1H-BINARY; do
    count=$(grep "$challenge" logs/miner.log | grep -c "submitted")
    avg_salience=$(grep "$challenge.*salience" logs/miner.log | grep -oP 'salience: \K[0-9.]+' | awk '{sum+=$1; n++} END {if(n>0) print sum/n; else print 0}')
    echo "  $challenge: $count submissions, avg salience: $avg_salience"
done
EOF
chmod +x weekly_analysis.sh
./weekly_analysis.sh
```

### Day 4-5: Identify Weak Challenges
```bash
# Find lowest performing challenges
grep "salience" logs/miner.log | \
    awk '{challenge=$3; salience=$NF; sum[challenge]+=salience; count[challenge]++} 
         END {for(c in sum) print c, sum[c]/count[c]}' | \
    sort -k2 -n | head -3

# These are candidates for retraining
```

### Day 6: Data Update
```bash
# Update data for all challenges
cd /home/ocean/MANTIS
source venv/bin/activate

python scripts/data_collection/update_all_data.py

# Check new data
ls -lh data/*.csv
```

### Day 7: Week 1 Review
```bash
./weekly_analysis.sh > week1_report.txt
cat week1_report.txt

# Decide:
# - Which challenges need retraining?
# - What's the current leaderboard position?
# - Any system issues to fix?
```

### 📋 Phase 3 Checklist

- [ ] Day 1: Successful deployment
- [ ] Day 2: 24-hour stable operation
- [ ] Day 3: Performance report generated
- [ ] Day 4-5: Weak challenges identified
- [ ] Day 6: Data updated
- [ ] Day 7: Week 1 review completed
- [ ] 100% uptime achieved
- [ ] Baseline performance established
- [ ] Rank: Top 20 (target)

**Next Phase:** Week 2 → Phase 4 (Optimization)

---

# PHASE 4: Week 2 Optimization 🔥

**Timeline:** Days 8-14  
**Goal:** Optimize high-weight challenges for maximum impact
**Target Rank:** Top 10

## Focus: High-Weight Challenges

### Priority 1: ETH-LBFGS (Weight: 3.5)

```bash
# Check current performance
grep "ETH-LBFGS.*salience" logs/miner.log | \
    tail -100 | \
    grep -oP 'salience: \K[0-9.]+' | \
    awk '{sum+=$1; n++; if($1>max) max=$1; if(min=="" || $1<min) min=$1} 
         END {print "Avg:", sum/n, "Min:", min, "Max:", max}'

# If average < 2.0, retrain with more trials
./run_training.sh --trials 100 --challenge ETH-LBFGS
```

### Priority 2: BTC-LBFGS-6H (Weight: 2.875)

```bash
# Same analysis
grep "BTC-LBFGS-6H.*salience" logs/miner.log | \
    tail -100 | \
    grep -oP 'salience: \K[0-9.]+' | \
    awk '{sum+=$1; n++} END {print "Avg:", sum/n}'

# Retrain if needed
./run_training.sh --trials 100 --challenge BTC-LBFGS-6H
```

### Priority 3: ETH-HITFIRST-100M (Weight: 2.5)

```bash
# Analysis
grep "ETH-HITFIRST.*salience" logs/miner.log | \
    tail -100 | \
    grep -oP 'salience: \K[0-9.]+' | \
    awk '{sum+=$1; n++} END {print "Avg:", sum/n}'

# Retrain if needed
./run_training.sh --trials 100 --challenge ETH-HITFIRST-100M
```

## Week 2 Daily Optimization

### Days 8-9: Retrain Weakest High-Weight Challenge
### Days 10-11: Deploy & Monitor Improvement
### Days 12-13: Retrain Second Weakest
### Day 14: Week 2 Review

## Advanced: Ensemble Methods

```python
# Create ensemble predictor (models/ensemble_predictor.py)
class EnsemblePredictor:
    def __init__(self, model_paths):
        self.models = [load_model(path) for path in model_paths]
    
    def predict(self, X):
        predictions = [model.predict(X) for model in self.models]
        # Weighted average (70% new, 30% old)
        return 0.7 * predictions[0] + 0.3 * predictions[1]

# Use in miner for high-weight challenges
```

### 📋 Phase 4 Checklist

- [ ] Analyzed all high-weight challenges
- [ ] Retrained ETH-LBFGS (if needed)
- [ ] Retrained BTC-LBFGS-6H (if needed)
- [ ] Retrained ETH-HITFIRST (if needed)
- [ ] Deployed improvements
- [ ] Measured performance increase
- [ ] Week 2 review completed
- [ ] Rank: Top 10 (target)

**Next Phase:** Week 3 → Phase 5 (Refinement)

---

# PHASE 5: Week 3 Refinement 💎

**Timeline:** Days 15-21  
**Goal:** Perfect binary challenges & advanced features
**Target Rank:** Top 5

## Binary Challenges Optimization

### Goal: 70%+ Accuracy on Each

```bash
# Check all binary challenges
for challenge in ETH-1H-BINARY EURUSD-1H-BINARY GBPUSD-1H-BINARY CADUSD-1H-BINARY NZDUSD-1H-BINARY CHFUSD-1H-BINARY XAUUSD-1H-BINARY XAGUSD-1H-BINARY; do
    echo "=== $challenge ==="
    grep "$challenge" logs/miner.log | tail -50 | \
        grep -oP 'accuracy: \K[0-9.]+' | \
        awk '{sum+=$1; n++} END {print "Avg accuracy:", sum/n}'
done
```

### Batch Retrain Binary Challenges

```bash
# Retrain all binary challenges with extended trials
for challenge in ETH-1H-BINARY EURUSD-1H-BINARY GBPUSD-1H-BINARY CADUSD-1H-BINARY NZDUSD-1H-BINARY CHFUSD-1H-BINARY XAUUSD-1H-BINARY XAGUSD-1H-BINARY; do
    echo "Training $challenge..."
    ./run_training.sh --trials 75 --challenge $challenge
    echo "Completed $challenge"
done
```

## Advanced Feature Engineering

### Add Custom Features

```python
# In scripts/feature_engineering/feature_extractor.py
# Add market regime detection:

def add_advanced_features(self, df):
    """Add cutting-edge features"""
    
    # 1. Market regime
    df['volatility_regime'] = df['close'].pct_change().rolling(20).std()
    df['trend_strength'] = (df['close'].rolling(50).mean() - 
                           df['close'].rolling(200).mean()) / df['close']
    
    # 2. Volume profile
    df['volume_regime'] = df['volume'] / df['volume'].rolling(50).mean()
    df['price_volume_trend'] = df['close'].pct_change() * df['volume_regime']
    
    # 3. Momentum indicators
    df['rsi'] = self.calculate_rsi(df['close'], 14)
    df['macd'] = self.calculate_macd(df['close'])
    
    # 4. Cross-asset correlation (if multiple assets available)
    # df['eth_btc_corr'] = df['eth_close'].rolling(50).corr(df['btc_close'])
    
    return df
```

## Latency Optimization

```python
# Optimize prediction speed
class FastPredictor:
    def __init__(self):
        self.model_cache = {}
        self.feature_cache = {}
        
    def predict(self, challenge, data):
        # Pre-compute features
        if challenge in self.feature_cache:
            features = self.update_incremental(self.feature_cache[challenge], data)
        else:
            features = self.extract_full(data)
            self.feature_cache[challenge] = features
        
        # Fast inference
        return self.model_cache[challenge].predict(features)
```

### 📋 Phase 5 Checklist

- [ ] Analyzed all binary challenges
- [ ] Retrained weak binary challenges
- [ ] All binary challenges >65% accuracy
- [ ] Advanced features implemented
- [ ] Latency optimized
- [ ] Week 3 review completed
- [ ] Rank: Top 5 (target)

**Next Phase:** Week 4 → Phase 6 (First Place Push)

---

# PHASE 6: Week 4 First Place Push 🏆

**Timeline:** Days 22-30  
**Goal:** REACH #1 POSITION
**Target Rank:** #1

## The Final Push Strategy

### Day 22-23: Final Model Polish

```bash
# Retrain ALL challenges with maximum trials
for challenge in ETH-LBFGS BTC-LBFGS-6H ETH-HITFIRST-100M; do
    ./run_training.sh --trials 150 --challenge $challenge
done
```

### Day 24-25: Deploy & Monitor

```bash
# Deploy all optimized models
# Monitor performance every 2 hours
# Track leaderboard position every 4 hours
```

### Day 26-27: Rapid Response System

```python
# Create auto-retraining script
def monitor_and_adapt():
    while True:
        # Check performance every hour
        recent_performance = get_last_hour_performance()
        
        for challenge in CHALLENGES:
            if recent_performance[challenge] < threshold:
                # Quick retrain
                retrain_challenge(challenge, trials=25)
                deploy_immediately(challenge)
        
        time.sleep(3600)  # 1 hour
```

### Day 28-29: Competitor Analysis

```bash
# Check top 3 miners
# Analyze their strategies
# Identify gaps in your performance
# Rapid improvements
```

### Day 30: Final Optimizations

- Perfect timing (submit at optimal moments)
- Maximize all challenge coverage
- Ensure 100% uptime
- Monitor continuously

## Final Week Metrics

Track every 4 hours:
```bash
cat > track_position.sh << 'EOF'
#!/bin/bash
while true; do
    echo "$(date): Checking position..."
    # Your leaderboard check here
    # Log position
    # Alert if dropped
    sleep 14400  # 4 hours
done
EOF
```

### 📋 Phase 6 Checklist

- [ ] Day 22-23: Final retraining completed
- [ ] Day 24-25: All models deployed and stable
- [ ] Day 26-27: Rapid response system active
- [ ] Day 28-29: Competitor analysis done
- [ ] Day 30: Final optimizations applied
- [ ] Rank: #1 ACHIEVED 🏆

**Next Phase:** Month 2+ → Phase 7 (Defend Position)

---

# PHASE 7: Defend #1 Position 🛡️

**Timeline:** Month 2 onwards  
**Goal:** STAY at #1  
**Target:** Maintain position continuously

## Daily Defense (10 minutes)

### Morning Routine (5 min)
```bash
# 1. Position check
check_leaderboard_position.sh

# 2. Performance check
./weekly_analysis.sh | tail -20

# 3. Competitor movement
track_top_5_miners.sh
```

### Evening Routine (5 min)
```bash
# 1. Any drops in performance?
check_daily_metrics.sh

# 2. Any new competitors?
identify_threats.sh

# 3. System health
system_health_check.sh
```

## Weekly Maintenance

### Every Monday: Data Update
```bash
python scripts/data_collection/update_all_data.py
```

### Every Wednesday: Performance Review
```bash
./weekly_analysis.sh > week_N_report.txt
# Review and identify any declining trends
```

### Every Friday: Competitive Analysis
```bash
# Check if anyone is catching up
# Prepare defensive strategies
```

## Monthly: Full Retraining

```bash
# First of each month
./run_training.sh --trials 100

# Deploy fresh models with latest data
```

## Defensive Strategies

### Strategy 1: Rapid Response
```python
# If position drops below #3
if current_position > 3:
    # Emergency retrain highest-weight challenges
    emergency_retrain(['ETH-LBFGS', 'BTC-LBFGS-6H'])
    deploy_immediately()
```

### Strategy 2: Continuous Improvement
```python
# Always be improving
def continuous_optimization():
    while True:
        weakest = identify_weakest_challenge()
        retrain(weakest, trials=50)
        deploy_if_better()
        sleep(7 * 24 * 3600)  # Weekly
```

### Strategy 3: Innovation
- Test new architectures
- Experiment with new features
- Try different optimization strategies
- Stay ahead of competitors

### 📋 Phase 7 Checklist

**Daily:**
- [ ] Morning position check
- [ ] Evening performance check
- [ ] System health verified

**Weekly:**
- [ ] Data updated (Monday)
- [ ] Performance reviewed (Wednesday)
- [ ] Competitive analysis (Friday)

**Monthly:**
- [ ] Full retraining completed
- [ ] Fresh models deployed
- [ ] Position maintained at #1

---

# 📊 SUCCESS METRICS

## Key Performance Indicators

Track these continuously:

### Overall Performance
```
Target: #1 Position
Current: Track daily
Trend: Should stay at #1
```

### Per-Challenge Performance
```
ETH-LBFGS: Salience > 2.5
BTC-LBFGS-6H: Salience > 2.0
ETH-HITFIRST: Salience > 1.8
Binary Challenges: Accuracy > 70%
```

### System Health
```
Uptime: 99.9%+
Latency: <5 seconds per prediction
Error Rate: <0.1%
GPU Utilization: 40-60% (during LSTM phases)
CPU Utilization: 400-600% (during XGBoost)
```

---

# 🎯 MILESTONE CHECKLIST

## Major Milestones

- [ ] **Phase 1:** Training completed (18-24 hours)
- [ ] **Phase 2:** Models deployed (Day 1)
- [ ] **Phase 3:** Week 1 - Stable operation, Top 20
- [ ] **Phase 4:** Week 2 - Optimized high-weight challenges, Top 10
- [ ] **Phase 5:** Week 3 - Refined all challenges, Top 5
- [ ] **Phase 6:** Week 4 - Reached #1 position 🏆
- [ ] **Phase 7:** Month 2+ - Defending #1 position 🛡️

---

# 🚨 CRITICAL SUCCESS FACTORS

## 1. Uptime (Most Important)
```
Target: 99.9%
Action: Monitoring + auto-restart scripts
Impact: Miss predictions = lose position
```

## 2. High-Weight Challenges (Maximum Impact)
```
Priority: ETH-LBFGS > BTC-LBFGS-6H > ETH-HITFIRST
Action: Extra attention, more trials, continuous optimization
Impact: These 3 = 45% of total weight
```

## 3. Consistency (Steady Performance)
```
Goal: No bad days
Action: Stable models, tested thoroughly
Impact: One bad day can drop many ranks
```

## 4. Rapid Response (Adapt Quickly)
```
Monitor: Performance drops, competitor improvements
Action: Quick retraining, immediate deployment
Impact: Stay ahead of competition
```

## 5. Continuous Improvement (Never Stop)
```
Mindset: Always improving
Action: Weekly optimizations, monthly retraining
Impact: Maintain edge over time
```

---

# 📞 QUICK REFERENCE

## Essential Commands

```bash
# Training
./run_training.sh --status           # Check training
./run_training.sh --trials 50        # Start training
./run_training.sh --stop             # Stop training

# Monitoring
tail -f logs/miner.log               # Watch miner
tail -f logs/training/training_current.log  # Watch training
./run_training.sh --status           # Quick status

# System Health
nvidia-smi                           # GPU status
ps aux | grep python                 # Python processes
df -h                                # Disk space
free -h                              # Memory

# Performance
./weekly_analysis.sh                 # Performance report
grep "salience" logs/miner.log       # Salience scores
```

## Critical Files

```
Models: models/tuned/
Logs: logs/miner.log, logs/training/
Scripts: ./run_training.sh
Guides: FIRST_PLACE_GUIDE.md, SALIENCE_OPTIMIZATION_GUIDE.md
```

---

# 🎓 ADDITIONAL RESOURCES

## Documentation
- `FIRST_PLACE_GUIDE.md` - Detailed strategy
- `SALIENCE_OPTIMIZATION_GUIDE.md` - Optimization techniques
- `TRAINING_TROUBLESHOOTING_GUIDE.md` - Fix issues
- `lbfgs_guide.md` - LBFGS challenge details
- `MINER_GUIDE.md` - Mining setup

## Support
- Check logs first
- Review troubleshooting guide
- Verify system resources
- Test individual components

---

# ✅ YOUR CURRENT ACTION

**RIGHT NOW (Phase 1):**

```bash
# 1. Verify training is running
cd /home/ocean/MANTIS
./run_training.sh --status

# 2. Wait for completion (18-24 hours)
# Check every 3-4 hours

# 3. When complete, start Phase 2
grep "Training Summary" logs/training/training_current.log
# If you see this → Move to Phase 2!
```

**THEN:** Follow each phase in order, completing all checklists.

---

# 🏆 FINAL THOUGHTS

## The Path to #1

```
Training (NOW) → Deploy (Day 1) → Stabilize (Week 1) →
Optimize (Week 2) → Refine (Week 3) → Push (Week 4) →
REACH #1 → Defend (Month 2+) → STAY #1
```

## Success Formula

```
First Place = 
    Better Models (training now) +
    Smart Optimization (weekly improvements) +
    System Reliability (99.9% uptime) +
    Continuous Improvement (never stop) +
    Rapid Response (adapt quickly)
```

## You Have Everything You Need

✅ Fixed training pipeline  
✅ Optimized hyperparameters  
✅ Complete roadmap (this file)  
✅ All necessary guides  
✅ Automated scripts  
✅ Clear action plan  

**Just follow this roadmap, phase by phase, and you'll reach #1!** 🚀

---

**Current Phase:** Phase 1 (Training) - 60% complete  
**Next Milestone:** Training completion in ~18-24 hours  
**Final Goal:** #1 Position in Subnet 123  

**Let's get that #1 spot!** 🏆💪

