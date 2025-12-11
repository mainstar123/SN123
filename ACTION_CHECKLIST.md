# 🎯 First Place Action Checklist

**Simple daily checklist to reach #1 in Subnet 123**

---

## ⏳ Phase 1: Waiting for Training [Current - Next 8-10 hours]

### What's Happening:
- ✅ Training running (60% complete on challenge 1/11)
- ⏳ ~8-10 hours until all models ready
- 💻 GPU working perfectly

### Your Tasks:

- [ ] **Check training progress** (every 2-3 hours)
  ```bash
  ./run_training.sh --status
  ```

- [ ] **Verify data is recent** (do once)
  ```bash
  ls -lh data/*.csv
  # Should show recent timestamps
  ```

- [ ] **Read the guides** (optional but helpful)
  - [ ] `FIRST_PLACE_GUIDE.md` - Complete strategy
  - [ ] `SALIENCE_OPTIMIZATION_GUIDE.md` - Optimization tips

- [ ] **Prepare wallet details** (write down)
  - Wallet name: `________________`
  - Hotkey name: `________________`
  - Netuid: `123`

---

## 🚀 Phase 2: Deploy Models [After training completes]

### Step 1: Verify Training Success

```bash
cd /home/ocean/MANTIS

# Check completion
grep "Training Summary" logs/training/training_current.log

# Verify all models exist
ls -l models/tuned/
# Should see 11 challenge folders
```

**Checklist:**
- [ ] Training completed successfully
- [ ] All 11 challenges have models
- [ ] No errors in final summary
- [ ] Models saved to `models/tuned/`

### Step 2: Test One Model

```bash
# Test prediction on ETH-LBFGS
source venv/bin/activate
python -c "
from scripts.training.model_architecture import VMDTMFGLSTMXGBoost
import pandas as pd

# Load test data
df = pd.read_csv('data/ETH_1h.csv').tail(100)

# Load model
model = VMDTMFGLSTMXGBoost.load('models/tuned/ETH-LBFGS')

# Test prediction
X, y, _ = model.prepare_data(df)
pred = model.predict_embeddings(X[-1:])
print('Prediction shape:', pred.shape)
print('Prediction:', pred[0])
"
```

**Checklist:**
- [ ] Model loads without errors
- [ ] Prediction completes successfully
- [ ] Output shape is correct (17 dims for LBFGS)

### Step 3: Deploy to Miner

```bash
# Stop any running miner
pkill -f "python.*miner"

# Update miner to use new models
# Edit neurons/miner.py or config:
# MODEL_DIR = "models/tuned"

# Start miner
python neurons/miner.py \
    --netuid 123 \
    --wallet.name YOUR_WALLET \
    --wallet.hotkey YOUR_HOTKEY \
    --logging.debug
```

**Checklist:**
- [ ] Old miner stopped
- [ ] Config updated to use `models/tuned/`
- [ ] New miner started successfully
- [ ] No errors in startup logs

### Step 4: Monitor First Hour

```bash
# Watch logs in real-time
tail -f logs/miner.log
```

**Look for:**
- [ ] Successful predictions for all challenges
- [ ] No error messages
- [ ] Regular submissions (every few minutes)
- [ ] Salience scores appearing

---

## 📊 Phase 3: Daily Monitoring [Day 1-7]

### Morning Routine (5 minutes)

```bash
cd /home/ocean/MANTIS

# 1. Check miner is running
ps aux | grep miner | grep -v grep

# 2. Check overnight performance
tail -200 logs/miner.log | grep -i "salience\|error"

# 3. Check GPU health
nvidia-smi

# 4. Check disk space
df -h
```

**Daily Checklist:**
- [ ] Miner running ✓
- [ ] No errors overnight ✓
- [ ] GPU temperature OK (<80°C) ✓
- [ ] Disk space OK (>10GB free) ✓

### Evening Routine (10 minutes)

```bash
# 1. Review daily performance
python scripts/analysis/daily_report.py  # Create this if needed

# 2. Check leaderboard position
# Visit validator dashboard or:
python scripts/analysis/check_rank.py

# 3. Identify weak challenges
grep "salience" logs/miner.log | tail -50 | sort -k2 -n
```

**Daily Goals:**
- [ ] All 11 challenges submitted successfully
- [ ] No prediction errors
- [ ] Salience scores stable or improving
- [ ] System uptime: 100%

---

## 🎯 Phase 4: Weekly Optimization [Week 1-4]

### Week 1 Goals:
- [ ] Stable mining operation (100% uptime)
- [ ] All challenges working
- [ ] Leaderboard position: Top 20
- [ ] Identify 2-3 weakest challenges

### Week 1 Tasks:

**Day 1-2:** Stabilize
```bash
# Just monitor, fix any issues
./run_training.sh --status  # Training should be done
tail -f logs/miner.log       # Watch for errors
```
- [ ] Mining stable for 48 hours
- [ ] All challenges submitting

**Day 3-4:** Analyze
```bash
# Find weakest challenges
grep -A 1 "Challenge:" logs/miner.log | grep salience | sort -k2 -n

# Focus on:
# 1. ETH-LBFGS (weight 3.5) - most important
# 2. BTC-LBFGS-6H (weight 2.875) - second most important
# 3. Any challenge with low salience
```
- [ ] Identified 2-3 weak challenges
- [ ] Noted which are high-weight (priority)

**Day 5-6:** Optimize
```bash
# Retrain weakest high-weight challenge
./run_training.sh --trials 100 --challenge WEAK_CHALLENGE_NAME

# Let it run overnight
# Deploy updated model in the morning
```
- [ ] Started retraining for weak challenge
- [ ] Updated model deployed

**Day 7:** Review
```bash
# Weekly performance report
echo "=== Week 1 Performance ==="
echo "Challenges completed: $(grep -c 'prediction submitted' logs/miner.log)"
echo "Errors: $(grep -c -i error logs/miner.log)"
echo "Avg salience (ETH-LBFGS): $(grep 'ETH-LBFGS.*salience' logs/miner.log | awk '{sum+=$NF; n++} END {print sum/n}')"
```
- [ ] Week 1 review completed
- [ ] Leaderboard position improved
- [ ] Plan for Week 2 ready

---

## 🏆 Path to #1 (Weekly Milestones)

### Week 1: Stabilize & Baseline
**Goal: Top 20**
- [X] Training completed (Done!)
- [ ] Models deployed
- [ ] 100% uptime
- [ ] All challenges working
- [ ] Baseline performance established

### Week 2: Optimize High-Weight Challenges
**Goal: Top 10**
- [ ] ETH-LBFGS optimized (weight 3.5)
- [ ] BTC-LBFGS-6H optimized (weight 2.875)
- [ ] ETH-HITFIRST optimized (weight 2.5)
- [ ] Data collection automated
- [ ] Monitoring dashboard setup

### Week 3: Perfect Binary Challenges
**Goal: Top 5**
- [ ] All 8 binary challenges >70% accuracy
- [ ] Fast submission times (<5 sec per prediction)
- [ ] Feature engineering improvements
- [ ] Ensemble methods implemented

### Week 4: Final Push
**Goal: #1**
- [ ] All challenges optimized
- [ ] Rapid response system (hourly updates)
- [ ] Competitor analysis
- [ ] System reliability: 99.9%
- [ ] Advanced features deployed

---

## 🔥 Advanced Optimizations (Week 2+)

### When You're in Top 10:

- [ ] **Ensemble Models**
  - Combine multiple model versions
  - Weighted predictions based on recent performance
  
- [ ] **Online Learning**
  - Update models with recent data daily
  - Fine-tune last layers only (fast)

- [ ] **Feature Engineering**
  - Add market regime detection
  - Include cross-asset correlations
  - Custom technical indicators

- [ ] **Latency Optimization**
  - Pre-compute features
  - Cache models in memory
  - Parallel predictions

### When You're in Top 3:

- [ ] **Ultra-Fast Updates**
  - Hourly model updates for high-weight challenges
  - Real-time feature computation
  
- [ ] **Competitor Monitoring**
  - Track top 5 miners daily
  - Analyze their strategies
  - Adapt quickly to changes

- [ ] **Perfect Reliability**
  - Redundant systems
  - Automatic failover
  - 100% uptime guarantee

---

## 🆘 Troubleshooting Checklist

### Training Issues:

**Problem:** Training stuck or slow
- [ ] Check GPU usage: `nvidia-smi`
- [ ] Check logs: `tail -100 logs/training/training_current.log`
- [ ] Restart if needed: `./run_training.sh --stop && ./run_training.sh`

**Problem:** Training completed but models missing
- [ ] Check: `ls -l models/tuned/`
- [ ] Review errors: `grep -i error logs/training/training_current.log`
- [ ] Retrain specific challenge: `./run_training.sh --challenge CHALLENGE_NAME`

### Mining Issues:

**Problem:** Miner not submitting predictions
- [ ] Check miner is running: `ps aux | grep miner`
- [ ] Check logs for errors: `tail -50 logs/miner.log`
- [ ] Verify wallet/hotkey: Check config
- [ ] Restart miner

**Problem:** Low salience scores
- [ ] Check prediction quality: Review logs
- [ ] Compare to baseline: Historical performance
- [ ] Retrain weak challenges: `./run_training.sh --challenge X`
- [ ] Update data: Fresh data collection

**Problem:** System crashes or stops
- [ ] Check system resources: `free -h`, `df -h`
- [ ] Check GPU: `nvidia-smi`
- [ ] Review crash logs: `dmesg | tail`
- [ ] Add monitoring alerts

---

## 📈 Success Metrics

### Track Daily:
- [ ] Miner uptime: ____%
- [ ] Predictions submitted: _____
- [ ] Average salience (all): _____
- [ ] Average salience (ETH-LBFGS): _____
- [ ] Errors today: _____

### Track Weekly:
- [ ] Leaderboard rank: #_____
- [ ] Rank change: _____ (up/down)
- [ ] Top challenge performance: _____
- [ ] Weakest challenge: _____
- [ ] Total rewards: _____

### Track Monthly:
- [ ] Overall rank improvement: _____
- [ ] Best challenge: _____ (salience: _____)
- [ ] Most improved challenge: _____
- [ ] System uptime: _____%
- [ ] Goal: #1 by month _____

---

## 🎯 Quick Reference Commands

```bash
# Training
./run_training.sh --status          # Check training status
./run_training.sh --stop            # Stop training
./run_training.sh                   # Start training (50 trials)
./run_training.sh --trials 100      # More thorough training

# Monitoring
tail -f logs/training/training_current.log   # Watch training
tail -f logs/miner.log                       # Watch mining
./run_training.sh --status                   # Quick status

# Mining
ps aux | grep miner                 # Check if running
pkill -f miner                      # Stop miner
python neurons/miner.py [args]      # Start miner

# System
nvidia-smi                          # Check GPU
df -h                               # Check disk space
free -h                             # Check memory
```

---

## 🏁 Current Status

**Training:**
- [X] Started successfully
- [X] Running with fixed code
- [X] 60% complete on challenge 1/11
- [ ] Completed (waiting...)

**Next Steps:**
1. Wait for training to complete (~8-10 hours)
2. Deploy models to miner
3. Monitor and optimize
4. Reach #1! 🏆

**You're on the right track! Keep going!** 💪

