# 🔧 Training Troubleshooting & Complete Solution Guide

**Your Complete Guide to Fixing Any Training Issues**

---

## 🎯 Your Current Situation

You're seeing:
```
Best trial: 20. Best value: 1.42448:  60%|██████    | 30/50 [3:09:08<2:26:21, 439.07s/it]
```

And it appears "stuck" at 60%.

---

## ✅ **GOOD NEWS: This is NORMAL!**

### What's Actually Happening:

**The progress bar shows 60% because:**
1. ✅ 30 trials have **completed**
2. 🔄 Trial 31 is **currently running** (you see "Training XGBoost")
3. ⏳ Progress bar **only updates when a trial finishes**
4. 📊 Each trial takes 5-10 minutes

**You're seeing XGBoost training:**
```
Training XGBoost on LSTM embeddings...
Class distribution: {0: 9201, 1: 9191}
Prediction distribution: Class 0: 9345 (50.8%), Class 1: 9047 (49.2%)
```

This means **training IS working** - it's just in the middle of Trial 31!

---

## 🔍 **How to Verify Training is Actually Working**

### Test 1: Check if Process is Active

```bash
cd /home/ocean/MANTIS

# Check CPU usage (should be high)
ps aux | grep train_all_challenges.py | grep -v grep

# Look for high CPU % (like 500-700%)
# If you see this, training IS running
```

**Expected Output:**
```
ocean  3804567  612  5.0  [...]  python scripts/training/train_all_challenges.py
                ^^^
           High CPU = Working!
```

### Test 2: Check if Log File is Growing

```bash
# Check log file size
ls -lh logs/training/training_current.log

# Wait 2 minutes, check again
sleep 120
ls -lh logs/training/training_current.log

# If size increased, training is working!
```

### Test 3: Watch for New Trial Completions

```bash
# Watch for new trials (wait 5-10 minutes)
tail -f logs/training/training_current.log | grep "Trial [0-9]*:"

# You should see Trial 31, 32, 33... appearing
# Each trial takes 5-10 minutes
```

### Test 4: Check GPU Usage

```bash
# Is GPU being used?
nvidia-smi

# Look for python process using GPU
# GPU Utilization should be 40-100%
```

---

## 📊 **Understanding Trial Progress**

### Why Progress Seems Slow:

Each trial goes through these steps:
```
Trial 31 starts
├─ 1. Prepare data with new hyperparameters (1-2 min)
├─ 2. Train LSTM on GPU (3-5 min)
│    └─ Multiple epochs with early stopping
├─ 3. Extract embeddings (30 sec)
├─ 4. Train XGBoost on CPU (1-2 min) ← YOU ARE HERE
├─ 5. Evaluate performance (10 sec)
└─ Trial 31 completes → Progress updates to 62%!
```

**Total time per trial:** 5-10 minutes

**Why you see the same progress:**
- Trial 30 finished → 60%
- Trial 31 running (currently training XGBoost)
- When Trial 31 finishes → 62%
- When Trial 32 finishes → 64%
- etc.

---

## 🚨 **When Training is ACTUALLY Stuck**

### Signs of Real Problems:

1. **No CPU usage** (below 100%)
2. **Log file not growing** for 30+ minutes
3. **No GPU activity** for 30+ minutes
4. **Same trial number** for 1+ hour
5. **Error messages** in log

### How to Check:

```bash
# 1. Check process status
ps aux | grep train_all_challenges.py | grep -v grep

# If OUTPUT: Process found with high CPU → WORKING ✅
# If NO OUTPUT: Process died → STUCK ❌

# 2. Check for errors
tail -100 logs/training/training_current.log | grep -i "error\|exception\|traceback"

# If OUTPUT: Errors found → PROBLEM ❌
# If NO OUTPUT: No errors → WORKING ✅

# 3. Monitor log growth
watch -n 60 "ls -lh logs/training/training_current.log"
# Watch for 5 minutes
# If size increases → WORKING ✅
# If size constant → STUCK ❌
```

---

## 🔧 **Solutions for Different Scenarios**

### Scenario 1: Training is Actually Working (Most Likely)

**Symptoms:**
- ✅ High CPU usage
- ✅ Log file growing
- ✅ GPU being used
- ✅ Just looks slow

**Solution:** **WAIT! Be patient.**

```bash
# Each trial takes 5-10 minutes
# 20 trials remaining × 7 minutes = 140 minutes = 2.3 hours
# Just let it run!

# Check back in 30 minutes
sleep 1800
./run_training.sh --status
```

**Why it seems slow:**
- LBFGS challenges have 17 dimensions (complex)
- Each trial trains full LSTM + XGBoost
- Hyperparameter optimization is thorough
- This investment pays off with better models!

---

### Scenario 2: Process Died (Crashed)

**Symptoms:**
- ❌ No process running
- ❌ Log file not growing
- ❌ No CPU/GPU usage

**Solution: Restart Training**

```bash
cd /home/ocean/MANTIS

# Verify process is dead
ps aux | grep train_all_challenges.py | grep -v grep
# Should show nothing

# Restart training
./run_training.sh

# Monitor restart
tail -f logs/training/training_current.log
```

**Note:** Training will resume from the current challenge (ETH-LBFGS) but will start trials from the beginning. Your previous best parameters are saved.

---

### Scenario 3: Out of Memory

**Symptoms:**
- ❌ Process killed
- ❌ "Killed" in logs
- ❌ `dmesg` shows OOM

**Check:**
```bash
# Check memory
free -h

# Check recent kills
dmesg | grep -i "killed\|oom" | tail -20
```

**Solution: Reduce Batch Size or Features**

```bash
# Stop current training
./run_training.sh --stop

# Option A: Reduce trials (faster, less thorough)
./run_training.sh --trials 25

# Option B: Edit train_all_challenges.py to use smaller batch sizes
# (Advanced - not recommended unless necessary)
```

---

### Scenario 4: GPU Out of Memory

**Symptoms:**
- ❌ CUDA out of memory errors
- ❌ GPU memory at 100%

**Check:**
```bash
nvidia-smi
# Look at memory usage
```

**Solution: Reduce Model Size**

```bash
# Stop training
./run_training.sh --stop

# The hyperparameter search will automatically find
# configurations that fit in GPU memory
# Just restart with fewer trials if needed
./run_training.sh --trials 25
```

---

### Scenario 5: Disk Full

**Symptoms:**
- ❌ "No space left on device"
- ❌ Training stops

**Check:**
```bash
df -h
# Look at / or /home usage
```

**Solution: Free Up Space**

```bash
# Clean old logs
cd /home/ocean/MANTIS
rm -f logs/training/training_202412*.log  # Keep only current

# Clean old models
rm -rf models/checkpoints/*  # Keep only tuned models

# Check Docker (if installed)
docker system prune -a

# Check available space
df -h
```

---

### Scenario 6: Network Issues (Data Download)

**Symptoms:**
- ❌ Hangs during data loading
- ❌ Network timeout errors

**Solution: Verify Data Exists**

```bash
# Check data files exist
ls -lh data/ETH_1h.csv

# If missing, data files should already exist
# Training uses local data, not network downloads
```

---

## 🎯 **Step-by-Step Diagnostic Protocol**

Follow these steps in order:

### Step 1: Is Training Running? (30 seconds)

```bash
cd /home/ocean/MANTIS
ps aux | grep train_all_challenges.py | grep -v grep
```

**If YES:** Go to Step 2  
**If NO:** Training died → See "Scenario 2: Restart Training"

### Step 2: Is It Making Progress? (2 minutes)

```bash
# Note log file size
ls -lh logs/training/training_current.log

# Wait 2 minutes
sleep 120

# Check again
ls -lh logs/training/training_current.log
```

**If size increased:** Training is working! → Be patient (Scenario 1)  
**If size same:** Go to Step 3

### Step 3: Check for Errors (10 seconds)

```bash
tail -100 logs/training/training_current.log | grep -i "error\|exception\|killed"
```

**If errors found:** Check specific error type (OOM, GPU, disk)  
**If no errors:** Go to Step 4

### Step 4: Check Resources (10 seconds)

```bash
# Memory
free -h | grep Mem

# Disk
df -h | grep -E "/$|/home$"

# GPU
nvidia-smi | grep -A 2 "python"
```

**If low memory (<2GB):** See Scenario 3  
**If low disk (<10GB):** See Scenario 5  
**If resources OK:** Training is likely just slow → Wait 30 min

### Step 5: Wait and Verify (30 minutes)

```bash
# Set reminder for 30 minutes
echo "Check training progress" | at now + 30 minutes

# Or just wait 30 min and check
sleep 1800
./run_training.sh --status
```

**If progress increased:** All good! Keep waiting  
**If still stuck:** Restart training (Scenario 2)

---

## 📋 **Your Specific Situation - Action Plan**

Based on your log showing XGBoost training, here's what to do:

### Immediate Actions (Next 5 Minutes)

```bash
cd /home/ocean/MANTIS

# 1. Verify process is running
ps aux | grep train_all_challenges.py | grep -v grep
# Expected: Should show process with high CPU

# 2. Check log is growing
ls -lh logs/training/training_current.log
# Note the size

# Wait 2 minutes
sleep 120

# Check again
ls -lh logs/training/training_current.log
# If size increased by ~5-10KB, it's working!

# 3. Watch for trial completion
timeout 600 tail -f logs/training/training_current.log | grep -E "Trial [0-9]+:"
# This will show when Trial 31 completes (wait up to 10 minutes)
```

### If Process is Running and Log Growing:

**✅ Training is WORKING!** Just be patient.

```bash
# Set up periodic checks
cat > check_training.sh << 'EOF'
#!/bin/bash
echo "=== Training Status Check ==="
echo "Time: $(date)"
./run_training.sh --status
echo ""
echo "Last 5 trials:"
grep "Trial [0-9]*:" logs/training/training_current.log | tail -5
echo ""
EOF

chmod +x check_training.sh

# Run this every 30 minutes
./check_training.sh
```

### If Process is NOT Running:

**❌ Training crashed.** Restart it:

```bash
cd /home/ocean/MANTIS

# Check what happened
tail -100 logs/training/training_current.log | grep -i "error\|killed\|exception"

# Restart training
./run_training.sh
```

---

## ⚡ **Quick Fix: Force Restart Training**

If you just want to restart fresh (nuclear option):

```bash
cd /home/ocean/MANTIS

# Stop everything
./run_training.sh --stop
pkill -9 -f train_all_challenges.py

# Clean up
rm -f logs/training/training.pid

# Start fresh
./run_training.sh

# Monitor
tail -f logs/training/training_current.log
```

**Warning:** This will restart from Trial 0 of Challenge 1. Only do this if training is truly stuck.

---

## 🎯 **Expected Timeline Reference**

To set expectations:

### ETH-LBFGS (Challenge 1/11) - Current Challenge

```
Total trials: 50
Trials per hour: ~6-8 (7-10 min each)
Current: Trial 30/50 (60%)
Remaining: 20 trials
Expected time: 2.5-3.5 hours
```

### All 11 Challenges

```
Challenge 1 (ETH-LBFGS):        2.5-3.5 hours remaining
Challenge 2 (BTC-LBFGS-6H):     4-6 hours (17 dim, 50 trials)
Challenge 3 (ETH-HITFIRST):     3-4 hours (3 dim, 50 trials)
Challenges 4-11 (Binary):       1-2 hours each × 8 = 8-16 hours

Total remaining: ~18-30 hours from now
```

**This is NORMAL for thorough hyperparameter optimization!**

---

## 📊 **How to Monitor Effectively**

### Good Monitoring (Low Effort)

```bash
# Check every 2-3 hours
cd /home/ocean/MANTIS
./run_training.sh --status

# Or set up cron job
crontab -e
# Add: 0 */2 * * * cd /home/ocean/MANTIS && ./run_training.sh --status >> /tmp/training_checks.log 2>&1
```

### Detailed Monitoring (If You Want)

```bash
# Watch progress in detail
watch -n 300 '
  echo "=== Training Monitor ==="
  echo "Time: $(date)"
  echo ""
  ps aux | grep train_all | grep -v grep | awk "{print \"CPU: \"\$3\"% Memory: \"\$4\"%\"}"
  echo ""
  tail -5 logs/training/training_current.log | grep -E "Trial|Best trial"
  echo ""
  nvidia-smi | grep python
'
# Updates every 5 minutes
```

---

## 🔍 **Advanced Diagnostics**

### Check Training Health Score

```bash
cd /home/ocean/MANTIS

# Run health check
cat > health_check.sh << 'EOF'
#!/bin/bash
echo "=== TRAINING HEALTH CHECK ==="
echo ""

# Process running?
if ps aux | grep -v grep | grep train_all_challenges.py > /dev/null; then
    echo "✅ Process: RUNNING"
    CPU=$(ps aux | grep -v grep | grep train_all_challenges.py | awk '{print $3}')
    MEM=$(ps aux | grep -v grep | grep train_all_challenges.py | awk '{print $4}')
    echo "   CPU: ${CPU}%"
    echo "   Memory: ${MEM}%"
else
    echo "❌ Process: NOT RUNNING"
    exit 1
fi

# Log growing?
SIZE_BEFORE=$(stat -f%z logs/training/training_current.log 2>/dev/null || stat -c%s logs/training/training_current.log)
sleep 10
SIZE_AFTER=$(stat -f%z logs/training/training_current.log 2>/dev/null || stat -c%s logs/training/training_current.log)

if [ "$SIZE_AFTER" -gt "$SIZE_BEFORE" ]; then
    GROWTH=$((SIZE_AFTER - SIZE_BEFORE))
    echo "✅ Log: GROWING (+${GROWTH} bytes in 10s)"
else
    echo "⚠️  Log: NOT GROWING (might be between trials)"
fi

# GPU active?
if nvidia-smi 2>/dev/null | grep python > /dev/null; then
    echo "✅ GPU: ACTIVE"
    nvidia-smi | grep python | awk '{print "   Usage: "$13" "$14" "$15}'
else
    echo "⚠️  GPU: IDLE (might be in XGBoost phase)"
fi

# Recent errors?
ERROR_COUNT=$(tail -100 logs/training/training_current.log | grep -i error | wc -l)
if [ "$ERROR_COUNT" -eq 0 ]; then
    echo "✅ Errors: NONE"
else
    echo "⚠️  Errors: $ERROR_COUNT found in last 100 lines"
fi

# Disk space?
DISK_FREE=$(df -h / | tail -1 | awk '{print $4}')
echo "✅ Disk Free: $DISK_FREE"

# Memory free?
MEM_FREE=$(free -h | grep Mem | awk '{print $4}')
echo "✅ Memory Free: $MEM_FREE"

echo ""
echo "=== CONCLUSION ==="
if ps aux | grep -v grep | grep train_all_challenges.py > /dev/null; then
    echo "Training appears to be HEALTHY and RUNNING"
    echo "Be patient - each trial takes 5-10 minutes"
else
    echo "Training is NOT RUNNING - restart required"
fi
EOF

chmod +x health_check.sh
./health_check.sh
```

---

## 🎯 **Final Recommendations**

### For Your Current Situation:

1. **Run the health check** (above script)
2. **If healthy:** Just wait patiently
3. **Check again in 30 minutes**
4. **If still at 60%:** Run diagnostic protocol
5. **If diagnostic fails:** Restart training

### Best Practice Going Forward:

```bash
# Morning check (30 seconds)
./run_training.sh --status

# If you want more detail
./health_check.sh

# If concerned
tail -20 logs/training/training_current.log
```

### When to Restart:

Only restart if:
- ❌ Process is dead
- ❌ No progress for 1+ hour
- ❌ Errors in logs
- ❌ Out of resources

**Don't restart just because progress looks slow!** Hyperparameter optimization takes time.

---

## 📞 **Quick Decision Tree**

```
Is process running?
├─ NO → Restart: ./run_training.sh
└─ YES → Is log growing?
    ├─ NO → Wait 30 min, check again
    │   └─ Still no? → Restart
    └─ YES → Training is working!
        └─ Be patient → Check in 2-3 hours
```

---

## ✅ **Summary: What to Do Right Now**

```bash
# 1. Quick check (30 seconds)
cd /home/ocean/MANTIS
ps aux | grep train_all_challenges.py | grep -v grep

# 2. If running, verify progress (2 minutes)
ls -lh logs/training/training_current.log
sleep 120
ls -lh logs/training/training_current.log
# Did size increase? → It's working!

# 3. If working, just wait
echo "Training is working. Check back in 2-3 hours."

# 4. If not working, restart
./run_training.sh --stop
./run_training.sh
```

**Most likely:** Training is working fine, just slow. Be patient! ⏳

---

**Your training is probably fine. Each trial takes 5-10 minutes. The progress bar only updates when a trial completes, not while it's running. Give it 30 minutes and check again!** 💪

