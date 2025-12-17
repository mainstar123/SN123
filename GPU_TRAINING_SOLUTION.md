# 🚀 GPU Training Setup - Complete Solution

## 🔍 Current Situation

**Problem:** Multiple processes competing for GPU access

```
✓ GPU Available: RTX 4090 (24GB)
❌ Blocked by: predictoor_bot process (PID 1483526, using 528MB)
⚠️  Current training: Running on CPU only (slow!)
```

---

## ✅ SOLUTION: Clear GPU & Start Fresh

### Step 1: Check What's Using GPU

```bash
nvidia-smi
```

**Look for:** Processes in the bottom table

### Step 2: Stop GPU-Blocking Processes

```bash
# Check what process is using GPU
nvidia-smi | grep python

# If you see predictoor_bot or other processes:
# Option A: Stop them if not needed
kill <PID_FROM_NVIDIA_SMI>

# Option B: If they're important, this training can share GPU
# (RTX 4090 has 24GB, plenty for multiple processes)
```

### Step 3: Stop Any Existing Training

```bash
cd /home/ocean/Nereus/SN123

# Stop all tune_all_challenges processes
pkill -f tune_all_challenges.py

# Verify stopped
ps aux | grep tune_all_challenges | grep -v grep
# Should show nothing
```

### Step 4: Clear GPU Memory

```bash
# If GPU still shows processes but they're dead:
sudo nvidia-smi --gpu-reset

# Or just wait 10 seconds for memory to clear
sleep 10 && nvidia-smi
```

### Step 5: Start Training with GPU

```bash
cd /home/ocean/Nereus/SN123
source venv/bin/activate

# Start training
nohup python scripts/training/tune_all_challenges.py \
    --data-dir data \
    --tuning-dir models/tuned \
    --epochs 100 \
    --batch-size 128 \
    > logs/training/training_$(date +%Y%m%d_%H%M%S).log 2>&1 &

# Save PID
echo $! > logs/training/tuning.pid
echo "Training PID: $(cat logs/training/tuning.pid)"
```

### Step 6: Verify GPU Usage (CRITICAL!)

Wait 2-3 minutes for initialization, then:

```bash
# Check GPU usage
nvidia-smi

# Should see your python process using GPU
# Example:
# |    0   N/A  N/A      1234567      C   python                   2000MiB |
```

**Expected GPU memory usage:** 1000-4000 MB

### Step 7: Monitor Training Progress

```bash
# Watch log
tail -f logs/training/training_*.log

# After 5-10 minutes, you should see:
# ✓ GPU detected
# ✓ Training Challenge: ETH-LBFGS
# ✓ Model will use GPU: /physical_device:GPU:0
# ✓ Epoch 1/100 ...
```

---

## 🎯 QUICK START (All-in-One)

```bash
cd /home/ocean/Nereus/SN123
source venv/bin/activate

# Stop conflicting processes
pkill -f tune_all_challenges.py
sleep 2

# Start fresh training
nohup python scripts/training/tune_all_challenges.py \
    --data-dir data \
    --tuning-dir models/tuned \
    > logs/training/training_$(date +%Y%m%d_%H%M%S).log 2>&1 &

echo $! > logs/training/tuning.pid

# Wait for initialization
echo "Waiting for GPU initialization..."
sleep 60

# Check GPU usage
nvidia-smi | grep python

# Watch progress
tail -f $(ls -t logs/training/training_*.log | head -1)
```

---

## 📊 How to Verify GPU is Working

### ✅ GOOD Signs (GPU Working)

```bash
# 1. nvidia-smi shows your process
nvidia-smi | grep tune_all_challenges
# Should show: python ... tune_all_challenges.py ... 1000-4000MiB

# 2. Log shows GPU detection
grep "GPU" logs/training/training_*.log
# Should show: "Model will use GPU: /physical_device:GPU:0"

# 3. High GPU utilization
nvidia-smi
# GPU-Util should show 80-100% during training

# 4. Fast training
# Each epoch should take ~30-60 seconds (not 3-5 minutes)
```

### ❌ BAD Signs (CPU Only)

```bash
# 1. nvidia-smi shows no process or very low memory (<100MB)
# 2. Log doesn't mention GPU
# 3. GPU-Util stays at 0%
# 4. Training very slow (5+ minutes per epoch)
```

---

## ⏱️ Expected Timeline with GPU

| Phase | Time | What's Happening |
|-------|------|------------------|
| Initialization | 2-5 min | Loading libraries, detecting GPU |
| First challenge | 1-2 hrs | ETH-LBFGS (7 configs × 10-15 min each) |
| All 11 challenges | **6-10 hrs** | Much faster than 18-24 hrs on CPU! |

**With GPU: 6-10 hours total ⚡**  
**Without GPU: 18-24 hours total 🐌**

---

## 🔧 Troubleshooting

### Problem: GPU Not Detected

```bash
# Check CUDA
python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
# Should show: [PhysicalDevice(name='/physical_device:GPU:0', ...)]
```

### Problem: Out of GPU Memory

```bash
# Reduce batch size
nohup python scripts/training/tune_all_challenges.py \
    --data-dir data \
    --tuning-dir models/tuned \
    --batch-size 64 \
    > logs/training/training_$(date +%Y%m%d_%H%M%S).log 2>&1 &
```

### Problem: Other Process Won't Stop

```bash
# Force kill
sudo kill -9 <PID>

# Or reset GPU
sudo nvidia-smi --gpu-reset
```

---

## 🎯 DECISION TREE

```
Is nvidia-smi working?
├─ NO → Install NVIDIA drivers
└─ YES → Continue

Is GPU free (no processes)?
├─ NO → Kill blocking processes or reduce batch size
└─ YES → Continue

Training started?
├─ NO → Run start command above
└─ YES → Continue

After 5 minutes, GPU showing your process?
├─ NO → Check logs for errors, restart
└─ YES → ✅ SUCCESS! Let it run 6-10 hours
```

---

## ✅ SUCCESS CHECKLIST

```
□ nvidia-smi shows your Python process
□ GPU memory usage: 1000-4000 MB
□ GPU utilization: 80-100% during epochs
□ Log shows: "Model will use GPU"
□ Training speed: ~30-60 sec/epoch
□ No errors in log
```

**All checked?** → You're good! Let it run 6-10 hours. 🚀

---

## 📞 FINAL NOTES

1. **RTX 4090 is POWERFUL:** Can handle multiple processes
2. **Don't worry about other processes:** Unless they use >20GB
3. **Batch size:** Start with 128, reduce to 64 if OOM
4. **Monitor first 30 minutes:** Make sure GPU kicks in
5. **Then leave it:** Will complete in 6-10 hours

---

**Quick command to start NOW:**

```bash
cd /home/ocean/Nereus/SN123 && source venv/bin/activate && pkill -f tune_all_challenges.py && sleep 2 && nohup python scripts/training/tune_all_challenges.py --data-dir data --tuning-dir models/tuned > logs/training/training_$(date +%Y%m%d_%H%M%S).log 2>&1 & echo $! > logs/training/tuning.pid && sleep 60 && nvidia-smi
```

**Then monitor:**

```bash
tail -f $(ls -t logs/training/training_*.log | head -1)
```

🎯 **You're ready to go!**

