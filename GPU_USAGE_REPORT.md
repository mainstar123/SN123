# GPU Usage Report
Generated: Wed Dec 10 20:18:49 UTC 2025

## Current GPU Status

**GPU Hardware:** NVIDIA GeForce RTX 3090 (24GB)

### Active GPU Processes

| PID | Process | GPU Memory | Status |
|-----|---------|------------|--------|
| 886796 | `python3 /home/ocean/ocean_predictoor_bot/pdr predictoor ...` | **4414 MiB** | RUNNING |
| 1013692 | `python3 /home/ocean/ocean_predictoor_bot/pdr multisim ...` | **510 MiB** | RUNNING |

**Total GPU Memory Used:** ~4924 MiB / 24576 MiB (20%)
**Current GPU Utilization:** 0% (idle at time of check)

---

## MANTIS Training Status

### Finding: MANTIS Training is NOT Currently Running

- **Training PID file:** `training_service.pid` contains PID 1460816, but this process is **NOT running**
- **Last training log activity:** `logs/training/training_20251210_011825.log` was last modified at **16:49:11** (about 3.5 hours ago)
- **Training progress when stopped:** Trial 42/50 for ETH-LBFGS challenge
- **Log ending:** Training appears to have stopped mid-trial (no completion message or error visible)

### Training Configuration
- **Script:** `scripts/training/train_all_challenges.py`
- **GPU Detection:** Yes - Log shows "Using GPU: /physical_device:GPU:0"
- **Mixed Precision:** Enabled
- **Challenges:** 11 total challenges configured

---

## Key Findings

1. **The 2 GPU processes are NOT from MANTIS training**
   - They are from `ocean_predictoor_bot` project
   - MANTIS training process appears to have stopped/crashed

2. **MANTIS training was configured for GPU but is not active**
   - Training logs show GPU was detected and being used
   - Process is no longer running

3. **GPU Availability**
   - ~20GB GPU memory still available (19,652 MiB free)
   - GPU is not being actively used by the ocean_predictoor processes (0% utilization)

---

## Recommendations

### To Check Why MANTIS Training Stopped:
```bash
# Check the end of the training log for errors
tail -100 logs/training/training_20251210_011825.log

# Check for any error logs
ls -lth logs/training/*.log
tail logs/training/training_service_error.log 2>/dev/null || echo "No error log found"
```

### To Restart MANTIS Training:
```bash
# Check training service status
./training_service.sh status

# If needed, restart training
./training_service.sh restart

# Or start training manually
python scripts/training/train_all_challenges.py
```

### To Monitor GPU Usage:
```bash
# Real-time GPU monitoring
watch -n 1 nvidia-smi

# Check which processes are using GPU
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv
```

---

## Summary

- **GPU Memory:** 20% used (4.9GB / 24GB) - Plenty available for MANTIS training
- **GPU Utilization:** Currently idle (0%)
- **MANTIS Training:** Not running (stopped ~3+ hours ago)
- **Other GPU Processes:** 2 processes from ocean_predictoor_bot (can coexist with MANTIS if needed)

MANTIS training should be able to run alongside the existing ocean_predictoor processes if restarted, as there's plenty of GPU memory available.

