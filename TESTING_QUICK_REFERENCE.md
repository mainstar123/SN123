# 🎯 Local Testing - Quick Reference Guide

**ONE COMMAND TO RUN EVERYTHING:**

```bash
cd /home/ocean/Nereus/SN123
./test_locally.sh
```

**Time:** 15-30 minutes  
**Result:** Know if you're ready for mainnet

---

## What Happens (Stage by Stage)

### Stage 1: Verify Training (30 seconds)
**What it does:** Checks if all 11 models exist  
**Expected output:**
```
Models found: 11
✅ All 11 models found
✓ ETH-LBFGS: Complete
✓ BTC-LBFGS-6H: Complete
...
✅ All models have complete files
```

### Stage 2: Load Models (30 seconds)
**What it does:** Tests if models can load into memory  
**Expected output:**
```
✓ ETH-LBFGS: Loaded successfully
✓ BTC-LBFGS-6H: Loaded successfully
...
Results: 11 passed, 0 failed
✅ ALL MODELS LOADED SUCCESSFULLY
```

### Stage 3: Backtest (5-10 minutes)
**What it does:** Tests models on last 30 days of data  
**Expected output for each challenge:**
```
============================================================
Backtesting: ETH-1H-BINARY
============================================================
✓ Model loaded
Test period: 2024-11-12 to 2024-12-12
Test samples: 720

📊 Results:
  Accuracy: 0.6714 (67.14%)
  AUC: 0.7023
  Estimated Salience: 0.6856
```

**Summary at end:**
```
📊 BACKTEST SUMMARY

Binary Avg Accuracy: 0.6687 (66.87%)
  ✓ GOOD - Should perform well on mainnet

Overall Est. Salience: 1.4521
  ✓ GOOD - Should rank well

✅ READY FOR MAINNET DEPLOYMENT
```

### Stage 4: Speed Test (30 seconds)
**What it does:** Checks prediction latency  
**Expected output:**
```
✓ Model loaded: 0.234s
✓ Data loaded: 0.089s
✓ Features prepared: 0.456s
✓ Prediction generated: 0.123s

Total time: 0.902s

✅ EXCELLENT - Very fast, ready for real-time
```

---

## Interpreting Results

### Binary Challenges (Accuracy)

| Result | Accuracy | Meaning | Action |
|--------|----------|---------|--------|
| ✅ Excellent | ≥70% | Top tier | Deploy now! |
| ✓ Good | 65-70% | Solid | Deploy confidently |
| ⚠️ Fair | 60-65% | Acceptable | Deploy or improve |
| ❌ Poor | <60% | Weak | Retrain first |

### Overall Salience

| Result | Salience | Expected Rank | Action |
|--------|----------|---------------|--------|
| ✅ Excellent | ≥2.0 | Top 5 | Deploy immediately |
| ✓ Good | 1.5-2.0 | Top 10-15 | Deploy confidently |
| ⚠️ Fair | 1.0-1.5 | Top 20-30 | Deploy or improve |
| ❌ Poor | <1.0 | >30 | Improve first |

### Speed

| Result | Time | Meaning | Action |
|--------|------|---------|--------|
| ✅ Excellent | <3s | Very fast | Perfect |
| ✓ Good | 3-5s | Fast enough | OK for mining |
| ⚠️ Acceptable | 5-10s | Slower | Monitor |
| ❌ Too Slow | >10s | Too slow | Optimize |

---

## Decision Tree

```
START
  ↓
All models exist? 
  YES → Continue
  NO → Fix training
  ↓
All models load?
  YES → Continue
  NO → Retrain broken models
  ↓
Binary Accuracy ≥ 65%?
  YES → Continue
  NO → Check below for <65%
  ↓
Salience ≥ 1.5?
  YES → Continue
  NO → Check below for <1.5
  ↓
Speed < 5s?
  YES → Continue
  NO → Acceptable if <10s
  ↓
✅ READY FOR MAINNET!
  → Proceed to Phase 2
```

### If Binary Accuracy < 65%

```
IF 60-65%:
  ⚠️ ACCEPTABLE
  → Can deploy OR improve weak challenges
  
IF 55-60%:
  ⚠️ NEEDS WORK
  → Recommend improving 2-3 weakest challenges
  
IF <55%:
  ❌ NOT READY
  → Retrain before mainnet
```

### If Salience < 1.5

```
IF 1.0-1.5:
  ⚠️ ACCEPTABLE
  → Will rank, but room for improvement
  → Deploy and optimize later
  
IF 0.5-1.0:
  ⚠️ WEAK
  → Recommend retraining high-weight challenges
  → ETH-LBFGS (weight 3.5) - most important
  → BTC-LBFGS-6H (weight 2.875) - second most
  
IF <0.5:
  ❌ NOT READY
  → Comprehensive retraining needed
```

---

## Quick Commands

### Run Full Test Suite
```bash
./test_locally.sh
```

### View Last Results
```bash
cat results/backtest_results_latest.txt
```

### Check Specific Challenge
```bash
grep "CHALLENGE-NAME" results/backtest_results_latest.txt
```

### Find Weakest Challenges
```bash
grep "Est. Salience" results/backtest_results_latest.txt | sort -k3 -n | head -3
```

### Retrain Weak Challenge
```bash
./run_training.sh --trials 100 --challenge CHALLENGE_NAME
```

### Re-test After Improvements
```bash
./test_locally.sh
```

---

## Common Scenarios & Actions

### Scenario 1: Everything Excellent (✅)
**Results:**
- Binary: 70%+
- Salience: 2.0+
- Speed: <3s

**Action:**
```bash
# Save baseline
cp results/backtest_results_latest.txt baseline_$(date +%Y%m%d).txt

# Deploy to mainnet
echo "Ready for Phase 2!"
# Follow COMPLETE_ROADMAP_TO_FIRST_PLACE.md Phase 2
```

### Scenario 2: Good but Some Weak Challenges (✓)
**Results:**
- Binary: 65-70%
- Salience: 1.5-2.0
- But 2-3 challenges < 60% accuracy

**Action:**
```bash
# Find weak ones
grep "Accuracy:" results/backtest_results_latest.txt | sort -k2 -n | head -3

# Retrain weakest 2
./run_training.sh --trials 100 --challenge WEAK_CHALLENGE_1
./run_training.sh --trials 100 --challenge WEAK_CHALLENGE_2

# Re-test
./test_locally.sh
```

### Scenario 3: Fair Performance (⚠️)
**Results:**
- Binary: 60-65%
- Salience: 1.0-1.5

**Action (Option A - Deploy Now):**
```bash
# Deploy and improve later
echo "Acceptable for deployment"
# Follow Phase 2, plan improvements for Week 2
```

**Action (Option B - Improve First):**
```bash
# Retrain high-weight challenges for maximum impact
./run_training.sh --trials 150 --challenge ETH-LBFGS
./run_training.sh --trials 150 --challenge BTC-LBFGS-6H
./run_training.sh --trials 150 --challenge ETH-HITFIRST-100M

# Re-test
./test_locally.sh
```

### Scenario 4: Poor Performance (❌)
**Results:**
- Binary: <60%
- Salience: <1.0

**Action:**
```bash
# Full retraining with more thorough hyperparameter search
./run_training.sh --trials 200

# Or retrain all individually
for challenge in ETH-LBFGS BTC-LBFGS-6H ETH-HITFIRST-100M \
                 ETH-1H-BINARY EURUSD-1H-BINARY GBPUSD-1H-BINARY \
                 CADUSD-1H-BINARY NZDUSD-1H-BINARY CHFUSD-1H-BINARY \
                 XAUUSD-1H-BINARY XAGUSD-1H-BINARY; do
    ./run_training.sh --trials 150 --challenge $challenge
done

# Re-test
./test_locally.sh
```

---

## Understanding Salience Scores

### What is Salience?
Salience measures how much **unique, non-redundant information** your model provides to validators.

**High salience = Your model captures patterns others miss**

### How It's Calculated

**Binary Challenges (2D embeddings):**
1. Top 20 miners selected by AUC
2. Ensemble model built using all selected miners
3. Your salience = How much ensemble performance drops if your model is removed
4. Normalized across all miners

**LBFGS Challenges (17D embeddings):**
1. Classifier salience (50%): 5-bucket probability prediction
2. Q salience (50%): Opposite-move probabilities
3. Final = Average, normalized

### Key Factors for High Salience

1. **Be Accurate** (get selected in top 20)
   - Binary: AUC > 0.55
   - Better accuracy → Higher chance of selection

2. **Be Unique** (provide non-redundant info)
   - Diverse embeddings (not collapsed)
   - Predict both classes (not majority-only)
   - Capture different patterns than competitors

3. **Be Consistent** (reliable performance)
   - Stable predictions over time
   - No NaN/Inf values
   - Valid embeddings always

### Expected Salience by Challenge Weight

| Challenge | Weight | Expected Salience | Importance |
|-----------|--------|------------------|------------|
| ETH-LBFGS | 3.5 | 1.5-2.5 | ⭐⭐⭐ Critical |
| BTC-LBFGS-6H | 2.875 | 1.5-2.5 | ⭐⭐⭐ High |
| ETH-HITFIRST | 2.5 | 1.0-2.0 | ⭐⭐ Important |
| Binary (8x) | 1.0 each | 0.4-1.2 each | ⭐ Standard |

**Strategy:** Focus on high-weight challenges for maximum impact!

---

## Files Created by Testing

```
/home/ocean/Nereus/SN123/
├── results/
│   ├── backtest_results_20251212_120000.txt  # Timestamped results
│   ├── backtest_results_latest.txt            # Latest results (symlink)
│   └── baseline_20251212.txt                  # Your baseline for mainnet
```

---

## What to Do After Testing

### If Ready (✅ or ✓)

1. **Save your baseline**
   ```bash
   cp results/backtest_results_latest.txt baseline_$(date +%Y%m%d).txt
   ```

2. **Document expected performance**
   ```bash
   cat > expected_performance.txt << EOF
   Date: $(date)
   Binary Accuracy: [YOUR_RESULT]
   Overall Salience: [YOUR_RESULT]
   Expected Mainnet Rank: Top [10-20]
   EOF
   ```

3. **Proceed to mainnet**
   - Follow `COMPLETE_ROADMAP_TO_FIRST_PLACE.md` Phase 2
   - Deploy with confidence!

### If Not Ready (⚠️ or ❌)

1. **Identify weakest challenges**
   ```bash
   grep "Est. Salience" results/backtest_results_latest.txt | sort -k3 -n
   ```

2. **Create improvement plan**
   ```bash
   cat > improvement_plan.txt << EOF
   Weak challenges to retrain:
   1. [CHALLENGE_1] - Current salience: [X]
   2. [CHALLENGE_2] - Current salience: [Y]
   3. [CHALLENGE_3] - Current salience: [Z]
   
   Action: Retrain with 150+ trials
   Timeline: [ESTIMATE]
   Retest after: [DATE]
   EOF
   ```

3. **Execute improvements**
   ```bash
   # Retrain weak challenges
   for challenge in WEAK_1 WEAK_2 WEAK_3; do
       ./run_training.sh --trials 150 --challenge $challenge
   done
   
   # Re-test
   ./test_locally.sh
   ```

---

## Troubleshooting

### "No such file or directory: data/XXX.csv"
**Solution:** Update data files
```bash
python scripts/data_collection/update_all_data.py
```

### "Model not found at models/tuned/XXX"
**Solution:** That challenge wasn't trained
```bash
./run_training.sh --trials 100 --challenge XXX
```

### "Out of memory"
**Solution:** Reduce batch size or test one challenge at a time
```bash
python scripts/testing/backtest_models.py --challenge ETH-LBFGS
```

### Low accuracy across all challenges
**Solution:** Check data quality, review training logs
```bash
tail -100 logs/training/training_current.log
ls -lh data/*.csv  # Check data freshness
```

---

## Time Estimates

| Task | Duration |
|------|----------|
| Stage 1: Verify Training | 30 sec |
| Stage 2: Load Models | 30 sec |
| Stage 3: Backtest All | 5-10 min |
| Stage 4: Speed Test | 30 sec |
| **Total** | **~15 min** |
| Retrain one challenge | 1-2 hours |
| Retrain all challenges | 12-24 hours |

---

## Success Checklist

```
Before running tests:
□ Training completed (check logs)
□ All 11 models exist in models/tuned/
□ Data files are recent (< 7 days old)
□ Virtual environment activated
□ Enough disk space (>10GB free)

After tests complete:
□ All models loaded successfully
□ Binary accuracy ≥ 60%
□ Overall salience ≥ 1.0
□ Speed < 5 seconds
□ Results saved to results/
□ Decision made: Deploy OR Improve
□ Next steps documented

Ready for mainnet if:
□ Binary accuracy ≥ 65%
□ Overall salience ≥ 1.5
□ No critical errors
□ Confident in performance
```

---

**ONE COMMAND TO START:**
```bash
./test_locally.sh
```

**Result:** Complete understanding of your models' readiness for mainnet! 🚀

