# 🧪 Local Testing Before Mainnet Deployment

**Test Your Models Locally → Gain Confidence → Deploy to Mainnet**

**Timeline:** After Phase 1 completes, BEFORE Phase 2 mainnet deployment  
**Duration:** 1-3 days  
**Goal:** Verify models work well locally before risking mainnet

---

## ✅ **YES - You Can and SHOULD Test Locally First!**

### **Testing Strategy:**

```
Phase 1: Training Complete (✅ Done)
         ↓
Phase 1.5: LOCAL TESTING (← We add this!)
         ├─ Backtest on historical data
         ├─ Validate predictions
         ├─ Estimate salience scores
         ├─ Test all 11 challenges
         └─ Gain confidence
         ↓
Phase 2: Deploy to Mainnet (when confident)
```

---

## 🎯 **Why Local Testing First?**

### **Benefits:**

1. ✅ **No Risk** - Test without spending TAO or affecting mainnet performance
2. ✅ **Fast Iteration** - Fix issues quickly without waiting for validator feedback
3. ✅ **Confidence** - Know your models work before going live
4. ✅ **Baseline** - Establish expected performance levels
5. ✅ **Debug** - Find and fix problems in safe environment

### **What You Can Test Locally:**

- ✅ Model loading and inference
- ✅ Prediction generation for all challenges
- ✅ Prediction quality (accuracy, distribution)
- ✅ System performance (latency, memory)
- ✅ Estimated salience scores (backtesting)
- ✅ All 11 challenges working correctly

---

## 📋 **Phase 1.5: Local Testing (Before Mainnet)**

**Add this phase after training completes, before mainnet deployment**

---

## Step 1: Verify All Models Load Correctly

```bash
cd /home/ocean/MANTIS
source venv/bin/activate

# Test loading all models
python << 'EOF'
from scripts.training.model_architecture import VMDTMFGLSTMXGBoost
import os

challenges = [
    'ETH-LBFGS', 'BTC-LBFGS-6H', 'ETH-HITFIRST-100M',
    'ETH-1H-BINARY', 'EURUSD-1H-BINARY', 'GBPUSD-1H-BINARY',
    'CADUSD-1H-BINARY', 'NZDUSD-1H-BINARY', 'CHFUSD-1H-BINARY',
    'XAUUSD-1H-BINARY', 'XAGUSD-1H-BINARY'
]

print("=== Model Loading Test ===\n")
for challenge in challenges:
    model_path = f'models/tuned/{challenge}'
    if os.path.exists(model_path):
        try:
            model = VMDTMFGLSTMXGBoost.load(model_path)
            print(f"✓ {challenge}: Loaded successfully")
        except Exception as e:
            print(f"✗ {challenge}: FAILED - {str(e)[:50]}")
    else:
        print(f"✗ {challenge}: Model directory not found")

print("\n✅ Model loading test complete!")
EOF
```

**Expected Output:**
```
✓ ETH-LBFGS: Loaded successfully
✓ BTC-LBFGS-6H: Loaded successfully
✓ ETH-HITFIRST-100M: Loaded successfully
✓ ETH-1H-BINARY: Loaded successfully
...
✅ Model loading test complete!
```

---

## Step 2: Backtest on Historical Data

### **Create Backtesting Script:**

```python
# scripts/testing/backtest_models.py

import pandas as pd
import numpy as np
from pathlib import Path
from scripts.training.model_architecture import VMDTMFGLSTMXGBoost
from sklearn.metrics import accuracy_score, roc_auc_score

def backtest_challenge(challenge_name, data_file, model_path):
    """Backtest a single challenge on recent historical data"""
    
    print(f"\n{'='*60}")
    print(f"Backtesting: {challenge_name}")
    print(f"{'='*60}")
    
    # Load model
    model = VMDTMFGLSTMXGBoost.load(model_path)
    
    # Load recent data (last 30 days)
    df = pd.read_csv(data_file)
    df['datetime'] = pd.to_datetime(df['datetime'])
    
    # Use last 30 days for backtesting
    cutoff_date = df['datetime'].max() - pd.Timedelta(days=30)
    test_df = df[df['datetime'] > cutoff_date].copy()
    
    print(f"Test period: {test_df['datetime'].min()} to {test_df['datetime'].max()}")
    print(f"Test samples: {len(test_df)}")
    
    # Prepare data
    X, y, features = model.prepare_data(test_df)
    
    if len(X) == 0:
        print("✗ No test sequences generated")
        return None
    
    print(f"Test sequences: {len(X)}")
    print(f"Features: {X.shape[2]}")
    
    # Generate predictions
    embeddings = model.predict_embeddings(X)
    print(f"Embedding shape: {embeddings.shape}")
    
    # For binary challenges, calculate accuracy
    if 'BINARY' in challenge_name:
        predictions = model.predict_binary(X)
        actual = (y > 0).astype(int)
        
        accuracy = accuracy_score(actual, predictions)
        
        if len(np.unique(actual)) > 1:
            auc = roc_auc_score(actual, embeddings[:, 1] if embeddings.shape[1] >= 2 else embeddings[:, 0])
        else:
            auc = 0.5
        
        print(f"\n📊 Results:")
        print(f"  Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
        print(f"  AUC: {auc:.4f}")
        print(f"  Predictions: Class 0: {(predictions==0).sum()}, Class 1: {(predictions==1).sum()}")
        
        # Estimate salience (higher accuracy = higher salience)
        estimated_salience = (accuracy - 0.5) * 4  # Rough estimate
        print(f"  Estimated Salience: {estimated_salience:.4f}")
        
        return {
            'challenge': challenge_name,
            'accuracy': accuracy,
            'auc': auc,
            'estimated_salience': estimated_salience,
            'samples': len(X)
        }
    
    # For LBFGS challenges, check embedding distribution
    else:
        print(f"\n📊 Embedding Statistics:")
        print(f"  Mean: {embeddings.mean(axis=0)[:5]}")
        print(f"  Std: {embeddings.std(axis=0)[:5]}")
        print(f"  Min: {embeddings.min(axis=0)[:5]}")
        print(f"  Max: {embeddings.max(axis=0)[:5]}")
        
        # Check if embeddings are valid (not all zeros, reasonable range)
        is_valid = (
            embeddings.std() > 0.01 and
            not np.isnan(embeddings).any() and
            not np.isinf(embeddings).any()
        )
        
        print(f"  Valid embeddings: {'✓' if is_valid else '✗'}")
        
        # Rough salience estimate based on embedding variance
        estimated_salience = min(embeddings.std() * 2, 3.0)
        print(f"  Estimated Salience: {estimated_salience:.4f}")
        
        return {
            'challenge': challenge_name,
            'valid': is_valid,
            'embedding_std': embeddings.std(),
            'estimated_salience': estimated_salience,
            'samples': len(X)
        }

def run_all_backtests():
    """Run backtests for all challenges"""
    
    challenges = {
        'ETH-LBFGS': 'data/ETH_1h.csv',
        'BTC-LBFGS-6H': 'data/BTC_6h.csv',
        'ETH-HITFIRST-100M': 'data/ETH_1h.csv',
        'ETH-1H-BINARY': 'data/ETH_1h.csv',
        'EURUSD-1H-BINARY': 'data/EURUSD_1h.csv',
        'GBPUSD-1H-BINARY': 'data/GBPUSD_1h.csv',
        'CADUSD-1H-BINARY': 'data/CADUSD_1h.csv',
        'NZDUSD-1H-BINARY': 'data/NZDUSD_1h.csv',
        'CHFUSD-1H-BINARY': 'data/CHFUSD_1h.csv',
        'XAUUSD-1H-BINARY': 'data/XAUUSD_1h.csv',
        'XAGUSD-1H-BINARY': 'data/XAGUSD_1h.csv',
    }
    
    results = []
    
    for challenge, data_file in challenges.items():
        model_path = f'models/tuned/{challenge}'
        
        if not Path(model_path).exists():
            print(f"✗ {challenge}: Model not found")
            continue
        
        if not Path(data_file).exists():
            print(f"✗ {challenge}: Data file not found")
            continue
        
        try:
            result = backtest_challenge(challenge, data_file, model_path)
            if result:
                results.append(result)
        except Exception as e:
            print(f"✗ {challenge}: Error - {str(e)[:100]}")
    
    # Summary
    print(f"\n{'='*60}")
    print("📊 BACKTEST SUMMARY")
    print(f"{'='*60}\n")
    
    for result in results:
        print(f"{result['challenge']}:")
        if 'accuracy' in result:
            print(f"  Accuracy: {result['accuracy']:.4f}")
            print(f"  Estimated Salience: {result['estimated_salience']:.4f}")
        else:
            print(f"  Valid: {result.get('valid', 'N/A')}")
            print(f"  Estimated Salience: {result['estimated_salience']:.4f}")
        print()
    
    # Overall assessment
    print(f"{'='*60}")
    print("✅ READINESS ASSESSMENT")
    print(f"{'='*60}\n")
    
    binary_results = [r for r in results if 'accuracy' in r]
    if binary_results:
        avg_accuracy = np.mean([r['accuracy'] for r in binary_results])
        print(f"Binary Challenges Avg Accuracy: {avg_accuracy:.4f}")
        
        if avg_accuracy >= 0.70:
            print("✅ EXCELLENT - Ready for mainnet!")
        elif avg_accuracy >= 0.60:
            print("✓ GOOD - Should perform well on mainnet")
        elif avg_accuracy >= 0.55:
            print("⚠️ FAIR - May want to retrain weak challenges")
        else:
            print("❌ POOR - Recommend retraining before mainnet")
    
    avg_salience = np.mean([r['estimated_salience'] for r in results])
    print(f"\nEstimated Average Salience: {avg_salience:.4f}")
    
    if avg_salience >= 2.0:
        print("✅ EXCELLENT - Competitive for top positions!")
    elif avg_salience >= 1.5:
        print("✓ GOOD - Should rank well")
    elif avg_salience >= 1.0:
        print("⚠️ FAIR - May need optimization")
    else:
        print("❌ POOR - Recommend improvement before mainnet")
    
    return results

if __name__ == "__main__":
    results = run_all_backtests()
```

### **Run Backtesting:**

```bash
cd /home/ocean/MANTIS
source venv/bin/activate

# Run complete backtest
python scripts/testing/backtest_models.py > backtest_results.txt 2>&1

# View results
cat backtest_results.txt
```

---

## Step 3: Analyze Backtest Results

### **Interpretation Guide:**

```bash
# Create analysis script
cat > analyze_backtest.sh << 'EOF'
#!/bin/bash
echo "=== Backtest Analysis ==="
echo ""

echo "Binary Challenges:"
grep -A 3 "BINARY:" backtest_results.txt | grep "Accuracy\|Estimated Salience"

echo ""
echo "LBFGS Challenges:"
grep -A 3 "LBFGS:" backtest_results.txt | grep "Valid\|Estimated Salience"

echo ""
echo "Overall Assessment:"
grep -A 10 "READINESS ASSESSMENT" backtest_results.txt
EOF

chmod +x analyze_backtest.sh
./analyze_backtest.sh
```

### **Decision Criteria:**

**✅ READY FOR MAINNET if:**
- Binary challenges: Average accuracy ≥ 65%
- LBFGS challenges: Valid embeddings, estimated salience ≥ 1.5
- No critical errors
- All 11 challenges working

**⚠️ NEEDS IMPROVEMENT if:**
- Binary challenges: Average accuracy < 60%
- LBFGS challenges: Estimated salience < 1.0
- Some challenges failing

**❌ NOT READY if:**
- Binary challenges: Average accuracy < 55%
- Multiple challenges failing
- Critical errors

---

## Step 4: Test Prediction Pipeline

### **Create End-to-End Test:**

```bash
cd /home/ocean/MANTIS
source venv/bin/activate

# Test complete prediction pipeline
python << 'EOF'
import time
import pandas as pd
from scripts.training.model_architecture import VMDTMFGLSTMXGBoost

print("=== Prediction Pipeline Test ===\n")

# Test ETH-LBFGS (most important)
challenge = 'ETH-LBFGS'
print(f"Testing {challenge}...")

start = time.time()

# 1. Load model
model = VMDTMFGLSTMXGBoost.load(f'models/tuned/{challenge}')
load_time = time.time() - start
print(f"✓ Model loaded: {load_time:.3f}s")

# 2. Load recent data
df = pd.read_csv('data/ETH_1h.csv').tail(100)
data_time = time.time() - start - load_time
print(f"✓ Data loaded: {data_time:.3f}s")

# 3. Prepare features
X, y, features = model.prepare_data(df)
prep_time = time.time() - start - load_time - data_time
print(f"✓ Features prepared: {prep_time:.3f}s")

# 4. Generate prediction
pred = model.predict_embeddings(X[-1:])
pred_time = time.time() - start - load_time - data_time - prep_time
print(f"✓ Prediction generated: {pred_time:.3f}s")

total_time = time.time() - start
print(f"\nTotal time: {total_time:.3f}s")

if total_time < 5.0:
    print("✅ EXCELLENT - Fast enough for real-time predictions")
elif total_time < 10.0:
    print("✓ GOOD - Acceptable latency")
else:
    print("⚠️ SLOW - May want to optimize")

print(f"\nPrediction shape: {pred.shape}")
print(f"Prediction (first 5): {pred[0][:5]}")
print("\n✅ Pipeline test complete!")
EOF
```

---

## Step 5: Test System Resources

```bash
# Monitor resource usage during predictions
echo "=== Resource Usage Test ==="

# Run predictions while monitoring
python << 'EOF' &
import time
from scripts.training.model_architecture import VMDTMFGLSTMXGBoost
import pandas as pd

for i in range(10):
    model = VMDTMFGLSTMXGBoost.load('models/tuned/ETH-LBFGS')
    df = pd.read_csv('data/ETH_1h.csv').tail(100)
    X, y, _ = model.prepare_data(df)
    pred = model.predict_embeddings(X[-1:])
    print(f"Prediction {i+1}/10 complete")
    time.sleep(2)
EOF

# Monitor resources
for i in {1..10}; do
    echo "Check $i:"
    nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader
    free -h | grep Mem | awk '{print "  Memory: "$3" / "$2}'
    sleep 2
done

echo "✅ Resource test complete"
```

---

## Step 6: Decision Point - Ready for Mainnet?

### **Checklist:**

```bash
cat > readiness_checklist.sh << 'EOF'
#!/bin/bash
echo "=== MAINNET READINESS CHECKLIST ==="
echo ""
echo "[ ] All 11 models load successfully"
echo "[ ] Backtest results show:"
echo "    [ ] Binary challenges: Avg accuracy ≥ 65%"
echo "    [ ] LBFGS challenges: Valid embeddings"
echo "    [ ] Estimated average salience ≥ 1.5"
echo "[ ] Prediction latency < 5 seconds"
echo "[ ] No critical errors in any test"
echo "[ ] Resource usage acceptable"
echo "[ ] Confident in model quality"
echo ""
echo "If ALL checked → ✅ READY FOR MAINNET"
echo "If MOST checked → ✓ Probably ready, proceed with caution"
echo "If FEW checked → ⚠️ Need more work"
EOF

chmod +x readiness_checklist.sh
./readiness_checklist.sh
```

---

## 🎯 **What to Do Based on Results**

### **Scenario A: Excellent Results (✅ READY)**

**Criteria:**
- Binary accuracy: 70%+
- Estimated salience: 2.0+
- All tests passing

**Action:**
```bash
# You're ready! Proceed to mainnet deployment
echo "✅ Models are ready for mainnet!"
echo "Next: Follow Phase 2 in COMPLETE_ROADMAP_TO_FIRST_PLACE.md"
echo "Deploy to mainnet with confidence!"
```

### **Scenario B: Good Results (✓ ACCEPTABLE)**

**Criteria:**
- Binary accuracy: 60-70%
- Estimated salience: 1.5-2.0
- Most tests passing

**Action:**
```bash
# Proceed to mainnet, but monitor closely
echo "✓ Models are ready for mainnet"
echo "Consider: Retrain 1-2 weakest challenges first"
echo "Then: Deploy to mainnet"
```

### **Scenario C: Fair Results (⚠️ NEEDS WORK)**

**Criteria:**
- Binary accuracy: 55-60%
- Estimated salience: 1.0-1.5
- Some failures

**Action:**
```bash
# Improve weak challenges before mainnet
echo "⚠️ Recommend improvements before mainnet"

# Identify weak challenges
grep "Estimated Salience" backtest_results.txt | sort -k3 -n | head -3

# Retrain weakest ones
./run_training.sh --trials 100 --challenge WEAKEST_CHALLENGE
```

### **Scenario D: Poor Results (❌ NOT READY)**

**Criteria:**
- Binary accuracy: <55%
- Estimated salience: <1.0
- Multiple failures

**Action:**
```bash
# Full retraining needed
echo "❌ Need to retrain before mainnet"
echo "Recommend:"
echo "1. Check data quality"
echo "2. Retrain all challenges with more trials"
echo "3. Re-run local tests"

# Comprehensive retraining
./run_training.sh --trials 100
```

---

## 📋 **Phase 1.5 Complete Checklist**

- [ ] All 11 models loaded successfully
- [ ] Backtesting completed
- [ ] Results analyzed
- [ ] Prediction pipeline tested
- [ ] Resource usage verified
- [ ] Readiness assessed
- [ ] Decision made: Ready for mainnet OR need improvements
- [ ] If ready: Proceed to Phase 2 (mainnet deployment)
- [ ] If not ready: Retrain weak challenges, re-test

---

## 🚀 **After Local Testing**

### **If Ready for Mainnet:**

```bash
# Save your test results
cp backtest_results.txt backtest_results_$(date +%Y%m%d).txt

# Document your baseline
echo "Local Testing Baseline:" > mainnet_baseline.txt
echo "Date: $(date)" >> mainnet_baseline.txt
./analyze_backtest.sh >> mainnet_baseline.txt

# Proceed to mainnet deployment
echo "✅ Ready for Phase 2: Mainnet Deployment"
echo "Follow: COMPLETE_ROADMAP_TO_FIRST_PLACE.md Phase 2"
```

### **Expected Mainnet Performance:**

Based on your local testing:
```
Local binary accuracy: 70% → Expect mainnet: 65-75%
Local estimated salience: 2.0 → Expect mainnet: 1.5-2.5

Local testing gives you a BASELINE to compare against.
If mainnet significantly underperforms, you know something's wrong.
```

---

## ✅ **Summary: Testing Strategy**

```
1. TRAIN MODELS (Phase 1)
   ↓
2. TEST LOCALLY (Phase 1.5) ← YOU ARE HERE
   ├─ Load all models
   ├─ Backtest on historical data
   ├─ Analyze results
   ├─ Estimate salience
   └─ Gain confidence
   ↓
3. DECISION POINT
   ├─ Ready? → Deploy to mainnet (Phase 2)
   └─ Not ready? → Improve and re-test
   ↓
4. MAINNET DEPLOYMENT (Phase 2)
   └─ Compare actual vs expected performance
```

---

## 🎯 **Key Benefits**

1. **Risk-Free Testing** - No mainnet exposure
2. **Fast Iteration** - Fix issues quickly
3. **Performance Baseline** - Know what to expect
4. **Confidence** - Deploy with certainty
5. **Quality Assurance** - Catch problems early

---

**This local testing phase ensures you deploy HIGH-QUALITY models to mainnet, maximizing your chances of success!** 🏆

**Next:** Based on test results, either deploy to mainnet (Phase 2) or improve models first!

