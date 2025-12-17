#!/bin/bash
#
# Quick Local Testing Script
# Run this after training completes to test all models locally
#

set -e  # Exit on error

echo "================================================================================"
echo "🧪 MANTIS Local Testing Suite"
echo "================================================================================"
echo ""
echo "This will test all your trained models locally before mainnet deployment"
echo "Estimated time: 15-30 minutes"
echo ""
read -p "Press Enter to start testing..."

# Change to project directory
cd "$(dirname "$0")"
PROJECT_DIR=$(pwd)

echo ""
echo "Project directory: $PROJECT_DIR"
echo ""

# Activate virtual environment
if [ -d "venv" ]; then
    source venv/bin/activate
    echo "✓ Virtual environment activated"
else
    echo "✗ Virtual environment not found"
    echo "Please create one: python3 -m venv venv && source venv/bin/activate && pip install -r requirements.txt"
    exit 1
fi

echo ""
echo "================================================================================"
echo "STAGE 1: Verify Training Completion"
echo "================================================================================"
echo ""

# Check if models directory exists
if [ ! -d "models/tuned" ]; then
    echo "✗ models/tuned/ directory not found"
    echo "Training may not be complete. Check logs/training/training_current.log"
    exit 1
fi

# Count models
MODEL_COUNT=$(ls -d models/tuned/*/ 2>/dev/null | wc -l)
echo "Models found: $MODEL_COUNT"

if [ "$MODEL_COUNT" -lt 11 ]; then
    echo "⚠️  Expected 11 models, found $MODEL_COUNT"
    echo "Some challenges may not have completed training"
    echo ""
    echo "Available models:"
    ls -d models/tuned/*/
    echo ""
    read -p "Continue anyway? (y/N): " CONTINUE
    if [ "$CONTINUE" != "y" ] && [ "$CONTINUE" != "Y" ]; then
        exit 1
    fi
else
    echo "✅ All 11 models found"
fi

echo ""
echo "Checking model files..."
INCOMPLETE=0
for MODEL_DIR in models/tuned/*/; do
    MODEL_NAME=$(basename "$MODEL_DIR")
    REQUIRED_FILES=("lstm_model.h5" "xgb_model.json" "scaler.pkl" "feature_indices.pkl" "config.json")
    MISSING=0
    
    for FILE in "${REQUIRED_FILES[@]}"; do
        if [ ! -f "$MODEL_DIR/$FILE" ]; then
            if [ $MISSING -eq 0 ]; then
                echo "⚠️  $MODEL_NAME: Missing files:"
            fi
            echo "  - $FILE"
            MISSING=1
        fi
    done
    
    if [ $MISSING -eq 0 ]; then
        echo "✓ $MODEL_NAME: Complete"
    else
        INCOMPLETE=$((INCOMPLETE + 1))
    fi
done

if [ $INCOMPLETE -gt 0 ]; then
    echo ""
    echo "⚠️  $INCOMPLETE models have missing files"
    read -p "Continue anyway? (y/N): " CONTINUE
    if [ "$CONTINUE" != "y" ] && [ "$CONTINUE" != "Y" ]; then
        exit 1
    fi
else
    echo ""
    echo "✅ All models have complete files"
fi

echo ""
echo "================================================================================"
echo "STAGE 2: Test Model Loading"
echo "================================================================================"
echo ""

python << 'STAGE2'
from scripts.training.model_architecture import VMDTMFGLSTMXGBoost
import os
import sys

challenges = [
    'ETH-LBFGS', 'BTC-LBFGS-6H', 'ETH-HITFIRST-100M',
    'ETH-1H-BINARY', 'EURUSD-1H-BINARY', 'GBPUSD-1H-BINARY',
    'CADUSD-1H-BINARY', 'NZDUSD-1H-BINARY', 'CHFUSD-1H-BINARY',
    'XAUUSD-1H-BINARY', 'XAGUSD-1H-BINARY'
]

passed = 0
failed = 0
failed_challenges = []

for challenge in challenges:
    model_path = f'models/tuned/{challenge}'
    if not os.path.exists(model_path):
        print(f"⏭️  {challenge}: Model directory not found, skipping")
        continue
        
    try:
        model = VMDTMFGLSTMXGBoost.load(model_path)
        print(f"✓ {challenge}: Loaded successfully")
        passed += 1
    except Exception as e:
        print(f"✗ {challenge}: FAILED - {str(e)[:60]}")
        failed += 1
        failed_challenges.append(challenge)

print()
print(f"Results: {passed} passed, {failed} failed")
if failed > 0:
    print(f"Failed challenges: {', '.join(failed_challenges)}")
    sys.exit(1)
else:
    print("✅ ALL MODELS LOADED SUCCESSFULLY")
STAGE2

if [ $? -ne 0 ]; then
    echo ""
    echo "❌ Stage 2 failed. Some models could not be loaded."
    echo "Fix the errors above and try again."
    exit 1
fi

echo ""
echo "================================================================================"
echo "STAGE 3: Backtest Performance (This will take 5-10 minutes)"
echo "================================================================================"
echo ""

# Check if backtest script exists
if [ ! -f "scripts/testing/backtest_models.py" ]; then
    echo "✗ scripts/testing/backtest_models.py not found"
    exit 1
fi

# Create results directory
mkdir -p results

# Run backtest
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RESULTS_FILE="results/backtest_results_${TIMESTAMP}.txt"

echo "Running comprehensive backtest..."
echo "Results will be saved to: $RESULTS_FILE"
echo ""

python scripts/testing/backtest_models.py | tee "$RESULTS_FILE"

if [ $? -ne 0 ]; then
    echo ""
    echo "❌ Stage 3 failed. Check errors above."
    exit 1
fi

# Save as latest
cp "$RESULTS_FILE" results/backtest_results_latest.txt

echo ""
echo "================================================================================"
echo "STAGE 4: Prediction Speed Test"
echo "================================================================================"
echo ""

python << 'STAGE4'
import time
import pandas as pd
from scripts.training.model_architecture import VMDTMFGLSTMXGBoost
import numpy as np

challenge = 'ETH-LBFGS'
print(f"Testing prediction speed for {challenge} (highest weight challenge)...")
print()

try:
    # Load model
    start = time.time()
    model = VMDTMFGLSTMXGBoost.load(f'models/tuned/{challenge}')
    load_time = time.time() - start
    
    # Load data
    start = time.time()
    df = pd.read_csv('data/ETH_1h.csv').tail(100)
    data_time = time.time() - start
    
    # Prepare features
    start = time.time()
    X, y, features = model.prepare_data(df)
    prep_time = time.time() - start
    
    # Generate prediction
    start = time.time()
    pred = model.predict_embeddings(X[-1:])
    pred_time = time.time() - start
    
    total_time = load_time + data_time + prep_time + pred_time
    
    print(f"✓ Model loaded: {load_time:.3f}s")
    print(f"✓ Data loaded: {data_time:.3f}s")
    print(f"✓ Features prepared: {prep_time:.3f}s")
    print(f"✓ Prediction generated: {pred_time:.3f}s")
    print()
    print(f"Total time: {total_time:.3f}s")
    print()
    
    if total_time < 3.0:
        print("✅ EXCELLENT - Very fast, ready for real-time")
    elif total_time < 5.0:
        print("✓ GOOD - Fast enough for mining")
    elif total_time < 10.0:
        print("⚠️  ACCEPTABLE - Should work but monitor")
    else:
        print("❌ TOO SLOW - May need optimization")
        
except Exception as e:
    print(f"✗ Speed test failed: {e}")
    import traceback
    traceback.print_exc()
STAGE4

echo ""
echo "================================================================================"
echo "TEST SUMMARY"
echo "================================================================================"
echo ""

# Extract key metrics from backtest results
if [ -f "$RESULTS_FILE" ]; then
    echo "Backtest Results:"
    echo ""
    
    # Binary challenges
    if grep -q "Binary Avg Accuracy" "$RESULTS_FILE"; then
        echo "📊 Binary Challenges:"
        grep "Binary Avg Accuracy" "$RESULTS_FILE" | tail -1
        grep -A 1 "Binary Avg Accuracy" "$RESULTS_FILE" | tail -1
        echo ""
    fi
    
    # Overall salience
    if grep -q "Overall Est. Salience" "$RESULTS_FILE"; then
        echo "📊 Overall Performance:"
        grep "Overall Est. Salience" "$RESULTS_FILE" | tail -1
        grep -A 1 "Overall Est. Salience" "$RESULTS_FILE" | tail -1
        echo ""
    fi
    
    # Recommendation
    if grep -q "RECOMMENDATION" "$RESULTS_FILE"; then
        echo "📋 Recommendation:"
        grep -A 10 "RECOMMENDATION" "$RESULTS_FILE" | grep -E "READY|ACCEPTABLE|IMPROVEMENTS|RECOMMEND" | head -3
        echo ""
    fi
fi

echo "================================================================================"
echo "DECISION MATRIX"
echo "================================================================================"
echo ""
echo "Review the results above. Your deployment readiness:"
echo ""
echo "✅ READY TO DEPLOY if:"
echo "   - All models loaded successfully (Stage 2)"
echo "   - Binary avg accuracy ≥ 65%"
echo "   - Overall estimated salience ≥ 1.5"
echo "   - Prediction time < 5 seconds"
echo ""
echo "⚠️  CONSIDER IMPROVEMENTS if:"
echo "   - Binary avg accuracy 60-65%"
echo "   - Overall estimated salience 1.0-1.5"
echo "   → You can deploy now OR improve weak challenges first"
echo ""
echo "❌ NOT READY if:"
echo "   - Binary avg accuracy < 60%"
echo "   - Overall estimated salience < 1.0"
echo "   → Recommend retraining before mainnet"
echo ""
echo "================================================================================"
echo ""
echo "📁 Results saved to: $RESULTS_FILE"
echo "📊 View full details: cat $RESULTS_FILE"
echo ""
echo "🚀 Next steps:"
echo "   - If ready: Follow Phase 2 in COMPLETE_ROADMAP_TO_FIRST_PLACE.md"
echo "   - If not ready: Retrain weak challenges with ./run_training.sh --trials 100 --challenge CHALLENGE_NAME"
echo ""
echo "✅ Local testing complete!"
echo ""

