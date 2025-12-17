#!/bin/bash
# Comprehensive Background Tuning Script with All Improvements
# This script runs hyperparameter tuning for ALL challenges in the background
# with all improvements: class weights, unique features, enhanced architecture

set -e

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

DATA_DIR="${DATA_DIR:-data/raw}"
TUNING_DIR="${TUNING_DIR:-models/tuning}"
EPOCHS="${EPOCHS:-100}"
# Use larger batch size for GPU (RTX 3090 can handle 128-256 easily)
# Check if GPU is available and adjust batch size accordingly
if command -v nvidia-smi &> /dev/null && nvidia-smi &> /dev/null; then
    BATCH_SIZE="${BATCH_SIZE:-128}"  # Larger batch for GPU
else
    BATCH_SIZE="${BATCH_SIZE:-64}"   # Smaller batch for CPU
fi
MIN_ACCURACY="${MIN_ACCURACY:-0.55}"

# Logging
LOG_DIR="logs/tuning"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="$LOG_DIR/tuning_background_${TIMESTAMP}.log"
PID_FILE="$LOG_DIR/tuning_background_${TIMESTAMP}.pid"
STATUS_FILE="$LOG_DIR/tuning_background_${TIMESTAMP}.status"

# Create directories
mkdir -p "$LOG_DIR"
mkdir -p "$DATA_DIR"
mkdir -p "$TUNING_DIR"

# Function to log messages
log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$LOG_FILE"
}

# Function to update status
update_status() {
    echo "$1" > "$STATUS_FILE"
    log "STATUS: $1"
}

# Function to cleanup on exit
cleanup() {
    if [ -f "$PID_FILE" ]; then
        rm -f "$PID_FILE"
    fi
    update_status "STOPPED"
}

trap cleanup EXIT INT TERM

# Check if already running
if [ -f "$PID_FILE" ]; then
    OLD_PID=$(cat "$PID_FILE")
    if ps -p "$OLD_PID" > /dev/null 2>&1; then
        log "ERROR: Tuning already running with PID $OLD_PID"
        log "Check status with: tail -f $LOG_FILE"
        exit 1
    else
        log "Removing stale PID file"
        rm -f "$PID_FILE"
    fi
fi

# Save PID
echo $$ > "$PID_FILE"

log "=================================================================================="
log "MANTIS - Background Hyperparameter Tuning (WITH ALL IMPROVEMENTS)"
log "=================================================================================="
log ""
log "Configuration:"
log "  Data directory: $DATA_DIR"
log "  Tuning directory: $TUNING_DIR"
log "  Epochs per config: $EPOCHS"
log "  Batch size: $BATCH_SIZE"
log "  Min accuracy: $MIN_ACCURACY"
log ""
log "Improvements enabled:"
log "  ✓ Class imbalance handling (automatic class weights)"
log "  ✓ Unique feature extraction (7 categories)"
log "  ✓ Enhanced model architecture (128→64→32 layers)"
log ""
log "This will tune ALL challenges:"
log "  - Binary: ETH, EURUSD, GBPUSD, CADUSD, NZDUSD, CHFUSD, AUDUSD, JPYUSD, XAUUSD, XAGUSD"
log "  - LBFGS: ETHLBFGS, BTCLBFGS"
log "  - HitFirst: ETHHITFIRST"
log ""
log "Estimated time: 30-40 hours"
log ""
log "Log file: $LOG_FILE"
log "Status file: $STATUS_FILE"
log "PID file: $PID_FILE"
log ""
log "To monitor progress:"
log "  tail -f $LOG_FILE"
log "  cat $STATUS_FILE"
log ""
log "To stop:"
log "  kill \$(cat $PID_FILE)"
log ""
log "=================================================================================="
log "Starting tuning process..."
log "=================================================================================="

# Update status
update_status "STARTING"

# Check Python environment
if ! command -v python &> /dev/null; then
    log "ERROR: Python not found"
    update_status "ERROR: Python not found"
    exit 1
fi

# Check if virtual environment exists and activate it
if [ -d "venv" ]; then
    log "Activating virtual environment..."
    source venv/bin/activate
fi

# Verify required packages
log "Checking dependencies..."
python -c "import tensorflow, xgboost, pandas, numpy, sklearn" 2>/dev/null || {
    log "ERROR: Missing required packages. Install with: pip install -r requirements.txt"
    update_status "ERROR: Missing dependencies"
    exit 1
}

# Set TensorFlow environment variables for GPU optimization
export TF_FORCE_GPU_ALLOW_GROWTH=true
export TF_GPU_THREAD_MODE=gpu_private
export TF_GPU_THREAD_COUNT=2
export CUDA_VISIBLE_DEVICES=0

log "GPU environment variables set for TensorFlow"

# Start tuning
update_status "RUNNING"

log ""
log "Starting hyperparameter tuning..."
log ""

# Run tuning with all improvements
# Use -u flag for unbuffered output so logs update in real-time
python -u scripts/training/tune_all_challenges.py \
    --data-dir "$DATA_DIR" \
    --tuning-dir "$TUNING_DIR" \
    --epochs "$EPOCHS" \
    --batch-size "$BATCH_SIZE" \
    --min-accuracy "$MIN_ACCURACY" \
    2>&1 | tee -a "$LOG_FILE"

EXIT_CODE=${PIPESTATUS[0]}

if [ $EXIT_CODE -eq 0 ]; then
    log ""
    log "=================================================================================="
    log "Tuning completed successfully!"
    log "=================================================================================="
    log ""
    log "Results saved to: $TUNING_DIR"
    log ""
    log "Next steps:"
    log "  1. Test salience: ./test_all_salience.sh"
    log "  2. Check best configs: cat $TUNING_DIR/best_configs_*.json"
    log ""
    update_status "COMPLETED"
else
    log ""
    log "=================================================================================="
    log "Tuning failed with exit code: $EXIT_CODE"
    log "=================================================================================="
    log ""
    log "Check the log file for details: $LOG_FILE"
    log ""
    update_status "FAILED: Exit code $EXIT_CODE"
    exit $EXIT_CODE
fi


