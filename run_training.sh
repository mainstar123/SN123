#!/bin/bash

################################################################################
# MANTIS Hyperparameter Tuning Runner
# Simplified script to run hyperparameter tuning
################################################################################

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Default values
DATA_DIR="data"
TUNING_DIR="models/tuned"
EPOCHS=100
BATCH_SIZE=128
TICKER=""
CHALLENGE_TYPE=""
MODE="all"

# Help message
show_help() {
    echo "Usage: ./run_training.sh [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  --ticker NAME       Train only specific ticker (e.g., ETH-LBFGS)"
    echo "  --challenge-type T  Train only specific type: binary, lbfgs, or hitfirst"
    echo "  --data-dir PATH     Data directory (default: data)"
    echo "  --tuning-dir PATH   Output directory (default: models/tuned)"
    echo "  --epochs N          Epochs per config (default: 100)"
    echo "  --batch-size N      Batch size (default: 128)"
    echo "  --help              Show this help message"
                echo ""
    echo "Examples:"
    echo "  ./run_training.sh                              # Train all challenges"
    echo "  ./run_training.sh --ticker ETH-LBFGS          # Train only ETH-LBFGS"
    echo "  ./run_training.sh --challenge-type binary     # Train only binary challenges"
    echo "  ./run_training.sh --epochs 150                # Train with 150 epochs"
            echo ""
    echo "Note: The script tests predefined configurations:"
    echo "  - Binary: 13 configs, LBFGS: 7 configs, HITFIRST: 5 configs"
            echo ""
}

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --ticker)
            TICKER="$2"
            MODE="single"
            shift 2
            ;;
        --challenge-type)
            CHALLENGE_TYPE="$2"
            shift 2
            ;;
        --epochs)
            EPOCHS="$2"
            shift 2
            ;;
        --batch-size)
            BATCH_SIZE="$2"
            shift 2
            ;;
        --data-dir)
            DATA_DIR="$2"
            shift 2
            ;;
        --tuning-dir)
            TUNING_DIR="$2"
            shift 2
            ;;
        --help)
            show_help
            exit 0
            ;;
        *)
            echo -e "${RED}Unknown option: $1${NC}"
            show_help
            exit 1
            ;;
    esac
done

echo "================================================================================"
echo "🚀 MANTIS Hyperparameter Tuning"
echo "================================================================================"
echo ""

# Pre-flight checks
echo "📋 Pre-flight Checks:"
echo ""

# Check if data directory exists
if [ ! -d "$DATA_DIR" ]; then
    echo -e "${RED}✗ Data directory not found: $DATA_DIR${NC}"
    exit 1
else
    echo -e "${GREEN}✓ Data directory found${NC}"
fi

# Check if training script exists
if [ ! -f "scripts/training/tune_all_challenges.py" ]; then
    echo -e "${RED}✗ Training script not found: scripts/training/tune_all_challenges.py${NC}"
    exit 1
else
    echo -e "${GREEN}✓ Training script found${NC}"
fi

# Check GPU
if command -v nvidia-smi &> /dev/null; then
    GPU_COUNT=$(nvidia-smi --query-gpu=count --format=csv,noheader | head -1)
    echo -e "${GREEN}✓ GPU available (Count: $GPU_COUNT)${NC}"
else
    echo -e "${YELLOW}⚠ GPU not detected (will use CPU - slower)${NC}"
fi

# Check disk space (need at least 10GB)
AVAILABLE_GB=$(df -BG . | tail -1 | awk '{print $4}' | sed 's/G//')
if [ "$AVAILABLE_GB" -lt 10 ]; then
    echo -e "${RED}✗ Low disk space: ${AVAILABLE_GB}GB available (need 10GB)${NC}"
    exit 1
else
    echo -e "${GREEN}✓ Sufficient disk space: ${AVAILABLE_GB}GB available${NC}"
fi

# Create output directory if needed
mkdir -p "$OUTPUT_DIR"
mkdir -p logs/training

echo ""
echo "================================================================================"
echo "📊 Training Configuration:"
echo "================================================================================"
echo "  Mode:          $MODE"
if [ "$MODE" == "single" ]; then
    echo "  Ticker:        $TICKER"
fi
if [ -n "$CHALLENGE_TYPE" ]; then
    echo "  Type Filter:   $CHALLENGE_TYPE"
fi
echo "  Epochs:        $EPOCHS"
echo "  Batch Size:    $BATCH_SIZE"
echo "  Data Dir:      $DATA_DIR"
echo "  Tuning Dir:    $TUNING_DIR"
echo ""
echo "  Configurations Tested:"
echo "    Binary: 13 configs, LBFGS: 7 configs, HITFIRST: 5 configs"
echo ""
if [ "$MODE" == "single" ]; then
    echo "  Estimated Time: 1.5-3 hours"
elif [ -n "$CHALLENGE_TYPE" ]; then
    echo "  Estimated Time: 4-8 hours (depends on type)"
else
    echo "  Estimated Time: 18-24 hours (all 11 challenges)"
fi
echo ""

# Ask for confirmation
read -p "Start training? (y/N): " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Training cancelled."
    exit 0
fi

echo ""
echo "================================================================================"
echo "🎯 Starting Training..."
echo "================================================================================"
echo ""

# Build command
CMD="python scripts/training/tune_all_challenges.py --data-dir $DATA_DIR --tuning-dir $TUNING_DIR --epochs $EPOCHS --batch-size $BATCH_SIZE"
if [ "$MODE" == "single" ]; then
    CMD="$CMD --ticker $TICKER"
fi
if [ -n "$CHALLENGE_TYPE" ]; then
    CMD="$CMD --challenge-type $CHALLENGE_TYPE"
fi

# Generate log filename
LOGFILE="logs/training/training_$(date +%Y%m%d_%H%M%S).log"

# Run training in background
nohup $CMD > "$LOGFILE" 2>&1 &
PID=$!

# Save PID
echo $PID > logs/training/training.pid

echo -e "${GREEN}✓ Training started successfully!${NC}"
echo ""
echo "Process ID: $PID"
echo "Log file:   $LOGFILE"
echo ""
echo "================================================================================"
echo "📈 Monitoring Commands:"
echo "================================================================================"
echo ""
echo "  Watch progress:"
echo "    tail -f $LOGFILE"
echo ""
echo "  Check status:"
echo "    ps aux | grep $PID"
echo ""
echo "  Check completed models:"
echo "    ls -d $TUNING_DIR/*/"
echo ""
echo "  Stop training (if needed):"
echo "    kill $PID"
echo ""
echo "================================================================================"
echo ""
echo -e "${YELLOW}Training is running in the background.${NC}"
if [ "$MODE" == "single" ]; then
    echo -e "${YELLOW}It will take 1.5-3 hours to complete.${NC}"
elif [ -n "$CHALLENGE_TYPE" ]; then
    echo -e "${YELLOW}It will take 4-8 hours to complete.${NC}"
else
    echo -e "${YELLOW}It will take 18-24 hours to complete.${NC}"
fi
echo ""
echo "You can now close this terminal. Training will continue."
echo ""

# Optionally show initial progress
read -p "Show live progress now? (y/N): " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo ""
    echo "Showing live progress (Ctrl+C to exit, training continues)..."
    echo ""
    sleep 2
    tail -f "$LOGFILE"
fi
