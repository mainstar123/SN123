#!/bin/bash
###############################################################################
# MANTIS Hyperparameter Tuning - Background Training Script
# 
# This script runs hyperparameter tuning for all challenges in the background
# with proper logging, monitoring, and error handling.
#
# Usage:
#   ./run_training.sh [OPTIONS]
#
# Options:
#   --trials N      Number of trials per challenge (default: 50)
#   --challenge X   Train only specific challenge (optional)
#   --quick         Quick mode: fewer trials for testing
#   --stop          Stop any running training
#   --status        Check training status
###############################################################################

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
PROJECT_DIR="/home/ocean/MANTIS"
VENV_DIR="$PROJECT_DIR/venv"
LOG_DIR="$PROJECT_DIR/logs/training"
SCRIPT="$PROJECT_DIR/scripts/training/train_all_challenges.py"
PID_FILE="$LOG_DIR/training.pid"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="$LOG_DIR/training_${TIMESTAMP}.log"
CURRENT_LOG="$LOG_DIR/training_current.log"

# Default options
TRIALS=50
CHALLENGE=""
QUICK_MODE=false
ACTION="start"

###############################################################################
# Helper Functions
###############################################################################

print_header() {
    echo -e "${BLUE}═══════════════════════════════════════════════════════════════${NC}"
    echo -e "${BLUE}  $1${NC}"
    echo -e "${BLUE}═══════════════════════════════════════════════════════════════${NC}"
}

print_success() {
    echo -e "${GREEN}✓ $1${NC}"
}

print_error() {
    echo -e "${RED}✗ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠ $1${NC}"
}

print_info() {
    echo -e "${BLUE}ℹ $1${NC}"
}

check_requirements() {
    print_header "Checking Requirements"
    
    # Check if virtual environment exists
    if [ ! -d "$VENV_DIR" ]; then
        print_error "Virtual environment not found at $VENV_DIR"
        print_info "Run: python3 -m venv venv && source venv/bin/activate && pip install -r requirements.txt"
        exit 1
    fi
    print_success "Virtual environment found"
    
    # Check if training script exists
    if [ ! -f "$SCRIPT" ]; then
        print_error "Training script not found at $SCRIPT"
        exit 1
    fi
    print_success "Training script found"
    
    # Create log directory if it doesn't exist
    mkdir -p "$LOG_DIR"
    print_success "Log directory ready: $LOG_DIR"
    
    echo ""
}

check_status() {
    print_header "Training Status"
    
    if [ -f "$PID_FILE" ]; then
        PID=$(cat "$PID_FILE")
        if ps -p "$PID" > /dev/null 2>&1; then
            print_success "Training is RUNNING (PID: $PID)"
            
            # Show current challenge
            if [ -f "$CURRENT_LOG" ]; then
                echo ""
                print_info "Current Progress:"
                tail -20 "$CURRENT_LOG" | grep -E "(Training Challenge|Progress:|Trial [0-9]+:|Best trial)" | tail -5
            fi
            
            # Show resource usage
            echo ""
            print_info "Resource Usage:"
            ps -p "$PID" -o pid,ppid,%cpu,%mem,etime,cmd --no-headers | awk '{printf "  PID: %s | CPU: %s%% | Memory: %s%% | Runtime: %s\n", $1, $3, $4, $5}'
            
            # Show log file
            echo ""
            print_info "Log file: $CURRENT_LOG"
            print_info "Monitor with: tail -f $CURRENT_LOG"
            
            return 0
        else
            print_warning "PID file exists but process not running"
            rm -f "$PID_FILE"
        fi
    fi
    
    print_info "No training currently running"
    return 1
}

stop_training() {
    print_header "Stopping Training"
    
    if [ -f "$PID_FILE" ]; then
        PID=$(cat "$PID_FILE")
        if ps -p "$PID" > /dev/null 2>&1; then
            print_info "Sending termination signal to PID $PID..."
            kill "$PID" 2>/dev/null || true
            
            # Wait for graceful shutdown
            print_info "Waiting for graceful shutdown (10 seconds)..."
            for i in {1..10}; do
                if ! ps -p "$PID" > /dev/null 2>&1; then
                    print_success "Training stopped gracefully"
                    rm -f "$PID_FILE"
                    return 0
                fi
                sleep 1
            done
            
            # Force kill if still running
            if ps -p "$PID" > /dev/null 2>&1; then
                print_warning "Force killing process..."
                kill -9 "$PID" 2>/dev/null || true
                sleep 1
            fi
            
            rm -f "$PID_FILE"
            print_success "Training stopped"
        else
            print_warning "PID file exists but process not running"
            rm -f "$PID_FILE"
        fi
    else
        print_info "No training currently running"
    fi
}

start_training() {
    print_header "Starting Hyperparameter Tuning"
    
    # Check if already running
    if [ -f "$PID_FILE" ]; then
        PID=$(cat "$PID_FILE")
        if ps -p "$PID" > /dev/null 2>&1; then
            print_error "Training already running (PID: $PID)"
            print_info "Use --stop to stop it first, or --status to check progress"
            exit 1
        else
            print_warning "Cleaning up stale PID file"
            rm -f "$PID_FILE"
        fi
    fi
    
    # Build command
    CMD="python $SCRIPT --trials $TRIALS"
    if [ -n "$CHALLENGE" ]; then
        CMD="$CMD --challenge $CHALLENGE"
        print_info "Training challenge: $CHALLENGE"
    else
        print_info "Training all challenges"
    fi
    print_info "Trials per challenge: $TRIALS"
    
    # Start training in background
    print_info "Starting training in background..."
    
    cd "$PROJECT_DIR"
    source "$VENV_DIR/bin/activate"
    
    # Start with logging wrapper
    (
        echo "=============================================="
        echo "MANTIS Training Started"
        echo "Time: $(date)"
        echo "PID: $$"
        echo "=============================================="
        echo ""
        
        # Run training
        $CMD 2>&1
        
        EXIT_CODE=$?
        
        echo ""
        echo "=============================================="
        echo "Training Completed"
        echo "Time: $(date)"
        echo "Exit Code: $EXIT_CODE"
        echo "=============================================="
        
        # Remove PID file on completion
        rm -f "$PID_FILE"
        
        exit $EXIT_CODE
    ) > "$LOG_FILE" 2>&1 &
    
    # Save PID
    TRAINING_PID=$!
    echo "$TRAINING_PID" > "$PID_FILE"
    
    # Create symlink to current log
    ln -sf "$LOG_FILE" "$CURRENT_LOG"
    
    sleep 2
    
    # Verify it started
    if ps -p "$TRAINING_PID" > /dev/null 2>&1; then
        print_success "Training started successfully!"
        echo ""
        print_info "PID: $TRAINING_PID"
        print_info "Log file: $LOG_FILE"
        print_info "Current log: $CURRENT_LOG"
        echo ""
        print_success "Monitor progress with:"
        echo "  tail -f $CURRENT_LOG"
        echo ""
        print_success "Check status with:"
        echo "  ./run_training.sh --status"
        echo ""
        print_success "Stop training with:"
        echo "  ./run_training.sh --stop"
    else
        print_error "Failed to start training"
        print_info "Check log file: $LOG_FILE"
        rm -f "$PID_FILE"
        exit 1
    fi
}

show_usage() {
    cat << EOF
Usage: ./run_training.sh [OPTIONS]

Options:
  --trials N       Number of trials per challenge (default: 50)
  --challenge X    Train only specific challenge (optional)
  --quick          Quick mode: 10 trials for testing
  --stop           Stop any running training
  --status         Check training status
  --help           Show this help message

Examples:
  # Start training all challenges with 50 trials each
  ./run_training.sh

  # Start with 100 trials per challenge
  ./run_training.sh --trials 100

  # Train only one challenge
  ./run_training.sh --challenge ETH-LBFGS

  # Quick test with 10 trials
  ./run_training.sh --quick

  # Check status
  ./run_training.sh --status

  # Stop training
  ./run_training.sh --stop

Monitor logs:
  tail -f $LOG_DIR/training_current.log

EOF
}

###############################################################################
# Parse Arguments
###############################################################################

while [[ $# -gt 0 ]]; do
    case $1 in
        --trials)
            TRIALS="$2"
            shift 2
            ;;
        --challenge)
            CHALLENGE="$2"
            shift 2
            ;;
        --quick)
            QUICK_MODE=true
            TRIALS=10
            shift
            ;;
        --stop)
            ACTION="stop"
            shift
            ;;
        --status)
            ACTION="status"
            shift
            ;;
        --help)
            show_usage
            exit 0
            ;;
        *)
            print_error "Unknown option: $1"
            show_usage
            exit 1
            ;;
    esac
done

###############################################################################
# Main
###############################################################################

clear
print_header "MANTIS Hyperparameter Tuning Manager"

case $ACTION in
    start)
        check_requirements
        start_training
        ;;
    stop)
        stop_training
        ;;
    status)
        check_status
        ;;
esac

echo ""

