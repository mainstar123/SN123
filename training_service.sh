#!/bin/bash
# MANTIS Training Service Manager
# Manages training script as a background service

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PID_FILE="$SCRIPT_DIR/training_service.pid"
LOG_FILE="$SCRIPT_DIR/logs/training_service.log"
ERROR_LOG="$SCRIPT_DIR/logs/training_service_error.log"
TRAINING_SCRIPT="$SCRIPT_DIR/scripts/training/train_model.py"

# Default arguments
TICKER="${TRAINING_TICKER:-BTC}"
DATA_DIR="${TRAINING_DATA_DIR:-data/raw}"
MODEL_DIR="${TRAINING_MODEL_DIR:-models/checkpoints}"
EPOCHS="${TRAINING_EPOCHS:-100}"
BATCH_SIZE="${TRAINING_BATCH_SIZE:-64}"
USE_GPU="${TRAINING_USE_GPU:-true}"

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Create logs directory
mkdir -p "$SCRIPT_DIR/logs"

# Function to check if service is running
is_running() {
    if [ -f "$PID_FILE" ]; then
        PID=$(cat "$PID_FILE")
        if ps -p "$PID" > /dev/null 2>&1; then
            return 0
        else
            # PID file exists but process is dead
            rm -f "$PID_FILE"
            return 1
        fi
    fi
    return 1
}

# Function to get process info
get_status() {
    if is_running; then
        PID=$(cat "$PID_FILE")
        echo "Status: ${GREEN}RUNNING${NC}"
        echo "PID: $PID"
        echo "Started: $(ps -p $PID -o lstart= 2>/dev/null || echo 'Unknown')"
        echo "CPU: $(ps -p $PID -o %cpu= 2>/dev/null || echo '0')%"
        echo "Memory: $(ps -p $PID -o %mem= 2>/dev/null || echo '0')%"
        return 0
    else
        echo "Status: ${RED}STOPPED${NC}"
        return 1
    fi
}

# Start service
start() {
    if is_running; then
        echo -e "${YELLOW}Service is already running${NC}"
        get_status
        return 1
    fi
    
    echo -e "${GREEN}Starting training service...${NC}"
    echo "Ticker: $TICKER"
    echo "Data dir: $DATA_DIR"
    echo "Model dir: $MODEL_DIR"
    echo "Epochs: $EPOCHS"
    echo "Batch size: $BATCH_SIZE"
    echo "GPU: $USE_GPU"
    echo "Log file: $LOG_FILE"
    echo ""
    
    # Activate virtual environment if it exists
    if [ -d "$SCRIPT_DIR/venv" ]; then
        PYTHON_CMD="$SCRIPT_DIR/venv/bin/python"
    elif [ -n "$VIRTUAL_ENV" ]; then
        PYTHON_CMD="$VIRTUAL_ENV/bin/python"
    else
        PYTHON_CMD="python3"
    fi
    
    # Build command
    CMD="$PYTHON_CMD $TRAINING_SCRIPT"
    CMD="$CMD --ticker $TICKER"
    CMD="$CMD --data-dir $DATA_DIR"
    CMD="$CMD --model-dir $MODEL_DIR"
    CMD="$CMD --epochs $EPOCHS"
    CMD="$CMD --batch-size $BATCH_SIZE"
    
    if [ "$USE_GPU" = "true" ]; then
        CMD="$CMD --use-gpu"
    else
        CMD="$CMD --use-cpu"
    fi
    
    # Start in background and redirect output
    cd "$SCRIPT_DIR"
    # Activate venv in the command if it exists
    if [ -d "$SCRIPT_DIR/venv" ]; then
        nohup bash -c "source $SCRIPT_DIR/venv/bin/activate && $CMD" >> "$LOG_FILE" 2>> "$ERROR_LOG" &
    else
        nohup $CMD >> "$LOG_FILE" 2>> "$ERROR_LOG" &
    fi
    PID=$!
    
    # Save PID
    echo $PID > "$PID_FILE"
    
    # Wait a moment to check if it started successfully
    sleep 2
    
    if is_running; then
        echo -e "${GREEN}✓ Service started successfully${NC}"
        echo "PID: $PID"
        echo "Logs: $LOG_FILE"
        echo "Errors: $ERROR_LOG"
        echo ""
        echo "View logs: $0 logs"
        echo "View real-time: $0 tail"
        return 0
    else
        echo -e "${RED}✗ Service failed to start${NC}"
        echo "Check error log: $ERROR_LOG"
        rm -f "$PID_FILE"
        return 1
    fi
}

# Stop service
stop() {
    if ! is_running; then
        echo -e "${YELLOW}Service is not running${NC}"
        return 1
    fi
    
    PID=$(cat "$PID_FILE")
    echo -e "${YELLOW}Stopping training service (PID: $PID)...${NC}"
    
    # Try graceful shutdown first
    kill -TERM "$PID" 2>/dev/null
    
    # Wait up to 30 seconds
    for i in {1..30}; do
        if ! ps -p "$PID" > /dev/null 2>&1; then
            break
        fi
        sleep 1
    done
    
    # Force kill if still running
    if ps -p "$PID" > /dev/null 2>&1; then
        echo -e "${YELLOW}Force killing process...${NC}"
        kill -KILL "$PID" 2>/dev/null
        sleep 1
    fi
    
    # Clean up
    rm -f "$PID_FILE"
    
    if ! ps -p "$PID" > /dev/null 2>&1; then
        echo -e "${GREEN}✓ Service stopped${NC}"
        return 0
    else
        echo -e "${RED}✗ Failed to stop service${NC}"
        return 1
    fi
}

# Restart service
restart() {
    echo -e "${YELLOW}Restarting training service...${NC}"
    stop
    sleep 2
    start
}

# Remove service (stop and clean up)
remove() {
    echo -e "${YELLOW}Removing training service...${NC}"
    stop
    
    # Optionally remove log files
    read -p "Remove log files? (y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        rm -f "$LOG_FILE" "$ERROR_LOG"
        echo -e "${GREEN}✓ Log files removed${NC}"
    fi
    
    echo -e "${GREEN}✓ Service removed${NC}"
}

# View logs
logs() {
    if [ ! -f "$LOG_FILE" ]; then
        echo -e "${YELLOW}No log file found${NC}"
        return 1
    fi
    
    echo -e "${GREEN}Training Service Logs${NC}"
    echo "File: $LOG_FILE"
    echo "Last 50 lines:"
    echo "----------------------------------------"
    tail -50 "$LOG_FILE"
}

# View error logs
errors() {
    if [ ! -f "$ERROR_LOG" ]; then
        echo -e "${YELLOW}No error log file found${NC}"
        return 1
    fi
    
    echo -e "${RED}Training Service Error Logs${NC}"
    echo "File: $ERROR_LOG"
    echo "Last 50 lines:"
    echo "----------------------------------------"
    tail -50 "$ERROR_LOG"
}

# Tail logs in real-time
tail() {
    echo -e "${GREEN}Tailing training service logs (Ctrl+C to exit)${NC}"
    echo "Log file: $LOG_FILE"
    echo "Error log: $ERROR_LOG"
    echo "----------------------------------------"
    
    # If log file doesn't exist, wait for it
    if [ ! -f "$LOG_FILE" ]; then
        echo -e "${YELLOW}Waiting for log file to be created...${NC}"
        while [ ! -f "$LOG_FILE" ]; do
            sleep 1
        done
    fi
    
    # Show last 20 lines first, then tail
    if [ -f "$LOG_FILE" ] && [ -s "$LOG_FILE" ]; then
        echo -e "${YELLOW}Last 20 lines:${NC}"
        tail -20 "$LOG_FILE"
        echo ""
        echo -e "${GREEN}Following new lines (Ctrl+C to exit):${NC}"
        echo "----------------------------------------"
    fi
    
    # Tail the log file
    tail -f "$LOG_FILE" 2>/dev/null
}

# Tail both logs
tail_all() {
    echo -e "${GREEN}Tailing all logs (Ctrl+C to exit)${NC}"
    echo "Output: $LOG_FILE"
    echo "Errors: $ERROR_LOG"
    echo "----------------------------------------"
    
    # Show last lines first
    if [ -f "$LOG_FILE" ] && [ -s "$LOG_FILE" ]; then
        echo -e "${GREEN}Last 10 lines of output:${NC}"
        tail -10 "$LOG_FILE"
        echo ""
    fi
    
    if [ -f "$ERROR_LOG" ] && [ -s "$ERROR_LOG" ]; then
        echo -e "${RED}Last 10 lines of errors:${NC}"
        tail -10 "$ERROR_LOG"
        echo ""
    fi
    
    echo -e "${GREEN}Following new lines (Ctrl+C to exit):${NC}"
    echo "----------------------------------------"
    
    # Tail both files
    if [ -f "$LOG_FILE" ] && [ -f "$ERROR_LOG" ]; then
        tail -f "$LOG_FILE" "$ERROR_LOG" 2>/dev/null
    elif [ -f "$LOG_FILE" ]; then
        tail -f "$LOG_FILE" 2>/dev/null
    elif [ -f "$ERROR_LOG" ]; then
        tail -f "$ERROR_LOG" 2>/dev/null
    else
        echo -e "${YELLOW}No log files found. Waiting...${NC}"
        while [ ! -f "$LOG_FILE" ] && [ ! -f "$ERROR_LOG" ]; do
            sleep 1
        done
        tail -f "$LOG_FILE" "$ERROR_LOG" 2>/dev/null
    fi
}

# Status
status() {
    echo "=========================================="
    echo "MANTIS Training Service Status"
    echo "=========================================="
    get_status
    echo ""
    echo "Configuration:"
    echo "  Ticker: $TICKER"
    echo "  Data dir: $DATA_DIR"
    echo "  Model dir: $MODEL_DIR"
    echo "  Epochs: $EPOCHS"
    echo "  Batch size: $BATCH_SIZE"
    echo "  GPU: $USE_GPU"
    echo ""
    echo "Logs:"
    echo "  Output: $LOG_FILE"
    echo "  Errors: $ERROR_LOG"
    echo "=========================================="
}

# Help
help() {
    echo "MANTIS Training Service Manager"
    echo ""
    echo "Usage: $0 {start|stop|restart|status|logs|errors|tail|tail-all|remove|help}"
    echo ""
    echo "Commands:"
    echo "  start      - Start the training service"
    echo "  stop       - Stop the training service"
    echo "  restart    - Restart the training service"
    echo "  status     - Show service status"
    echo "  logs       - View last 50 lines of output log"
    echo "  errors     - View last 50 lines of error log"
    echo "  tail       - Follow output log in real-time"
    echo "  tail-all   - Follow both logs in real-time"
    echo "  remove     - Stop and remove service (optionally remove logs)"
    echo "  help       - Show this help message"
    echo ""
    echo "Environment Variables:"
    echo "  TRAINING_TICKER      - Ticker to train (default: BTC)"
    echo "  TRAINING_DATA_DIR    - Data directory (default: data/raw)"
    echo "  TRAINING_MODEL_DIR   - Model directory (default: models/checkpoints)"
    echo "  TRAINING_EPOCHS      - Number of epochs (default: 100)"
    echo "  TRAINING_BATCH_SIZE  - Batch size (default: 64)"
    echo "  TRAINING_USE_GPU     - Use GPU (default: true)"
    echo ""
    echo "Examples:"
    echo "  # Start with defaults"
    echo "  $0 start"
    echo ""
    echo "  # Start with custom ticker"
    echo "  TRAINING_TICKER=ETH $0 start"
    echo ""
    echo "  # View logs in real-time"
    echo "  $0 tail"
    echo ""
    echo "  # Check status"
    echo "  $0 status"
}

# Main
case "$1" in
    start)
        start
        ;;
    stop)
        stop
        ;;
    restart)
        restart
        ;;
    status)
        status
        ;;
    logs)
        logs
        ;;
    errors)
        errors
        ;;
    tail)
        tail
        ;;
    tail-all)
        tail_all
        ;;
    remove)
        remove
        ;;
    help|--help|-h)
        help
        ;;
    *)
        echo "Usage: $0 {start|stop|restart|status|logs|errors|tail|tail-all|remove|help}"
        exit 1
        ;;
esac

exit $?

