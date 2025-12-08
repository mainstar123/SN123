#!/bin/bash

# Real-time training log viewer
# Usage: ./tail_training_log.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_FILE="${SCRIPT_DIR}/logs/training_service.log"
ERROR_LOG="${SCRIPT_DIR}/logs/training_service_error.log"

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}MANTIS Training Log Viewer${NC}"
echo -e "${GREEN}========================================${NC}"
echo -e "Log file: ${LOG_FILE}"
echo -e "Error log: ${ERROR_LOG}"
echo -e "${GREEN}========================================${NC}"
echo ""

# Check if log file exists
if [ ! -f "$LOG_FILE" ]; then
    echo -e "${YELLOW}Log file not found. Waiting for it to be created...${NC}"
    while [ ! -f "$LOG_FILE" ]; do
        sleep 1
    done
    echo -e "${GREEN}Log file found! Starting to tail...${NC}"
    echo ""
fi

# Show last 30 lines if file exists and has content
if [ -f "$LOG_FILE" ] && [ -s "$LOG_FILE" ]; then
    echo -e "${YELLOW}Last 30 lines:${NC}"
    echo "----------------------------------------"
    tail -30 "$LOG_FILE"
    echo ""
    echo -e "${GREEN}Following new lines (Ctrl+C to exit):${NC}"
    echo "----------------------------------------"
    echo ""
fi

# Tail the log file in real-time
tail -f "$LOG_FILE" 2>/dev/null

