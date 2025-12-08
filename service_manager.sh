#!/bin/bash

# MANTIS Service Manager
# Manages all MANTIS services (training, mining, validator, etc.)

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SERVICES_DIR="${SCRIPT_DIR}/.services"

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Create services directory if it doesn't exist
mkdir -p "$SERVICES_DIR"

# List all services
list() {
    echo -e "${BLUE}========================================${NC}"
    echo -e "${BLUE}MANTIS Services${NC}"
    echo -e "${BLUE}========================================${NC}"
    echo ""
    
    local found=0
    
    # Check for PID files in services directory
    if [ -d "$SERVICES_DIR" ] && [ "$(ls -A $SERVICES_DIR/*.pid 2>/dev/null)" ]; then
        for pid_file in "$SERVICES_DIR"/*.pid; do
            if [ -f "$pid_file" ]; then
                local service_name=$(basename "$pid_file" .pid)
                local pid=$(cat "$pid_file" 2>/dev/null)
                
                if [ -n "$pid" ] && kill -0 "$pid" 2>/dev/null; then
                    # Process is running
                    local cmd=$(ps -p "$pid" -o cmd= 2>/dev/null | head -1)
                    local cpu=$(ps -p "$pid" -o %cpu= 2>/dev/null | tr -d ' ')
                    local mem=$(ps -p "$pid" -o %mem= 2>/dev/null | tr -d ' ')
                    local runtime=$(ps -p "$pid" -o etime= 2>/dev/null | tr -d ' ')
                    
                    echo -e "${GREEN}✓${NC} ${BLUE}$service_name${NC} (PID: $pid)"
                    echo "    Status: ${GREEN}Running${NC}"
                    echo "    CPU: ${cpu}% | Memory: ${mem}% | Runtime: ${runtime}"
                    echo "    Command: ${cmd:0:80}..."
                    echo ""
                    found=1
                else
                    # Stale PID file
                    echo -e "${YELLOW}⚠${NC} ${BLUE}$service_name${NC} (PID: $pid)"
                    echo "    Status: ${YELLOW}Stale PID file${NC} (process not running)"
                    echo ""
                    found=1
                fi
            fi
        done
    fi
    
    # Check for legacy PID files in root directory
    for pid_file in "$SCRIPT_DIR"/*.pid; do
        if [ -f "$pid_file" ]; then
            local service_name=$(basename "$pid_file" .pid)
            local pid=$(cat "$pid_file" 2>/dev/null)
            
            if [ -n "$pid" ] && kill -0 "$pid" 2>/dev/null; then
                local cmd=$(ps -p "$pid" -o cmd= 2>/dev/null | head -1)
                local cpu=$(ps -p "$pid" -o %cpu= 2>/dev/null | tr -d ' ')
                local mem=$(ps -p "$pid" -o %mem= 2>/dev/null | tr -d ' ')
                local runtime=$(ps -p "$pid" -o etime= 2>/dev/null | tr -d ' ')
                
                echo -e "${GREEN}✓${NC} ${BLUE}$service_name${NC} (PID: $pid) [Legacy]"
                echo "    Status: ${GREEN}Running${NC}"
                echo "    CPU: ${cpu}% | Memory: ${mem}% | Runtime: ${runtime}"
                echo "    Command: ${cmd:0:80}..."
                echo ""
                found=1
            fi
        fi
    done
    
    # Check for Python processes that might be MANTIS services
    echo -e "${BLUE}Python MANTIS Processes:${NC}"
    local python_procs=$(ps aux | grep -E "python.*(train_model|miner|validator)" | grep -v grep)
    if [ -n "$python_procs" ]; then
        echo "$python_procs" | while read line; do
            local pid=$(echo "$line" | awk '{print $2}')
            local cmd=$(echo "$line" | awk '{for(i=11;i<=NF;i++) printf "%s ", $i; print ""}')
            local cpu=$(echo "$line" | awk '{print $3}')
            local mem=$(echo "$line" | awk '{print $4}')
            
            # Try to extract service name from command
            local service_name="unknown"
            if echo "$cmd" | grep -q "train_model"; then
                service_name="training"
            elif echo "$cmd" | grep -q "miner"; then
                service_name="mining"
            elif echo "$cmd" | grep -q "validator"; then
                service_name="validator"
            fi
            
            echo -e "${GREEN}✓${NC} ${BLUE}$service_name${NC} (PID: $pid)"
            echo "    Status: ${GREEN}Running${NC} (no PID file)"
            echo "    CPU: ${cpu}% | Memory: ${mem}%"
            echo "    Command: ${cmd:0:80}..."
            echo ""
        done
        found=1
    fi
    
    if [ $found -eq 0 ]; then
        echo -e "${YELLOW}No services found${NC}"
    fi
    
    echo -e "${BLUE}========================================${NC}"
}

# Remove specific service
remove() {
    local service_name=$1
    
    if [ -z "$service_name" ]; then
        echo -e "${RED}Error: Service name required${NC}"
        echo "Usage: $0 remove <service_name>"
        echo ""
        echo "Available services:"
        list
        return 1
    fi
    
    echo -e "${YELLOW}Removing service: $service_name${NC}"
    
    # Try to find PID file
    local pid_file=""
    local pid=""
    
    # Check in services directory
    if [ -f "$SERVICES_DIR/${service_name}.pid" ]; then
        pid_file="$SERVICES_DIR/${service_name}.pid"
        pid=$(cat "$pid_file" 2>/dev/null)
    # Check in root directory (legacy)
    elif [ -f "$SCRIPT_DIR/${service_name}.pid" ]; then
        pid_file="$SCRIPT_DIR/${service_name}.pid"
        pid=$(cat "$pid_file" 2>/dev/null)
    fi
    
    # If no PID file, try to find by process name
    if [ -z "$pid" ]; then
        case "$service_name" in
            training|train)
                pid=$(ps aux | grep -E "python.*train_model" | grep -v grep | awk '{print $2}' | head -1)
                ;;
            mining|miner)
                pid=$(ps aux | grep -E "python.*miner" | grep -v grep | awk '{print $2}' | head -1)
                ;;
            validator)
                pid=$(ps aux | grep -E "python.*validator" | grep -v grep | awk '{print $2}' | head -1)
                ;;
            *)
                echo -e "${RED}Error: Could not find service '$service_name'${NC}"
                echo "Use '$0 list' to see available services"
                return 1
                ;;
        esac
    fi
    
    if [ -z "$pid" ]; then
        echo -e "${YELLOW}Service '$service_name' not found or not running${NC}"
        # Clean up stale PID file if exists
        if [ -n "$pid_file" ] && [ -f "$pid_file" ]; then
            rm -f "$pid_file"
            echo -e "${GREEN}✓ Removed stale PID file${NC}"
        fi
        return 1
    fi
    
    # Check if process is running
    if ! kill -0 "$pid" 2>/dev/null; then
        echo -e "${YELLOW}Process $pid is not running${NC}"
        # Clean up stale PID file
        if [ -n "$pid_file" ] && [ -f "$pid_file" ]; then
            rm -f "$pid_file"
            echo -e "${GREEN}✓ Removed stale PID file${NC}"
        fi
        return 1
    fi
    
    # Stop the process
    echo -e "${YELLOW}Stopping process $pid...${NC}"
    kill -TERM "$pid" 2>/dev/null
    
    # Wait for process to stop
    local count=0
    while kill -0 "$pid" 2>/dev/null && [ $count -lt 10 ]; do
        sleep 1
        count=$((count + 1))
    done
    
    # Force kill if still running
    if kill -0 "$pid" 2>/dev/null; then
        echo -e "${YELLOW}Process did not stop, force killing...${NC}"
        kill -KILL "$pid" 2>/dev/null
        sleep 1
    fi
    
    # Remove PID file
    if [ -n "$pid_file" ] && [ -f "$pid_file" ]; then
        rm -f "$pid_file"
        echo -e "${GREEN}✓ Removed PID file${NC}"
    fi
    
    # Ask about log files
    local log_file=""
    local error_log=""
    
    if [ -f "$SCRIPT_DIR/logs/${service_name}_service.log" ]; then
        log_file="$SCRIPT_DIR/logs/${service_name}_service.log"
    elif [ -f "$SCRIPT_DIR/logs/${service_name}.log" ]; then
        log_file="$SCRIPT_DIR/logs/${service_name}.log"
    fi
    
    if [ -f "$SCRIPT_DIR/logs/${service_name}_service_error.log" ]; then
        error_log="$SCRIPT_DIR/logs/${service_name}_service_error.log"
    elif [ -f "$SCRIPT_DIR/logs/${service_name}_error.log" ]; then
        error_log="$SCRIPT_DIR/logs/${service_name}_error.log"
    fi
    
    if [ -n "$log_file" ] || [ -n "$error_log" ]; then
        read -p "Remove log files? (y/N): " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            [ -n "$log_file" ] && rm -f "$log_file" && echo -e "${GREEN}✓ Removed log file${NC}"
            [ -n "$error_log" ] && rm -f "$error_log" && echo -e "${GREEN}✓ Removed error log${NC}"
        fi
    fi
    
    echo -e "${GREEN}✓ Service '$service_name' removed${NC}"
}

# Stop all services
stop_all() {
    echo -e "${YELLOW}Stopping all MANTIS services...${NC}"
    echo ""
    
    # Get all PID files
    local pids=()
    
    if [ -d "$SERVICES_DIR" ]; then
        for pid_file in "$SERVICES_DIR"/*.pid; do
            if [ -f "$pid_file" ]; then
                local pid=$(cat "$pid_file" 2>/dev/null)
                if [ -n "$pid" ] && kill -0 "$pid" 2>/dev/null; then
                    pids+=("$pid")
                fi
            fi
        done
    fi
    
    # Also check legacy PID files
    for pid_file in "$SCRIPT_DIR"/*.pid; do
        if [ -f "$pid_file" ]; then
            local pid=$(cat "$pid_file" 2>/dev/null)
            if [ -n "$pid" ] && kill -0 "$pid" 2>/dev/null; then
                pids+=("$pid")
            fi
        fi
    done
    
    if [ ${#pids[@]} -eq 0 ]; then
        echo -e "${YELLOW}No running services found${NC}"
        return 0
    fi
    
    # Stop all processes
    for pid in "${pids[@]}"; do
        echo -e "${YELLOW}Stopping PID $pid...${NC}"
        kill -TERM "$pid" 2>/dev/null
    done
    
    # Wait
    sleep 2
    
    # Force kill any remaining
    for pid in "${pids[@]}"; do
        if kill -0 "$pid" 2>/dev/null; then
            echo -e "${YELLOW}Force killing PID $pid...${NC}"
            kill -KILL "$pid" 2>/dev/null
        fi
    done
    
    # Clean up PID files
    rm -f "$SERVICES_DIR"/*.pid "$SCRIPT_DIR"/*.pid 2>/dev/null
    
    echo -e "${GREEN}✓ All services stopped${NC}"
}

# Help
help() {
    echo "MANTIS Service Manager"
    echo ""
    echo "Usage: $0 {list|remove|stop-all|help} [service_name]"
    echo ""
    echo "Commands:"
    echo "  list              - List all running services"
    echo "  remove <name>     - Remove a specific service"
    echo "  stop-all          - Stop all running services"
    echo "  help              - Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0 list                    # List all services"
    echo "  $0 remove training         # Remove training service"
    echo "  $0 remove training_service # Remove training_service"
    echo "  $0 stop-all                # Stop all services"
}

# Main
case "${1:-help}" in
    list)
        list
        ;;
    remove)
        remove "$2"
        ;;
    stop-all)
        stop_all
        ;;
    help|--help|-h)
        help
        ;;
    *)
        echo -e "${RED}Unknown command: $1${NC}"
        echo ""
        help
        exit 1
        ;;
esac

