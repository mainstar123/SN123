#!/bin/bash
# Script to stop/kill background MANTIS processes

echo "=================================================================================="
echo "MANTIS - Stop Background Processes"
echo "=================================================================================="
echo ""

# PID files
TRAIN_PID_FILE="logs/training/train_all.pid"
TUNE_PID_FILE="logs/tuning/tuning.pid"
TUNE_PROBLEMATIC_PID_FILE="logs/tuning/tune_problematic.pid"

# Function to kill process by PID file
kill_by_pid_file() {
    local pid_file=$1
    local process_name=$2
    
    if [ -f "$pid_file" ]; then
        PID=$(cat "$pid_file")
        if ps -p "$PID" > /dev/null 2>&1; then
            echo "  Stopping $process_name (PID: $PID)..."
            kill "$PID" 2>/dev/null
            sleep 2
            if ps -p "$PID" > /dev/null 2>&1; then
                echo "    Process still running, force killing..."
                kill -9 "$PID" 2>/dev/null
            fi
            echo "    ✓ Stopped"
            rm -f "$pid_file"
        else
            echo "  $process_name: Process not running (stale PID file)"
            rm -f "$pid_file"
        fi
    else
        echo "  $process_name: No PID file found"
    fi
}

# Function to kill processes by name pattern
kill_by_pattern() {
    local pattern=$1
    local process_name=$2
    
    PIDS=$(ps aux | grep "$pattern" | grep -v grep | awk '{print $2}')
    if [ -n "$PIDS" ]; then
        echo "  Stopping $process_name processes..."
        for PID in $PIDS; do
            echo "    Killing PID: $PID"
            kill "$PID" 2>/dev/null
        done
        sleep 2
        # Force kill if still running
        PIDS=$(ps aux | grep "$pattern" | grep -v grep | awk '{print $2}')
        if [ -n "$PIDS" ]; then
            echo "    Force killing remaining processes..."
            for PID in $PIDS; do
                kill -9 "$PID" 2>/dev/null
            done
        fi
        echo "    ✓ Stopped"
    else
        echo "  $process_name: No processes found"
    fi
}

# Menu
echo "Select what to stop:"
echo "  1) Stop all MANTIS processes"
echo "  2) Stop training processes only"
echo "  3) Stop tuning processes only"
echo "  4) Stop specific process by PID"
echo "  5) Stop all Python processes (MANTIS related)"
echo ""

read -p "Select option (1-5): " -n 1 -r
echo ""

case $REPLY in
    1)
        echo ""
        echo "Stopping all MANTIS processes..."
        echo ""
        
        # Stop by PID files
        kill_by_pid_file "$TRAIN_PID_FILE" "Training"
        kill_by_pid_file "$TUNE_PID_FILE" "Tuning"
        kill_by_pid_file "$TUNE_PROBLEMATIC_PID_FILE" "Problematic Tuning"
        
        # Stop by pattern
        kill_by_pattern "train_model.py" "Training"
        kill_by_pattern "tune_all_challenges.py" "Tuning"
        ;;
    2)
        echo ""
        echo "Stopping training processes..."
        echo ""
        kill_by_pid_file "$TRAIN_PID_FILE" "Training"
        kill_by_pattern "train_model.py" "Training"
        ;;
    3)
        echo ""
        echo "Stopping tuning processes..."
        echo ""
        kill_by_pid_file "$TUNE_PID_FILE" "Tuning"
        kill_by_pid_file "$TUNE_PROBLEMATIC_PID_FILE" "Problematic Tuning"
        kill_by_pattern "tune_all_challenges.py" "Tuning"
        ;;
    4)
        read -p "Enter PID to kill: " PID
        if ps -p "$PID" > /dev/null 2>&1; then
            echo "Killing PID: $PID"
            kill "$PID" 2>/dev/null
            sleep 2
            if ps -p "$PID" > /dev/null 2>&1; then
                echo "Force killing PID: $PID"
                kill -9 "$PID" 2>/dev/null
            fi
            echo "✓ Stopped"
        else
            echo "✗ Process $PID not found"
        fi
        ;;
    5)
        echo ""
        echo "Stopping all MANTIS-related Python processes..."
        echo ""
        kill_by_pattern "python.*MANTIS" "MANTIS Python"
        kill_by_pattern "python.*training" "Training Python"
        kill_by_pattern "python.*tuning" "Tuning Python"
        ;;
    *)
        echo "Invalid option"
        exit 1
        ;;
esac

echo ""
echo "=================================================================================="
echo "Done!"
echo "=================================================================================="
echo ""
echo "To verify processes are stopped:"
echo "  ./check_background_processes.sh"
echo ""

