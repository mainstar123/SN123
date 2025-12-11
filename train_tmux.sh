#!/bin/bash
#
# MANTIS Training - Tmux Session Runner
#
# This runs training in a detached tmux session
# Usage:
#   ./train_tmux.sh [options]
#

set -e

SESSION_NAME="mantis-training"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_DIR="$SCRIPT_DIR/logs/training"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

# Create log directory
mkdir -p "$LOG_DIR"

# Check if tmux is installed
if ! command -v tmux &> /dev/null; then
    echo "❌ tmux is not installed!"
    echo "   Install with: sudo apt-get install tmux"
    echo ""
    echo "   Or use: ./train_background.sh (doesn't require tmux)"
    exit 1
fi

# Check if session already exists
if tmux has-session -t "$SESSION_NAME" 2>/dev/null; then
    echo "⚠️  Training session already exists!"
    echo ""
    echo "Options:"
    echo "  1. Attach to existing session: tmux attach -t $SESSION_NAME"
    echo "  2. Kill existing session: tmux kill-session -t $SESSION_NAME"
    echo "  3. Use different session name: SESSION_NAME=mantis-2 $0"
    exit 1
fi

echo "=============================================="
echo "🚀 Starting MANTIS Training in Tmux"
echo "=============================================="
echo "Session name: $SESSION_NAME"
echo ""

# Create tmux session
tmux new-session -d -s "$SESSION_NAME" -c "$SCRIPT_DIR"

# Send commands to session
tmux send-keys -t "$SESSION_NAME" "source venv/bin/activate" Enter
tmux send-keys -t "$SESSION_NAME" "echo '============================================'" Enter
tmux send-keys -t "$SESSION_NAME" "echo '🚀 MANTIS Training Started'" Enter
tmux send-keys -t "$SESSION_NAME" "echo 'Time: \$(date)'" Enter
tmux send-keys -t "$SESSION_NAME" "echo '============================================'" Enter
tmux send-keys -t "$SESSION_NAME" "echo ''" Enter
tmux send-keys -t "$SESSION_NAME" "python scripts/training/train_all_challenges.py $@ 2>&1 | tee logs/training/training_${TIMESTAMP}.log" Enter

echo "✅ Training started in tmux session!"
echo ""
echo "📊 Attach to session (watch live):"
echo "   tmux attach -t $SESSION_NAME"
echo ""
echo "🔌 Detach from session:"
echo "   Press: Ctrl+B, then D"
echo ""
echo "📋 List all sessions:"
echo "   tmux list-sessions"
echo ""
echo "🛑 Stop training:"
echo "   tmux send-keys -t $SESSION_NAME C-c"
echo "   tmux kill-session -t $SESSION_NAME"
echo ""
echo "💡 The session will continue running even if you:"
echo "   - Close your terminal"
echo "   - Disconnect from SSH"
echo "   - Log out"
echo ""
echo "=============================================="

