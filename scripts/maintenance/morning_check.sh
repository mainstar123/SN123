#!/bin/bash
# Morning Check Script - Run daily to verify system health

echo "=========================================="
echo "Morning Check - $(date)"
echo "=========================================="

cd /home/ocean/Nereus/SN123

# 1. Miner Status
echo ""
echo "1. Miner Status:"
if ps aux | grep "python.*miner.py" | grep -v grep > /dev/null; then
    PID=$(ps aux | grep "python.*miner.py" | grep -v grep | awk '{print $2}')
    echo "   ✅ Miner Running (PID: $PID)"
else
    echo "   ❌ MINER DOWN"
    echo "   To restart: nohup python neurons/miner.py [your flags] > logs/miner_\$(date +%Y%m%d).log 2>&1 &"
fi

# 2. Overnight Performance
echo ""
echo "2. Overnight Performance:"
if [ -f logs/miner.log ] || ls logs/miner_*.log 1> /dev/null 2>&1; then
    yesterday=$(date -d '1 day ago' +%Y-%m-%d 2>/dev/null || date -v-1d +%Y-%m-%d 2>/dev/null || echo "")
    if [ -n "$yesterday" ]; then
        overnight_avg=$(grep "$yesterday" logs/miner*.log 2>/dev/null | grep "Salience" | awk '{print $NF}' | awk '{sum+=$1; n++} END {if(n>0) printf "%.4f", sum/n; else print "N/A"}')
        prediction_count=$(grep "$yesterday" logs/miner*.log 2>/dev/null | grep -c "Salience")
        echo "   Average Salience: $overnight_avg"
        echo "   Predictions: $prediction_count"
    else
        echo "   (Date command not working, skipping)"
    fi
else
    echo "   No miner logs found yet"
fi

# 3. System Health
echo ""
echo "3. System Health:"
if command -v nvidia-smi &> /dev/null; then
    gpu_util=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits 2>/dev/null | head -1)
    echo "   GPU: ${gpu_util}%"
else
    echo "   GPU: Not available"
fi
disk_usage=$(df -h / | tail -1 | awk '{print $5}')
echo "   Disk: $disk_usage"
mem_usage=$(free | grep Mem | awk '{printf "%.0f%%", $3/$2 * 100}')
echo "   Memory: $mem_usage"

# 4. Errors Check
echo ""
echo "4. Recent Errors:"
today=$(date +%Y-%m-%d)
if [ -f logs/miner.log ] || ls logs/miner_*.log 1> /dev/null 2>&1; then
    error_count=$(grep "$today" logs/miner*.log 2>/dev/null | grep -ic "error" || echo "0")
    if [ "$error_count" -gt "0" ]; then
        echo "   ⚠️  $error_count errors today"
        echo "   Recent errors:"
        grep "$today" logs/miner*.log 2>/dev/null | grep -i "error" | tail -3
    else
        echo "   ✅ No errors today"
    fi
else
    echo "   No logs to check"
fi

echo ""
echo "=========================================="
echo "Morning check complete!"
echo "=========================================="

