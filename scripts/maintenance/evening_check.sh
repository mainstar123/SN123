#!/bin/bash
# Evening Check Script - Run daily to review performance

echo "=========================================="
echo "Evening Check - $(date)"
echo "=========================================="

cd /home/ocean/Nereus/SN123

# 1. Today's Performance
echo ""
echo "1. Today's Performance:"
today=$(date +%Y-%m-%d)

if [ -f logs/miner.log ] || ls logs/miner_*.log 1> /dev/null 2>&1; then
    today_avg=$(grep "$today" logs/miner*.log 2>/dev/null | grep "Salience" | awk '{print $NF}' | awk '{sum+=$1; n++} END {if(n>0) printf "%.4f", sum/n; else print "N/A"}')
    prediction_count=$(grep "$today" logs/miner*.log 2>/dev/null | grep -c "Salience")
    echo "   Average Salience: $today_avg"
    echo "   Predictions: $prediction_count"
else
    echo "   No miner logs found yet"
fi

# 2. Top Performers
echo ""
echo "2. Top Challenges Today:"
for challenge in ETH-LBFGS BTC-LBFGS-6H ETH-HITFIRST-100M; do
    if [ -f logs/miner.log ] || ls logs/miner_*.log 1> /dev/null 2>&1; then
        avg=$(grep "$today" logs/miner*.log 2>/dev/null | grep "$challenge" | grep "Salience" | awk '{print $NF}' | awk '{sum+=$1; n++} END {if(n>0) printf "%.4f", sum/n; else print "N/A"}')
        echo "   $challenge: $avg"
    fi
done

# 3. Week-to-Date
echo ""
echo "3. Week-to-Date Average:"
week_start=$(date -d 'last monday' +%Y-%m-%d 2>/dev/null || date -v-mon +%Y-%m-%d 2>/dev/null || echo "")
if [ -n "$week_start" ] && ([ -f logs/miner.log ] || ls logs/miner_*.log 1> /dev/null 2>&1); then
    week_avg=$(grep "$week_start" logs/miner*.log 2>/dev/null | grep "Salience" | awk '{print $NF}' | awk '{sum+=$1; n++} END {if(n>0) printf "%.4f", sum/n; else print "N/A"}')
    echo "   $week_avg (since $week_start)"
else
    echo "   (Unable to calculate)"
fi

# 4. Uptime
echo ""
echo "4. System Uptime:"
uptime_info=$(uptime -p 2>/dev/null || uptime | awk '{print $3,$4}')
echo "   $uptime_info"

echo ""
echo "=========================================="
echo "Evening check complete!"
echo "=========================================="

