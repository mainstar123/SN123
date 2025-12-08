# Training Service Management Guide

Complete guide for managing the MANTIS training script as a background service.

## Quick Start

```bash
# Start service
./training_service.sh start

# View logs in real-time
./training_service.sh tail

# Check status
./training_service.sh status

# Stop service
./training_service.sh stop
```

## Commands

### Start Service
```bash
./training_service.sh start
```

Starts the training service in the background with default settings:
- Ticker: BTC
- Data dir: data/raw
- Model dir: models/checkpoints
- Epochs: 100
- Batch size: 64
- GPU: enabled

### Stop Service
```bash
./training_service.sh stop
```

Gracefully stops the training service (sends SIGTERM, then SIGKILL if needed).

### Restart Service
```bash
./training_service.sh restart
```

Stops and starts the service.

### Status
```bash
./training_service.sh status
```

Shows:
- Running status
- Process ID (PID)
- Start time
- CPU/Memory usage
- Configuration
- Log file locations

### View Logs
```bash
# View last 50 lines of output
./training_service.sh logs

# View last 50 lines of errors
./training_service.sh errors

# Follow output log in real-time
./training_service.sh tail

# Follow both logs in real-time
./training_service.sh tail-all
```

### Remove Service
```bash
./training_service.sh remove
```

Stops the service and optionally removes log files.

## Custom Configuration

Set environment variables before starting:

```bash
# Train ETH instead of BTC
TRAINING_TICKER=ETH ./training_service.sh start

# Use CPU instead of GPU
TRAINING_USE_GPU=false ./training_service.sh start

# Custom epochs and batch size
TRAINING_EPOCHS=50 TRAINING_BATCH_SIZE=32 ./training_service.sh start

# Full custom configuration
TRAINING_TICKER=ETH \
TRAINING_DATA_DIR=data/raw \
TRAINING_MODEL_DIR=models/checkpoints \
TRAINING_EPOCHS=100 \
TRAINING_BATCH_SIZE=64 \
TRAINING_USE_GPU=true \
./training_service.sh start
```

## Permanent Configuration

Add to your `~/.bashrc` or `~/.zshrc`:

```bash
# MANTIS Training Service Configuration
export TRAINING_TICKER=BTC
export TRAINING_DATA_DIR=/home/ocean/MANTIS/data/raw
export TRAINING_MODEL_DIR=/home/ocean/MANTIS/models/checkpoints
export TRAINING_EPOCHS=100
export TRAINING_BATCH_SIZE=64
export TRAINING_USE_GPU=true
```

## File Locations

- **PID File**: `training_service.pid`
- **Output Log**: `logs/training_service.log`
- **Error Log**: `logs/training_service_error.log`

## Examples

### Train BTC Model
```bash
./training_service.sh start
./training_service.sh tail  # Watch progress
```

### Train All Tickers Sequentially
```bash
for ticker in BTC ETH EURUSD GBPUSD; do
    echo "Training $ticker..."
    TRAINING_TICKER=$ticker ./training_service.sh start
    
    # Wait for completion (check status)
    while ./training_service.sh status | grep -q RUNNING; do
        sleep 60
    done
    
    echo "$ticker training complete!"
done
```

### Monitor Training Progress
```bash
# Start training
./training_service.sh start

# In another terminal, watch logs
./training_service.sh tail

# Or check status periodically
watch -n 30 './training_service.sh status'
```

## Troubleshooting

### Service Won't Start
```bash
# Check error log
./training_service.sh errors

# Check if port/process is already in use
ps aux | grep train_model.py
```

### Service Won't Stop
```bash
# Force kill
PID=$(cat training_service.pid)
kill -KILL $PID
rm -f training_service.pid
```

### View Full Logs
```bash
# View entire log file
cat logs/training_service.log

# Search for errors
grep -i error logs/training_service.log
grep -i error logs/training_service_error.log
```

### Check if Service is Running
```bash
./training_service.sh status

# Or manually
if [ -f training_service.pid ]; then
    PID=$(cat training_service.pid)
    ps -p $PID
fi
```

## Integration with systemd (Optional)

For system-level service management, create `/etc/systemd/system/mantis-training.service`:

```ini
[Unit]
Description=MANTIS Training Service
After=network.target

[Service]
Type=simple
User=ocean
WorkingDirectory=/home/ocean/MANTIS
Environment="TRAINING_TICKER=BTC"
Environment="TRAINING_USE_GPU=true"
ExecStart=/home/ocean/MANTIS/training_service.sh start
ExecStop=/home/ocean/MANTIS/training_service.sh stop
Restart=on-failure

[Install]
WantedBy=multi-user.target
```

Then:
```bash
sudo systemctl daemon-reload
sudo systemctl enable mantis-training
sudo systemctl start mantis-training
sudo systemctl status mantis-training
```

---

**Last Updated**: 2025-01-15

