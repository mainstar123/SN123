#!/bin/bash
# Quick Start Script for MANTIS Mining Implementation

set -e

echo "=========================================="
echo "MANTIS SN123 Mining - Quick Start"
echo "=========================================="

# Step 1: Install dependencies
echo ""
echo "Step 1: Installing dependencies..."
echo "This may take a while due to large packages..."

# Increase pip timeout and install in batches
PIP_TIMEOUT=300  # 5 minutes timeout

# Core dependencies first (smaller packages)
echo "Installing core dependencies..."
pip install --timeout=$PIP_TIMEOUT --default-timeout=$PIP_TIMEOUT numpy pandas requests tqdm python-dotenv aiohttp

# Data collection dependencies
echo "Installing data collection dependencies..."
pip install --timeout=$PIP_TIMEOUT --default-timeout=$PIP_TIMEOUT yfinance python-binance

# ML dependencies (can be slow)
echo "Installing ML dependencies..."
pip install --timeout=$PIP_TIMEOUT --default-timeout=$PIP_TIMEOUT scikit-learn xgboost

# Deep learning (very large, may take time)
echo "Installing deep learning dependencies (this may take several minutes)..."
pip install --timeout=$PIP_TIMEOUT --default-timeout=$PIP_TIMEOUT tensorflow torch

# Bittensor and cloud storage
echo "Installing Bittensor and cloud storage..."
pip install --timeout=$PIP_TIMEOUT --default-timeout=$PIP_TIMEOUT bittensor boto3 botocore aiobotocore "cloudflare>=0.7.0"

# Additional packages
echo "Installing additional packages..."
pip install --timeout=$PIP_TIMEOUT --default-timeout=$PIP_TIMEOUT ccxt pywt matplotlib seaborn ipykernel

# Install additional mining requirements if file exists
if [ -f requirements_mining.txt ]; then
    echo "Installing mining-specific requirements..."
    pip install --timeout=$PIP_TIMEOUT --default-timeout=$PIP_TIMEOUT -r requirements_mining.txt
fi

# Try to install vmdpy (may fail, that's OK)
echo "Installing vmdpy (optional)..."
pip install --timeout=$PIP_TIMEOUT --default-timeout=$PIP_TIMEOUT vmdpy || echo "Warning: vmdpy installation failed, will use fallback"

# Step 2: Create directories
echo ""
echo "Step 2: Creating directory structure..."
mkdir -p data/{raw,processed}
mkdir -p models/{checkpoints,embeddings}
mkdir -p mining/payloads
mkdir -p logs

# Step 3: Data collection (optional - can be run separately)
echo ""
echo "Step 3: Data collection..."
read -p "Do you want to collect data now? (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "Starting data collection (this may take 1-2 hours)..."
    python scripts/data_collection/collect_all_data.py \
        --start-date 2019-09-01 \
        --output-dir data/raw
else
    echo "Skipping data collection. Run manually with:"
    echo "  python scripts/data_collection/collect_all_data.py --start-date 2019-09-01"
fi

# Step 4: Training (optional - can be run separately)
echo ""
echo "Step 4: Model training..."
read -p "Do you want to train models now? (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "Training models (this may take several hours)..."
    echo "Training ETH model as example..."
    python scripts/training/train_model.py \
        --ticker ETH \
        --data-dir data/raw \
        --train-start 2020-01-01 \
        --train-end 2023-12-31 \
        --val-start 2024-01-01 \
        --val-end 2024-12-31 \
        --epochs 100
else
    echo "Skipping training. Run manually with:"
    echo "  python scripts/training/train_model.py --ticker ETH --data-dir data/raw"
fi

echo ""
echo "=========================================="
echo "Quick Start Complete!"
echo "=========================================="
echo ""
echo "Next steps:"
echo "1. Collect data: python scripts/data_collection/collect_all_data.py"
echo "2. Train models: python scripts/training/train_model.py --ticker ETH"
echo "3. Backtest: python scripts/testing/backtest_accuracy.py --ticker ETH --model-path models/checkpoints/ETH"
echo "4. Start mining: python mining/miner.py --wallet.name your_wallet --wallet.hotkey your_hotkey"
echo ""
echo "See MINING_IMPLEMENTATION_GUIDE.md for detailed instructions."

