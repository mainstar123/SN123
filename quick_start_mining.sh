#!/bin/bash
# Quick Start Script for MANTIS Mining Implementation

set -e

echo "=========================================="
echo "MANTIS Mining Quick Start"
echo "=========================================="

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Step 1: Install dependencies
echo -e "${YELLOW}Step 1: Installing dependencies...${NC}"
# Try editable install, fallback to regular install if it fails
pip install -e . 2>/dev/null || {
    echo -e "${YELLOW}Editable install failed, trying regular install...${NC}"
    pip install . || echo -e "${RED}Warning: pip install failed${NC}"
}
pip install -r requirements_mining.txt || echo -e "${RED}Warning: pip install requirements_mining.txt failed${NC}"

# Step 2: Create directories
echo -e "${YELLOW}Step 2: Creating directories...${NC}"
mkdir -p data/raw
mkdir -p models/checkpoints
mkdir -p test_storage

# Step 3: Download sample data (BTC)
echo -e "${YELLOW}Step 3: Downloading sample data (BTC)...${NC}"
python scripts/data_collection/data_fetcher.py || echo -e "${RED}Warning: Data download failed${NC}"

# Step 4: Train sample model (BTC)
echo -e "${YELLOW}Step 4: Training sample model (BTC)...${NC}"
echo -e "${YELLOW}This may take 30-60 minutes...${NC}"
python scripts/training/train_model.py \
    --ticker BTC \
    --data-dir data/raw \
    --model-dir models/checkpoints \
    --epochs 50 \
    --batch-size 64 || echo -e "${RED}Warning: Training failed${NC}"

# Step 5: Test accuracy
echo -e "${YELLOW}Step 5: Testing accuracy...${NC}"
python scripts/testing/backtest_accuracy.py \
    --ticker BTC \
    --model-dir models/checkpoints \
    --min-accuracy 0.55 || echo -e "${RED}Warning: Accuracy test failed${NC}"

# Step 6: Test salience
echo -e "${YELLOW}Step 6: Testing salience...${NC}"
python scripts/testing/test_salience.py \
    --ticker BTC \
    --model-path models/checkpoints/BTC \
    --data-dir data/raw \
    --n-samples 5000 \
    --n-competitors 5 || echo -e "${RED}Warning: Salience test failed${NC}"

echo ""
echo -e "${GREEN}=========================================="
echo "Quick Start Complete!"
echo "==========================================${NC}"
echo ""
echo "Next steps:"
echo "1. Review IMPLEMENTATION_GUIDE.md for detailed instructions"
echo "2. Train models for all tickers:"
echo "   python scripts/training/train_model.py --data-dir data/raw"
echo "3. Set up R2 storage and start mining:"
echo "   python mining/miner.py --wallet.name YOUR_WALLET --wallet.hotkey YOUR_HOTKEY"
echo ""


