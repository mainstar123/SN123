#!/bin/bash

# Quick Start Script - Automates initial setup
# Execute this first, then follow EXECUTE_STEP_BY_STEP.md

set -e

echo "=========================================="
echo "  MANTIS First Place - Quick Start"
echo "=========================================="
echo ""

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Step 1: Install dependencies
echo -e "${YELLOW}[1/5] Installing dependencies...${NC}"
pip install --upgrade pip > /dev/null 2>&1
pip install optuna lightgbm scikit-learn pandas numpy scipy tensorflow xgboost > /dev/null 2>&1
echo -e "${GREEN}✓ Dependencies installed${NC}"
echo ""

# Step 2: Verify bug fix
echo -e "${YELLOW}[2/5] Verifying bug fix...${NC}"
if grep -q "np.percentile" scripts/training/model_architecture.py; then
    echo -e "${GREEN}✓ Bug fix confirmed${NC}"
else
    echo -e "${RED}✗ Bug fix not found!${NC}"
    echo "Please check scripts/training/model_architecture.py"
    exit 1
fi
echo ""

# Step 3: Create directories
echo -e "${YELLOW}[3/5] Creating directories...${NC}"
mkdir -p logs
mkdir -p models/baseline
mkdir -p models/tuned
mkdir -p models/ensemble
mkdir -p scripts/testing
echo -e "${GREEN}✓ Directories created${NC}"
echo ""

# Step 4: Check data
echo -e "${YELLOW}[4/5] Checking data files...${NC}"
if [ -d "data" ]; then
    data_count=$(ls data/*.csv 2>/dev/null | wc -l)
    if [ $data_count -gt 0 ]; then
        echo -e "${GREEN}✓ Found $data_count data files${NC}"
        echo "Data files:"
        ls -1 data/*.csv | head -5
    else
        echo -e "${RED}✗ No CSV files found in data/${NC}"
        echo "Please add OHLCV data files"
    fi
else
    echo -e "${RED}✗ data/ directory not found${NC}"
    mkdir -p data
    echo "Created data/ directory - please add data files"
fi
echo ""

# Step 5: Check if training script exists
echo -e "${YELLOW}[5/5] Checking training script...${NC}"
if [ -f "scripts/training/train_all_challenges.py" ]; then
    echo -e "${GREEN}✓ Training script exists${NC}"
else
    echo -e "${YELLOW}⚠ Training script not found${NC}"
    echo "TODO: Copy code from TRAIN_ALL_CHALLENGES_GUIDE.md to:"
    echo "      scripts/training/train_all_challenges.py"
fi
echo ""

# Summary
echo "=========================================="
echo "  SETUP COMPLETE!"
echo "=========================================="
echo ""
echo "Next steps:"
echo ""
echo "1. Create training script (if not done):"
echo "   Copy code from TRAIN_ALL_CHALLENGES_GUIDE.md"
echo "   to scripts/training/train_all_challenges.py"
echo ""
echo "2. Test single challenge:"
echo "   python scripts/training/train_all_challenges.py --single ETHUSDT"
echo ""
echo "3. Start full training:"
echo "   python scripts/training/train_all_challenges.py --tune --trials 30"
echo ""
echo "4. Follow EXECUTE_STEP_BY_STEP.md for complete guide"
echo ""
echo "Good luck! 🚀"
