# MANTIS SN123 Mining Implementation Guide

Complete implementation guide for first-place mining strategy on MANTIS subnet 123.

## Table of Contents

1. [Overview](#overview)
2. [Directory Structure](#directory-structure)
3. [Setup](#setup)
4. [Data Collection](#data-collection)
5. [Model Training](#model-training)
6. [Local Testing](#local-testing)
7. [Mining Workflow](#mining-workflow)
8. [Pre-Registration Checklist](#pre-registration-checklist)

## Overview

This implementation provides a complete mining system for MANTIS SN123 using:

- **Hybrid VMD-TMFG-LSTM + XGBoost** model architecture
- **Orthogonal signal generation** via OI/funding rate interactions and cross-exchange features
- **5+ years of historical data** from Binance, Bybit, and Yahoo Finance
- **Local testing infrastructure** for salience evaluation
- **Automated mining workflow** with encryption and R2 upload

### Key Features

- **VMD (Variational Mode Decomposition)**: Decomposes prices into 8 IMF components
- **TMFG Feature Selection**: Reduces 67 features to 10 most important
- **LSTM Temporal Modeling**: 2-layer LSTM with 128 hidden units
- **XGBoost Ensemble**: Binary classification with probability calibration
- **Orthogonal Signals**: OI/funding interactions, cross-exchange arbitrage signals

## Directory Structure

```
MANTIS/
├── data/
│   ├── raw/              # Raw collected data
│   │   ├── binance/      # Binance OHLCV, funding rates
│   │   ├── bybit/        # Bybit OHLCV, funding rates
│   │   └── yfinance/     # Forex and commodities
│   └── processed/        # Processed features
├── models/
│   ├── checkpoints/      # Trained models (one per ticker)
│   └── embeddings/       # Generated embeddings
├── scripts/
│   ├── data_collection/  # Data collection scripts
│   ├── training/         # Model training scripts
│   └── testing/          # Testing and evaluation
├── mining/               # Mining workflow
├── tests/                # Unit and integration tests
└── logs/                 # Log files
```

## Setup

### 1. Install Dependencies

```bash
# Install base dependencies
pip install -e .

# Install additional dependencies
pip install python-binance vmdpy tensorflow xgboost scikit-learn
```

### 2. Environment Variables

Create a `.env` file (optional, for R2 credentials):

```bash
R2_BUCKET=your-bucket-name
R2_ACCESS_KEY=your-access-key
R2_SECRET_KEY=your-secret-key
R2_ENDPOINT=https://<account-id>.r2.cloudflarestorage.com
```

### 3. Verify Setup

```bash
python -c "import tensorflow as tf; import xgboost; print('Setup OK')"
```

## Data Collection

### Step 1: Collect All Data

Run the master data collection script:

```bash
python scripts/data_collection/collect_all_data.py \
    --start-date 2019-09-01 \
    --end-date 2025-01-01 \
    --output-dir data/raw
```

This will:
- Download BTCUSDT and ETHUSDT 1h data from Binance (2019-09-01 to present)
- Download funding rates from Binance
- Download BTCUSDT data from Bybit (for cross-exchange features)
- Download funding rates from Bybit
- Download forex pairs (EURUSD, GBPUSD, CADUSD, NZDUSD, CHFUSD) from Yahoo Finance
- Download commodities (XAUUSD, XAGUSD) from Yahoo Finance

**Expected time**: 1-2 hours depending on internet speed

### Step 2: Verify Data

Check that data files were created:

```bash
ls -lh data/raw/binance/*.csv
ls -lh data/raw/bybit/*.csv
ls -lh data/raw/yfinance/*.csv
```

## Model Training

### Step 1: Train Model for Each Challenge

Train models for each ticker in `config.CHALLENGES`:

```bash
# Train BTC model (for ETH challenge, uses BTC price_key)
python scripts/training/train_model.py \
    --ticker BTC \
    --data-dir data/raw \
    --train-start 2020-01-01 \
    --train-end 2023-12-31 \
    --val-start 2024-01-01 \
    --val-end 2024-12-31 \
    --epochs 100 \
    --batch-size 64

# Train ETH model
python scripts/training/train_model.py \
    --ticker ETH \
    --data-dir data/raw \
    --train-start 2020-01-01 \
    --train-end 2023-12-31 \
    --val-start 2024-01-01 \
    --val-end 2024-12-31 \
    --epochs 100 \
    --batch-size 64

# Train forex pairs
for ticker in EURUSD GBPUSD CADUSD NZDUSD CHFUSD XAUUSD XAGUSD; do
    python scripts/training/train_model.py \
        --ticker $ticker \
        --data-dir data/raw \
        --train-start 2020-01-01 \
        --train-end 2023-12-31 \
        --val-start 2024-01-01 \
        --val-end 2024-12-31 \
        --epochs 100 \
        --batch-size 64
done
```

**Expected time**: 2-4 hours per model (depending on GPU)

### Step 2: Verify Training Results

Check training logs for accuracy:

- **Target**: Validation accuracy > 55%
- **Ideal**: Validation accuracy > 57%

Models are saved to `models/checkpoints/{TICKER}/`

## Local Testing

### Step 1: Backtest Accuracy

Test model accuracy on held-out 2025 data:

```bash
python scripts/testing/backtest_accuracy.py \
    --ticker ETH \
    --model-path models/checkpoints/ETH \
    --test-start 2025-01-01 \
    --test-end 2025-06-30 \
    --data-dir data/raw
```

**Pass Criteria**:
- Accuracy > 55%
- 95% CI lower bound > 55%

### Step 2: Test Salience (Optional)

Create a test script to evaluate salience:

```python
from scripts.testing.test_salience import test_model_salience
from scripts.training.model_architecture import VMDTMFGLSTMXGBoost
from scripts.training.train_model import load_data
import numpy as np

# Load model
model = VMDTMFGLSTMXGBoost(embedding_dim=2)
model.load("models/checkpoints/ETH")

# Load test data
df, funding, oi, cross_exchange = load_data("ETH", "data/raw", "2024-01-01", "2024-12-31")
X_seq, y_delta, y_binary = model.prepare_data(df, funding, oi, cross_exchange)

# Test salience
stats, scores = test_model_salience(
    model, (X_seq, y_delta, y_binary), df['close'].values, "ETH", 2
)

print(f"Your Salience Score: {stats['your_score']:.4f}")
print(f"90th Percentile: {stats['percentile_90']:.4f}")
print(f"Rank: {stats['rank']}/{stats['total_miners']}")
```

**Target**: Salience score > 90th percentile

## Mining Workflow

### Step 1: Setup R2 Bucket

1. Create Cloudflare R2 bucket
2. Configure CORS for public access
3. Get access keys
4. Set environment variables or pass as arguments

### Step 2: Register on Subnet

```bash
# Register hotkey on subnet 123
btcli subnet register --netuid 123 --wallet.name your_wallet --wallet.hotkey your_hotkey
```

### Step 3: Run Miner

#### Option A: Run Once (Test)

```bash
python mining/miner.py \
    --wallet.name your_wallet \
    --wallet.hotkey your_hotkey \
    --model-dir models/checkpoints \
    --data-dir data/raw \
    --r2-bucket your-bucket \
    --r2-access-key your-key \
    --r2-secret-key your-secret \
    --r2-endpoint https://<account-id>.r2.cloudflarestorage.com \
    --commit \
    --once
```

#### Option B: Run Continuously

```bash
python mining/miner.py \
    --wallet.name your_wallet \
    --wallet.hotkey your_hotkey \
    --model-dir models/checkpoints \
    --data-dir data/raw \
    --r2-bucket your-bucket \
    --r2-access-key your-key \
    --r2-secret-key your-secret \
    --r2-endpoint https://<account-id>.r2.cloudflarestorage.com \
    --commit \
    --interval 60
```

The miner will:
1. Generate embeddings for all challenges
2. Encrypt with time-lock (30 seconds)
3. Upload to R2
4. Commit URL to subtensor (once, or when URL changes)
5. Repeat every 60 seconds

### Step 4: Monitor

Check logs for:
- Embedding generation success
- Upload success
- Any errors

## Pre-Registration Checklist

### Technical Readiness

- [ ] Models trained for all challenges in `config.CHALLENGES`
- [ ] Validation accuracy > 55% for all models
- [ ] Backtest accuracy (2025 H1) > 55% with 95% CI lower bound > 55%
- [ ] Salience score > 90th percentile (if tested)
- [ ] Embedding generation pipeline works end-to-end
- [ ] Encryption/decryption tested
- [ ] R2 upload tested (payload size < 25MB)
- [ ] URL commit tested

### Infrastructure

- [ ] Data pipeline running (Binance/Bybit APIs)
- [ ] Cloud VM provisioned (4+ CPU, 16GB RAM)
- [ ] Bittensor wallet funded (>100 TAO)
- [ ] R2 bucket configured with CORS
- [ ] Automated retraining schedule (weekly)

### Competitive Positioning

- [ ] Feature set includes OI/funding interactions
- [ ] Cross-exchange features implemented (BTC)
- [ ] Model differentiation validated (correlation < 0.3 with random)
- [ ] Single UID strategy (no sybil)

### Financial Planning

- [ ] Registration cost budgeted (~5 TAO)
- [ ] Expected emissions calculated (0.75% share ≈ 54 TAO/day)
- [ ] Break-even timeline understood
- [ ] Operating capital secured (3+ months)

## Troubleshooting

### Data Collection Issues

**Problem**: Binance historical data download fails
**Solution**: Use API method fallback (slower but works)

**Problem**: Yahoo Finance data missing
**Solution**: Check ticker mapping in `yfinance_collector.py`

### Training Issues

**Problem**: Out of memory during training
**Solution**: Reduce batch size or sequence length

**Problem**: Low accuracy (< 55%)
**Solution**: 
- Check data quality (no gaps, proper alignment)
- Increase training epochs
- Tune hyperparameters (LSTM hidden units, dropout)

### Mining Issues

**Problem**: R2 upload fails
**Solution**: Check credentials, bucket name, CORS settings

**Problem**: Embeddings are all zeros
**Solution**: Check model loading, data availability

## Performance Targets

### Accuracy Targets

- **Minimum**: 55% (pass threshold)
- **Target**: 57-60% (competitive)
- **Ideal**: > 60% (top tier)

### Salience Targets

- **Minimum**: 80th percentile
- **Target**: 90th percentile
- **Ideal**: 95th+ percentile

### Expected Rewards

- **Current emissions**: ~54 TAO/day (0.75% subnet share)
- **Top 10% miners**: ~5.4 TAO/day
- **First place**: 10-15% of total emissions

## Next Steps

1. **Collect data** (1-2 hours)
2. **Train models** (1-2 days for all challenges)
3. **Backtest** (verify > 55% accuracy)
4. **Test locally** (salience evaluation)
5. **Register on subnet** (commit URL)
6. **Start mining** (continuous loop)

## Support

For issues or questions:
- Check MANTIS GitHub: https://github.com/Barbariandev/MANTIS
- Review `MINER_GUIDE.md` in repository
- Check validator code in `validator.py` and `model.py`

## License

MIT License - See LICENSE file in repository

