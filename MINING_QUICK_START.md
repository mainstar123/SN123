# MANTIS Mining Quick Start

This is a quick reference guide. For complete details, see [IMPLEMENTATION_GUIDE.md](IMPLEMENTATION_GUIDE.md).

## 🚀 Quick Start (5 Steps)

### 1. Install Dependencies

```bash
pip install -e .
pip install -r requirements_mining.txt
```

### 2. Download Data

```bash
# Download BTC data (example)
python scripts/data_collection/data_fetcher.py
```

### 3. Train Model

```bash
# Train BTC model
python scripts/training/train_model.py \
    --ticker BTC \
    --data-dir data/raw \
    --model-dir models/checkpoints \
    --epochs 100
```

### 4. Test Accuracy

```bash
# Verify >55% accuracy
python scripts/testing/backtest_accuracy.py \
    --ticker BTC \
    --model-dir models/checkpoints \
    --min-accuracy 0.55
```

### 5. Test Salience

```bash
# Verify top 10% salience
python scripts/testing/test_salience.py \
    --ticker BTC \
    --model-path models/checkpoints/BTC \
    --data-dir data/raw \
    --n-samples 10000
```

## 📋 Or Use Automated Script

```bash
./quick_start_mining.sh
```

## 🎯 Key Files

| File | Purpose |
|------|---------|
| `scripts/data_collection/data_fetcher.py` | Download free data (Binance, Bybit, yfinance) |
| `scripts/feature_engineering/feature_extractor.py` | VMD + TMFG + technical indicators |
| `scripts/training/model_architecture.py` | LSTM + XGBoost hybrid model |
| `scripts/training/train_model.py` | Training pipeline |
| `scripts/testing/backtest_accuracy.py` | Accuracy evaluation |
| `scripts/testing/test_salience.py` | Salience evaluation |
| `mining/miner.py` | Mining loop (generate + encrypt + upload) |

## 📊 Data Sources (All Free)

- **Binance**: Public data repository (2019+)
- **Bybit**: Public API (funding rates)
- **yfinance**: Forex & commodities

## 🎓 Model Architecture

1. **VMD**: Decompose prices into 8 IMF components
2. **TMFG**: Select top 10 features from 67+ indicators
3. **LSTM**: Temporal modeling (128 hidden, 2 layers, 20 timesteps)
4. **XGBoost**: Final embedding generation

## ✅ Pass Criteria

- **Accuracy**: >55% (95% CI lower bound)
- **Salience**: >90th percentile
- **Rank**: Top 10% of competitors

## 🔧 Common Issues

### VMD Import Error
```bash
pip install vmdpy
```

### Insufficient Data
```bash
# Fetch more data
python scripts/data_collection/data_fetcher.py
```

### Low Accuracy
- Increase training data (5+ years)
- Adjust `tmfg_n_features` (try 15-20)
- Add regularization

## 📚 Full Documentation

See [IMPLEMENTATION_GUIDE.md](IMPLEMENTATION_GUIDE.md) for:
- Detailed architecture explanation
- Feature engineering details
- Troubleshooting guide
- Mining setup instructions

## 🚦 Next Steps

1. ✅ Complete quick start
2. 📖 Read full implementation guide
3. 🎯 Train all tickers
4. 🧪 Test thoroughly
5. ⛏️ Start mining!

---

**Need Help?** Check [IMPLEMENTATION_GUIDE.md](IMPLEMENTATION_GUIDE.md) troubleshooting section.


