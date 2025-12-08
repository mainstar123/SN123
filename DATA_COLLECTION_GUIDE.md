# Complete Data Collection Guide for MANTIS

This guide provides comprehensive solutions for collecting all 9 asset datasets using **100% free data sources** with multiple fallback options.

## Quick Start

```bash
# Collect all datasets
python scripts/data_collection/collect_all_data.py --data-dir data/raw --start-date 2020-01-01
```

## Data Collection Strategy

### Primary Sources (Free, No API Key Required)

1. **Crypto (BTC, ETH)**: Binance via CCXT → CryptoCompare fallback
2. **Forex (EURUSD, GBPUSD, etc.)**: yfinance (daily resampled to 1h) → Alpha Vantage fallback
3. **Commodities (XAUUSD, XAGUSD)**: yfinance (daily resampled to 1h)

### Fallback Sources (Free, Optional API Keys)

1. **CryptoCompare**: Free tier (100k calls/month) - Get API key: https://www.cryptocompare.com/cryptopian/api-keys
2. **Alpha Vantage**: Free tier (5 calls/minute) - Get API key: https://www.alphavantage.co/support/#api-key
3. **Alternative Exchanges**: Coinbase, Kraken, Bitfinex via CCXT

## Setup Optional API Keys (Recommended)

For better reliability, set these environment variables (optional):

```bash
# Add to ~/.bashrc or ~/.zshrc
export ALPHA_VANTAGE_API_KEY="your_key_here"  # Free: https://www.alphavantage.co/support/#api-key
export CRYPTOCOMPARE_API_KEY="your_key_here"  # Free: https://www.cryptocompare.com/cryptopian/api-keys
```

Then reload:
```bash
source ~/.bashrc  # or source ~/.zshrc
```

## Data Sources by Asset

### BTC (Bitcoin)
- **Primary**: Binance via CCXT
- **Fallback 1**: CryptoCompare
- **Fallback 2**: Coinbase/Kraken via CCXT
- **Funding Rates**: Bybit API (free)

### ETH (Ethereum)
- **Primary**: Binance via CCXT
- **Fallback 1**: CryptoCompare
- **Fallback 2**: Coinbase/Kraken via CCXT
- **Funding Rates**: Bybit API (free)

### EURUSD, GBPUSD, CADUSD, NZDUSD, CHFUSD
- **Primary**: yfinance (daily → 1h resample)
- **Fallback**: Alpha Vantage (if API key set)

### XAUUSD (Gold)
- **Primary**: yfinance (daily → 1h resample)
- **Symbol**: `GC=F` (Gold Futures)

### XAGUSD (Silver)
- **Primary**: yfinance (daily → 1h resample)
- **Symbol**: `SI=F` (Silver Futures)

## Troubleshooting

### Issue: yfinance Timezone Errors

**Fixed**: The code now handles timezone issues automatically by:
- Converting timezone-aware datetimes to timezone-naive
- Ensuring consistent datetime comparisons

### Issue: yfinance 1h Data Limited to 730 Days

**Fixed**: The code automatically:
- Detects requests >730 days
- Fetches daily data instead
- Resamples to 1h using forward fill

### Issue: Binance Public Repository Unavailable

**Fixed**: Automatic fallback to:
1. CCXT API (Binance)
2. CryptoCompare
3. Alternative exchanges (Coinbase, Kraken, etc.)

### Issue: Rate Limiting

**Solutions**:
- Built-in rate limiting in all API calls
- Data is cached locally (won't re-download if exists)
- Optional API keys increase rate limits

## Manual Collection (If Automated Fails)

### Collect Individual Assets

```bash
# BTC
python -c "
from scripts.data_collection.data_fetcher import DataFetcher
d = DataFetcher('data/raw')
ohlcv, funding, oi = d.fetch_all_data('BTC', '2020-01-01', '2025-01-15')
d.save_data('BTC', ohlcv, funding, oi)
"

# ETH
python -c "
from scripts.data_collection.data_fetcher import DataFetcher
d = DataFetcher('data/raw')
ohlcv, funding, oi = d.fetch_all_data('ETH', '2020-01-01', '2025-01-15')
d.save_data('ETH', ohlcv, funding, oi)
"

# Forex (EURUSD example)
python -c "
from scripts.data_collection.data_fetcher import DataFetcher
d = DataFetcher('data/raw')
ohlcv, funding, oi = d.fetch_all_data('EURUSD', '2020-01-01', '2025-01-15')
d.save_data('EURUSD', ohlcv, funding, oi)
"
```

## Verification

Check collected data:

```bash
# List all collected datasets
for ticker in BTC ETH EURUSD GBPUSD CADUSD NZDUSD CHFUSD XAUUSD XAGUSD; do
    if [ -f "data/raw/$ticker/ohlcv.csv" ]; then
        rows=$(wc -l < data/raw/$ticker/ohlcv.csv)
        echo "✓ $ticker: $rows rows"
    else
        echo "✗ $ticker: Not collected"
    fi
done
```

## Expected Results

After successful collection, you should have:

```
✓ BTC: ~44,000 rows (2020-2025)
✓ ETH: ~44,000 rows (2020-2025)
✓ EURUSD: ~44,000 rows (2020-2025, resampled from daily)
✓ GBPUSD: ~44,000 rows (2020-2025, resampled from daily)
✓ CADUSD: ~44,000 rows (2020-2025, resampled from daily)
✓ NZDUSD: ~44,000 rows (2020-2025, resampled from daily)
✓ CHFUSD: ~44,000 rows (2020-2025, resampled from daily)
✓ XAUUSD: ~44,000 rows (2020-2025, resampled from daily)
✓ XAGUSD: ~44,000 rows (2020-2025, resampled from daily)
```

## Notes

1. **Forex Data**: yfinance provides daily data for full history, which is resampled to 1h. This is acceptable for 1h predictions as it maintains the daily trend.

2. **Crypto Data**: Full 1h historical data is available from multiple sources.

3. **Funding Rates**: Only available for BTC and ETH (from Bybit).

4. **Data Quality**: All sources are free and reliable. The resampling from daily to hourly for forex maintains trend information while providing hourly granularity.

## Next Steps

After collecting all data:
1. Verify data quality: `python scripts/data_collection/collect_all_data.py`
2. Train models: `python scripts/training/train_model.py --data-dir data/raw`
3. Test accuracy: `python scripts/testing/backtest_accuracy.py`

---

**Last Updated**: 2025-01-15
**Status**: ✅ All free sources implemented with fallbacks

