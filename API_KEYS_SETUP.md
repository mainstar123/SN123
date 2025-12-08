# API Keys Setup Guide

Your API keys have been configured! Here's how to use them:

## Quick Setup (Current Session)

```bash
# Set API keys for current session
export ALPHA_VANTAGE_API_KEY="RWZWSSADTQOVK4T1"
export CRYPTOCOMPARE_API_KEY="a214d146a26d6d78fed6eb014c2a3a402fabee89ea992d46db9f9184162716df"
```

## Permanent Setup (Recommended)

Add to your `~/.bashrc` or `~/.zshrc`:

```bash
# Add these lines to ~/.bashrc or ~/.zshrc
export ALPHA_VANTAGE_API_KEY="RWZWSSADTQOVK4T1"
export CRYPTOCOMPARE_API_KEY="a214d146a26d6d78fed6eb014c2a3a402fabee89ea992d46db9f9184162716df"
```

Then reload:
```bash
source ~/.bashrc  # or source ~/.zshrc
```

## Using .env File (Alternative)

A `.env` file has been created in the project root with your keys. The code will automatically load it if `python-dotenv` is installed:

```bash
# Install python-dotenv (if not already installed)
pip install python-dotenv
```

## Test Your API Keys

```bash
# Test if API keys are working
python scripts/data_collection/test_api_keys.py
```

## Benefits of Using API Keys

1. **Alpha Vantage**: 
   - 5 calls/minute (vs 1 without key)
   - Better reliability for forex data
   - Access to more data endpoints

2. **CryptoCompare**:
   - 100,000 calls/month (free tier)
   - Better rate limits
   - More reliable crypto data

## Data Collection with API Keys

Once keys are set, run data collection:

```bash
# Make sure keys are set
export ALPHA_VANTAGE_API_KEY="RWZWSSADTQOVK4T1"
export CRYPTOCOMPARE_API_KEY="a214d146a26d6d78fed6eb014c2a3a402fabee89ea992d46db9f9184162716df"

# Collect all datasets
python scripts/data_collection/collect_all_data.py --data-dir data/raw --start-date 2020-01-01
```

The system will automatically use your API keys for better data collection reliability!

## Security Note

- Never commit `.env` file to git (it's in .gitignore)
- Never share your API keys publicly
- Keys are stored locally only

