"""
Prepare Training Data - Consolidate and Format Data for Training

This script:
1. Consolidates multi-file crypto data (ETHUSDT, BTCUSDT)
2. Formats forex data (EURUSD, XAUUSD, etc.)
3. Creates training-ready CSV files in data/ directory
"""

import os
import pandas as pd
import glob
from pathlib import Path

def consolidate_binance_data(ticker: str, output_path: str):
    """Consolidate multiple Binance CSV files into one"""
    
    pattern = f"data/raw/binance/{ticker}-1h-*.csv"
    files = sorted(glob.glob(pattern))
    
    if not files:
        print(f"  ✗ No files found for {ticker}")
        return False
    
    print(f"  📁 Found {len(files)} files for {ticker}")
    
    # Read and concatenate all files
    dfs = []
    for file in files:
        try:
            df = pd.read_csv(file)
            dfs.append(df)
        except Exception as e:
            print(f"  ⚠️  Error reading {file}: {e}")
    
    if not dfs:
        return False
    
    # Concatenate
    combined = pd.concat(dfs, ignore_index=True)
    
    # Ensure datetime column
    if 'timestamp' in combined.columns:
        combined['datetime'] = pd.to_datetime(combined['timestamp'], unit='ms')
    elif 'datetime' not in combined.columns and 'open_time' in combined.columns:
        combined['datetime'] = pd.to_datetime(combined['open_time'], unit='ms')
    
    # Sort by datetime
    if 'datetime' in combined.columns:
        combined = combined.sort_values('datetime')
        combined = combined.drop_duplicates(subset=['datetime'], keep='first')
    
    # Ensure required columns
    required_cols = ['open', 'high', 'low', 'close', 'volume']
    if all(col in combined.columns for col in required_cols):
        # Select and reorder columns
        output_cols = ['datetime'] + required_cols
        if 'datetime' in combined.columns:
            combined = combined[output_cols]
        
        # Save
        combined.to_csv(output_path, index=False)
        print(f"  ✓ Saved {len(combined)} rows to {output_path}")
        return True
    else:
        missing = [col for col in required_cols if col not in combined.columns]
        print(f"  ✗ Missing columns: {missing}")
        return False


def prepare_forex_data(ticker: str, output_path: str):
    """Prepare forex data from raw directory"""
    
    input_path = f"data/raw/{ticker}/ohlcv.csv"
    
    if not os.path.exists(input_path):
        print(f"  ✗ File not found: {input_path}")
        return False
    
    try:
        df = pd.read_csv(input_path)
        
        # Ensure datetime column
        if 'datetime' not in df.columns and 'timestamp' in df.columns:
            df['datetime'] = pd.to_datetime(df['timestamp'])
        elif 'datetime' in df.columns:
            df['datetime'] = pd.to_datetime(df['datetime'])
        
        # Check required columns
        required_cols = ['open', 'high', 'low', 'close', 'volume']
        if all(col in df.columns for col in required_cols):
            # Sort and deduplicate
            if 'datetime' in df.columns:
                df = df.sort_values('datetime')
                df = df.drop_duplicates(subset=['datetime'], keep='first')
            
            # Select columns
            output_cols = ['datetime'] + required_cols
            if 'datetime' in df.columns:
                df = df[output_cols]
            
            # Save
            df.to_csv(output_path, index=False)
            print(f"  ✓ Saved {len(df)} rows to {output_path}")
            return True
        else:
            missing = [col for col in required_cols if col not in df.columns]
            print(f"  ✗ Missing columns: {missing}")
            return False
            
    except Exception as e:
        print(f"  ✗ Error: {e}")
        return False


def main():
    """Prepare all training data"""
    
    print("="*80)
    print("📊 Preparing Training Data")
    print("="*80)
    print()
    
    # Create data directory if it doesn't exist
    os.makedirs("data", exist_ok=True)
    
    successful = 0
    failed = 0
    
    # Crypto data (from Binance multi-files)
    crypto_tickers = {
        'ETHUSDT': 'ETH_1h.csv',
        'BTCUSDT': 'BTC_1h.csv'
    }
    
    print("🔹 Consolidating Crypto Data:")
    print("-" * 80)
    for ticker, output_file in crypto_tickers.items():
        print(f"\n{ticker}:")
        output_path = f"data/{output_file}"
        if consolidate_binance_data(ticker, output_path):
            successful += 1
        else:
            failed += 1
    
    print("\n")
    print("🔹 Preparing Forex Data:")
    print("-" * 80)
    
    # Forex data (from raw single files)
    forex_tickers = {
        'EURUSD': 'EURUSD_1h.csv',
        'GBPUSD': 'GBPUSD_1h.csv',
        'CADUSD': 'CADUSD_1h.csv',
        'NZDUSD': 'NZDUSD_1h.csv',
        'CHFUSD': 'CHFUSD_1h.csv',
        'XAUUSD': 'XAUUSD_1h.csv',
        'XAGUSD': 'XAGUSD_1h.csv'
    }
    
    for ticker, output_file in forex_tickers.items():
        print(f"\n{ticker}:")
        output_path = f"data/{output_file}"
        if prepare_forex_data(ticker, output_path):
            successful += 1
        else:
            failed += 1
    
    # Summary
    print("\n")
    print("="*80)
    print("📊 Data Preparation Complete")
    print("="*80)
    print(f"✓ Successful: {successful}")
    print(f"✗ Failed: {failed}")
    print()
    
    if successful > 0:
        print("✅ Training data ready in data/ directory:")
        print()
        for file in sorted(glob.glob("data/*.csv")):
            size_mb = os.path.getsize(file) / (1024 * 1024)
            print(f"  {os.path.basename(file):30s} {size_mb:>8.2f} MB")
        print()
        print("🚀 Ready to train! Run:")
        print("   ./train_background.sh")
    else:
        print("⚠️  No data prepared. Check error messages above.")
    
    print("="*80)


if __name__ == '__main__':
    main()

