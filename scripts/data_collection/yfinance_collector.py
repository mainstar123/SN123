"""
Yahoo Finance Data Collector
Collects forex and commodity data from yfinance
"""
import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Optional, List
import yfinance as yf
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


class YFinanceCollector:
    """Collects data from Yahoo Finance"""
    
    # Mapping of MANTIS tickers to Yahoo Finance symbols
    TICKER_MAP = {
        'EURUSD': 'EURUSD=X',
        'GBPUSD': 'GBPUSD=X',
        'CADUSD': 'CADUSD=X',
        'NZDUSD': 'NZDUSD=X',
        'CHFUSD': 'CHFUSD=X',
        'XAUUSD': 'GC=F',  # Gold futures
        'XAGUSD': 'SI=F',  # Silver futures
    }
    
    def __init__(self):
        """Initialize yfinance collector"""
        pass
    
    def download_historical_data(
        self,
        ticker: str,
        start_date: str = "2019-01-01",
        end_date: Optional[str] = None,
        interval: str = "1h",
        output_dir: str = "data/raw/yfinance"
    ) -> pd.DataFrame:
        """
        Download historical data from Yahoo Finance
        
        Args:
            ticker: MANTIS ticker (EURUSD, GBPUSD, etc.)
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format (default: today)
            interval: Data interval (1h, 1d, etc.)
            output_dir: Output directory
            
        Returns:
            DataFrame with OHLCV data
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # Map ticker to Yahoo Finance symbol
        yf_symbol = self.TICKER_MAP.get(ticker, ticker)
        
        if end_date is None:
            end_date = datetime.now().strftime("%Y-%m-%d")
        
        print(f"Downloading {ticker} ({yf_symbol}) from {start_date} to {end_date}...")
        
        try:
            # Download data
            ticker_obj = yf.Ticker(yf_symbol)
            df = ticker_obj.history(
                start=start_date,
                end=end_date,
                interval=interval,
                auto_adjust=True,
                prepost=False
            )
            
            if df.empty:
                print(f"  ✗ No data available for {ticker}")
                return pd.DataFrame()
            
            # Rename columns to match our format
            df = df.reset_index()
            df = df.rename(columns={
                'Date': 'datetime',
                'Open': 'open',
                'High': 'high',
                'Low': 'low',
                'Close': 'close',
                'Volume': 'volume'
            })
            
            # Select and reorder columns
            df = df[['datetime', 'open', 'high', 'low', 'close', 'volume']]
            df = df.astype({
                'open': float, 'high': float, 'low': float, 'close': float,
                'volume': float
            })
            
            # Remove duplicates and sort
            df = df.drop_duplicates(subset=['datetime']).sort_values('datetime').reset_index(drop=True)
            
            # Save to file
            output_file = os.path.join(output_dir, f"{ticker}_{interval}_{start_date}_{end_date}.csv")
            df.to_csv(output_file, index=False)
            print(f"  ✓ Saved {len(df)} rows to {output_file}")
            
            return df
            
        except Exception as e:
            print(f"  ✗ Error downloading {ticker}: {e}")
            return pd.DataFrame()
    
    def download_all_forex(
        self,
        start_date: str = "2019-01-01",
        end_date: Optional[str] = None,
        interval: str = "1h"
    ) -> dict:
        """
        Download all forex pairs
        
        Args:
            start_date: Start date
            end_date: End date
            interval: Data interval
            
        Returns:
            Dictionary of DataFrames keyed by ticker
        """
        forex_tickers = ['EURUSD', 'GBPUSD', 'CADUSD', 'NZDUSD', 'CHFUSD']
        results = {}
        
        for ticker in forex_tickers:
            print(f"\n{'='*60}")
            print(f"Downloading {ticker}")
            print('='*60)
            df = self.download_historical_data(
                ticker=ticker,
                start_date=start_date,
                end_date=end_date,
                interval=interval
            )
            if not df.empty:
                results[ticker] = df
        
        return results
    
    def download_all_commodities(
        self,
        start_date: str = "2019-01-01",
        end_date: Optional[str] = None,
        interval: str = "1h"
    ) -> dict:
        """
        Download all commodity data
        
        Args:
            start_date: Start date
            end_date: End date
            interval: Data interval
            
        Returns:
            Dictionary of DataFrames keyed by ticker
        """
        commodity_tickers = ['XAUUSD', 'XAGUSD']
        results = {}
        
        for ticker in commodity_tickers:
            print(f"\n{'='*60}")
            print(f"Downloading {ticker}")
            print('='*60)
            df = self.download_historical_data(
                ticker=ticker,
                start_date=start_date,
                end_date=end_date,
                interval=interval
            )
            if not df.empty:
                results[ticker] = df
        
        return results


if __name__ == "__main__":
    collector = YFinanceCollector()
    
    # Download EURUSD as example
    print("=" * 60)
    print("Downloading EURUSD 1h data from Yahoo Finance")
    print("=" * 60)
    df = collector.download_historical_data(
        ticker="EURUSD",
        start_date="2019-01-01",
        end_date=None,
        interval="1h"
    )
    
    if not df.empty:
        print(f"\nDownloaded {len(df)} rows")
        print(f"Date range: {df['datetime'].min()} to {df['datetime'].max()}")
        print(f"\nFirst few rows:")
        print(df.head())
        print(f"\nLast few rows:")
        print(df.tail())

