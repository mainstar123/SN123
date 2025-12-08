"""
Binance Data Collector
Collects OHLCV, funding rates, and other market data from Binance
"""
import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Optional, List
import requests
import time
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

try:
    from binance.client import Client
    from binance.exceptions import BinanceAPIException
except ImportError:
    print("Installing python-binance...")
    os.system("pip install python-binance")
    from binance.client import Client
    from binance.exceptions import BinanceAPIException


class BinanceCollector:
    """Collects data from Binance API"""
    
    def __init__(self, api_key: Optional[str] = None, api_secret: Optional[str] = None):
        """
        Initialize Binance client
        
        Args:
            api_key: Binance API key (optional for public data)
            api_secret: Binance API secret (optional for public data)
        """
        self.client = Client(api_key, api_secret) if api_key else Client()
        self.base_url = "https://data.binance.vision/data/futures/um/monthly/klines"
        
    def download_historical_klines(
        self,
        symbol: str = "BTCUSDT",
        interval: str = "1h",
        start_date: str = "2019-09-01",
        end_date: Optional[str] = None,
        output_dir: str = "data/raw/binance"
    ) -> pd.DataFrame:
        """
        Download historical klines from Binance historical data
        
        Args:
            symbol: Trading pair (e.g., BTCUSDT)
            interval: Kline interval (1h, 4h, 1d, etc.)
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format (default: today)
            output_dir: Directory to save data
            
        Returns:
            DataFrame with OHLCV data
        """
        os.makedirs(output_dir, exist_ok=True)
        
        if end_date is None:
            end_date = datetime.now().strftime("%Y-%m-%d")
            
        start = datetime.strptime(start_date, "%Y-%m-%d")
        end = datetime.strptime(end_date, "%Y-%m-%d")
        
        all_data = []
        current = start
        
        print(f"Downloading {symbol} {interval} data from {start_date} to {end_date}...")
        
        while current < end:
            year_month = current.strftime("%Y-%m")
            filename = f"{symbol}-{interval}-{year_month}.zip"
            url = f"{self.base_url}/{symbol}/{interval}/{filename}"
            
            try:
                print(f"Downloading {filename}...")
                response = requests.get(url, timeout=30)
                
                if response.status_code == 200:
                    # Save zip file
                    zip_path = os.path.join(output_dir, filename)
                    with open(zip_path, 'wb') as f:
                        f.write(response.content)
                    
                    # Extract and read CSV
                    import zipfile
                    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                        csv_name = filename.replace('.zip', '.csv')
                        zip_ref.extract(csv_name, output_dir)
                        csv_path = os.path.join(output_dir, csv_name)
                        
                        # Read CSV
                        df = pd.read_csv(
                            csv_path,
                            names=['open_time', 'open', 'high', 'low', 'close', 'volume',
                                   'close_time', 'quote_volume', 'trades', 'taker_buy_base',
                                   'taker_buy_quote', 'ignore']
                        )
                        df['datetime'] = pd.to_datetime(df['open_time'], unit='ms')
                        df = df[['datetime', 'open', 'high', 'low', 'close', 'volume',
                                'quote_volume', 'trades', 'taker_buy_base', 'taker_buy_quote']]
                        df = df.astype({
                            'open': float, 'high': float, 'low': float, 'close': float,
                            'volume': float, 'quote_volume': float, 'trades': int,
                            'taker_buy_base': float, 'taker_buy_quote': float
                        })
                        all_data.append(df)
                        print(f"  ✓ Loaded {len(df)} rows")
                    
                    # Clean up
                    os.remove(zip_path)
                    if os.path.exists(csv_path):
                        os.remove(csv_path)
                else:
                    print(f"  ✗ Not found: {filename}")
                    
            except Exception as e:
                print(f"  ✗ Error downloading {filename}: {e}")
            
            # Move to next month
            if current.month == 12:
                current = current.replace(year=current.year + 1, month=1)
            else:
                current = current.replace(month=current.month + 1)
            
            time.sleep(0.5)  # Rate limiting
        
        if not all_data:
            print("No data downloaded. Trying API method...")
            return self._download_via_api(symbol, interval, start_date, end_date, output_dir)
        
        # Combine all data
        df = pd.concat(all_data, ignore_index=True)
        df = df.sort_values('datetime').reset_index(drop=True)
        df = df.drop_duplicates(subset=['datetime']).reset_index(drop=True)
        
        # Save combined data
        output_file = os.path.join(output_dir, f"{symbol}_{interval}_{start_date}_{end_date}.csv")
        df.to_csv(output_file, index=False)
        print(f"\n✓ Saved {len(df)} rows to {output_file}")
        
        return df
    
    def _download_via_api(
        self,
        symbol: str,
        interval: str,
        start_date: str,
        end_date: str,
        output_dir: str
    ) -> pd.DataFrame:
        """Fallback: Download via API (slower, has limits)"""
        print("Using API method (slower, may hit rate limits)...")
        
        start_ts = int(datetime.strptime(start_date, "%Y-%m-%d").timestamp() * 1000)
        end_ts = int(datetime.strptime(end_date, "%Y-%m-%d").timestamp() * 1000)
        
        all_data = []
        current_ts = start_ts
        
        while current_ts < end_ts:
            try:
                klines = self.client.get_historical_klines(
                    symbol, interval, current_ts, limit=1000
                )
                
                if not klines:
                    break
                
                df = pd.DataFrame(klines, columns=[
                    'open_time', 'open', 'high', 'low', 'close', 'volume',
                    'close_time', 'quote_volume', 'trades', 'taker_buy_base',
                    'taker_buy_quote', 'ignore'
                ])
                
                df['datetime'] = pd.to_datetime(df['open_time'].astype(int), unit='ms')
                df = df[['datetime', 'open', 'high', 'low', 'close', 'volume',
                        'quote_volume', 'trades', 'taker_buy_base', 'taker_buy_quote']]
                df = df.astype({
                    'open': float, 'high': float, 'low': float, 'close': float,
                    'volume': float, 'quote_volume': float, 'trades': int,
                    'taker_buy_base': float, 'taker_buy_quote': float
                })
                
                all_data.append(df)
                current_ts = int(df['close_time'].iloc[-1]) + 1
                
                print(f"  Downloaded {len(df)} rows, total: {sum(len(d) for d in all_data)}")
                time.sleep(0.2)  # Rate limiting
                
            except BinanceAPIException as e:
                print(f"API Error: {e}")
                time.sleep(5)
                continue
        
        if not all_data:
            return pd.DataFrame()
        
        df = pd.concat(all_data, ignore_index=True)
        df = df.sort_values('datetime').reset_index(drop=True)
        df = df.drop_duplicates(subset=['datetime']).reset_index(drop=True)
        
        output_file = os.path.join(output_dir, f"{symbol}_{interval}_{start_date}_{end_date}.csv")
        df.to_csv(output_file, index=False)
        print(f"\n✓ Saved {len(df)} rows to {output_file}")
        
        return df
    
    def get_funding_rate_history(
        self,
        symbol: str = "BTCUSDT",
        start_date: Optional[str] = None,
        limit: int = 1000
    ) -> pd.DataFrame:
        """
        Get funding rate history
        
        Args:
            symbol: Trading pair
            start_date: Start date (optional)
            limit: Number of records (max 1000)
            
        Returns:
            DataFrame with funding rates
        """
        try:
            if start_date:
                start_ts = int(datetime.strptime(start_date, "%Y-%m-%d").timestamp() * 1000)
                funding_rates = self.client.futures_funding_rate(
                    symbol=symbol, startTime=start_ts, limit=limit
                )
            else:
                funding_rates = self.client.futures_funding_rate(symbol=symbol, limit=limit)
            
            df = pd.DataFrame(funding_rates)
            if not df.empty:
                df['fundingTime'] = pd.to_datetime(df['fundingTime'], unit='ms')
                df['fundingRate'] = df['fundingRate'].astype(float)
                df = df.rename(columns={'fundingTime': 'datetime', 'fundingRate': 'funding_rate'})
                df = df[['datetime', 'funding_rate']].sort_values('datetime').reset_index(drop=True)
            
            return df
            
        except Exception as e:
            print(f"Error fetching funding rates: {e}")
            return pd.DataFrame()
    
    def get_open_interest(
        self,
        symbol: str = "BTCUSDT",
        period: str = "5m",
        limit: int = 500
    ) -> pd.DataFrame:
        """
        Get open interest history
        
        Args:
            symbol: Trading pair
            period: Period (5m, 15m, 30m, 1h, 2h, 4h, 12h, 1d)
            limit: Number of records
            
        Returns:
            DataFrame with open interest
        """
        try:
            oi = self.client.futures_open_interest_hist(symbol=symbol, period=period, limit=limit)
            df = pd.DataFrame(oi)
            if not df.empty:
                df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
                df['sumOpenInterest'] = df['sumOpenInterest'].astype(float)
                df = df.rename(columns={'timestamp': 'datetime', 'sumOpenInterest': 'open_interest'})
                df = df[['datetime', 'open_interest']].sort_values('datetime').reset_index(drop=True)
            return df
        except Exception as e:
            print(f"Error fetching open interest: {e}")
            return pd.DataFrame()


if __name__ == "__main__":
    collector = BinanceCollector()
    
    # Download BTCUSDT 1h data
    print("=" * 60)
    print("Downloading BTCUSDT 1h data from Binance")
    print("=" * 60)
    df = collector.download_historical_klines(
        symbol="BTCUSDT",
        interval="1h",
        start_date="2019-09-01",
        end_date=None
    )
    
    print(f"\nDownloaded {len(df)} rows")
    print(f"Date range: {df['datetime'].min()} to {df['datetime'].max()}")
    print(f"\nFirst few rows:")
    print(df.head())
    print(f"\nLast few rows:")
    print(df.tail())

