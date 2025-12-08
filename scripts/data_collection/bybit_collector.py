"""
Bybit Data Collector
Collects OHLCV, funding rates, and open interest from Bybit API
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


class BybitCollector:
    """Collects data from Bybit API"""
    
    def __init__(self, api_key: Optional[str] = None, api_secret: Optional[str] = None):
        """
        Initialize Bybit collector
        
        Args:
            api_key: Bybit API key (optional for public data)
            api_secret: Bybit API secret (optional for public data)
        """
        self.base_url = "https://api.bybit.com"
        self.api_key = api_key
        self.api_secret = api_secret
        
    def _make_request(self, endpoint: str, params: dict = None) -> dict:
        """Make API request"""
        url = f"{self.base_url}{endpoint}"
        try:
            response = requests.get(url, params=params, timeout=30)
            response.raise_for_status()
            data = response.json()
            if data.get('retCode') == 0:
                return data.get('result', {})
            else:
                print(f"API Error: {data.get('retMsg')}")
                return {}
        except Exception as e:
            print(f"Request error: {e}")
            return {}
    
    def get_klines(
        self,
        symbol: str = "BTCUSDT",
        interval: str = "60",  # 60 = 1h
        start_time: Optional[str] = None,
        end_time: Optional[str] = None,
        limit: int = 200
    ) -> pd.DataFrame:
        """
        Get klines (OHLCV) data
        
        Args:
            symbol: Trading pair
            interval: Kline interval (60=1h, 240=4h, D=1d)
            start_time: Start time (YYYY-MM-DD or timestamp)
            end_time: End time (YYYY-MM-DD or timestamp)
            limit: Number of records (max 200)
            
        Returns:
            DataFrame with OHLCV data
        """
        params = {
            'category': 'linear',
            'symbol': symbol,
            'interval': interval,
            'limit': limit
        }
        
        if start_time:
            if isinstance(start_time, str) and '-' in start_time:
                params['start'] = int(datetime.strptime(start_time, "%Y-%m-%d").timestamp() * 1000)
            else:
                params['start'] = int(start_time)
        
        if end_time:
            if isinstance(end_time, str) and '-' in end_time:
                params['end'] = int(datetime.strptime(end_time, "%Y-%m-%d").timestamp() * 1000)
            else:
                params['end'] = int(end_time)
        
        all_data = []
        current_start = params.get('start')
        
        print(f"Downloading {symbol} {interval} data from Bybit...")
        
        while True:
            if current_start:
                params['start'] = current_start
            
            result = self._make_request('/v5/market/kline', params)
            
            if not result or 'list' not in result:
                break
            
            klines = result['list']
            if not klines:
                break
            
            # Convert to DataFrame
            df = pd.DataFrame(klines, columns=[
                'start_time', 'open', 'high', 'low', 'close', 'volume',
                'turnover'
            ])
            
            df['datetime'] = pd.to_datetime(df['start_time'].astype(int), unit='ms')
            df = df[['datetime', 'open', 'high', 'low', 'close', 'volume', 'turnover']]
            df = df.astype({
                'open': float, 'high': float, 'low': float, 'close': float,
                'volume': float, 'turnover': float
            })
            
            all_data.append(df)
            print(f"  Downloaded {len(df)} rows, total: {sum(len(d) for d in all_data)}")
            
            # Check if we need to paginate
            if len(klines) < limit:
                break
            
            # Update start time for next request
            current_start = int(df['start_time'].iloc[-1]) + 1
            
            # Check if we've reached end_time
            if end_time and current_start >= params.get('end', float('inf')):
                break
            
            time.sleep(0.2)  # Rate limiting
        
        if not all_data:
            return pd.DataFrame()
        
        df = pd.concat(all_data, ignore_index=True)
        df = df.sort_values('datetime').reset_index(drop=True)
        df = df.drop_duplicates(subset=['datetime']).reset_index(drop=True)
        
        return df
    
    def download_historical_klines(
        self,
        symbol: str = "BTCUSDT",
        interval: str = "60",
        start_date: str = "2018-01-01",
        end_date: Optional[str] = None,
        output_dir: str = "data/raw/bybit"
    ) -> pd.DataFrame:
        """
        Download historical klines with pagination
        
        Args:
            symbol: Trading pair
            interval: Kline interval
            start_date: Start date
            end_date: End date
            output_dir: Output directory
            
        Returns:
            DataFrame with OHLCV data
        """
        os.makedirs(output_dir, exist_ok=True)
        
        if end_date is None:
            end_date = datetime.now().strftime("%Y-%m-%d")
        
        # Download in chunks to avoid rate limits
        chunk_days = 30  # Download 30 days at a time
        start = datetime.strptime(start_date, "%Y-%m-%d")
        end = datetime.strptime(end_date, "%Y-%m-%d")
        
        all_data = []
        current = start
        
        while current < end:
            chunk_end = min(current + timedelta(days=chunk_days), end)
            
            print(f"Downloading {current.strftime('%Y-%m-%d')} to {chunk_end.strftime('%Y-%m-%d')}...")
            
            df_chunk = self.get_klines(
                symbol=symbol,
                interval=interval,
                start_time=current.strftime("%Y-%m-%d"),
                end_time=chunk_end.strftime("%Y-%m-%d"),
                limit=200
            )
            
            if not df_chunk.empty:
                all_data.append(df_chunk)
            
            current = chunk_end
            time.sleep(1)  # Rate limiting
        
        if not all_data:
            return pd.DataFrame()
        
        df = pd.concat(all_data, ignore_index=True)
        df = df.sort_values('datetime').reset_index(drop=True)
        df = df.drop_duplicates(subset=['datetime']).reset_index(drop=True)
        
        # Save to file
        output_file = os.path.join(output_dir, f"{symbol}_{interval}_{start_date}_{end_date}.csv")
        df.to_csv(output_file, index=False)
        print(f"\n✓ Saved {len(df)} rows to {output_file}")
        
        return df
    
    def get_funding_rate_history(
        self,
        symbol: str = "BTCUSDT",
        start_date: Optional[str] = None,
        limit: int = 200
    ) -> pd.DataFrame:
        """
        Get funding rate history
        
        Args:
            symbol: Trading pair
            start_date: Start date
            limit: Number of records (max 200)
            
        Returns:
            DataFrame with funding rates
        """
        params = {
            'category': 'linear',
            'symbol': symbol,
            'limit': limit
        }
        
        if start_date:
            params['startTime'] = int(datetime.strptime(start_date, "%Y-%m-%d").timestamp() * 1000)
        
        all_data = []
        current_start = params.get('startTime')
        
        print(f"Downloading funding rates for {symbol}...")
        
        while True:
            if current_start:
                params['startTime'] = current_start
            
            result = self._make_request('/v5/market/funding/history', params)
            
            if not result or 'list' not in result:
                break
            
            rates = result['list']
            if not rates:
                break
            
            df = pd.DataFrame(rates)
            if not df.empty:
                df['fundingRateTimestamp'] = pd.to_datetime(df['fundingRateTimestamp'].astype(int), unit='ms')
                df['fundingRate'] = df['fundingRate'].astype(float)
                df = df.rename(columns={'fundingRateTimestamp': 'datetime', 'fundingRate': 'funding_rate'})
                df = df[['datetime', 'funding_rate']].sort_values('datetime').reset_index(drop=True)
                all_data.append(df)
                
                print(f"  Downloaded {len(df)} rows, total: {sum(len(d) for d in all_data)}")
                
                # Paginate
                if len(rates) < limit:
                    break
                
                current_start = int(df['datetime'].iloc[-1].timestamp() * 1000) + 1
            else:
                break
            
            time.sleep(0.2)
        
        if not all_data:
            return pd.DataFrame()
        
        df = pd.concat(all_data, ignore_index=True)
        df = df.sort_values('datetime').reset_index(drop=True)
        df = df.drop_duplicates(subset=['datetime']).reset_index(drop=True)
        
        return df
    
    def get_open_interest(
        self,
        symbol: str = "BTCUSDT",
        interval: str = "60",  # 60 = 1h
        limit: int = 200
    ) -> pd.DataFrame:
        """
        Get open interest history
        
        Args:
            symbol: Trading pair
            interval: Interval (60=1h, 240=4h, D=1d)
            limit: Number of records
            
        Returns:
            DataFrame with open interest
        """
        params = {
            'category': 'linear',
            'symbol': symbol,
            'interval': interval,
            'limit': limit
        }
        
        result = self._make_request('/v5/market/open-interest', params)
        
        if not result or 'list' not in result:
            return pd.DataFrame()
        
        df = pd.DataFrame(result['list'])
        if not df.empty:
            df['openInterest'] = df['openInterest'].astype(float)
            if 'timestamp' in df.columns:
                df['datetime'] = pd.to_datetime(df['timestamp'].astype(int), unit='ms')
                df = df[['datetime', 'openInterest']].rename(columns={'openInterest': 'open_interest'})
                df = df.sort_values('datetime').reset_index(drop=True)
        
        return df


if __name__ == "__main__":
    collector = BybitCollector()
    
    # Download BTCUSDT 1h data
    print("=" * 60)
    print("Downloading BTCUSDT 1h data from Bybit")
    print("=" * 60)
    df = collector.download_historical_klines(
        symbol="BTCUSDT",
        interval="60",
        start_date="2020-01-01",
        end_date=None
    )
    
    print(f"\nDownloaded {len(df)} rows")
    if not df.empty:
        print(f"Date range: {df['datetime'].min()} to {df['datetime'].max()}")
        print(f"\nFirst few rows:")
        print(df.head())
        print(f"\nLast few rows:")
        print(df.tail())

