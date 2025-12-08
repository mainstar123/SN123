"""
Free Data Collection Module for MANTIS Mining
Supports Binance, Bybit (via CCXT), and yfinance for historical data
"""

import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Optional, Tuple, Dict
import time
import requests
from pathlib import Path
import yfinance as yf
import ccxt
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# Optional: Add API keys for additional free services
ALPHA_VANTAGE_API_KEY = os.getenv('ALPHA_VANTAGE_API_KEY', None)  # Free: https://www.alphavantage.co/support/#api-key
CRYPTOCOMPARE_API_KEY = os.getenv('CRYPTOCOMPARE_API_KEY', None)  # Free tier available


class DataFetcher:
    """Free data collection from multiple sources"""
    
    def __init__(self, data_dir: str = "data/raw"):
        """
        Initialize data fetcher
        
        Args:
            data_dir: Directory to store downloaded data
        """
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize exchanges
        self.binance = ccxt.binance({
            'enableRateLimit': True,
            'options': {'defaultType': 'future'}  # For perpetual futures
        })
        self.bybit = ccxt.bybit({
            'enableRateLimit': True,
            'options': {'defaultType': 'swap'}  # For perpetual swaps
        })
        
    def _get_ticker_mapping(self, ticker: str) -> Dict[str, str]:
        """
        Map MANTIS tickers to exchange symbols
        
        Args:
            ticker: MANTIS ticker symbol
            
        Returns:
            Dictionary with exchange symbols
        """
        mapping = {
            'BTC': {'binance': 'BTC/USDT:USDT', 'bybit': 'BTC/USDT:USDT', 'yfinance': None},
            'ETH': {'binance': 'ETH/USDT:USDT', 'bybit': 'ETH/USDT:USDT', 'yfinance': None},
            'EURUSD': {'binance': None, 'bybit': None, 'yfinance': 'EURUSD=X'},
            'GBPUSD': {'binance': None, 'bybit': None, 'yfinance': 'GBPUSD=X'},
            'CADUSD': {'binance': None, 'bybit': None, 'yfinance': 'CADUSD=X'},
            'NZDUSD': {'binance': None, 'bybit': None, 'yfinance': 'NZDUSD=X'},
            'CHFUSD': {'binance': None, 'bybit': None, 'yfinance': 'CHFUSD=X'},
            'XAUUSD': {'binance': None, 'bybit': None, 'yfinance': 'GC=F'},  # Gold futures
            'XAGUSD': {'binance': None, 'bybit': None, 'yfinance': 'SI=F'},  # Silver futures
        }
        return mapping.get(ticker, {})
    
    def fetch_binance_historical(self, symbol: str, start_date: str, end_date: str, 
                                 timeframe: str = '1h', use_ccxt_fallback: bool = True) -> pd.DataFrame:
        """
        Fetch historical data from Binance public data repository (FREE)
        
        Args:
            symbol: Symbol like 'BTCUSDT'
            start_date: Start date 'YYYY-MM-DD'
            end_date: End date 'YYYY-MM-DD'
            timeframe: '1h', '1d', etc.
            
        Returns:
            DataFrame with OHLCV data
        """
        print(f"Fetching Binance historical data for {symbol} ({timeframe})...")
        print(f"  Attempting Binance public data repository (will fallback to CCXT API if needed)...")
        
        # Binance public data repository
        # Format: https://data.binance.vision/data/futures/um/monthly/klines/{SYMBOL}/{TIMEFRAME}/{SYMBOL}-{TIMEFRAME}-{YEAR}-{MONTH}.zip
        base_url = "https://data.binance.vision/data/futures/um/monthly/klines"
        
        start = pd.to_datetime(start_date)
        end = pd.to_datetime(end_date)
        
        all_data = []
        success_count = 0
        
        # Generate monthly date ranges
        current = start
        while current <= end:
            year_month = current.strftime('%Y-%m')
            year = current.strftime('%Y')
            month = current.strftime('%m')
            
            # Try different filename formats
            filename_formats = [
                f"{symbol}-{timeframe}-{year_month}.zip",  # BTCUSDT-1h-2020-01.zip
                f"{symbol}-{timeframe}-{year}-{month}.zip",  # BTCUSDT-1h-2020-01.zip (alternative)
            ]
            
            downloaded = False
            for filename in filename_formats:
                url = f"{base_url}/{symbol}/{timeframe}/{filename}"
                
                try:
                    # Try to download
                    response = requests.get(url, timeout=30, allow_redirects=True)
                    if response.status_code == 200 and len(response.content) > 1000:  # Check if actually got data
                        # Save to temp file
                        temp_path = self.data_dir / f"temp_{filename}"
                        with open(temp_path, 'wb') as f:
                            f.write(response.content)
                        
                        # Read CSV from zip
                        df = pd.read_csv(temp_path, compression='zip', header=None)
                        df.columns = ['open_time', 'open', 'high', 'low', 'close', 'volume',
                                     'close_time', 'quote_volume', 'trades', 'taker_buy_base',
                                     'taker_buy_quote', 'ignore']
                        
                        # Convert timestamps
                        df['datetime'] = pd.to_datetime(df['open_time'], unit='ms')
                        df = df[['datetime', 'open', 'high', 'low', 'close', 'volume',
                                'taker_buy_base', 'taker_buy_quote']].copy()
                        
                        # Convert to float
                        for col in ['open', 'high', 'low', 'close', 'volume', 
                                   'taker_buy_base', 'taker_buy_quote']:
                            df[col] = pd.to_numeric(df[col], errors='coerce')
                        
                        all_data.append(df)
                        print(f"  ✓ Downloaded {year_month}: {len(df)} candles")
                        success_count += 1
                        downloaded = True
                        
                        # Clean up
                        temp_path.unlink()
                        break  # Success, no need to try other formats
                        
                except Exception as e:
                    # Try next format
                    continue
            
            if not downloaded:
                # Suppress verbose "Not available" messages - CCXT fallback will handle it
                # Only show first few to indicate we're trying
                if success_count == 0 and current <= start + pd.Timedelta(days=90):
                    print(f"  ⚠ Public repo not available for {year_month}, will use CCXT API...")
            
            # Move to next month
            if current.month == 12:
                current = current.replace(year=current.year + 1, month=1)
            else:
                current = current.replace(month=current.month + 1)
            
            time.sleep(0.5)  # Rate limiting
        
        if not all_data or success_count < 3:  # If we got less than 3 months, use CCXT
            # Fallback to CCXT API (works reliably for all data)
            if all_data:
                print(f"  Got {success_count} months from public repo, supplementing with CCXT API...")
            else:
                print(f"  Public repository unavailable, using CCXT API (this is normal and works fine)...")
            # Construct CCXT symbol format
            if not "/" in symbol:
                ccxt_symbol = f"{symbol}/USDT:USDT"
            else:
                ccxt_symbol = symbol
            
            # Get data from CCXT
            try:
                ccxt_data = self._fetch_ccxt_historical(self.binance, ccxt_symbol, start_date, end_date, timeframe)
            except Exception as e:
                print(f"  CCXT API error: {e}")
                ccxt_data = pd.DataFrame()
            
            if not all_data:
                return ccxt_data
            
            # Merge with existing data
            if not ccxt_data.empty:
                combined = pd.concat([pd.concat(all_data, ignore_index=True), ccxt_data], ignore_index=True)
                combined = combined.sort_values('datetime').drop_duplicates('datetime').reset_index(drop=True)
                # Filter date range
                combined = combined[(combined['datetime'] >= start) & (combined['datetime'] <= end)].copy()
                return combined
            else:
                # If CCXT failed but we have some data, return what we have
                if all_data:
                    df = pd.concat(all_data, ignore_index=True)
                    df = df.sort_values('datetime').drop_duplicates('datetime').reset_index(drop=True)
                    df = df[(df['datetime'] >= start) & (df['datetime'] <= end)].copy()
                    return df
                return pd.DataFrame()
        
        # Combine all downloaded data
        if all_data:
            # Combine and sort
            df = pd.concat(all_data, ignore_index=True)
            df = df.sort_values('datetime').drop_duplicates('datetime').reset_index(drop=True)
            
            # Filter date range
            df = df[(df['datetime'] >= start) & (df['datetime'] <= end)].copy()
            
            return df
        
        return pd.DataFrame()
    
    def _fetch_ccxt_historical(self, exchange, symbol: str, start_date: str, 
                               end_date: str, timeframe: str = '1h') -> pd.DataFrame:
        """
        Fetch historical data using CCXT API (for recent data or fallback)
        
        Args:
            exchange: CCXT exchange instance
            symbol: Symbol like 'BTC/USDT:USDT'
            start_date: Start date 'YYYY-MM-DD'
            end_date: End date 'YYYY-MM-DD'
            timeframe: '1h', '1d', etc.
            
        Returns:
            DataFrame with OHLCV data
        """
        print(f"  Fetching via CCXT API...")
        
        start_ts = int(pd.to_datetime(start_date).timestamp() * 1000)
        end_ts = int(pd.to_datetime(end_date).timestamp() * 1000)
        
        all_ohlcv = []
        current_ts = start_ts
        
        while current_ts < end_ts:
            try:
                ohlcv = exchange.fetch_ohlcv(symbol, timeframe, since=current_ts, limit=1000)
                if not ohlcv:
                    break
                
                all_ohlcv.extend(ohlcv)
                current_ts = ohlcv[-1][0] + 1
                
                # Rate limiting
                time.sleep(exchange.rateLimit / 1000)
                
            except Exception as e:
                print(f"    Error: {e}")
                break
        
        if not all_ohlcv:
            return pd.DataFrame()
        
        df = pd.DataFrame(all_ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['datetime'] = pd.to_datetime(df['timestamp'], unit='ms')
        df = df[['datetime', 'open', 'high', 'low', 'close', 'volume']].copy()
        df = df.sort_values('datetime').drop_duplicates('datetime').reset_index(drop=True)
        
        return df
    
    def fetch_bybit_funding_rates(self, symbol: str, start_date: str, 
                                  end_date: str) -> pd.DataFrame:
        """
        Fetch funding rates from Bybit (FREE API)
        
        Args:
            symbol: Symbol like 'BTCUSDT'
            start_date: Start date 'YYYY-MM-DD'
            end_date: End date 'YYYY-MM-DD'
            
        Returns:
            DataFrame with funding rates
        """
        print(f"Fetching Bybit funding rates for {symbol}...")
        
        # Bybit funding rate API
        base_url = "https://api.bybit.com/v5/market/funding/history"
        
        start_ts = int(pd.to_datetime(start_date).timestamp() * 1000)
        end_ts = int(pd.to_datetime(end_date).timestamp() * 1000)
        
        all_data = []
        cursor = None
        
        while True:
            params = {
                'category': 'linear',
                'symbol': symbol,
                'startTime': start_ts,
                'endTime': end_ts,
                'limit': 200
            }
            if cursor:
                params['cursor'] = cursor
            
            try:
                response = requests.get(base_url, params=params, timeout=30)
                data = response.json()
                
                if data.get('retCode') != 0:
                    break
                
                result = data.get('result', {})
                funding_list = result.get('list', [])
                
                if not funding_list:
                    break
                
                for item in funding_list:
                    all_data.append({
                        'datetime': pd.to_datetime(int(item['fundingRateTimestamp']), unit='ms'),
                        'funding_rate': float(item['fundingRate']),
                        'funding_rate_timestamp': int(item['fundingRateTimestamp'])
                    })
                
                # Check for next page
                cursor = result.get('nextPageCursor')
                if not cursor:
                    break
                
                time.sleep(0.2)  # Rate limiting
                
            except Exception as e:
                print(f"  Error fetching funding rates: {e}")
                break
        
        if not all_data:
            return pd.DataFrame()
        
        df = pd.DataFrame(all_data)
        df = df.sort_values('datetime').drop_duplicates('datetime').reset_index(drop=True)
        
        print(f"  ✓ Fetched {len(df)} funding rate records")
        return df
    
    def fetch_yfinance_historical(self, symbol: str, start_date: str, 
                                  end_date: str, interval: str = '1h') -> pd.DataFrame:
        """
        Fetch historical data from yfinance (FREE)
        
        For 1h data, yfinance only provides last 730 days, so we fetch daily
        and resample to 1h for longer histories.
        
        Args:
            symbol: yfinance symbol like 'EURUSD=X'
            start_date: Start date 'YYYY-MM-DD'
            end_date: End date 'YYYY-MM-DD'
            interval: '1h', '1d', etc.
            
        Returns:
            DataFrame with OHLCV data
        """
        print(f"Fetching yfinance data for {symbol} ({interval})...")
        
        try:
            ticker = yf.Ticker(symbol)
            
            # For 1h interval, yfinance only provides last 730 days
            # So we fetch daily data and resample to 1h for longer histories
            days_diff = (pd.to_datetime(end_date) - pd.to_datetime(start_date)).days
            
            if interval == '1h' and days_diff > 730:
                print(f"  Note: 1h data limited to 730 days, fetching daily and resampling...")
                # Fetch daily data (available for full history)
                df = ticker.history(start=start_date, end=end_date, interval='1d')
                
                if df.empty:
                    return pd.DataFrame()
                
                # Reset index
                df = df.reset_index()
                df.columns = [col.lower().replace(' ', '_') for col in df.columns]
                
                # Set datetime index
                if 'date' in df.columns:
                    df['datetime'] = pd.to_datetime(df['date'])
                elif 'datetime' in df.columns:
                    df['datetime'] = pd.to_datetime(df['datetime'])
                else:
                    df['datetime'] = df.index
                
                df = df.set_index('datetime')
                
                # Handle timezone issues - ensure timezone-naive
                if df.index.tz is not None:
                    df.index = df.index.tz_localize(None)
                
                # Resample daily to hourly (forward fill)
                df_hourly = df.resample('1H').ffill()
                
                # Filter to requested date range (timezone-naive comparison)
                start_dt = pd.to_datetime(start_date)
                end_dt = pd.to_datetime(end_date)
                if start_dt.tz is not None:
                    start_dt = start_dt.tz_localize(None)
                if end_dt.tz is not None:
                    end_dt = end_dt.tz_localize(None)
                
                df_hourly = df_hourly[
                    (df_hourly.index >= start_dt) &
                    (df_hourly.index <= end_dt)
                ]
                
                # Reset index
                df_hourly = df_hourly.reset_index()
                
            else:
                # Use requested interval directly
                yf_interval = interval
                if interval == '1h':
                    yf_interval = '1h'
                elif interval == '1d':
                    yf_interval = '1d'
                
                df = ticker.history(start=start_date, end=end_date, interval=yf_interval)
                
                if df.empty:
                    return pd.DataFrame()
                
                # Reset index and rename
                df = df.reset_index()
                df.columns = [col.lower().replace(' ', '_') for col in df.columns]
                
                # Standardize column names
                if 'datetime' not in df.columns and 'date' in df.columns:
                    df['datetime'] = pd.to_datetime(df['date'])
                elif 'datetime' in df.columns:
                    df['datetime'] = pd.to_datetime(df['datetime'])
                else:
                    df['datetime'] = df.index
                
                # Handle timezone issues - ensure timezone-naive
                if df['datetime'].dt.tz is not None:
                    df['datetime'] = df['datetime'].dt.tz_localize(None)
                
                df_hourly = df
            
            # Select required columns
            required_cols = ['datetime', 'open', 'high', 'low', 'close', 'volume']
            available_cols = [col for col in required_cols if col in df_hourly.columns]
            df_hourly = df_hourly[available_cols].copy()
            
            # Fill missing volume with 0
            if 'volume' not in df_hourly.columns:
                df_hourly['volume'] = 0.0
            
            df_hourly = df_hourly.sort_values('datetime').drop_duplicates('datetime').reset_index(drop=True)
            
            print(f"  ✓ Fetched {len(df_hourly)} candles")
            return df_hourly
            
        except Exception as e:
            print(f"  ✗ Error fetching yfinance data: {e}")
            import traceback
            traceback.print_exc()
            return pd.DataFrame()
    
    def fetch_all_data(self, ticker: str, start_date: str, end_date: str,
                      timeframe: str = '1h') -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Fetch all data for a ticker (OHLCV, funding rates, OI)
        
        Args:
            ticker: MANTIS ticker symbol
            start_date: Start date 'YYYY-MM-DD'
            end_date: End date 'YYYY-MM-DD'
            timeframe: '1h', '1d', etc.
            
        Returns:
            Tuple of (ohlcv_df, funding_df, oi_df)
        """
        mapping = self._get_ticker_mapping(ticker)
        
        # Fetch OHLCV data with multiple fallbacks
        ohlcv_df = pd.DataFrame()
        
        # Import alternative sources
        try:
            from scripts.data_collection.alternative_sources import (
                fetch_alphavantage_forex,
                fetch_cryptocompare_historical,
                fetch_ccxt_alternative_exchange
            )
        except ImportError:
            fetch_alphavantage_forex = None
            fetch_cryptocompare_historical = None
            fetch_ccxt_alternative_exchange = None
        
        if mapping.get('binance'):
            # Crypto: Try Binance first
            symbol = mapping['binance'].replace('/USDT:USDT', '').replace('/', '')
            ohlcv_df = self.fetch_binance_historical(symbol, start_date, end_date, timeframe)
            
            # Fallback 1: CryptoCompare for crypto
            if ohlcv_df.empty and ticker in ['BTC', 'ETH'] and fetch_cryptocompare_historical:
                print(f"  Trying CryptoCompare as fallback...")
                ohlcv_df = fetch_cryptocompare_historical(ticker, start_date, end_date, timeframe)
            
            # Fallback 2: Alternative CCXT exchanges
            if ohlcv_df.empty and fetch_ccxt_alternative_exchange:
                print(f"  Trying alternative exchanges as fallback...")
                for exchange_name in ['coinbase', 'kraken', 'bitfinex']:
                    ccxt_symbol = mapping['binance']
                    ohlcv_df = fetch_ccxt_alternative_exchange(
                        ccxt_symbol, start_date, end_date, timeframe, exchange_name
                    )
                    if not ohlcv_df.empty:
                        break
                        
        elif mapping.get('yfinance'):
            # Forex/Commodities: Try yfinance first
            ohlcv_df = self.fetch_yfinance_historical(mapping['yfinance'], start_date, end_date, timeframe)
            
            # Fallback: Alpha Vantage for forex (if API key available)
            if ohlcv_df.empty and ticker in ['EURUSD', 'GBPUSD', 'CADUSD', 'NZDUSD', 'CHFUSD'] and fetch_alphavantage_forex:
                print(f"  Trying Alpha Vantage as fallback...")
                base_currency = ticker.replace('USD', '')
                ohlcv_df = fetch_alphavantage_forex(base_currency, 'USD', start_date, end_date)
        
        # Fetch funding rates (only for crypto perpetuals)
        funding_df = pd.DataFrame()
        if mapping.get('bybit') and ticker in ['BTC', 'ETH']:
            symbol = ticker + 'USDT'
            funding_df = self.fetch_bybit_funding_rates(symbol, start_date, end_date)
        
        # OI data (placeholder - would need paid API for historical)
        oi_df = pd.DataFrame()
        
        return ohlcv_df, funding_df, oi_df
    
    def save_data(self, ticker: str, ohlcv_df: pd.DataFrame, 
                  funding_df: pd.DataFrame, oi_df: pd.DataFrame):
        """
        Save data to disk
        
        Args:
            ticker: Ticker symbol
            ohlcv_df: OHLCV DataFrame
            funding_df: Funding rates DataFrame
            oi_df: Open interest DataFrame
        """
        ticker_dir = self.data_dir / ticker
        ticker_dir.mkdir(exist_ok=True)
        
        if not ohlcv_df.empty:
            ohlcv_df.to_csv(ticker_dir / 'ohlcv.csv', index=False)
            print(f"  ✓ Saved OHLCV: {len(ohlcv_df)} rows")
        
        if not funding_df.empty:
            funding_df.to_csv(ticker_dir / 'funding_rates.csv', index=False)
            print(f"  ✓ Saved funding rates: {len(funding_df)} rows")
        
        if not oi_df.empty:
            oi_df.to_csv(ticker_dir / 'open_interest.csv', index=False)
            print(f"  ✓ Saved OI: {len(oi_df)} rows")
    
    def load_data(self, ticker: str) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Load saved data from disk
        
        Args:
            ticker: Ticker symbol
            
        Returns:
            Tuple of (ohlcv_df, funding_df, oi_df)
        """
        ticker_dir = self.data_dir / ticker
        
        ohlcv_df = pd.DataFrame()
        if (ticker_dir / 'ohlcv.csv').exists():
            ohlcv_df = pd.read_csv(ticker_dir / 'ohlcv.csv', parse_dates=['datetime'])
        
        funding_df = pd.DataFrame()
        if (ticker_dir / 'funding_rates.csv').exists():
            funding_df = pd.read_csv(ticker_dir / 'funding_rates.csv', parse_dates=['datetime'])
        
        oi_df = pd.DataFrame()
        if (ticker_dir / 'open_interest.csv').exists():
            oi_df = pd.read_csv(ticker_dir / 'open_interest.csv', parse_dates=['datetime'])
        
        return ohlcv_df, funding_df, oi_df


def main():
    """Example usage"""
    fetcher = DataFetcher(data_dir="data/raw")
    
    # Example: Fetch BTC data
    print("=" * 80)
    print("Fetching BTC Historical Data")
    print("=" * 80)
    
    ohlcv, funding, oi = fetcher.fetch_all_data(
        ticker='BTC',
        start_date='2020-01-01',
        end_date='2024-12-31',
        timeframe='1h'
    )
    
    if not ohlcv.empty:
        fetcher.save_data('BTC', ohlcv, funding, oi)
        print(f"\n✓ Data saved successfully")
        print(f"  OHLCV: {len(ohlcv)} rows")
        print(f"  Funding: {len(funding)} rows")
        print(f"  Date range: {ohlcv['datetime'].min()} to {ohlcv['datetime'].max()}")


if __name__ == "__main__":
    main()

