"""
Alternative free data sources for MANTIS data collection
Includes Alpha Vantage, CryptoCompare, and other free APIs
"""

import os
import pandas as pd
import numpy as np
import requests
import time
from typing import Optional
from datetime import datetime, timedelta

# Try to load .env file if python-dotenv is available
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # python-dotenv not installed, use environment variables directly

# Optional API keys (free tier available)
ALPHA_VANTAGE_API_KEY = os.getenv('ALPHA_VANTAGE_API_KEY', None)
CRYPTOCOMPARE_API_KEY = os.getenv('CRYPTOCOMPARE_API_KEY', None)


def fetch_alphavantage_forex(base_currency: str, quote_currency: str = 'USD',
                             start_date: str = '2020-01-01',
                             end_date: str = None) -> Optional[pd.DataFrame]:
    """
    Fetch forex data from Alpha Vantage (FREE - 5 calls/minute)
    
    Note: Alpha Vantage forex API is limited, mainly for current rates.
    For historical data, we'll use daily data and resample.
    
    Args:
        base_currency: Base currency (e.g., 'EUR', 'GBP')
        quote_currency: Quote currency (default: 'USD')
        start_date: Start date
        end_date: End date
        
    Returns:
        DataFrame with OHLCV data or None
    """
    if not ALPHA_VANTAGE_API_KEY:
        return None
    
    try:
        # Alpha Vantage FX Daily endpoint
        url = "https://www.alphavantage.co/query"
        params = {
            'function': 'FX_DAILY',
            'from_symbol': base_currency,
            'to_symbol': quote_currency,
            'apikey': ALPHA_VANTAGE_API_KEY,
            'outputsize': 'full'
        }
        
        response = requests.get(url, params=params, timeout=30)
        data = response.json()
        
        if 'Time Series FX (Daily)' not in data:
            return None
        
        # Convert to DataFrame
        time_series = data['Time Series FX (Daily)']
        df = pd.DataFrame.from_dict(time_series, orient='index')
        df.index = pd.to_datetime(df.index)
        df.columns = ['open', 'high', 'low', 'close']
        
        # Convert to float
        for col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Add volume (0 for forex)
        df['volume'] = 0.0
        
        # Filter date range
        start_dt = pd.to_datetime(start_date)
        if end_date:
            end_dt = pd.to_datetime(end_date)
        else:
            end_dt = datetime.now()
        
        df = df[(df.index >= start_dt) & (df.index <= end_dt)]
        
        # Reset index and rename
        df = df.reset_index()
        df.columns = ['datetime', 'open', 'high', 'low', 'close', 'volume']
        
        # Resample to hourly (forward fill)
        df = df.set_index('datetime')
        df_hourly = df.resample('1H').ffill()
        df_hourly = df_hourly.reset_index()
        
        return df_hourly
        
    except Exception as e:
        print(f"  Alpha Vantage error: {e}")
        return None


def fetch_cryptocompare_historical(symbol: str, start_date: str, end_date: str,
                                   timeframe: str = '1h') -> Optional[pd.DataFrame]:
    """
    Fetch historical data from CryptoCompare (FREE tier available)
    
    Args:
        symbol: Symbol like 'BTC', 'ETH'
        start_date: Start date 'YYYY-MM-DD'
        end_date: End date 'YYYY-MM-DD'
        timeframe: '1h', '1d', etc.
        
    Returns:
        DataFrame with OHLCV data or None
    """
    try:
        # Map timeframe
        limit_map = {
            '1h': ('histohour', 2000),
            '1d': ('histoday', 2000)
        }
        
        if timeframe not in limit_map:
            return None
        
        endpoint, limit = limit_map[timeframe]
        
        # Convert dates to timestamps
        start_ts = int(pd.to_datetime(start_date).timestamp())
        end_ts = int(pd.to_datetime(end_date).timestamp())
        
        all_data = []
        current_ts = end_ts
        
        # CryptoCompare API
        base_url = "https://min-api.cryptocompare.com/data/v2"
        
        while current_ts > start_ts:
            url = f"{base_url}/{endpoint}"
            params = {
                'fsym': symbol,
                'tsym': 'USD',
                'limit': limit,
                'toTs': current_ts
            }
            
            if CRYPTOCOMPARE_API_KEY:
                params['api_key'] = CRYPTOCOMPARE_API_KEY
            
            try:
                response = requests.get(url, params=params, timeout=30)
                data = response.json()
                
                if data.get('Response') != 'Success':
                    break
                
                records = data.get('Data', {}).get('Data', [])
                if not records:
                    break
                
                all_data.extend(records)
                
                # Move to earlier time
                current_ts = records[0]['time'] - 1
                
                # Rate limiting (free tier: 100k calls/month)
                time.sleep(0.2)
                
            except Exception as e:
                print(f"    CryptoCompare API error: {e}")
                break
        
        if not all_data:
            return None
        
        # Convert to DataFrame
        df = pd.DataFrame(all_data)
        df['datetime'] = pd.to_datetime(df['time'], unit='s')
        df = df[['datetime', 'open', 'high', 'low', 'close', 'volumefrom']].copy()
        df.columns = ['datetime', 'open', 'high', 'low', 'close', 'volume']
        
        # Filter date range
        start_dt = pd.to_datetime(start_date)
        end_dt = pd.to_datetime(end_date)
        df = df[(df['datetime'] >= start_dt) & (df['datetime'] <= end_dt)]
        
        df = df.sort_values('datetime').drop_duplicates('datetime').reset_index(drop=True)
        
        return df
        
    except Exception as e:
        print(f"  CryptoCompare error: {e}")
        return None


def fetch_ccxt_alternative_exchange(symbol: str, start_date: str, end_date: str,
                                    timeframe: str = '1h',
                                    exchange_name: str = 'coinbase') -> Optional[pd.DataFrame]:
    """
    Fetch data from alternative CCXT exchanges as fallback
    
    Args:
        symbol: Symbol like 'BTC/USDT' or 'ETH/USDT'
        start_date: Start date
        end_date: End date
        timeframe: '1h', '1d', etc.
        exchange_name: Exchange name ('coinbase', 'kraken', 'bitfinex', etc.)
        
    Returns:
        DataFrame with OHLCV data or None
    """
    try:
        import ccxt
        
        # Initialize exchange
        exchange_class = getattr(ccxt, exchange_name)
        exchange = exchange_class({
            'enableRateLimit': True
        })
        
        # Ensure symbol format
        if not '/' in symbol:
            symbol = f"{symbol}/USD"
        
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
                
                time.sleep(exchange.rateLimit / 1000)
                
            except Exception as e:
                break
        
        if not all_ohlcv:
            return None
        
        df = pd.DataFrame(all_ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['datetime'] = pd.to_datetime(df['timestamp'], unit='ms')
        df = df[['datetime', 'open', 'high', 'low', 'close', 'volume']].copy()
        df = df.sort_values('datetime').drop_duplicates('datetime').reset_index(drop=True)
        
        return df
        
    except Exception as e:
        print(f"  {exchange_name} error: {e}")
        return None

