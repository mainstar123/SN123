#!/usr/bin/env python3
"""
Collect all datasets for MANTIS challenges
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from scripts.data_collection.data_fetcher import DataFetcher
from datetime import datetime
import config

def collect_all_datasets(data_dir: str = "data/raw", start_date: str = "2020-01-01"):
    """
    Collect all datasets needed for MANTIS challenges
    
    Args:
        data_dir: Data directory
        start_date: Start date for data collection
    """
    fetcher = DataFetcher(data_dir=data_dir)
    
    # Get unique assets from challenges
    assets = set()
    for challenge in config.CHALLENGES:
        # Use price_key if available, otherwise use ticker
        asset = challenge.get('price_key', challenge['ticker'])
        assets.add(asset)
    
    # Remove challenge-specific tickers (keep only base assets)
    assets = {a for a in assets if a not in ['ETHHITFIRST', 'ETHLBFGS', 'BTCLBFGS']}
    
    print("=" * 80)
    print("MANTIS Data Collection - All Assets")
    print("=" * 80)
    print(f"Assets to collect: {sorted(assets)}")
    print(f"Start date: {start_date}")
    print(f"End date: {datetime.now().strftime('%Y-%m-%d')}")
    print("=" * 80)
    print()
    
    results = {}
    
    for asset in sorted(assets):
        print(f"\n{'='*80}")
        print(f"Collecting {asset}")
        print(f"{'='*80}")
        
        try:
            # Check if already collected
            ohlcv, funding, oi = fetcher.load_data(asset)
            
            if not ohlcv.empty:
                print(f"  ✓ {asset} already collected: {len(ohlcv)} rows")
                print(f"    Date range: {ohlcv['datetime'].min()} to {ohlcv['datetime'].max()}")
                results[asset] = {'status': 'already_collected', 'rows': len(ohlcv)}
                continue
            
            # Fetch data
            ohlcv, funding, oi = fetcher.fetch_all_data(
                ticker=asset,
                start_date=start_date,
                end_date=datetime.now().strftime('%Y-%m-%d'),
                timeframe='1h'
            )
            
            if ohlcv.empty:
                print(f"  ✗ Failed to collect {asset}")
                results[asset] = {'status': 'failed', 'rows': 0}
                continue
            
            # Save data
            fetcher.save_data(asset, ohlcv, funding, oi)
            
            print(f"  ✓ Successfully collected {asset}")
            print(f"    OHLCV: {len(ohlcv)} rows")
            print(f"    Funding: {len(funding)} rows")
            print(f"    Date range: {ohlcv['datetime'].min()} to {ohlcv['datetime'].max()}")
            
            results[asset] = {
                'status': 'success',
                'rows': len(ohlcv),
                'funding_rows': len(funding)
            }
            
        except Exception as e:
            print(f"  ✗ Error collecting {asset}: {e}")
            import traceback
            traceback.print_exc()
            results[asset] = {'status': 'error', 'error': str(e)}
    
    # Summary
    print("\n" + "=" * 80)
    print("Collection Summary")
    print("=" * 80)
    
    successful = [a for a, r in results.items() if r.get('status') in ['success', 'already_collected']]
    failed = [a for a, r in results.items() if r.get('status') not in ['success', 'already_collected']]
    
    print(f"Successful: {len(successful)}/{len(assets)}")
    for asset in successful:
        result = results[asset]
        rows = result.get('rows', 0)
        status = result.get('status', '')
        if status == 'already_collected':
            print(f"  ✓ {asset}: {rows} rows (already collected)")
        else:
            print(f"  ✓ {asset}: {rows} rows")
    
    if failed:
        print(f"\nFailed: {len(failed)}")
        for asset in failed:
            error = results[asset].get('error', 'Unknown error')
            print(f"  ✗ {asset}: {error}")
    
    print("=" * 80)
    
    return results


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Collect all MANTIS challenge datasets")
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data/raw",
        help="Data directory"
    )
    parser.add_argument(
        "--start-date",
        type=str,
        default="2020-01-01",
        help="Start date (YYYY-MM-DD)"
    )
    
    args = parser.parse_args()
    
    collect_all_datasets(
        data_dir=args.data_dir,
        start_date=args.start_date
    )
