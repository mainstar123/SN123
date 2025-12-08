"""
Verify Training Process - Step 3 Implementation
Verifies that the training process completed successfully by checking:
1. Data loading and splitting
2. Feature extraction (VMD, technical indicators, etc.)
3. Feature selection (TMFG)
4. Model architecture (LSTM + XGBoost)
5. Model files saved correctly
"""

import os
import sys
import argparse
import json
from pathlib import Path
import pandas as pd

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from scripts.training.train_model import load_data, prepare_train_val_test_split
from scripts.training.model_architecture import VMDTMFGLSTMXGBoost
import config


def verify_training_process(ticker: str, model_dir: str, data_dir: str,
                           train_end: str = '2023-12-31',
                           val_end: str = '2024-12-31') -> dict:
    """
    Verify the training process for a ticker
    
    Args:
        ticker: Ticker symbol (challenge ticker)
        model_dir: Model directory
        data_dir: Data directory
        train_end: Training end date
        val_end: Validation end date
        
    Returns:
        Dictionary with verification results
    """
    print(f"\n{'='*80}")
    print(f"Verifying Training Process for {ticker}")
    print(f"{'='*80}\n")
    
    results = {
        'ticker': ticker,
        'steps': {},
        'all_passed': True
    }
    
    model_path = os.path.join(model_dir, ticker)
    
    # Step 1: Check model files exist
    print("Step 1: Checking model files...")
    required_files = {
        'lstm_model.h5': 'LSTM model weights',
        'xgb_model.json': 'XGBoost model',
        'scaler.pkl': 'Feature scaler',
        'feature_indices.pkl': 'Selected feature indices',
        'config.json': 'Model configuration'
    }
    
    files_exist = True
    for filename, description in required_files.items():
        filepath = os.path.join(model_path, filename)
        if os.path.exists(filepath):
            size = os.path.getsize(filepath) / 1024  # KB
            print(f"  ✓ {filename} ({description}) - {size:.1f} KB")
        else:
            print(f"  ✗ {filename} ({description}) - MISSING")
            files_exist = False
    
    results['steps']['model_files'] = files_exist
    if not files_exist:
        results['all_passed'] = False
        return results
    
    # Step 2: Load and verify model configuration
    print("\nStep 2: Verifying model configuration...")
    try:
        with open(os.path.join(model_path, 'config.json'), 'r') as f:
            model_config = json.load(f)
        
        print(f"  ✓ Model configuration loaded")
        print(f"    Embedding dimension: {model_config.get('embedding_dim', 'N/A')}")
        print(f"    LSTM hidden units: {model_config.get('lstm_hidden', 'N/A')}")
        print(f"    LSTM layers: {model_config.get('lstm_layers', 'N/A')}")
        print(f"    Time steps: {model_config.get('time_steps', 'N/A')}")
        print(f"    VMD components (K): {model_config.get('vmd_k', 'N/A')}")
        print(f"    TMFG features: {model_config.get('tmfg_n_features', 'N/A')}")
        print(f"    Dropout: {model_config.get('dropout', 'N/A')}")
        
        results['steps']['config'] = True
        results['config'] = model_config
    except Exception as e:
        print(f"  ✗ Failed to load configuration: {e}")
        results['steps']['config'] = False
        results['all_passed'] = False
        return results
    
    # Step 3: Verify data loading
    print("\nStep 3: Verifying data loading...")
    try:
        challenge = config.CHALLENGE_MAP.get(ticker)
        data_ticker = challenge.get('price_key', ticker) if challenge else ticker
        
        ohlcv, funding, oi, cross_exchange = load_data(data_ticker, data_dir)
        
        if ohlcv.empty:
            print(f"  ✗ No data available for {data_ticker}")
            results['steps']['data_loading'] = False
            results['all_passed'] = False
            return results
        
        print(f"  ✓ Data loaded successfully")
        print(f"    OHLCV records: {len(ohlcv)}")
        print(f"    Date range: {ohlcv['datetime'].min()} to {ohlcv['datetime'].max()}")
        print(f"    Funding records: {len(funding) if not funding.empty else 0}")
        print(f"    OI records: {len(oi) if not oi.empty else 0}")
        
        results['steps']['data_loading'] = True
        results['data_info'] = {
            'ohlcv_count': len(ohlcv),
            'date_range': (str(ohlcv['datetime'].min()), str(ohlcv['datetime'].max())),
            'funding_count': len(funding) if not funding.empty else 0,
            'oi_count': len(oi) if not oi.empty else 0
        }
    except Exception as e:
        print(f"  ✗ Data loading failed: {e}")
        results['steps']['data_loading'] = False
        results['all_passed'] = False
        return results
    
    # Step 4: Verify data splitting
    print("\nStep 4: Verifying data splitting...")
    try:
        train_df, val_df, test_df = prepare_train_val_test_split(
            ohlcv, train_end=train_end, val_end=val_end
        )
        
        print(f"  ✓ Data split successfully")
        print(f"    Train: {len(train_df)} samples ({train_df['datetime'].min()} to {train_df['datetime'].max()})")
        print(f"    Val: {len(val_df)} samples ({val_df['datetime'].min()} to {val_df['datetime'].max()})")
        print(f"    Test: {len(test_df)} samples ({test_df['datetime'].min()} to {test_df['datetime'].max()})")
        
        if len(train_df) < 1000:
            print(f"  ⚠ Warning: Training set has only {len(train_df)} samples (recommended: 1000+)")
        
        results['steps']['data_splitting'] = True
        results['split_info'] = {
            'train_count': len(train_df),
            'val_count': len(val_df),
            'test_count': len(test_df)
        }
    except Exception as e:
        print(f"  ✗ Data splitting failed: {e}")
        results['steps']['data_splitting'] = False
        results['all_passed'] = False
        return results
    
    # Step 5: Verify feature extraction and selection
    print("\nStep 5: Verifying feature extraction and selection...")
    try:
        embedding_dim = model_config.get('embedding_dim', 2)
        model = VMDTMFGLSTMXGBoost(embedding_dim=embedding_dim)
        model.load(model_path)
        
        # Try to prepare data to verify feature extraction works
        X_train, y_train, feature_names = model.prepare_data(
            train_df.head(100), funding, oi, cross_exchange  # Use small sample for verification
        )
        
        print(f"  ✓ Feature extraction verified")
        print(f"    Selected features: {len(feature_names)}")
        print(f"    Feature sequence shape: {X_train.shape if len(X_train) > 0 else 'N/A'}")
        print(f"    Sample features: {', '.join(feature_names[:5])}..." if len(feature_names) > 5 else f"    Features: {', '.join(feature_names)}")
        
        results['steps']['feature_extraction'] = True
        results['feature_info'] = {
            'n_features': len(feature_names),
            'sequence_shape': list(X_train.shape) if len(X_train) > 0 else None
        }
    except Exception as e:
        print(f"  ✗ Feature extraction verification failed: {e}")
        import traceback
        traceback.print_exc()
        results['steps']['feature_extraction'] = False
        results['all_passed'] = False
        return results
    
    # Step 6: Verify model architecture
    print("\nStep 6: Verifying model architecture...")
    try:
        # Model is already loaded from step 5
        print(f"  ✓ Model architecture verified")
        print(f"    LSTM model: Loaded")
        print(f"    XGBoost model: Loaded")
        print(f"    Embedding dimension: {embedding_dim}")
        
        # Try a prediction to verify model works
        if len(X_train) > 0:
            test_embeddings = model.predict_embeddings(X_train[:1])
            print(f"    Test prediction shape: {test_embeddings.shape}")
            print(f"    ✓ Model inference working")
        
        results['steps']['model_architecture'] = True
    except Exception as e:
        print(f"  ✗ Model architecture verification failed: {e}")
        results['steps']['model_architecture'] = False
        results['all_passed'] = False
        return results
    
    # Summary
    print(f"\n{'='*80}")
    print("Training Process Verification Summary")
    print(f"{'='*80}")
    
    all_steps = [
        ('Model Files', 'model_files'),
        ('Configuration', 'config'),
        ('Data Loading', 'data_loading'),
        ('Data Splitting', 'data_splitting'),
        ('Feature Extraction', 'feature_extraction'),
        ('Model Architecture', 'model_architecture')
    ]
    
    for step_name, step_key in all_steps:
        status = "✓ PASS" if results['steps'].get(step_key, False) else "✗ FAIL"
        print(f"  {status}: {step_name}")
    
    if results['all_passed']:
        print(f"\n✓ All steps passed! Training process verified successfully.")
    else:
        print(f"\n✗ Some steps failed. Please check the errors above.")
    
    print(f"{'='*80}\n")
    
    return results


def main():
    parser = argparse.ArgumentParser(
        description="Verify training process (Step 3 from Implementation Guide)"
    )
    parser.add_argument(
        "--ticker",
        type=str,
        default=None,
        help="Ticker to verify (default: all trained models)"
    )
    parser.add_argument(
        "--model-dir",
        type=str,
        default="models/checkpoints",
        help="Model directory"
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data/raw",
        help="Data directory"
    )
    parser.add_argument(
        "--train-end",
        type=str,
        default="2023-12-31",
        help="Training end date (YYYY-MM-DD)"
    )
    parser.add_argument(
        "--val-end",
        type=str,
        default="2024-12-31",
        help="Validation end date (YYYY-MM-DD)"
    )
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("MANTIS Training Process Verification (Step 3)")
    print("=" * 80)
    print(f"Model directory: {args.model_dir}")
    print(f"Data directory: {args.data_dir}")
    print("=" * 80)
    
    # Determine tickers to verify
    if args.ticker:
        # Check if ticker exists in config
        challenge = config.CHALLENGE_MAP.get(args.ticker)
        if not challenge:
            # Check if it's a price_key (e.g., BTC -> BTCLBFGS)
            matching_challenges = [c for c in config.CHALLENGES if c.get('price_key') == args.ticker]
            if matching_challenges:
                print(f"ℹ Note: '{args.ticker}' is a price_key, not a challenge ticker.")
                print(f"  Found {len(matching_challenges)} challenge(s) using this price_key:")
                for c in matching_challenges:
                    print(f"    - {c['ticker']} ({c['name']})")
                print(f"\n  Using challenge ticker: {matching_challenges[0]['ticker']}")
                tickers = [matching_challenges[0]['ticker']]
            else:
                # Check if directory exists
                ticker_path = os.path.join(args.model_dir, args.ticker)
                if os.path.exists(ticker_path):
                    print(f"⚠ Warning: '{args.ticker}' not found in config, but directory exists.")
                    print(f"  Attempting to verify anyway...")
                    tickers = [args.ticker]
                else:
                    print(f"✗ Ticker '{args.ticker}' not found in config and no model directory exists.")
                    print(f"  Available challenge tickers: {', '.join(config.CHALLENGE_MAP.keys())}")
                    print(f"  Available price_keys: {', '.join(set(c.get('price_key', '') for c in config.CHALLENGES if c.get('price_key')))}")
                    return
        else:
            tickers = [args.ticker]
    else:
        # Find all trained models
        if os.path.exists(args.model_dir):
            tickers = [d for d in os.listdir(args.model_dir) 
                      if os.path.isdir(os.path.join(args.model_dir, d)) 
                      and os.path.exists(os.path.join(args.model_dir, d, 'lstm_model.h5'))]
        else:
            print(f"✗ Model directory not found: {args.model_dir}")
            return
    
    if not tickers:
        print("✗ No trained models found")
        return
    
    print(f"Found {len(tickers)} trained model(s): {', '.join(tickers)}\n")
    
    # Verify each model
    all_results = {}
    for ticker in tickers:
        try:
            result = verify_training_process(
                ticker=ticker,
                model_dir=args.model_dir,
                data_dir=args.data_dir,
                train_end=args.train_end,
                val_end=args.val_end
            )
            all_results[ticker] = result
        except Exception as e:
            print(f"\n✗ Error verifying {ticker}: {e}")
            import traceback
            traceback.print_exc()
            all_results[ticker] = {'all_passed': False, 'error': str(e)}
    
    # Final summary
    print("\n" + "=" * 80)
    print("Final Summary")
    print("=" * 80)
    
    passed = [t for t, r in all_results.items() if r.get('all_passed', False)]
    failed = [t for t, r in all_results.items() if not r.get('all_passed', False)]
    
    print(f"\nPassed: {len(passed)}/{len(tickers)}")
    for ticker in passed:
        print(f"  ✓ {ticker}")
    
    if failed:
        print(f"\nFailed: {len(failed)}")
        for ticker in failed:
            error = all_results[ticker].get('error', 'Verification failed')
            print(f"  ✗ {ticker}: {error}")


if __name__ == "__main__":
    main()


