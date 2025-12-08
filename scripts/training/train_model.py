"""
Training Pipeline for MANTIS Mining Models
"""

import os
import sys
import argparse
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from scripts.data_collection.data_fetcher import DataFetcher
from scripts.training.model_architecture import VMDTMFGLSTMXGBoost
import config


def load_data(ticker: str, data_dir: str, start_date: str = None, 
              end_date: str = None) -> tuple:
    """
    Load data for a ticker
    
    Args:
        ticker: Ticker symbol
        data_dir: Data directory
        start_date: Start date (optional, for fetching)
        end_date: End date (optional, for fetching)
        
    Returns:
        Tuple of (ohlcv_df, funding_df, oi_df, cross_exchange_df)
    """
    fetcher = DataFetcher(data_dir=data_dir)
    
    # Try to load from disk first
    ohlcv, funding, oi = fetcher.load_data(ticker)
    
    # If not found or date range specified, fetch
    if ohlcv.empty or (start_date and end_date):
        print(f"Fetching data for {ticker}...")
        ohlcv, funding, oi = fetcher.fetch_all_data(
            ticker=ticker,
            start_date=start_date or '2020-01-01',
            end_date=end_date or datetime.now().strftime('%Y-%m-%d'),
            timeframe='1h'
        )
        
        if not ohlcv.empty:
            fetcher.save_data(ticker, ohlcv, funding, oi)
    
    # Cross-exchange data (placeholder)
    cross_exchange = pd.DataFrame()
    
    return ohlcv, funding, oi, cross_exchange


def prepare_train_val_test_split(df: pd.DataFrame, 
                                 train_end: str = '2023-12-31',
                                 val_end: str = '2024-12-31') -> tuple:
    """
    Split data into train/val/test sets by time
    
    Args:
        df: DataFrame with datetime index
        train_end: End date for training set
        val_end: End date for validation set
        
    Returns:
        Tuple of (train_df, val_df, test_df)
    """
    df = df.copy()
    df['datetime'] = pd.to_datetime(df['datetime'])
    df = df.sort_values('datetime').reset_index(drop=True)
    
    train_end_dt = pd.to_datetime(train_end)
    val_end_dt = pd.to_datetime(val_end)
    
    train_df = df[df['datetime'] <= train_end_dt].copy()
    val_df = df[(df['datetime'] > train_end_dt) & (df['datetime'] <= val_end_dt)].copy()
    test_df = df[df['datetime'] > val_end_dt].copy()
    
    return train_df, val_df, test_df


def train_ticker_model(ticker: str, data_dir: str, model_dir: str,
                      embedding_dim: int = 2,
                      train_end: str = '2023-12-31',
                      val_end: str = '2024-12-31',
                      epochs: int = 100,
                      batch_size: int = 64,
                      challenge_ticker: str = None,
                      use_gpu: bool = True,
                      lstm_hidden: int = 256,
                      tmfg_n_features: int = 25,  # Best from tuning: 25 features
                      dropout: float = 0.3,
                      learning_rate: float = 0.0005) -> dict:
    """
    Train model for a single ticker
    
    Args:
        ticker: Ticker symbol
        data_dir: Data directory
        model_dir: Model save directory
        embedding_dim: Embedding dimension
        train_end: Training end date
        val_end: Validation end date
        epochs: Training epochs
        batch_size: Batch size
        challenge_ticker: Challenge ticker name (for model saving)
        use_gpu: Whether to use GPU
        lstm_hidden: LSTM hidden units
        tmfg_n_features: Number of TMFG features
        dropout: Dropout rate
        learning_rate: Learning rate for Adam optimizer
        
    Returns:
        Dictionary with training results
    """
    print(f"\n{'='*80}")
    print(f"Training model for {ticker}")
    print(f"{'='*80}")
    
    # Load data
    print("Loading data...")
    ohlcv, funding, oi, cross_exchange = load_data(ticker, data_dir)
    
    if ohlcv.empty:
        print(f"✗ No data available for {ticker}")
        return {'success': False, 'error': 'No data'}
    
    print(f"  ✓ Loaded {len(ohlcv)} OHLCV records")
    print(f"  ✓ Date range: {ohlcv['datetime'].min()} to {ohlcv['datetime'].max()}")
    
    # Split data
    print("Splitting data...")
    train_df, val_df, test_df = prepare_train_val_test_split(
        ohlcv, train_end=train_end, val_end=val_end
    )
    
    print(f"  Train: {len(train_df)} samples ({train_df['datetime'].min()} to {train_df['datetime'].max()})")
    print(f"  Val: {len(val_df)} samples ({val_df['datetime'].min()} to {val_df['datetime'].max()})")
    print(f"  Test: {len(test_df)} samples ({test_df['datetime'].min()} to {test_df['datetime'].max()})")
    
    if len(train_df) < 1000:
        print(f"✗ Insufficient training data: {len(train_df)} samples")
        return {'success': False, 'error': 'Insufficient data'}
    
    # Initialize model
    print("Initializing model...")
    model = VMDTMFGLSTMXGBoost(
        embedding_dim=embedding_dim,
        lstm_hidden=lstm_hidden,
        lstm_layers=2,
        time_steps=20,
        vmd_k=8,
        tmfg_n_features=tmfg_n_features,
        dropout=dropout,
        learning_rate=learning_rate,
        use_gpu=use_gpu
    )
    
    # Prepare training data
    print("Preparing training data...")
    X_train, y_train, feature_names = model.prepare_data(
        train_df, funding, oi, cross_exchange
    )
    
    if len(X_train) == 0:
        print("✗ Failed to prepare training data")
        return {'success': False, 'error': 'Data preparation failed'}
    
    print(f"  ✓ Training sequences: {len(X_train)}")
    print(f"  ✓ Feature dimension: {X_train.shape[2]}")
    print(f"  ✓ Selected features: {len(feature_names)}")
    
    # Prepare validation data
    X_val, y_val, _ = model.prepare_data(val_df, funding, oi, cross_exchange)
    
    if len(X_val) == 0:
        print("  Warning: No validation data, using training set")
        X_val, y_val = None, None
    
    # Train model
    print("Training model...")
    model.train(
        X_train, y_train,
        X_val=X_val, y_val=y_val,
        epochs=epochs,
        batch_size=batch_size,
        verbose=1
    )
    
    # Evaluate on test set
    print("Evaluating on test set...")
    if len(test_df) > 0:
        X_test, y_test, _ = model.prepare_data(test_df, funding, oi, cross_exchange)
        
        if len(X_test) > 0:
            # Generate embeddings
            test_embeddings = model.predict_embeddings(X_test)
            
            # Predict binary direction
            y_test_binary = (y_test > 0).astype(int)  # Price increase
            y_pred_binary = model.predict_binary(X_test)
            
            # Calculate metrics
            test_acc = accuracy_score(y_test_binary, y_pred_binary)
            test_probs = test_embeddings[:, 1] if test_embeddings.shape[1] >= 2 else (test_embeddings[:, 0] + 1) / 2
            
            # Calculate AUC only if we have both classes
            try:
                if len(np.unique(y_test_binary)) > 1 and len(np.unique(y_pred_binary)) > 1:
                    test_auc = roc_auc_score(y_test_binary, test_probs)
                else:
                    test_auc = float('nan')
                    print("  Warning: Test AUC cannot be calculated (all predictions or labels are the same class)")
            except Exception as e:
                test_auc = float('nan')
                print(f"  Warning: Test AUC calculation failed: {str(e)[:60]}")
            
            print(f"\nTest Set Results:")
            print(f"  Accuracy: {test_acc:.4f} ({test_acc*100:.2f}%)")
            print(f"  AUC: {test_auc:.4f}")
            
            # Bootstrap confidence interval
            bootstrap_accs = []
            n_bootstrap = 1000
            for _ in range(n_bootstrap):
                sample_idx = np.random.choice(len(y_test_binary), len(y_test_binary), replace=True)
                bootstrap_accs.append(
                    accuracy_score(y_test_binary[sample_idx], y_pred_binary[sample_idx])
                )
            
            ci_lower = np.percentile(bootstrap_accs, 2.5)
            ci_upper = np.percentile(bootstrap_accs, 97.5)
            
            print(f"  95% CI: [{ci_lower:.4f}, {ci_upper:.4f}]")
            
            results = {
                'success': True,
                'test_accuracy': float(test_acc),
                'test_auc': float(test_auc),
                'ci_lower': float(ci_lower),
                'ci_upper': float(ci_upper),
                'n_test_samples': len(X_test)
            }
        else:
            results = {'success': True, 'test_accuracy': None}
    else:
        results = {'success': True, 'test_accuracy': None}
    
    # Save model - use challenge_ticker if provided (for proper challenge naming)
    # otherwise use data ticker
    save_ticker = challenge_ticker if challenge_ticker else ticker
    ticker_model_dir = os.path.join(model_dir, save_ticker)
    model.save(ticker_model_dir)
    
    results['model_path'] = ticker_model_dir
    results['data_ticker'] = ticker  # Store the data ticker used
    results['challenge_ticker'] = save_ticker  # Store the challenge ticker
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Train MANTIS mining models")
    parser.add_argument(
        "--ticker",
        type=str,
        default=None,
        help="Ticker to train (default: all from config)"
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data/raw",
        help="Data directory"
    )
    parser.add_argument(
        "--model-dir",
        type=str,
        default="models/checkpoints",
        help="Model save directory"
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
    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
        help="Training epochs"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Batch size"
    )
    parser.add_argument(
        "--fetch-data",
        action="store_true",
        help="Force fetch data even if exists"
    )
    parser.add_argument(
        "--use-gpu",
        action="store_true",
        default=True,
        help="Use GPU if available (default: True)"
    )
    parser.add_argument(
        "--use-cpu",
        action="store_true",
        help="Force CPU usage (overrides --use-gpu)"
    )
    parser.add_argument(
        "--lstm-hidden",
        type=int,
        default=None,
        help="LSTM hidden units (default: 256)"
    )
    parser.add_argument(
        "--tmfg-n-features",
        type=int,
        default=None,
        help="Number of TMFG features (default: 25)"
    )
    parser.add_argument(
        "--dropout",
        type=float,
        default=None,
        help="Dropout rate (default: 0.3)"
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=None,
        help="Learning rate (default: 0.0005)"
    )
    
    args = parser.parse_args()
    
    # Determine GPU usage
    use_gpu = args.use_gpu and not args.use_cpu
    
    # Create directories
    os.makedirs(args.data_dir, exist_ok=True)
    os.makedirs(args.model_dir, exist_ok=True)
    
    # Determine tickers to train
    if args.ticker:
        tickers = [args.ticker]
    else:
        # Train for all challenges in config
        tickers = [c['ticker'] for c in config.CHALLENGES]
    
    print("=" * 80)
    print("MANTIS Model Training")
    print("=" * 80)
    print(f"Tickers: {tickers}")
    print(f"Data directory: {args.data_dir}")
    print(f"Model directory: {args.model_dir}")
    print(f"Train end: {args.train_end}")
    print(f"Val end: {args.val_end}")
    print("=" * 80)
    
    # Train each ticker
    all_results = {}
    for ticker in tickers:
        try:
            # First try direct ticker lookup
            challenge = config.CHALLENGE_MAP.get(ticker)
            
            # If not found, try to find challenge by price_key
            if not challenge:
                for ch in config.CHALLENGES:
                    if ch.get('price_key') == ticker:
                        challenge = ch
                        print(f"\n  Note: Found challenge '{challenge['name']}' for price_key '{ticker}'")
                        break
            
            if not challenge:
                print(f"\n✗ Challenge not found for {ticker}")
                print(f"  Available tickers: {list(config.CHALLENGE_MAP.keys())}")
                print(f"  Available price_keys: {[c.get('price_key') for c in config.CHALLENGES if c.get('price_key')]}")
                continue
            
            embedding_dim = challenge['dim']
            # Use price_key for data loading if available, otherwise use ticker
            data_ticker = challenge.get('price_key', ticker)
            challenge_ticker = challenge['ticker']
            
            # Temporarily modify train_ticker_model to save with challenge ticker
            # We'll pass challenge_ticker separately
            # Use command-line hyperparameters if provided, otherwise use defaults
            result = train_ticker_model(
                ticker=data_ticker,  # Use price_key for data loading
                data_dir=args.data_dir,
                model_dir=args.model_dir,
                embedding_dim=embedding_dim,
                train_end=args.train_end,
                val_end=args.val_end,
                epochs=args.epochs,
                batch_size=args.batch_size,
                challenge_ticker=challenge_ticker,  # For model directory naming
                use_gpu=use_gpu,
                lstm_hidden=args.lstm_hidden if args.lstm_hidden else 256,
                tmfg_n_features=args.tmfg_n_features if args.tmfg_n_features else 25,
                dropout=args.dropout if args.dropout else 0.3,
                learning_rate=args.learning_rate if args.learning_rate else 0.0005
            )
            
            # Save result with challenge ticker as key
            all_results[challenge_ticker] = result
            result['challenge_name'] = challenge['name']
            
        except Exception as e:
            print(f"\n✗ Error training {ticker}: {e}")
            import traceback
            traceback.print_exc()
            all_results[ticker] = {'success': False, 'error': str(e)}
    
    # Summary
    print("\n" + "=" * 80)
    print("Training Summary")
    print("=" * 80)
    
    successful = [t for t, r in all_results.items() if r.get('success')]
    failed = [t for t, r in all_results.items() if not r.get('success')]
    
    print(f"Successful: {len(successful)}/{len(tickers)}")
    for ticker in successful:
        result = all_results[ticker]
        if result.get('test_accuracy') is not None:
            acc = result['test_accuracy']
            ci_lower = result.get('ci_lower', 0)
            print(f"  {ticker}: {acc:.4f} (95% CI: [{ci_lower:.4f}, {result.get('ci_upper', 1):.4f}])")
        else:
            print(f"  {ticker}: Model saved (no test data)")
    
    if failed:
        print(f"\nFailed: {len(failed)}")
        for ticker in failed:
            error = all_results[ticker].get('error', 'Unknown error')
            print(f"  {ticker}: {error}")


if __name__ == "__main__":
    main()
