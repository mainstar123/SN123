"""
Check Training Results - Step 4 Implementation
Displays training results from saved models and verifies pass/fail criteria
"""

import os
import sys
import argparse
import json
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import accuracy_score, roc_auc_score
from tqdm import tqdm

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from scripts.training.model_architecture import VMDTMFGLSTMXGBoost
from scripts.training.train_model import load_data, prepare_train_val_test_split
import config


def bootstrap_confidence_interval(y_true, y_pred, n_bootstrap=1000, confidence=0.95):
    """
    Calculate bootstrap confidence interval
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        n_bootstrap: Number of bootstrap samples
        confidence: Confidence level (0.95 for 95%)
        
    Returns:
        Tuple of (lower_bound, upper_bound)
    """
    bootstrap_accs = []
    
    for _ in range(n_bootstrap):
        sample_idx = np.random.choice(len(y_true), len(y_true), replace=True)
        acc = accuracy_score(y_true[sample_idx], y_pred[sample_idx])
        bootstrap_accs.append(acc)
    
    alpha = 1 - confidence
    lower = np.percentile(bootstrap_accs, alpha / 2 * 100)
    upper = np.percentile(bootstrap_accs, (1 - alpha / 2) * 100)
    
    return lower, upper


def check_model_results(ticker: str, model_dir: str, data_dir: str,
                       train_end: str = '2023-12-31',
                       val_end: str = '2024-12-31',
                       min_accuracy: float = 0.55) -> dict:
    """
    Check training results for a single model
    
    Args:
        ticker: Ticker symbol (challenge ticker)
        model_dir: Model directory
        data_dir: Data directory
        train_end: Training end date
        val_end: Validation end date
        min_accuracy: Minimum required accuracy
        
    Returns:
        Dictionary with results
    """
    print(f"\n{'='*80}")
    print(f"Checking results for {ticker}")
    print(f"{'='*80}")
    
    model_path = os.path.join(model_dir, ticker)
    
    # Check if model exists
    if not os.path.exists(model_path):
        print(f"✗ Model not found at {model_path}")
        return {'success': False, 'error': 'Model not found'}
    
    # Check required files
    required_files = ['lstm_model.h5', 'xgb_model.json', 'scaler.pkl', 
                     'feature_indices.pkl', 'config.json']
    missing_files = []
    for f in required_files:
        if not os.path.exists(os.path.join(model_path, f)):
            missing_files.append(f)
    
    if missing_files:
        print(f"✗ Missing required files: {', '.join(missing_files)}")
        return {'success': False, 'error': f'Missing files: {missing_files}'}
    
    print("✓ Model files found")
    
    # Load model config
    try:
        with open(os.path.join(model_path, 'config.json'), 'r') as f:
            model_config = json.load(f)
        print(f"✓ Model config loaded")
        print(f"  Embedding dim: {model_config.get('embedding_dim', 'N/A')}")
        print(f"  LSTM hidden: {model_config.get('lstm_hidden', 'N/A')}")
        print(f"  LSTM layers: {model_config.get('lstm_layers', 'N/A')}")
    except Exception as e:
        print(f"⚠ Warning: Could not load config: {e}")
        model_config = {}
    
    # Get challenge info
    challenge = config.CHALLENGE_MAP.get(ticker)
    if not challenge:
        print(f"⚠ Warning: Challenge not found for {ticker}, using default embedding_dim=2")
        embedding_dim = model_config.get('embedding_dim', 2)
    else:
        embedding_dim = challenge['dim']
        print(f"✓ Challenge: {challenge['name']}")
        print(f"  Embedding dimension: {embedding_dim}")
    
    # Load model
    print("Loading model...")
    try:
        model = VMDTMFGLSTMXGBoost(embedding_dim=embedding_dim)
        model.load(model_path)
        print("✓ Model loaded successfully")
    except Exception as e:
        print(f"✗ Failed to load model: {e}")
        return {'success': False, 'error': f'Model load failed: {str(e)}'}
    
    # Load data
    print("Loading data...")
    try:
        # Use price_key if available, otherwise use ticker
        data_ticker = challenge.get('price_key', ticker) if challenge else ticker
        ohlcv, funding, oi, cross_exchange = load_data(data_ticker, data_dir)
        
        if ohlcv.empty:
            print(f"✗ No data available for {data_ticker}")
            return {'success': False, 'error': 'No data'}
        
        print(f"✓ Loaded {len(ohlcv)} records")
        print(f"  Date range: {ohlcv['datetime'].min()} to {ohlcv['datetime'].max()}")
    except Exception as e:
        print(f"✗ Failed to load data: {e}")
        return {'success': False, 'error': f'Data load failed: {str(e)}'}
    
    # Split data
    print("Splitting data...")
    train_df, val_df, test_df = prepare_train_val_test_split(
        ohlcv, train_end=train_end, val_end=val_end
    )
    
    print(f"  Train: {len(train_df)} samples")
    print(f"  Val: {len(val_df)} samples")
    print(f"  Test: {len(test_df)} samples")
    
    if len(test_df) == 0:
        print("⚠ No test data available")
        return {
            'success': True,
            'test_accuracy': None,
            'message': 'No test data available'
        }
    
    # Evaluate on test set
    print("Evaluating on test set...")
    try:
        X_test, y_test, _ = model.prepare_data(test_df, funding, oi, cross_exchange)
        
        if len(X_test) == 0:
            print("✗ Failed to prepare test data")
            return {'success': False, 'error': 'Data preparation failed'}
        
        print(f"  Test sequences: {len(X_test)}")
        
        # Calculate actual price changes for binary labels
        # y_test contains absolute prices, we need to calculate price changes
        test_df_sorted = test_df.sort_values('datetime').reset_index(drop=True)
        if 'close' in test_df_sorted.columns:
            # Calculate forward-looking price change (next price - current price)
            # This matches what the model should predict
            price_changes = test_df_sorted['close'].diff().shift(-1).dropna()
            # Align with sequences (sequences start at index time_steps)
            # Each sequence predicts the price change from the last timestep to the next
            # Get time_steps from model config or default
            time_steps = 20
            if hasattr(model, 'time_steps'):
                time_steps = model.time_steps
            elif 'config' in locals() and model_config:
                time_steps = model_config.get('time_steps', 20)
            # Get price changes corresponding to each sequence
            # Sequence i uses data from [i-time_steps:i] and predicts change at i
            y_test_changes = price_changes.iloc[time_steps-1:time_steps-1+len(X_test)].values
            
            # Create binary labels: 1 if price goes up, 0 if down
            y_test_binary = (y_test_changes > 0).astype(int)
            
            print(f"  Price changes: Up={np.sum(y_test_changes > 0)}, Down={np.sum(y_test_changes < 0)}, No change={np.sum(y_test_changes == 0)}")
        else:
            # Fallback: use y_test > 0 (but this is wrong for prices)
            print(f"  ⚠ Warning: 'close' column not found, using y_test > 0 (may be incorrect)")
            y_test_binary = (y_test > 0).astype(int)
        
        # Generate predictions
        y_pred_binary = model.predict_binary(X_test)
        
        # Diagnostic information
        unique_preds = np.unique(y_pred_binary)
        unique_labels = np.unique(y_test_binary)
        pred_counts = {int(k): int(v) for k, v in zip(*np.unique(y_pred_binary, return_counts=True))}
        label_counts = {int(k): int(v) for k, v in zip(*np.unique(y_test_binary, return_counts=True))}
        
        print(f"  Prediction distribution: {pred_counts}")
        print(f"  Label distribution: {label_counts}")
        
        if len(unique_preds) == 1:
            print(f"  ⚠ Warning: Model predicts only class {unique_preds[0]} for all samples")
            print(f"     This may indicate the model is not learning properly or the embeddings are not diverse")
        if len(unique_labels) == 1:
            print(f"  ⚠ Warning: Test set contains only class {unique_labels[0]}")
            print(f"     This makes evaluation unreliable - test set should have both classes (up/down)")
            print(f"     Consider using a different test period or checking data quality")
        
        # Calculate metrics
        test_acc = accuracy_score(y_test_binary, y_pred_binary)
        
        # Get probabilities for AUC
        test_embeddings = model.predict_embeddings(X_test)
        if test_embeddings.shape[1] >= 2:
            test_probs = test_embeddings[:, 1]
        else:
            test_probs = (test_embeddings[:, 0] + 1) / 2
        
        # Calculate AUC
        try:
            if len(np.unique(y_test_binary)) > 1 and len(np.unique(y_pred_binary)) > 1:
                test_auc = roc_auc_score(y_test_binary, test_probs)
            else:
                test_auc = float('nan')
                print("  Warning: AUC cannot be calculated (all predictions or labels are the same class)")
        except Exception as e:
            test_auc = float('nan')
            print(f"  Warning: AUC calculation failed: {str(e)[:60]}")
        
        # Bootstrap confidence interval
        print("Calculating confidence intervals...")
        ci_lower, ci_upper = bootstrap_confidence_interval(
            y_test_binary, y_pred_binary, n_bootstrap=1000
        )
        
        # Display results
        print(f"\n{'='*80}")
        print("Test Set Results:")
        print(f"{'='*80}")
        print(f"  Accuracy: {test_acc:.4f} ({test_acc*100:.2f}%)")
        print(f"  AUC: {test_auc:.4f}")
        print(f"  95% CI: [{ci_lower:.4f}, {ci_upper:.4f}]")
        print(f"  Test samples: {len(X_test)}")
        
        # Pass/fail
        passed = ci_lower >= min_accuracy
        if passed:
            print(f"  ✓ PASS: Lower CI ({ci_lower:.4f}) >= {min_accuracy:.4f}")
        else:
            print(f"  ✗ FAIL: Lower CI ({ci_lower:.4f}) < {min_accuracy:.4f}")
        
        print(f"{'='*80}")
        
        results = {
            'success': True,
            'ticker': ticker,
            'test_accuracy': float(test_acc),
            'test_auc': float(test_auc),
            'ci_lower': float(ci_lower),
            'ci_upper': float(ci_upper),
            'n_test_samples': len(X_test),
            'passed': passed,
            'min_accuracy': min_accuracy
        }
        
        return results
        
    except Exception as e:
        print(f"✗ Evaluation failed: {e}")
        import traceback
        traceback.print_exc()
        return {'success': False, 'error': f'Evaluation failed: {str(e)}'}


def main():
    parser = argparse.ArgumentParser(
        description="Check training results (Step 4 from Implementation Guide)"
    )
    parser.add_argument(
        "--ticker",
        type=str,
        default=None,
        help="Ticker to check (default: all trained models)"
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
    parser.add_argument(
        "--min-accuracy",
        type=float,
        default=0.55,
        help="Minimum required accuracy (default: 0.55)"
    )
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("MANTIS Training Results Check (Step 4)")
    print("=" * 80)
    print(f"Model directory: {args.model_dir}")
    print(f"Data directory: {args.data_dir}")
    print(f"Minimum accuracy: {args.min_accuracy:.2%}")
    print("=" * 80)
    
    # Determine tickers to check
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
                    print(f"  Attempting to check anyway...")
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
    
    print(f"Found {len(tickers)} trained model(s): {', '.join(tickers)}")
    
    # Check each model
    all_results = {}
    for ticker in tickers:
        try:
            result = check_model_results(
                ticker=ticker,
                model_dir=args.model_dir,
                data_dir=args.data_dir,
                train_end=args.train_end,
                val_end=args.val_end,
                min_accuracy=args.min_accuracy
            )
            all_results[ticker] = result
        except Exception as e:
            print(f"\n✗ Error checking {ticker}: {e}")
            import traceback
            traceback.print_exc()
            all_results[ticker] = {'success': False, 'error': str(e)}
    
    # Summary
    print("\n" + "=" * 80)
    print("Summary")
    print("=" * 80)
    
    successful = [t for t, r in all_results.items() if r.get('success')]
    failed = [t for t, r in all_results.items() if not r.get('success')]
    passed = [t for t in successful if all_results[t].get('passed', False)]
    not_passed = [t for t in successful if not all_results[t].get('passed', False)]
    
    print(f"\nTotal models checked: {len(tickers)}")
    print(f"Successful evaluations: {len(successful)}")
    print(f"Failed evaluations: {len(failed)}")
    
    if passed:
        print(f"\n✓ Passed ({len(passed)}/{len(successful)}):")
        for ticker in passed:
            r = all_results[ticker]
            print(f"  {ticker}: {r['test_accuracy']:.4f} (CI: [{r['ci_lower']:.4f}, {r['ci_upper']:.4f}])")
    
    if not_passed:
        print(f"\n✗ Did not pass ({len(not_passed)}/{len(successful)}):")
        for ticker in not_passed:
            r = all_results[ticker]
            if r.get('test_accuracy') is not None:
                print(f"  {ticker}: {r['test_accuracy']:.4f} (CI: [{r['ci_lower']:.4f}, {r['ci_upper']:.4f}])")
            else:
                print(f"  {ticker}: {r.get('message', 'No test data')}")
    
    if failed:
        print(f"\n✗ Failed to evaluate ({len(failed)}):")
        for ticker in failed:
            error = all_results[ticker].get('error', 'Unknown error')
            print(f"  {ticker}: {error}")


if __name__ == "__main__":
    main()


