"""
Backtest Accuracy Evaluation
Verify >55% accuracy on held-out test data
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report
from tqdm import tqdm

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from scripts.training.model_architecture import VMDTMFGLSTMXGBoost
from scripts.training.train_model import load_data
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


def evaluate_model(ticker: str, model_path: str, data_dir: str,
                  test_start: str = '2025-01-01',
                  test_end: str = '2025-06-30') -> dict:
    """
    Evaluate model on test set
    
    Args:
        ticker: Ticker symbol
        model_path: Path to trained model
        data_dir: Data directory
        test_start: Test start date
        test_end: Test end date
        
    Returns:
        Dictionary with evaluation results
    """
    print(f"Evaluating {ticker} model...")
    
    # Load model
    challenge = config.CHALLENGE_MAP.get(ticker)
    if not challenge:
        raise ValueError(f"Challenge not found for {ticker}")
    
    embedding_dim = challenge['dim']
    model = VMDTMFGLSTMXGBoost(embedding_dim=embedding_dim)
    model.load(model_path)
    
    # Load data
    ohlcv, funding, oi, cross_exchange = load_data(ticker, data_dir)
    if ohlcv.empty:
        raise ValueError(f"No data available for {ticker}")
    
    # Filter test period
    ohlcv['datetime'] = pd.to_datetime(ohlcv['datetime'])
    test_df = ohlcv[
        (ohlcv['datetime'] >= test_start) & 
        (ohlcv['datetime'] <= test_end)
    ].copy()
    
    if len(test_df) < 100:
        raise ValueError(f"Insufficient test data: {len(test_df)} samples")
    
    print(f"  Test period: {test_start} to {test_end}")
    print(f"  Test samples: {len(test_df)}")
    
    # Prepare test data
    X_test, y_test, _ = model.prepare_data(test_df, funding, oi, cross_exchange)
    
    if len(X_test) == 0:
        raise ValueError("Failed to prepare test data")
    
    # Generate predictions
    print("Generating predictions...")
    y_pred_binary = model.predict_binary(X_test)
    
    # Calculate true binary labels (price increase)
    y_test_binary = (y_test > 0).astype(int)
    
    # Calculate metrics
    accuracy = accuracy_score(y_test_binary, y_pred_binary)
    
    # Get probabilities for AUC
    embeddings = model.predict_embeddings(X_test)
    if embeddings.shape[1] >= 2:
        y_probs = embeddings[:, 1]
    else:
        y_probs = (embeddings[:, 0] + 1) / 2  # Convert [-1, 1] to [0, 1]
    
    auc = roc_auc_score(y_test_binary, y_probs)
    
    # Bootstrap confidence interval
    print("Calculating confidence intervals...")
    ci_lower, ci_upper = bootstrap_confidence_interval(
        y_test_binary, y_pred_binary, n_bootstrap=1000
    )
    
    # Classification report
    report = classification_report(
        y_test_binary, y_pred_binary,
        target_names=['Down', 'Up'],
        output_dict=True
    )
    
    results = {
        'ticker': ticker,
        'accuracy': float(accuracy),
        'auc': float(auc),
        'ci_lower': float(ci_lower),
        'ci_upper': float(ci_upper),
        'n_samples': len(X_test),
        'report': report
    }
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Backtest model accuracy")
    parser.add_argument(
        "--ticker",
        type=str,
        default=None,
        help="Ticker to evaluate (default: all)"
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
        "--test-start",
        type=str,
        default="2025-01-01",
        help="Test start date (YYYY-MM-DD)"
    )
    parser.add_argument(
        "--test-end",
        type=str,
        default="2025-06-30",
        help="Test end date (YYYY-MM-DD)"
    )
    parser.add_argument(
        "--min-accuracy",
        type=float,
        default=0.55,
        help="Minimum required accuracy (default: 0.55)"
    )
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("MANTIS Backtest Accuracy Evaluation")
    print("=" * 80)
    print(f"Test period: {args.test_start} to {args.test_end}")
    print(f"Minimum accuracy: {args.min_accuracy:.2%}")
    print("=" * 80)
    
    # Determine tickers
    if args.ticker:
        tickers = [args.ticker]
    else:
        tickers = [c['ticker'] for c in config.CHALLENGES]
    
    # Evaluate each ticker
    all_results = {}
    for ticker in tickers:
        model_path = os.path.join(args.model_dir, ticker)
        
        if not os.path.exists(model_path):
            print(f"\n✗ Model not found for {ticker} at {model_path}, skipping")
            continue
        
        try:
            result = evaluate_model(
                ticker=ticker,
                model_path=model_path,
                data_dir=args.data_dir,
                test_start=args.test_start,
                test_end=args.test_end
            )
            
            all_results[ticker] = result
            
            # Display results
            print(f"\n{ticker} Results:")
            print(f"  Accuracy: {result['accuracy']:.4f} ({result['accuracy']*100:.2f}%)")
            print(f"  AUC: {result['auc']:.4f}")
            print(f"  95% CI: [{result['ci_lower']:.4f}, {result['ci_upper']:.4f}]")
            print(f"  Samples: {result['n_samples']}")
            
            # Pass/fail
            if result['ci_lower'] >= args.min_accuracy:
                print(f"  ✓ PASS: Lower CI ({result['ci_lower']:.4f}) >= {args.min_accuracy:.4f}")
            else:
                print(f"  ✗ FAIL: Lower CI ({result['ci_lower']:.4f}) < {args.min_accuracy:.4f}")
            
        except Exception as e:
            print(f"\n✗ Error evaluating {ticker}: {e}")
            import traceback
            traceback.print_exc()
    
    # Summary
    print("\n" + "=" * 80)
    print("Summary")
    print("=" * 80)
    
    passed = []
    failed = []
    
    for ticker, result in all_results.items():
        if result['ci_lower'] >= args.min_accuracy:
            passed.append(ticker)
        else:
            failed.append(ticker)
    
    print(f"Passed: {len(passed)}/{len(all_results)}")
    for ticker in passed:
        acc = all_results[ticker]['accuracy']
        ci_lower = all_results[ticker]['ci_lower']
        print(f"  ✓ {ticker}: {acc:.4f} (CI: [{ci_lower:.4f}, {all_results[ticker]['ci_upper']:.4f}])")
    
    if failed:
        print(f"\nFailed: {len(failed)}")
        for ticker in failed:
            acc = all_results[ticker]['accuracy']
            ci_lower = all_results[ticker]['ci_lower']
            print(f"  ✗ {ticker}: {acc:.4f} (CI: [{ci_lower:.4f}, {all_results[ticker]['ci_upper']:.4f}])")


if __name__ == "__main__":
    main()
