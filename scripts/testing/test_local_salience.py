"""
Local Salience Testing Script
Test your model's salience score locally before deploying to mainnet
"""

import os
import sys
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple
import json

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from scripts.training.model_architecture import VMDTMFGLSTMXGBoost
from scripts.training.train_model import load_data, prepare_train_val_test_split
from scripts.training.check_training_results import check_model_results
import config


def generate_diverse_embeddings(model: VMDTMFGLSTMXGBoost, X_test: np.ndarray) -> np.ndarray:
    """
    Generate embeddings and analyze their diversity
    
    Args:
        model: Trained model
        X_test: Test sequences
        
    Returns:
        Embeddings array
    """
    embeddings = model.predict_embeddings(X_test)
    
    # Calculate embedding diversity metrics
    embedding_std = np.std(embeddings, axis=0)
    embedding_range = np.ptp(embeddings, axis=0)  # Peak-to-peak (max - min)
    embedding_mean = np.mean(embeddings, axis=0)
    
    print("\n" + "="*80)
    print("Embedding Diversity Analysis")
    print("="*80)
    print(f"Embedding shape: {embeddings.shape}")
    print(f"Mean embeddings: {embedding_mean}")
    print(f"Std embeddings: {embedding_std}")
    print(f"Range embeddings: {embedding_range}")
    
    # Check for diversity (non-zero variance)
    diversity_score = np.mean(embedding_std) / (np.abs(embedding_mean).mean() + 1e-8)
    print(f"\nDiversity Score: {diversity_score:.4f}")
    print("  (Higher = more diverse embeddings, better for salience)")
    
    # Check for class separation
    if embeddings.shape[1] >= 2:
        # For 2D embeddings, check separation
        embedding_diff = embeddings[:, 1] - embeddings[:, 0]
        separation = np.std(embedding_diff)
        print(f"Class Separation: {separation:.4f}")
        print("  (Higher = better separation between classes)")
    
    return embeddings


def test_salience_metrics(ticker: str, model_dir: str, data_dir: str) -> Dict:
    """
    Test model's potential salience by analyzing:
    1. Embedding diversity
    2. Prediction diversity (not always predicting same class)
    3. Feature importance uniqueness
    
    Args:
        ticker: Ticker symbol
        model_dir: Model directory
        data_dir: Data directory
        
    Returns:
        Dictionary with salience metrics
    """
    print(f"\n{'='*80}")
    print(f"Testing Salience Metrics for {ticker}")
    print(f"{'='*80}")
    
    # Load model
    challenge = config.CHALLENGE_MAP.get(ticker)
    if not challenge:
        print(f"✗ Challenge not found for {ticker}")
        return {}
    
    embedding_dim = challenge['dim']
    
    # Check if model exists
    model_path = os.path.join(model_dir, ticker)
    if not os.path.exists(model_path):
        print(f"✗ Model not found at {model_path}")
        return {}
    
    # Load data
    ohlcv, funding, oi, _ = load_data(ticker, data_dir)
    if ohlcv.empty:
        print(f"✗ No data found for {ticker}")
        return {}
    
    # Split data
    train_df, val_df, test_df = prepare_train_val_test_split(ohlcv)
    
    if test_df.empty:
        print(f"✗ No test data available")
        return {}
    
    # Load model
    from scripts.training.model_architecture import VMDTMFGLSTMXGBoost
    model = VMDTMFGLSTMXGBoost(embedding_dim=embedding_dim)
    model.load(model_path)
    
    # Prepare test data
    X_test, y_test, _ = model.prepare_data(test_df, funding, oi)
    
    if len(X_test) == 0:
        print(f"✗ No test sequences available")
        return {}
    
    # Generate embeddings
    embeddings = generate_diverse_embeddings(model, X_test)
    
    # Analyze predictions
    y_pred_binary = model.predict_binary(X_test)
    y_test_binary = (y_test > 0).astype(int)
    
    # Prediction diversity
    unique_predictions = len(np.unique(y_pred_binary))
    prediction_distribution = {int(k): int(v) for k, v in zip(*np.unique(y_pred_binary, return_counts=True))}
    
    print("\n" + "="*80)
    print("Prediction Diversity Analysis")
    print("="*80)
    print(f"Unique predictions: {unique_predictions} (should be 2 for binary)")
    print(f"Prediction distribution: {prediction_distribution}")
    
    if unique_predictions < 2:
        print("  ⚠ WARNING: Model predicts only one class - LOW SALIENCE!")
        print("  → This means your embeddings are not diverse enough")
        print("  → Validators will give you zero salience")
    else:
        class_balance = min(prediction_distribution.values()) / max(prediction_distribution.values())
        print(f"  Class balance ratio: {class_balance:.4f}")
        print("  (Closer to 1.0 = more balanced, better for salience)")
    
    # Calculate accuracy metrics
    from sklearn.metrics import accuracy_score, roc_auc_score, classification_report
    
    accuracy = accuracy_score(y_test_binary, y_pred_binary)
    
    try:
        if len(np.unique(y_test_binary)) > 1 and len(np.unique(y_pred_binary)) > 1:
            auc = roc_auc_score(y_test_binary, embeddings[:, 1] if embeddings.shape[1] >= 2 else embeddings[:, 0])
        else:
            auc = float('nan')
    except:
        auc = float('nan')
    
    print("\n" + "="*80)
    print("Performance Metrics")
    print("="*80)
    print(f"Accuracy: {accuracy:.4f}")
    print(f"AUC: {auc:.4f}")
    
    if not np.isnan(auc):
        print(f"\nClassification Report:")
        print(classification_report(y_test_binary, y_pred_binary, target_names=['Down', 'Up']))
    
    # Salience score estimate
    salience_score = 0.0
    salience_factors = []
    
    # Factor 1: Embedding diversity (0-0.3)
    diversity_score = np.mean(np.std(embeddings, axis=0)) / (np.abs(np.mean(embeddings)).mean() + 1e-8)
    diversity_factor = min(0.3, diversity_score * 0.1)
    salience_score += diversity_factor
    salience_factors.append(f"Embedding diversity: {diversity_factor:.4f}")
    
    # Factor 2: Prediction diversity (0-0.3)
    if unique_predictions >= 2:
        pred_diversity_factor = 0.3 * class_balance
    else:
        pred_diversity_factor = 0.0
    salience_score += pred_diversity_factor
    salience_factors.append(f"Prediction diversity: {pred_diversity_factor:.4f}")
    
    # Factor 3: AUC performance (0-0.4)
    if not np.isnan(auc):
        auc_factor = 0.4 * max(0, (auc - 0.5) / 0.5)  # Normalize: 0.5 -> 0, 1.0 -> 0.4
    else:
        auc_factor = 0.0
    salience_score += auc_factor
    salience_factors.append(f"AUC performance: {auc_factor:.4f}")
    
    print("\n" + "="*80)
    print("Estimated Salience Score")
    print("="*80)
    print(f"Total Score: {salience_score:.4f} / 1.0")
    print("\nBreakdown:")
    for factor in salience_factors:
        print(f"  {factor}")
    
    if salience_score < 0.3:
        print("\n⚠ LOW SALIENCE - Your model needs improvement!")
        print("  Recommendations:")
        print("  1. Check class imbalance handling")
        print("  2. Increase model capacity")
        print("  3. Add more unique features")
        print("  4. Ensure embeddings are diverse")
    elif salience_score < 0.6:
        print("\n✓ MODERATE SALIENCE - Good but can improve")
    else:
        print("\n✓✓ HIGH SALIENCE - Excellent! You should rank well")
    
    return {
        'ticker': ticker,
        'salience_score': float(salience_score),
        'diversity_score': float(diversity_score),
        'prediction_diversity': unique_predictions,
        'accuracy': float(accuracy),
        'auc': float(auc) if not np.isnan(auc) else None,
        'factors': salience_factors
    }


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Test local salience metrics")
    parser.add_argument("--ticker", type=str, required=True, help="Ticker to test")
    parser.add_argument("--model-dir", type=str, default="models/checkpoints", help="Model directory")
    parser.add_argument("--data-dir", type=str, default="data/raw", help="Data directory")
    parser.add_argument("--output", type=str, default=None, help="Output JSON file")
    
    args = parser.parse_args()
    
    results = test_salience_metrics(args.ticker, args.model_dir, args.data_dir)
    
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()



