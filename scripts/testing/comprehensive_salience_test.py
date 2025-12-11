"""
Comprehensive Local Salience Testing Suite
Tests all aspects needed to take first place:
1. Permutation importance (removal impact)
2. Correlation with other miners
3. Embedding diversity and orthogonality
4. Challenge-specific salience breakdown
"""

import os
import sys
import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from collections import defaultdict
import argparse

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from scripts.training.model_architecture import VMDTMFGLSTMXGBoost
from scripts.training.train_model import load_data, prepare_train_val_test_split
from scripts.training.check_training_results import check_model_results
from ledger import DataLog
from model import multi_salience, salience_binary_prediction
import config


def test_permutation_importance(
    datalog: DataLog,
    your_hotkey: str,
    ticker: str,
    challenge_type: str = 'binary'
) -> Dict:
    """
    Test permutation importance: How much does removing your embeddings hurt the ensemble?
    This is the KEY metric for salience!
    
    Returns:
        Dictionary with removal impact scores
    """
    print(f"\n{'='*80}")
    print(f"Testing Permutation Importance for {ticker}")
    print(f"{'='*80}")
    
    # Get training data
    training_data = datalog.get_training_data_sync()
    
    if ticker not in training_data:
        return {'error': f'No training data for {ticker}'}
    
    hist, prices = training_data[ticker]
    X_flat, hk2idx = hist
    
    if your_hotkey not in hk2idx:
        return {'error': f'Your hotkey {your_hotkey} not found in data'}
    
    your_idx = hk2idx[your_hotkey]
    dim = config.CHALLENGE_MAP[ticker]['dim']
    H = X_flat.shape[1] // dim
    
    if your_idx >= H:
        return {'error': f'Your index {your_idx} >= H {H}'}
    
    # Calculate baseline salience (with your embeddings)
    baseline_salience = multi_salience(training_data)
    baseline_score = baseline_salience.get(your_hotkey, 0.0)
    
    print(f"Baseline salience: {baseline_score:.6f}")
    
    # Remove your embeddings (set to zero)
    X_flat_removed = X_flat.copy()
    start_col = your_idx * dim
    end_col = start_col + dim
    X_flat_removed[:, start_col:end_col] = 0.0
    
    # Recalculate salience without your embeddings
    hist_removed = (X_flat_removed, hk2idx)
    training_data_removed = {ticker: (hist_removed, prices)}
    
    removed_salience = multi_salience(training_data_removed)
    removed_score = removed_salience.get(your_hotkey, 0.0)
    
    # Calculate impact
    impact = baseline_score - removed_score
    
    print(f"Salience after removal: {removed_score:.6f}")
    print(f"Removal impact: {impact:.6f}")
    print(f"Impact ratio: {impact / (baseline_score + 1e-8):.4f}")
    
    # For binary challenges, also test individual AUC impact
    auc_impact = None
    if challenge_type == 'binary':
        try:
            # Get challenge returns
            challenge = config.CHALLENGE_MAP[ticker]
            blocks_ahead = challenge['blocks_ahead']
            
            # Calculate returns
            price_arr = np.asarray(prices, dtype=np.float32)
            if len(price_arr) > blocks_ahead:
                returns = (price_arr[blocks_ahead:] - price_arr[:-blocks_ahead]) / price_arr[:-blocks_ahead]
                
                # Test with and without your embeddings
                baseline_auc = _test_individual_auc(X_flat, hk2idx, returns, your_idx, dim)
                removed_auc = _test_individual_auc(X_flat_removed, hk2idx, returns, your_idx, dim)
                
                auc_impact = baseline_auc - removed_auc
                print(f"Individual AUC impact: {auc_impact:.6f}")
        except Exception as e:
            print(f"Could not calculate AUC impact: {e}")
    
    return {
        'baseline_salience': float(baseline_score),
        'removed_salience': float(removed_score),
        'removal_impact': float(impact),
        'impact_ratio': float(impact / (baseline_score + 1e-8)),
        'auc_impact': float(auc_impact) if auc_impact is not None else None
    }


def _test_individual_auc(X_flat, hk2idx, returns, your_idx, dim):
    """Test individual AUC for a specific miner"""
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    
    # Extract your embeddings
    start_col = your_idx * dim
    end_col = start_col + dim
    your_embeddings = X_flat[:, start_col:end_col]
    
    # Remove zero rows
    non_zero_mask = np.any(your_embeddings != 0, axis=1)
    if non_zero_mask.sum() < 100:
        return 0.5
    
    your_embeddings_clean = your_embeddings[non_zero_mask]
    returns_clean = returns[non_zero_mask]
    
    # Binary classification
    y_binary = (returns_clean > 0).astype(int)
    
    if len(np.unique(y_binary)) < 2:
        return 0.5
    
    # Train simple classifier
    clf = LogisticRegression(max_iter=1000, class_weight='balanced')
    clf.fit(your_embeddings_clean, y_binary)
    
    # Predict
    probs = clf.predict_proba(your_embeddings_clean)[:, 1]
    auc = roc_auc_score(y_binary, probs)
    
    return auc


def test_correlation_with_others(
    datalog: DataLog,
    your_hotkey: str,
    ticker: str
) -> Dict:
    """
    Test correlation with other miners' embeddings.
    Low correlation = unique information = high salience potential
    """
    print(f"\n{'='*80}")
    print(f"Testing Correlation with Other Miners for {ticker}")
    print(f"{'='*80}")
    
    training_data = datalog.get_training_data_sync()
    
    if ticker not in training_data:
        return {'error': f'No training data for {ticker}'}
    
    hist, prices = training_data[ticker]
    X_flat, hk2idx = hist
    
    if your_hotkey not in hk2idx:
        return {'error': f'Your hotkey {your_hotkey} not found'}
    
    your_idx = hk2idx[your_hotkey]
    dim = config.CHALLENGE_MAP[ticker]['dim']
    
    # Extract your embeddings
    start_col = your_idx * dim
    end_col = start_col + dim
    your_embeddings = X_flat[:, start_col:end_col]
    
    # Remove zero rows
    non_zero_mask = np.any(your_embeddings != 0, axis=1)
    your_embeddings_clean = your_embeddings[non_zero_mask]
    
    if len(your_embeddings_clean) < 100:
        return {'error': 'Insufficient data'}
    
    # Calculate correlations with all other miners
    correlations = {}
    max_correlations = []
    
    for other_hotkey, other_idx in hk2idx.items():
        if other_hotkey == your_hotkey or other_idx >= X_flat.shape[1] // dim:
            continue
        
        other_start = other_idx * dim
        other_end = other_start + dim
        other_embeddings = X_flat[:, other_start:other_end]
        other_embeddings_clean = other_embeddings[non_zero_mask]
        
        # Calculate correlation for each dimension
        dim_correlations = []
        for d in range(dim):
            if np.std(your_embeddings_clean[:, d]) > 1e-6 and np.std(other_embeddings_clean[:, d]) > 1e-6:
                corr = np.corrcoef(
                    your_embeddings_clean[:, d],
                    other_embeddings_clean[:, d]
                )[0, 1]
                dim_correlations.append(abs(corr))
        
        if dim_correlations:
            max_corr = max(dim_correlations)
            correlations[other_hotkey] = max_corr
            max_correlations.append(max_corr)
    
    if not correlations:
        return {'error': 'No other miners to compare'}
    
    avg_correlation = np.mean(max_correlations)
    max_correlation = np.max(max_correlations)
    min_correlation = np.min(max_correlations)
    
    # Find most similar miner
    most_similar = max(correlations.items(), key=lambda x: x[1]) if correlations else None
    
    print(f"Average correlation: {avg_correlation:.4f}")
    print(f"Max correlation: {max_correlation:.4f}")
    print(f"Min correlation: {min_correlation:.4f}")
    if most_similar:
        print(f"Most similar miner: {most_similar[0]} (corr: {most_similar[1]:.4f})")
    
    # Orthogonality score (1 - avg_correlation)
    orthogonality = 1.0 - avg_correlation
    print(f"Orthogonality score: {orthogonality:.4f} (higher = more unique)")
    
    return {
        'average_correlation': float(avg_correlation),
        'max_correlation': float(max_correlation),
        'min_correlation': float(min_correlation),
        'orthogonality_score': float(orthogonality),
        'most_similar_miner': most_similar[0] if most_similar else None,
        'most_similar_correlation': float(most_similar[1]) if most_similar else None,
        'n_comparisons': len(correlations)
    }


def test_challenge_breakdown(
    datalog: DataLog,
    your_hotkey: str
) -> Dict:
    """
    Test salience breakdown by challenge type.
    Shows which challenges you're strongest/weakest in.
    """
    print(f"\n{'='*80}")
    print(f"Challenge Breakdown for {your_hotkey}")
    print(f"{'='*80}")
    
    training_data = datalog.get_training_data_sync()
    all_salience = multi_salience(training_data)
    
    your_total = all_salience.get(your_hotkey, 0.0)
    
    # Group by challenge type
    binary_salience = {}
    lbfgs_salience = {}
    hitfirst_salience = {}
    
    for ticker, challenge in config.CHALLENGE_MAP.items():
        if ticker not in training_data:
            continue
        
        challenge_type = challenge.get('loss_func', 'binary')
        weight = challenge.get('weight', 1.0)
        
        # Calculate per-challenge salience (simplified)
        # In reality, salience is computed across all challenges
        # This is an approximation
        
        if challenge_type == 'binary':
            binary_salience[ticker] = {
                'weight': weight,
                'name': challenge['name']
            }
        elif challenge_type == 'lbfgs':
            lbfgs_salience[ticker] = {
                'weight': weight,
                'name': challenge['name']
            }
        elif challenge_type == 'hitfirst':
            hitfirst_salience[ticker] = {
                'weight': weight,
                'name': challenge['name']
            }
    
    print(f"Total salience: {your_total:.6f}")
    print(f"\nBinary challenges: {len(binary_salience)}")
    print(f"LBFGS challenges: {len(lbfgs_salience)}")
    print(f"HitFirst challenges: {len(hitfirst_salience)}")
    
    # Calculate weighted contribution estimate
    total_weight = sum(c['weight'] for c in binary_salience.values()) + \
                   sum(c['weight'] for c in lbfgs_salience.values()) + \
                   sum(c['weight'] for c in hitfirst_salience.values())
    
    print(f"\nTotal challenge weight: {total_weight}")
    print(f"Average salience per unit weight: {your_total / total_weight:.6f}")
    
    return {
        'total_salience': float(your_total),
        'binary_challenges': len(binary_salience),
        'lbfgs_challenges': len(lbfgs_salience),
        'hitfirst_challenges': len(hitfirst_salience),
        'total_weight': float(total_weight),
        'avg_per_weight': float(your_total / total_weight)
    }


def comprehensive_test(
    datalog_path: str,
    your_hotkey: str,
    ticker: Optional[str] = None,
    output_file: Optional[str] = None
) -> Dict:
    """
    Run comprehensive local testing suite
    """
    print("="*80)
    print("COMPREHENSIVE LOCAL SALIENCE TESTING SUITE")
    print("="*80)
    print(f"Your hotkey: {your_hotkey}")
    print(f"Datalog: {datalog_path}")
    print("="*80)
    
    # Load datalog
    from ledger import DataLog
    datalog = DataLog.load(datalog_path)
    
    results = {
        'hotkey': your_hotkey,
        'tests': {}
    }
    
    # Test each challenge
    if ticker:
        tickers_to_test = [ticker]
    else:
        # Test all challenges
        training_data = datalog.get_training_data_sync()
        tickers_to_test = list(training_data.keys())
    
    for ticker in tickers_to_test:
        print(f"\n{'='*80}")
        print(f"Testing {ticker}")
        print(f"{'='*80}")
        
        challenge = config.CHALLENGE_MAP.get(ticker)
        if not challenge:
            continue
        
        challenge_type = challenge.get('loss_func', 'binary')
        
        ticker_results = {}
        
        # 1. Permutation importance
        try:
            perm_results = test_permutation_importance(
                datalog, your_hotkey, ticker, challenge_type
            )
            ticker_results['permutation_importance'] = perm_results
        except Exception as e:
            ticker_results['permutation_importance'] = {'error': str(e)}
        
        # 2. Correlation with others
        try:
            corr_results = test_correlation_with_others(
                datalog, your_hotkey, ticker
            )
            ticker_results['correlation'] = corr_results
        except Exception as e:
            ticker_results['correlation'] = {'error': str(e)}
        
        results['tests'][ticker] = ticker_results
    
    # 3. Challenge breakdown
    try:
        breakdown = test_challenge_breakdown(datalog, your_hotkey)
        results['challenge_breakdown'] = breakdown
    except Exception as e:
        results['challenge_breakdown'] = {'error': str(e)}
    
    # Summary
    print(f"\n{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}")
    
    # Calculate overall scores
    avg_impact = []
    avg_orthogonality = []
    
    for ticker, test_results in results['tests'].items():
        if 'permutation_importance' in test_results:
            impact = test_results['permutation_importance'].get('removal_impact', 0)
            if impact > 0:
                avg_impact.append(impact)
        
        if 'correlation' in test_results:
            ortho = test_results['correlation'].get('orthogonality_score', 0)
            avg_orthogonality.append(ortho)
    
    if avg_impact:
        print(f"Average removal impact: {np.mean(avg_impact):.6f}")
        print(f"  (Higher = more important to ensemble)")
    
    if avg_orthogonality:
        print(f"Average orthogonality: {np.mean(avg_orthogonality):.4f}")
        print(f"  (Higher = more unique, less redundant)")
    
    # Recommendations
    print(f"\n{'='*80}")
    print("RECOMMENDATIONS")
    print(f"{'='*80}")
    
    if avg_impact and np.mean(avg_impact) < 0.0001:
        print("⚠ LOW IMPACT: Your embeddings don't significantly help the ensemble")
        print("  → Improve model quality, add unique features")
    
    if avg_orthogonality and np.mean(avg_orthogonality) < 0.3:
        print("⚠ LOW ORTHOGONALITY: Your embeddings are too similar to others")
        print("  → Add more unique features, use different model architecture")
    
    if avg_impact and avg_orthogonality and np.mean(avg_impact) > 0.001 and np.mean(avg_orthogonality) > 0.5:
        print("✓ EXCELLENT: High impact and high orthogonality!")
        print("  → You should rank very well!")
    
    # Save results
    if output_file:
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to: {output_file}")
    
    return results


def main():
    parser = argparse.ArgumentParser(
        description="Comprehensive local salience testing suite"
    )
    parser.add_argument(
        '--datalog',
        type=str,
        default=os.path.join(config.STORAGE_DIR, 'mantis_datalog.pkl'),
        help='Path to datalog file'
    )
    parser.add_argument(
        '--hotkey',
        type=str,
        required=True,
        help='Your hotkey to test'
    )
    parser.add_argument(
        '--ticker',
        type=str,
        default=None,
        help='Specific ticker to test (default: all)'
    )
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Output JSON file'
    )
    
    args = parser.parse_args()
    
    comprehensive_test(
        args.datalog,
        args.hotkey,
        args.ticker,
        args.output
    )


if __name__ == '__main__':
    main()

